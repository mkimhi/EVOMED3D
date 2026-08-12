# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import math

import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from projects.mmdet3d_plugin.models.utils.misc import locations

@DETECTORS.register_module()
class Petr3D(MVXTwoStageDetector):
    """Petr3D."""

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 num_frame_head_grads=2,
                 num_frame_backbone_grads=2,
                 num_frame_losses=2,
                 stride=16,
                 position_level=0,
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 temporal_memory=None,
                 tm_debug_shapes=False,
                 tm_debug_latency=False,
                 tm_debug_compare_full_path=False,
                 tm_assert_no_teacher_in_inference=True):
        super(Petr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.prev_scene_token = None
        self.num_frame_head_grads = num_frame_head_grads
        self.num_frame_backbone_grads = num_frame_backbone_grads
        self.num_frame_losses = num_frame_losses
        self.single_test = single_test
        self.stride = stride
        self.position_level = position_level
        self.aux_2d_only = aux_2d_only
        self.test_flag = False
        self.temporal_memory = temporal_memory or {}
        self.temporal_memory_enabled = bool(self.temporal_memory.get('enabled', False))
        self.tm_debug_shapes = tm_debug_shapes
        self.tm_debug_latency = tm_debug_latency
        self.tm_debug_compare_full_path = tm_debug_compare_full_path
        self.tm_assert_no_teacher_in_inference = tm_assert_no_teacher_in_inference
        self.temporal_memory_adapter = None
        self.temporal_memory_controller = None
        self.temporal_memory_projector = None
        self.temporal_memory_updater = None
        self.temporal_memory_policy = None
        self.register_buffer(
            '_tm_train_step_buffer', torch.zeros((), dtype=torch.long),
            persistent=True)
        if self.temporal_memory_enabled:
            from projects.mmdet3d_plugin.models.temporal_memory import (
                TemporalMemoryController,
                build_detector_adapter,
            )
            updater = self.temporal_memory.get('updater', 'gated_residual')
            updater_name = (
                updater.get('type', updater.get('name', 'gated_residual'))
                if isinstance(updater, dict) else updater)
            updater_cfg = dict(self.temporal_memory.get('updater_cfg', {}) or {})
            if updater_name == 'gated_residual':
                updater_cfg.setdefault(
                    'channels', self.temporal_memory.get('channels', 256))
                updater_cfg.setdefault('feature_format', 'spatial')
            self.temporal_memory_adapter = build_detector_adapter(
                self.temporal_memory.get('detector_adapter', 'streampetr'),
                debug_shapes=tm_debug_shapes)
            self.temporal_memory_controller = TemporalMemoryController(
                enabled=True,
                keyframe_interval=self.temporal_memory.get(
                    'keyframe_interval', 1),
                updater=updater,
                updater_cfg=updater_cfg,
                policy=self.temporal_memory.get('policy', None),
                partial_layer=self.temporal_memory.get(
                    'partial_layer', 'block12'),
                train_with_teacher=self.temporal_memory.get(
                    'train_with_teacher', False),
                distill_weight=self.temporal_memory.get('distill_weight', 1.0),
                inference_teacher_forbidden=self.temporal_memory.get(
                    'inference_teacher_forbidden',
                    tm_assert_no_teacher_in_inference),
                max_frames_between_full=self.temporal_memory.get(
                    'max_frames_between_full', None),
                adaptive_reset=self.temporal_memory.get(
                    'adaptive_reset', None),
                debug=tm_debug_shapes)
            if isinstance(self.temporal_memory_controller.updater, torch.nn.Module):
                self.temporal_memory_updater = (
                    self.temporal_memory_controller.updater)
            if isinstance(self.temporal_memory_controller.policy, torch.nn.Module):
                self.temporal_memory_policy = (
                    self.temporal_memory_controller.policy)
            # The StreamPETR adapter already projects intermediate EVA
            # features through sim_fpn at the detector feature boundary.
            self.temporal_memory_projector = torch.nn.Identity()
            if self.temporal_memory.get('freeze_pretrained_detector', False):
                self._freeze_temporal_memory_pretrained_detector()

    def _maybe_log_tm_shapes(self, img_feats):
        if not (self.temporal_memory_enabled and self.tm_debug_shapes):
            return
        print(f'[temporal_memory] Petr3D img_feats shape: {tuple(img_feats.shape)}')

    @staticmethod
    def _temporal_memory_sample_key(img_meta, batch_index):
        """Identify one temporal stream without keying on changing frame IDs."""
        scene_token = img_meta.get('scene_token') if isinstance(img_meta, dict) else None
        stream_index = (
            img_meta.get('temporal_stream_index', batch_index)
            if isinstance(img_meta, dict) else batch_index)
        return scene_token, stream_index

    @staticmethod
    def _temporal_memory_sample_prev_exists(prev_exists, batch_index):
        if prev_exists is None:
            return True
        if torch.is_tensor(prev_exists):
            if prev_exists.numel() == 0:
                return True
            if prev_exists.numel() == 1:
                return bool(prev_exists.detach().bool().item())
            return bool(
                prev_exists.reshape(-1)[batch_index].detach().bool().item())
        if isinstance(prev_exists, (list, tuple)):
            if not prev_exists:
                return True
            value = prev_exists[min(batch_index, len(prev_exists) - 1)]
            return Petr3D._temporal_memory_sample_prev_exists(value, 0)
        return bool(prev_exists)

    @staticmethod
    def _temporal_memory_layer_for_action(action, default_layer):
        if action == 'L24':
            return None
        text = str(action).lower()
        if text.startswith('l'):
            return 'block' + text[1:]
        return default_layer

    def _temporal_memory_sample_action(
            self, sample_key, scene_token, prev_exists, policy_features=None):
        controller = self.temporal_memory_controller
        state = controller.states.get(sample_key)
        has_memory = state is not None and state.get('memory') is not None
        scene_changed = state is not None and state.get('scene_token') != scene_token
        is_reset_frame = scene_changed or not prev_exists
        frame_index = 0 if state is None else int(state.get('frame_index', 0))
        action = controller.policy.decide(
            frame_index=frame_index,
            has_memory=has_memory,
            is_reset_frame=is_reset_frame,
            scene_token=scene_token,
            prev_exists=prev_exists,
            policy_features=policy_features)
        max_frames = controller._max_frames_for_policy_features(
            policy_features)
        frames_since_full = 0 if state is None else int(
            state.get('frames_since_full', 0))
        if (
            not has_memory
            or scene_changed
            or not prev_exists
            or (max_frames is not None
                and frames_since_full >= int(max_frames) - 1)
        ):
            return 'L24'
        return action

    @staticmethod
    def _temporal_memory_policy_value(data, name, batch_index):
        value = data.get(name, None)
        if value is None:
            return None
        if torch.is_tensor(value):
            if value.numel() == 1:
                return float(value.detach().item())
            selected = value[batch_index]
            if selected.numel() == 1:
                return float(selected.detach().item())
            return selected
        if isinstance(value, (list, tuple)):
            return value[min(batch_index, len(value) - 1)] if value else None
        return value

    @staticmethod
    def _temporal_memory_box_tensor(boxes):
        if boxes is None:
            return None
        if hasattr(boxes, 'tensor'):
            boxes = boxes.tensor
        if torch.is_tensor(boxes):
            return boxes.detach()
        return None

    @staticmethod
    def _temporal_memory_zero_offline_features():
        names = (
            'num_annotations', 'avg_object_distance_m',
            'median_object_distance_m', 'min_object_distance_m',
            'p10_object_distance_m', 'p90_object_distance_m',
            'avg_box_length_m', 'avg_box_width_m', 'avg_box_height_m',
            'avg_box_volume_m3', 'avg_box_area_m2',
            'avg_object_speed_mps', 'max_object_speed_mps',
            'camera_annotation_count', 'max_camera_annotation_count',
            'min_camera_annotation_count', 'avg_camera_annotation_count',
            'avg_2d_box_area_px', 'max_2d_box_area_px',
            'new_object_proxy_count', 'matched_object_proxy_count',
            'disappeared_object_proxy_count', 'new_object_proxy_fraction',
        )
        features = {name: 0.0 for name in names}
        for distance in (10, 20, 30, 50):
            features[f'near_object_count_{distance}m'] = 0.0
        return features

    @staticmethod
    def _temporal_memory_object_proxy_features(
            previous_boxes, current_boxes, previous_labels=None,
            current_labels=None):
        previous = Petr3D._temporal_memory_box_tensor(previous_boxes)
        current = Petr3D._temporal_memory_box_tensor(current_boxes)
        previous_count = 0 if previous is None else len(previous.reshape(-1, previous.shape[-1]))
        current_count = 0 if current is None else len(current.reshape(-1, current.shape[-1]))
        if previous_count == 0 or current_count == 0:
            return dict(
                new_object_proxy_count=float(current_count),
                matched_object_proxy_count=0.0,
                disappeared_object_proxy_count=float(previous_count),
                new_object_proxy_fraction=float(bool(current_count)),
            )
        previous = previous.reshape(-1, previous.shape[-1]).float().cpu()
        current = current.reshape(-1, current.shape[-1]).float().cpu()
        previous_labels = Petr3D._temporal_memory_box_tensor(previous_labels)
        current_labels = Petr3D._temporal_memory_box_tensor(current_labels)
        if previous_labels is not None:
            previous_labels = previous_labels.reshape(-1).cpu()
        if current_labels is not None:
            current_labels = current_labels.reshape(-1).cpu()
        used = set()
        matched = 0
        for current_index, box in enumerate(current):
            distances = torch.linalg.vector_norm(previous[:, :2] - box[:2], dim=1)
            candidates = [
                index for index in range(previous_count)
                if index not in used and (
                    previous_labels is None or current_labels is None
                    or previous_labels[index] == current_labels[current_index])
            ]
            if not candidates:
                continue
            candidate_distances = distances[candidates]
            best = candidates[int(candidate_distances.argmin())]
            if float(distances[best]) <= 2.0:
                used.add(best)
                matched += 1
        new_count = current_count - matched
        return dict(
            new_object_proxy_count=float(new_count),
            matched_object_proxy_count=float(matched),
            disappeared_object_proxy_count=float(previous_count - matched),
            new_object_proxy_fraction=float(new_count / current_count),
        )

    @staticmethod
    def _temporal_memory_global_boxes(boxes, ego_pose):
        boxes = Petr3D._temporal_memory_box_tensor(boxes)
        pose = Petr3D._temporal_memory_box_tensor(ego_pose)
        if boxes is None or pose is None or pose.numel() < 16:
            return boxes
        transformed = boxes.clone()
        flat = transformed.reshape(-1, transformed.shape[-1])
        if flat.numel() == 0 or flat.shape[1] < 3:
            return transformed
        pose = pose.reshape(-1, 4, 4)[0].to(
            device=flat.device, dtype=flat.dtype)
        homogeneous = torch.cat(
            [flat[:, :3], torch.ones_like(flat[:, :1])], dim=1)
        flat[:, :3] = (homogeneous @ pose.transpose(0, 1))[:, :3]
        return transformed

    @staticmethod
    def _temporal_memory_project_boxes_to_cameras(boxes, img_meta):
        """Project 3D box corners into each camera for deployable 2D stats."""
        boxes = Petr3D._temporal_memory_box_tensor(boxes)
        if boxes is None or boxes.numel() == 0 or boxes.shape[-1] < 6:
            return None
        lidar2img = img_meta.get('lidar2img') if img_meta else None
        image_shapes = None if not img_meta else img_meta.get(
            'img_shape', img_meta.get('pad_shape'))
        if lidar2img is None or image_shapes is None:
            return None
        if not isinstance(lidar2img, (list, tuple)):
            try:
                lidar2img = torch.as_tensor(lidar2img)
            except (TypeError, ValueError):
                return None
            if lidar2img.dim() == 2:
                lidar2img = [lidar2img]
            elif lidar2img.dim() == 3:
                lidar2img = list(lidar2img)
        if not isinstance(lidar2img, (list, tuple)):
            return None
        if (
            isinstance(image_shapes, tuple) and image_shapes
            and isinstance(image_shapes[0], (int, float))
        ):
            image_shapes = [image_shapes] * len(lidar2img)
        elif not isinstance(image_shapes, (list, tuple)):
            return None

        boxes = boxes.reshape(-1, boxes.shape[-1]).float().cpu()
        signs = torch.tensor([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ], dtype=boxes.dtype)
        offsets = signs[None] * boxes[:, None, 3:6].abs() / 2.0
        if boxes.shape[1] >= 7:
            yaw = boxes[:, 6]
            cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
            x = offsets[:, :, 0].clone()
            y = offsets[:, :, 1].clone()
            offsets[:, :, 0] = cos_yaw[:, None] * x - sin_yaw[:, None] * y
            offsets[:, :, 1] = sin_yaw[:, None] * x + cos_yaw[:, None] * y
        corners = offsets + boxes[:, None, :3]
        homogeneous = torch.cat(
            [corners, torch.ones_like(corners[:, :, :1])], dim=2)

        per_camera = []
        for camera_index, matrix in enumerate(lidar2img):
            matrix = torch.as_tensor(matrix, dtype=boxes.dtype).reshape(4, 4)
            projected = homogeneous @ matrix.transpose(0, 1)
            depth = projected[:, :, 2]
            shape = image_shapes[min(camera_index, len(image_shapes) - 1)]
            height, width = float(shape[0]), float(shape[1])
            camera_boxes = []
            for box_index in range(len(boxes)):
                visible = depth[box_index] > 1e-5
                if not bool(visible.any()):
                    continue
                pixels = (
                    projected[box_index, visible, :2]
                    / depth[box_index, visible, None])
                x1 = pixels[:, 0].min().clamp(0.0, width)
                y1 = pixels[:, 1].min().clamp(0.0, height)
                x2 = pixels[:, 0].max().clamp(0.0, width)
                y2 = pixels[:, 1].max().clamp(0.0, height)
                if float(x2 - x1) > 0.0 and float(y2 - y1) > 0.0:
                    camera_boxes.append(torch.stack([x1, y1, x2, y2]))
            per_camera.append(
                torch.stack(camera_boxes) if camera_boxes
                else torch.empty((0, 4), dtype=boxes.dtype))
        return per_camera

    def _temporal_memory_box_statistics(
            self, boxes, previous_boxes=None, camera_boxes=None,
            transition_boxes=None, previous_labels=None,
            current_labels=None):
        features = self._temporal_memory_zero_offline_features()
        boxes = self._temporal_memory_box_tensor(boxes)
        if boxes is not None and boxes.numel() > 0:
            boxes = boxes.reshape(-1, boxes.shape[-1]).float().cpu()
            distances = torch.linalg.vector_norm(boxes[:, :2], dim=1)
            features.update(
                num_annotations=float(len(boxes)),
                avg_object_distance_m=float(distances.mean()),
                median_object_distance_m=float(distances.median()),
                min_object_distance_m=float(distances.min()),
                p10_object_distance_m=float(torch.quantile(distances, 0.1)),
                p90_object_distance_m=float(torch.quantile(distances, 0.9)),
            )
            for distance in (10, 20, 30, 50):
                features[f'near_object_count_{distance}m'] = float(
                    (distances < distance).sum())
            if boxes.shape[1] >= 6:
                dims = boxes[:, 3:6].abs()
                features.update(
                    avg_box_length_m=float(dims[:, 0].mean()),
                    avg_box_width_m=float(dims[:, 1].mean()),
                    avg_box_height_m=float(dims[:, 2].mean()),
                    avg_box_volume_m3=float(dims.prod(dim=1).mean()),
                    avg_box_area_m2=float((dims[:, 0] * dims[:, 1]).mean()),
                )
            if boxes.shape[1] >= 9:
                speeds = torch.linalg.vector_norm(boxes[:, 7:9], dim=1)
                features.update(
                    avg_object_speed_mps=float(speeds.mean()),
                    max_object_speed_mps=float(speeds.max()),
                )
        if isinstance(camera_boxes, (list, tuple)):
            counts = []
            areas = []
            for item in camera_boxes:
                item = self._temporal_memory_box_tensor(item)
                if item is None or item.numel() == 0:
                    counts.append(0)
                    continue
                item = item.reshape(-1, item.shape[-1]).float()
                counts.append(len(item))
                if item.shape[1] >= 4:
                    areas.extend(((item[:, 2] - item[:, 0]).abs()
                                  * (item[:, 3] - item[:, 1]).abs()).tolist())
            if counts:
                features.update(
                    camera_annotation_count=float(sum(counts)),
                    max_camera_annotation_count=float(max(counts)),
                    min_camera_annotation_count=float(min(counts)),
                    avg_camera_annotation_count=float(sum(counts) / len(counts)),
                )
            if areas:
                features.update(
                    avg_2d_box_area_px=float(sum(areas) / len(areas)),
                    max_2d_box_area_px=float(max(areas)),
                )
        features.update(self._temporal_memory_object_proxy_features(
            previous_boxes,
            boxes if transition_boxes is None else transition_boxes,
            previous_labels=previous_labels,
            current_labels=current_labels))
        return features

    def _temporal_memory_offline_features(
            self, img_meta, policy_data, previous_boxes=None,
            previous_labels=None):
        boxes = self._temporal_memory_box_tensor(
            policy_data.get('gt_bboxes_3d', img_meta.get('gt_bboxes_3d')))
        labels = self._temporal_memory_box_tensor(
            policy_data.get('gt_labels_3d', img_meta.get('gt_labels_3d')))
        ego_pose = policy_data.get('ego_pose', img_meta.get('ego_pose'))
        projected_camera_boxes = self._temporal_memory_project_boxes_to_cameras(
            boxes, img_meta)
        return self._temporal_memory_box_statistics(
            boxes,
            previous_boxes=previous_boxes,
            camera_boxes=(
                projected_camera_boxes if projected_camera_boxes is not None
                else policy_data.get('gt_bboxes')),
            transition_boxes=self._temporal_memory_global_boxes(
                boxes, ego_pose),
            previous_labels=previous_labels,
            current_labels=labels)

    @staticmethod
    def _temporal_memory_pose(policy_data, img_meta=None):
        img_meta = img_meta or {}
        pose = Petr3D._temporal_memory_box_tensor(
            policy_data.get('ego_pose', img_meta.get('ego_pose')))
        if pose is None or pose.numel() < 16:
            return 0.0, 0.0, 0.0, 0.0
        pose = pose.reshape(-1, 4, 4)[0].float().cpu()
        return (
            float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3]),
            math.atan2(float(pose[1, 0]), float(pose[0, 0])))

    @staticmethod
    def _temporal_memory_yaw_delta(previous, current):
        return (current - previous + math.pi) % (2.0 * math.pi) - math.pi

    def _temporal_memory_policy_features(
            self, img_meta, policy_data, state, frame_number, training):
        features = {
            key: value for key, value in img_meta.items()
            if isinstance(value, (bool, int, float, str))
        }
        features['frame_number'] = float(frame_number)
        features.setdefault('info_frame_idx', float(
            img_meta.get('frame_idx', frame_number)))
        features['is_scene_start'] = float(not state or state.get('memory') is None)
        timestamp = self._temporal_memory_policy_value(
            policy_data, 'timestamp', 0)
        timestamp = float(img_meta.get('timestamp', 0.0)
                          if timestamp is None else timestamp)
        x, y, z, yaw = self._temporal_memory_pose(policy_data, img_meta)
        previous = state.get('policy_state') or {}
        dt = delta = speed = accel = yaw_rate = yaw_accel = 0.0
        if previous:
            raw_dt = timestamp - float(previous.get('timestamp', timestamp))
            dt = raw_dt / 1e6 if abs(raw_dt) > 1000.0 else raw_dt
            if dt > 0.0:
                delta = math.hypot(
                    x - float(previous.get('ego_x_m', x)),
                    y - float(previous.get('ego_y_m', y)))
                speed = delta / dt
                yaw_rate = math.degrees(self._temporal_memory_yaw_delta(
                    float(previous.get('ego_yaw_rad', yaw)), yaw)) / dt
                accel = (speed - float(
                    previous.get('ego_speed_mps', 0.0))) / dt
                yaw_accel = (yaw_rate - float(
                    previous.get('yaw_rate_deg_s', 0.0))) / dt
        features.update(
            dt_s=float(dt), ego_x_m=x, ego_y_m=y, ego_z_m=z,
            ego_yaw_deg=float(math.degrees(yaw)),
            delta_ego_translation_m=float(delta), ego_speed_mps=float(speed),
            ego_accel_mps2=float(accel), yaw_rate_deg_s=float(yaw_rate),
            abs_yaw_rate_deg_s=float(abs(yaw_rate)),
            yaw_accel_deg_s2=float(yaw_accel),
            abs_yaw_accel_deg_s2=float(abs(yaw_accel)),
        )
        if 'ego_pose' not in policy_data and 'ego_speed_mps' in img_meta:
            features['ego_speed_mps'] = float(img_meta['ego_speed_mps'])
        current_offline = self._temporal_memory_offline_features(
            img_meta, policy_data, previous_boxes=state.get('gt_boxes'),
            previous_labels=state.get('gt_labels'))
        mode = self.temporal_memory.get('policy_offline_features', None)
        if mode == 'teacher_forcing':
            source = (
                state.get('offline_gt_features') if training
                else state.get('prediction_features'))
            features.update(source or self._temporal_memory_zero_offline_features())
        elif mode == 'current_gt':
            features.update(current_offline)
        features['_current_offline_features'] = current_offline
        features['_current_policy_state'] = dict(
            timestamp=timestamp, ego_x_m=x, ego_y_m=y, ego_yaw_rad=yaw,
            ego_speed_mps=features['ego_speed_mps'],
            yaw_rate_deg_s=features['yaw_rate_deg_s'])
        current_gt_boxes = self._temporal_memory_box_tensor(
            policy_data.get('gt_bboxes_3d', img_meta.get('gt_bboxes_3d')))
        features['_current_gt_boxes'] = self._temporal_memory_global_boxes(
            current_gt_boxes,
            policy_data.get('ego_pose', img_meta.get('ego_pose')))
        features['_current_gt_labels'] = self._temporal_memory_box_tensor(
            policy_data.get('gt_labels_3d', img_meta.get('gt_labels_3d')))
        return features

    def _freeze_temporal_memory_pretrained_detector(self):
        for param in self.parameters():
            param.requires_grad = False
        for module in (
            self.temporal_memory_projector,
            self.temporal_memory_updater,
            self.temporal_memory_policy,
        ):
            if not isinstance(module, torch.nn.Module):
                continue
            for param in module.parameters():
                param.requires_grad = True

    def _temporal_memory_prediction_state(
            self, result, previous_boxes=None, previous_labels=None,
            ego_pose=None, img_meta=None):
        if isinstance(result, dict) and 'pts_bbox' in result:
            result = result['pts_bbox']
        boxes = None if not isinstance(result, dict) else result.get(
            'boxes_3d', result.get('bboxes_3d'))
        boxes = self._temporal_memory_box_tensor(boxes)
        scores = None if not isinstance(result, dict) else result.get(
            'scores_3d')
        scores = self._temporal_memory_box_tensor(scores)
        labels = None if not isinstance(result, dict) else result.get(
            'labels_3d')
        labels = self._temporal_memory_box_tensor(labels)
        if (
            boxes is not None and scores is not None
            and scores.numel() == len(boxes)
        ):
            threshold = float(self.temporal_memory.get(
                'policy_prediction_score_thr', 0.1))
            keep = scores.reshape(-1).to(boxes.device) >= threshold
            boxes = boxes[keep]
            scores = scores.reshape(-1)[keep.to(scores.device)]
            if labels is not None and labels.numel() == len(keep):
                labels = labels.reshape(-1)[keep.to(labels.device)]
        camera_boxes = None if not isinstance(result, dict) else result.get(
            'bboxes_2d', result.get('boxes_2d'))
        projected_camera_boxes = self._temporal_memory_project_boxes_to_cameras(
            boxes, img_meta or {})
        if projected_camera_boxes is not None:
            camera_boxes = projected_camera_boxes
        transition_boxes = self._temporal_memory_global_boxes(boxes, ego_pose)
        return dict(
            features=self._temporal_memory_box_statistics(
                boxes, previous_boxes=previous_boxes,
                camera_boxes=camera_boxes,
                transition_boxes=transition_boxes,
                previous_labels=previous_labels,
                current_labels=labels),
            boxes=transition_boxes,
            scores=scores,
            labels=labels,
            camera_boxes=camera_boxes,
        )

    def _temporal_memory_prediction_features(self, result):
        return self._temporal_memory_prediction_state(result)['features']

    def _temporal_memory_record_predictions(
            self, bbox_results, img_metas, policy_data=None):
        controller = self.temporal_memory_controller
        if controller is None:
            return
        for index, (result, img_meta) in enumerate(zip(bbox_results, img_metas)):
            sample_key = self._temporal_memory_sample_key(img_meta, index)
            state = controller.states.get(sample_key)
            if state is not None:
                ego_pose = img_meta.get('ego_pose')
                if ego_pose is None and policy_data is not None:
                    ego_pose = self._temporal_memory_policy_value(
                        policy_data, 'ego_pose', index)
                prediction = self._temporal_memory_prediction_state(
                    result,
                    previous_boxes=state.get('prediction_boxes'),
                    previous_labels=state.get('prediction_labels'),
                    ego_pose=ego_pose,
                    img_meta=img_meta)
                state['prediction_features'] = prediction['features']
                state['prediction_boxes'] = prediction['boxes']
                state['prediction_scores'] = prediction['scores']
                state['prediction_labels'] = prediction['labels']
                state['prediction_camera_boxes'] = prediction['camera_boxes']

    def _temporal_memory_policy_supervision_loss(
            self, policy_features, previous_teacher, current_teacher):
        policy = self.temporal_memory_policy
        if not isinstance(policy, torch.nn.Module) or not hasattr(policy, 'logits'):
            return None
        similarity = F.cosine_similarity(
            previous_teacher.float().flatten(1),
            current_teacher.float().flatten(1), dim=1).mean()
        spec = self.temporal_memory.get('policy_supervision', {}) or {}
        thresholds = tuple(spec.get('thresholds', (0.85, 0.75, 0.65)))
        if len(thresholds) != 3:
            raise ValueError('policy_supervision.thresholds must contain three values')
        value = float(similarity.detach())
        target_index = (
            0 if value >= thresholds[0] else
            1 if value >= thresholds[1] else
            2 if value >= thresholds[2] else 3)
        policy_inputs = dict(policy_features)
        for key in list(policy_inputs):
            if key.startswith('_'):
                policy_inputs.pop(key)
        feature_noise_std = float(spec.get('feature_noise_std', 0.0) or 0.0)
        if feature_noise_std > 0.0:
            scale = policy.scale.detach().float().cpu()
            for index, name in enumerate(policy.feature_names):
                policy_inputs[name] = float(policy_inputs.get(name, 0.0) or 0.0) + float(
                    torch.randn(()) * feature_noise_std * scale[index])
        logits = policy.logits(policy_inputs)
        target = torch.as_tensor(
            [target_index], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, target, reduction='none')[0] * float(
            spec.get('weight', 1.0))
        class_weights = spec.get('class_weights', None)
        sample_weight = 1.0
        if class_weights is not None:
            sample_weight = float(class_weights[target_index])
        return loss, sample_weight

    @staticmethod
    def _temporal_memory_aggregate_policy_losses(weighted_losses):
        numerator = sum(loss * weight for loss, weight in weighted_losses)
        return numerator / max(len(weighted_losses), 1)

    def _temporal_memory_policy_supervision_active(self, training):
        if not training:
            return False
        spec = self.temporal_memory.get('policy_supervision', {}) or {}
        if not spec or spec.get('enabled', True) is False:
            return False
        counter = getattr(self, '_tm_train_step_buffer', None)
        step = (
            int(counter.item()) if counter is not None
            else int(getattr(self, '_tm_train_step', 0)))
        max_steps = int(spec.get('max_train_steps', 0) or 0)
        if 'start_train_step' in spec:
            start = int(spec.get('start_train_step', 0) or 0)
            return step >= start and (max_steps <= 0 or step < max_steps)
        if max_steps <= 0:
            return True
        warmup_steps = max(
            1, int(max_steps * float(spec.get('warmup_ratio', 0.1))))
        return step < warmup_steps

    def _temporal_memory_parameter_anchor_loss(self):
        anchor = None
        for module in (
            getattr(self, 'temporal_memory_projector', None),
            getattr(self, 'temporal_memory_updater', None),
            getattr(self, 'temporal_memory_policy', None),
        ):
            if not isinstance(module, torch.nn.Module):
                continue
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                value = param.float().sum() * 0.0
                anchor = value if anchor is None else anchor + value
        return anchor

    def _temporal_memory_evict_stale_stream(self, sample_key):
        """Drop the previous scene's cache for the same batch stream slot."""
        _, stream_index = sample_key
        stale_keys = [
            key for key in self.temporal_memory_controller.states
            if key != sample_key
            and isinstance(key, tuple)
            and len(key) == 2
            and key[1] == stream_index
        ]
        for key in stale_keys:
            del self.temporal_memory_controller.states[key]

    def _extract_img_feat_temporal_memory(
            self, img, img_metas, training=False, prev_exists=None,
            teacher_features=None, policy_data=None):
        """Route one StreamPETR frame through dense pre-PE temporal memory."""
        if not self.temporal_memory_enabled:
            features = self.extract_img_feat(
                img, 1, training_mode=training)
            if features.dim() == 6 and features.size(1) == 1:
                features = features[:, 0]
            return features, {}

        batch_size = img.size(0)
        if len(img_metas) != batch_size:
            raise ValueError(
                'StreamPETR temporal routing requires one metadata dict per sample.')

        policy_data = policy_data or {}
        routes = []
        actions = []
        for batch_index, img_meta in enumerate(img_metas):
            scene_token = img_meta.get('scene_token')
            sample_key = self._temporal_memory_sample_key(
                img_meta, batch_index)
            self._temporal_memory_evict_stale_stream(sample_key)
            sample_prev_exists = self._temporal_memory_sample_prev_exists(
                prev_exists, batch_index)
            state = self.temporal_memory_controller.states.get(
                sample_key, {})
            if state and (
                state.get('scene_token') != scene_token
                or not sample_prev_exists
            ):
                state.clear()
                state.update(
                    scene_token=scene_token, frame_index=0,
                    frames_since_full=0, memory=None)
            sample_policy_data = {
                name: self._temporal_memory_policy_value(
                    policy_data, name, batch_index)
                for name in policy_data
            }
            policy_features = self._temporal_memory_policy_features(
                img_meta,
                sample_policy_data,
                state,
                frame_number=int(state.get('frame_index', 0)),
                training=training)
            action = self._temporal_memory_sample_action(
                sample_key, scene_token, sample_prev_exists,
                policy_features=policy_features)
            actions.append(action)
            routes.append(dict(
                batch_index=batch_index,
                sample_key=sample_key,
                scene_token=scene_token,
                prev_exists=sample_prev_exists,
                action=action,
                policy_features=policy_features,
                previous_full_teacher=state.get('full_teacher'),
            ))

        def select_batch(indices):
            index = torch.as_tensor(
                indices, dtype=torch.long, device=img.device)
            return {'img': img.index_select(0, index)}

        full_indices = [
            route['batch_index'] for route in routes
            if route['action'] == 'L24'
        ]
        full_positions = {
            batch_index: position
            for position, batch_index in enumerate(full_indices)
        }
        full_group = []

        def extract_full_group():
            if not full_group:
                full_group.append(
                    self.temporal_memory_adapter.extract_full_features(
                        self, select_batch(full_indices)))
            return full_group[0]

        partial_groups = {}
        partial_positions = {}
        updater_name = self.temporal_memory_controller.updater_name
        for route in routes:
            if route['action'] == 'L24':
                continue
            if updater_name == 'reuse':
                continue
            layer_id = self._temporal_memory_layer_for_action(
                route['action'],
                self.temporal_memory.get('partial_layer', 'block12'))
            partial_groups.setdefault(layer_id, []).append(
                route['batch_index'])
        for layer_id, indices in list(partial_groups.items()):
            partial_positions[layer_id] = {
                batch_index: position
                for position, batch_index in enumerate(indices)
            }
            partial_groups[layer_id] = (
                self.temporal_memory_adapter.
                extract_projected_partial_features(
                    self, select_batch(indices), layer_id))

        requests = []
        for route in routes:
            batch_index = route['batch_index']
            request = dict(
                sample_key=route['sample_key'],
                scene_token=route['scene_token'],
                prev_exists=route['prev_exists'],
                action=route['action'],
            )
            if route['action'] == 'L24':
                position = full_positions[batch_index]
                request['extract_full'] = (
                    lambda index=position: extract_full_group()[
                        index:index + 1])
            elif updater_name != 'reuse':
                layer_id = self._temporal_memory_layer_for_action(
                    route['action'],
                    self.temporal_memory.get('partial_layer', 'block12'))
                position = partial_positions[layer_id][batch_index]
                request['partial_current'] = partial_groups[layer_id][
                    position:position + 1]
            requests.append(request)

        if (
            not training
            and teacher_features is not None
            and any(action != 'L24' for action in actions)
            and self.temporal_memory_controller.inference_teacher_forbidden
        ):
            raise AssertionError(
                'Full-feature teacher is forbidden on non-keyframe inference.')

        policy_spec = self.temporal_memory.get('policy_supervision', {}) or {}
        policy_supervision = bool(
            self._temporal_memory_policy_supervision_active(training)
            and policy_spec.get('target', 'similarity') in (
                'similarity', 'similarity_L24_mean'))
        policy_teacher = None
        if policy_supervision:
            with torch.no_grad():
                policy_teacher = (
                    self.temporal_memory_adapter.extract_full_features(
                        self, {'img': img})).detach()

        outputs = self.temporal_memory_controller.get_or_update_samples(
            requests, training=training)
        policy_losses = []
        for route in routes:
            state = self.temporal_memory_controller.states[route['sample_key']]
            state['offline_gt_features'] = route['policy_features'].get(
                '_current_offline_features', {})
            state['policy_state'] = route['policy_features'].get(
                '_current_policy_state', {})
            state['gt_boxes'] = route['policy_features'].get(
                '_current_gt_boxes')
            state['gt_labels'] = route['policy_features'].get(
                '_current_gt_labels')
            if policy_teacher is not None:
                batch_index = route['batch_index']
                previous_teacher = route['previous_full_teacher']
                current_teacher = policy_teacher[batch_index:batch_index + 1]
                if previous_teacher is not None:
                    policy_loss = self._temporal_memory_policy_supervision_loss(
                        route['policy_features'], previous_teacher,
                        current_teacher)
                    if policy_loss is not None:
                        policy_losses.append(policy_loss)
                state['full_teacher'] = current_teacher.detach()
        features = torch.cat(outputs, dim=0)
        losses = {}
        if policy_losses:
            losses['loss_tm_policy'] = (
                self._temporal_memory_aggregate_policy_losses(policy_losses))
        if (
            training
            and self.temporal_memory_controller.train_with_teacher
            and any(action != 'L24' for action in actions)
        ):
            teacher_indices = [
                index for index, action in enumerate(actions)
                if action != 'L24'
            ]
            if teacher_features is None:
                with torch.no_grad():
                    teacher_features = (
                        self.temporal_memory_adapter.extract_full_features(
                            self, select_batch(teacher_indices)))
            else:
                index = torch.as_tensor(
                    teacher_indices, dtype=torch.long,
                    device=teacher_features.device)
                teacher_features = teacher_features.index_select(0, index)
            teacher_features = teacher_features.detach()
            distill_losses = [
                self.temporal_memory_controller.distillation_loss(
                    outputs[batch_index],
                    teacher_features[position:position + 1])
                for position, batch_index in enumerate(teacher_indices)
            ]
            losses['loss_tm_distill'] = (
                sum(distill_losses) / len(distill_losses))
        return features, losses


    def extract_img_feat(self, img, len_queue=1, training_mode=False):
        """Extract features of images."""
        B = img.size(0)

        if img is not None:
            if img.dim() == 6:
                img = img.flatten(1, 2)
            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        BN, C, H, W = img_feats[self.position_level].size()
        if self.training or training_mode:
            img_feats_reshaped = img_feats[self.position_level].view(B, len_queue, int(BN/B / len_queue), C, H, W)
        else:
            img_feats_reshaped = img_feats[self.position_level].view(B, int(BN/B/len_queue), C, H, W)


        self._maybe_log_tm_shapes(img_feats_reshaped)
        return img_feats_reshaped

    def extract_img_feat_intermediate(self, img, layer_id, len_queue=1):
        """Extract one EVA block feature through the detector's projection."""
        B = img.size(0)
        if img.dim() == 6:
            img = img.flatten(1, 2)
        if img.dim() == 5 and img.size(0) == 1:
            img = img.squeeze(0)
        elif img.dim() == 5 and img.size(0) > 1:
            _, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)

        feature = self.img_backbone.forward_intermediate(img, layer_id)
        sim_fpn = getattr(self.img_backbone, 'adapter', None)
        if sim_fpn is None:
            raise ValueError('StreamPETR intermediate EVA extraction requires sim_fpn.')
        img_feats = sim_fpn(feature)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        feature = img_feats[self.position_level]
        BN, C, H, W = feature.size()
        num_cameras = int(BN / B / len_queue)
        if len_queue == 1:
            feature = feature.view(B, num_cameras, C, H, W)
        else:
            feature = feature.view(B, len_queue, num_cameras, C, H, W)
        self._maybe_log_tm_shapes(feature)
        return feature


    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, T, training_mode=False):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, T, training_mode)
        return img_feats

    def obtain_history_memory(self,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_bboxes=None,
                            gt_labels=None,
                            img_metas=None,
                            centers2d=None,
                            depths=None,
                            gt_bboxes_ignore=None,
                            **data):
        losses = dict()
        T = data['img'].size(1)
        num_nograd_frames = T - self.num_frame_head_grads
        num_grad_losses = T - self.num_frame_losses
        for i in range(T):
            requires_grad = False
            return_losses = False
            data_t = dict()
            for key in data:
                data_t[key] = data[key][:, i] 

            data_t['img_feats'] = data_t['img_feats']
            if i >= num_nograd_frames:
                requires_grad = True
            if i >= num_grad_losses:
                return_losses = True
            loss = self.forward_pts_train(gt_bboxes_3d[i],
                                        gt_labels_3d[i], gt_bboxes[i],
                                        gt_labels[i], img_metas[i], centers2d[i], depths[i], requires_grad=requires_grad,return_losses=return_losses,**data_t)
            if loss is not None:
                for key, value in loss.items():
                    losses['frame_'+str(i)+"_"+key] = value
        return losses


    def prepare_location(self, img_metas, **data):
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def forward_roi_head(self, location, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(location, **data)
            return outs_roi


    def forward_pts_train(self,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          gt_bboxes,
                          gt_labels,
                          img_metas,
                          centers2d,
                          depths,
                          requires_grad=True,
                          return_losses=False,
                          **data):
        """Forward function for point cloud branch.
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
        Returns:
            dict: Losses of each branch.
        """
        location = self.prepare_location(img_metas, **data)

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(location, img_metas, None, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(location, **data)
            topk_indexes = outs_roi['topk_indexes']
            outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas]
                losses2d = self.img_roi_head.loss(*loss2d_inputs)
                losses.update(losses2d) 

            return losses
        else:
            return None

    @force_fp32(apply_to=('img'))
    def forward(self, return_loss=True, **data):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        if return_loss:
            for key in ['gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'img_metas']:
                data[key] = list(zip(*data[key]))
            return self.forward_train(**data)
        else:
            return self.forward_test(**data)

    def forward_train(self,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      gt_bboxes_ignore=None,
                      depths=None,
                      centers2d=None,
                      **data):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        if self.test_flag: #for interval evaluation
            self.pts_bbox_head.reset_memory()
            self.test_flag = False

        T = data['img'].size(1)

        tm_losses = []
        tm_policy_losses = []
        if self.temporal_memory_enabled:
            frame_features = []
            num_nograd_frames = T - self.num_frame_backbone_grads
            for i in range(T):
                frame_prev_exists = data.get('prev_exists', None)
                if frame_prev_exists is not None:
                    frame_prev_exists = frame_prev_exists[:, i]
                frame_policy_data = {
                    key: value[:, i]
                    for key, value in data.items()
                    if key != 'img' and torch.is_tensor(value)
                    and value.dim() >= 2
                }
                if gt_bboxes_3d is not None:
                    frame_policy_data['gt_bboxes_3d'] = gt_bboxes_3d[i]
                if gt_labels_3d is not None:
                    frame_policy_data['gt_labels_3d'] = gt_labels_3d[i]
                if gt_bboxes is not None:
                    frame_policy_data['gt_bboxes'] = gt_bboxes[i]
                if i < num_nograd_frames:
                    self.eval()
                    with torch.no_grad():
                        features, frame_tm_losses = (
                            self._extract_img_feat_temporal_memory(
                                data['img'][:, i], img_metas[i],
                                training=True,
                                prev_exists=frame_prev_exists,
                                policy_data=frame_policy_data))
                    self.train()
                else:
                    features, frame_tm_losses = (
                        self._extract_img_feat_temporal_memory(
                            data['img'][:, i], img_metas[i],
                            training=True,
                            prev_exists=frame_prev_exists,
                            policy_data=frame_policy_data))
                frame_features.append(features)
                tm_loss = frame_tm_losses.get('loss_tm_distill')
                if tm_loss is not None and tm_loss.requires_grad:
                    tm_losses.append(tm_loss)
                tm_policy_loss = frame_tm_losses.get('loss_tm_policy')
                if tm_policy_loss is not None and tm_policy_loss.requires_grad:
                    tm_policy_losses.append(tm_policy_loss)
            data['img_feats'] = torch.stack(frame_features, dim=1)
        else:
            prev_img = data['img'][:, :-self.num_frame_backbone_grads]
            rec_img = data['img'][:, -self.num_frame_backbone_grads:]
            rec_img_feats = self.extract_feat(
                rec_img, self.num_frame_backbone_grads)

            if T-self.num_frame_backbone_grads > 0:
                self.eval()
                with torch.no_grad():
                    prev_img_feats = self.extract_feat(
                        prev_img, T-self.num_frame_backbone_grads, True)
                self.train()
                data['img_feats'] = torch.cat(
                    [prev_img_feats, rec_img_feats], dim=1)
            else:
                data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)

        if tm_losses:
            losses['loss_tm_distill'] = sum(tm_losses) / len(tm_losses)
        if tm_policy_losses:
            losses['loss_tm_policy'] = (
                sum(tm_policy_losses) / len(tm_policy_losses))
        if self.temporal_memory_enabled:
            anchor = self._temporal_memory_parameter_anchor_loss()
            if anchor is not None:
                losses['loss_tm_param_anchor'] = anchor
            counter = getattr(self, '_tm_train_step_buffer', None)
            if counter is not None:
                counter.add_(1)
            else:
                self._tm_train_step = int(
                    getattr(self, '_tm_train_step', 0)) + 1

        return losses
  
  
    def forward_test(self, img_metas, rescale, **data):
        self.test_flag = True
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        for key in data:
            if key != 'img':
                data[key] = data[key][0][0].unsqueeze(0)
            else:
                data[key] = data[key][0]
        return self.simple_test(img_metas[0], **data)

    def simple_test_pts(self, img_metas, **data):
        """Test function of point cloud branch."""
        location = self.prepare_location(img_metas, **data)
        outs_roi = self.forward_roi_head(location, **data)
        topk_indexes = outs_roi['topk_indexes']

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs = self.pts_bbox_head(location, img_metas, topk_indexes, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        if self.temporal_memory_enabled:
            data['img_feats'], _ = self._extract_img_feat_temporal_memory(
                data['img'], img_metas, training=False,
                prev_exists=data.get('prev_exists', None),
                policy_data=data)
        else:
            data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        if self.temporal_memory_enabled:
            self._temporal_memory_record_predictions(
                bbox_list, img_metas, policy_data=data)
        return bbox_list

    
