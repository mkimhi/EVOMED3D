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
import torch
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
                debug=tm_debug_shapes)
            if isinstance(self.temporal_memory_controller.updater, torch.nn.Module):
                self.temporal_memory_updater = (
                    self.temporal_memory_controller.updater)
            # The StreamPETR adapter already projects intermediate EVA
            # features through sim_fpn at the detector feature boundary.
            self.temporal_memory_projector = torch.nn.Identity()

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
            self, sample_key, scene_token, prev_exists):
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
            prev_exists=prev_exists)
        max_frames = controller.max_frames_between_full
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
            teacher_features=None):
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

        routes = []
        actions = []
        for batch_index, img_meta in enumerate(img_metas):
            scene_token = img_meta.get('scene_token')
            sample_key = self._temporal_memory_sample_key(
                img_meta, batch_index)
            self._temporal_memory_evict_stale_stream(sample_key)
            sample_prev_exists = self._temporal_memory_sample_prev_exists(
                prev_exists, batch_index)
            action = self._temporal_memory_sample_action(
                sample_key, scene_token, sample_prev_exists)
            actions.append(action)
            routes.append(dict(
                batch_index=batch_index,
                sample_key=sample_key,
                scene_token=scene_token,
                prev_exists=sample_prev_exists,
                action=action,
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
        for route in routes:
            if route['action'] == 'L24':
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
            else:
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

        outputs = self.temporal_memory_controller.get_or_update_samples(
            requests, training=training)
        features = torch.cat(outputs, dim=0)
        losses = {}
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
        if self.temporal_memory_enabled:
            frame_features = []
            num_nograd_frames = T - self.num_frame_backbone_grads
            for i in range(T):
                frame_prev_exists = data.get('prev_exists', None)
                if frame_prev_exists is not None:
                    frame_prev_exists = frame_prev_exists[:, i]
                if i < num_nograd_frames:
                    self.eval()
                    with torch.no_grad():
                        features, frame_tm_losses = (
                            self._extract_img_feat_temporal_memory(
                                data['img'][:, i], img_metas[i],
                                training=True,
                                prev_exists=frame_prev_exists))
                    self.train()
                else:
                    features, frame_tm_losses = (
                        self._extract_img_feat_temporal_memory(
                            data['img'][:, i], img_metas[i],
                            training=True,
                            prev_exists=frame_prev_exists))
                frame_features.append(features)
                tm_loss = frame_tm_losses.get('loss_tm_distill')
                if tm_loss is not None and tm_loss.requires_grad:
                    tm_losses.append(tm_loss)
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
                prev_exists=data.get('prev_exists', None))
        else:
            data['img_feats'] = self.extract_img_feat(data['img'], 1)

        bbox_list = [dict() for i in range(len(img_metas))]
        bbox_pts = self.simple_test_pts(
            img_metas, **data)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return bbox_list

    
