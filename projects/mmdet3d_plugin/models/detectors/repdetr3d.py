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
from contextlib import nullcontext as _nullcontext
import csv
import inspect
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
class RepDetr3D(MVXTwoStageDetector):
    """RepDetr3D."""

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
                 stride=[16],
                 position_level=[0],
                 aux_2d_only=True,
                 single_test=False,
                 pretrained=None,
                 temporal_memory=None,
                 tm_debug_shapes=False,
                 tm_debug_latency=False,
                 tm_debug_compare_full_path=False,
                 tm_assert_no_teacher_in_inference=True):
        super(RepDetr3D, self).__init__(pts_voxel_layer, pts_voxel_encoder,
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
        self.temporal_memory_depth_router = None
        self._tm_train_step = 0
        self._tm_previous_full_teacher = None
        self._tm_policy_prev_state = None
        self._tm_policy_prev_l8_features = None
        self._tm_policy_previous_l8_similarity = None
        self._tm_policy_prev_l12_features = None
        self._tm_policy_previous_l12_similarity = None
        self._tm_policy_prev_prediction_features = None
        self._tm_policy_offline_lookup = self._load_tm_policy_offline_lookup(
            self.temporal_memory.get('policy_offline_csv', None))
        self._tm_counterfactual_targets = None
        policy_supervision = self.temporal_memory.get('policy_supervision', {}) or {}
        if policy_supervision.get('target') == 'counterfactual_action':
            from projects.mmdet3d_plugin.models.temporal_memory import (
                CounterfactualActionTargets,
            )
            self._tm_counterfactual_targets = CounterfactualActionTargets(
                policy_supervision.get('target_csv'),
                token_column=policy_supervision.get('token_column', 'token'),
                action_column=policy_supervision.get('action_column', 'oracle_action'))
        self.tm_profiler = None
        if self.tm_debug_latency:
            from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryProfiler
            self.tm_profiler = TemporalMemoryProfiler(enabled=True)
        if self.temporal_memory_enabled:
            from projects.mmdet3d_plugin.models.temporal_memory import (
                DepthExitRouter,
                LayeredPartialFeatureProjector,
                PartialFeatureProjector,
                TemporalMemoryController,
                build_detector_adapter,
            )
            updater_cfg = dict(self.temporal_memory.get('updater_cfg', {}) or {})
            if 'shared_memory_gate' in self.temporal_memory:
                updater_cfg.setdefault(
                    'shared_memory_gate',
                    self.temporal_memory.get('shared_memory_gate'))
            if 'partial_layers' in self.temporal_memory:
                updater_cfg.setdefault('layers', self.temporal_memory.get('partial_layers'))
            self.temporal_memory_adapter = build_detector_adapter(
                self.temporal_memory.get('detector_adapter', 'repdetr3d'),
                debug_shapes=tm_debug_shapes)
            self.temporal_memory_controller = TemporalMemoryController(
                enabled=True,
                keyframe_interval=self.temporal_memory.get('keyframe_interval', 1),
                updater=self.temporal_memory.get('updater', 'reuse'),
                updater_cfg=updater_cfg,
                policy=self.temporal_memory.get('policy', None),
                partial_layer=self.temporal_memory.get('partial_layer', 'block8'),
                train_with_teacher=self.temporal_memory.get('train_with_teacher', False),
                distill_weight=self.temporal_memory.get('distill_weight', 1.0),
                inference_teacher_forbidden=self.temporal_memory.get(
                    'inference_teacher_forbidden',
                    tm_assert_no_teacher_in_inference),
                max_frames_between_full=self.temporal_memory.get(
                    'max_frames_between_full', None),
                adaptive_reset=self.temporal_memory.get('adaptive_reset', None),
                profiler=self.tm_profiler,
                debug=tm_debug_shapes)
            if isinstance(self.temporal_memory_controller.updater, torch.nn.Module):
                self.temporal_memory_updater = self.temporal_memory_controller.updater
            if isinstance(self.temporal_memory_controller.policy, torch.nn.Module):
                self.temporal_memory_policy = self.temporal_memory_controller.policy
            router_cfg = dict(self.temporal_memory.get('router', {}) or {})
            use_depth_router = router_cfg.get('type') in ('depth_exit', 'depth_router')
            if self.temporal_memory.get('updater', 'reuse') != 'reuse' or use_depth_router:
                projector_cfg = dict(self.temporal_memory.get('projector_cfg', {}))
                if self.temporal_memory.get('projectors_per_layer', True):
                    self.temporal_memory_projector = LayeredPartialFeatureProjector(
                        layers=self.temporal_memory.get('partial_layers', (8, 12, 16)),
                        projector_cfg=projector_cfg,
                        per_layer_cfg=self.temporal_memory.get('projector_per_layer_cfg', None),
                    )
                else:
                    self.temporal_memory_projector = PartialFeatureProjector(**projector_cfg)
            if use_depth_router:
                feature_dim = int(router_cfg.get(
                    'feature_dim', projector_cfg.get('out_channels', 256)))
                self.temporal_memory_depth_router = DepthExitRouter(
                    feature_dim=feature_dim,
                    metadata_dim=len(tuple(router_cfg.get('feature_names', ()))),
                    hidden_dim=int(router_cfg.get('hidden_dim', 128)),
                    temperature=float(router_cfg.get('temperature', 1.0)),
                    costs=tuple(router_cfg.get('costs', (8 / 24, 12 / 24, 16 / 24, 1.0))),
                )
            if self.temporal_memory.get('freeze_pretrained_detector', False):
                self._freeze_temporal_memory_pretrained_detector()

    def _maybe_log_tm_shapes(self, img_feats):
        if not (self.temporal_memory_enabled and self.tm_debug_shapes):
            return
        shapes = [tuple(feat.shape) for feat in img_feats]
        print(f'[temporal_memory] RepDetr3D img_feats shapes: {shapes}')

    def _extract_img_feat_temporal_memory(self, img_metas, **data):
        if (
            not self.temporal_memory_enabled
            or self.training
        ):
            if self.tm_profiler is None:
                return self.extract_img_feat(data['img'], 1)
            with self.tm_profiler.measure('backbone_full'):
                return self.extract_img_feat(data['img'], 1)

        if self._tm_depth_router_enabled():
            features, _ = self._tm_route_depth_batch(data, img_metas, training=False)
            return features

        scene_token = img_metas[0].get('scene_token') if img_metas else None
        prev_exists = data.get('prev_exists', None)
        if prev_exists is not None and hasattr(prev_exists, 'numel') and prev_exists.numel() == 1:
            prev_exists = bool(prev_exists.item())

        partial_current = None
        l8_features = None
        l12_features = None
        policy_features = self._temporal_memory_policy_features(
            img_metas, data, scene_token=scene_token)
        action = self._temporal_memory_next_action(
            scene_token=scene_token,
            prev_exists=prev_exists,
            policy_features=policy_features)
        if (
            self._temporal_memory_policy_needs_l8_probe()
            and getattr(self.temporal_memory_controller, 'memory', None) is not None
        ):
            l8_features = self._profile_extract_partial_features(data, 'block8')
            policy_features.update(self._temporal_memory_l8_similarity_features(
                getattr(self, '_tm_policy_prev_l8_features', None), l8_features))
            action = self._temporal_memory_next_action(
                scene_token=scene_token,
                prev_exists=prev_exists,
                policy_features=policy_features)
        if (
            self.temporal_memory.get('updater', 'reuse') != 'reuse'
            and action != 'L24'
        ):
            layer_id = self._temporal_memory_action_layer(action)
            partial_features = l8_features if layer_id == 'block8' and l8_features is not None else self._profile_extract_partial_features(
                data,
                layer_id)
            if layer_id == 'block8':
                l8_features = partial_features
            elif layer_id == 'block12':
                l12_features = partial_features
            partial_current = self._project_temporal_memory_partial(
                partial_features,
                layer_id=layer_id,
                action=action)
        if self._temporal_memory_policy_needs_previous_l8_drift() and l8_features is None:
            l8_features = self._profile_extract_partial_features(data, 'block8')
        if self._temporal_memory_policy_needs_previous_l12_drift() and l12_features is None:
            l12_features = self._extract_temporal_memory_policy_probe_features(data, 'block12')

        output = self.temporal_memory_controller.get_or_update(
            lambda: self._profile_extract_full_features(data, len_queue=1, training_mode=False),
            scene_token=scene_token,
            prev_exists=prev_exists,
            partial_current=partial_current,
            policy_features=policy_features)
        self._temporal_memory_commit_policy_state(
            policy_features, l8_features=l8_features, l12_features=l12_features)
        return output

    def _profile_extract_full_features(self, data, len_queue=1, training_mode=False):
        if self.tm_profiler is None:
            return self.temporal_memory_adapter.extract_full_features(
                self, data, len_queue=len_queue, training_mode=training_mode)
        with self.tm_profiler.measure('backbone_full'):
            return self.temporal_memory_adapter.extract_full_features(
                self, data, len_queue=len_queue, training_mode=training_mode)

    def _profile_extract_partial_features(self, data, layer_id, **kwargs):
        if self.tm_profiler is None:
            return self.temporal_memory_adapter.extract_partial_features(
                self, data, layer_id, **kwargs)
        with self.tm_profiler.measure('backbone_partial'):
            return self.temporal_memory_adapter.extract_partial_features(
                self, data, layer_id, **kwargs)

    def _extract_temporal_memory_policy_probe_features(self, data, layer_id):
        adapter = getattr(self, 'temporal_memory_adapter', None)
        if adapter is None:
            return self._profile_extract_partial_features(data, layer_id)
        return adapter.extract_partial_features(self, data, layer_id)

    def _extract_img_feat_temporal_memory_train(self, img_metas, policy_annotation_data=None, **data):
        if self._temporal_memory_batch_size_from_data(data) > 1:
            return self._extract_img_feat_temporal_memory_train_batched(
                img_metas, policy_annotation_data=policy_annotation_data, **data)

        frame_features = []
        distill_losses = []
        policy_losses = []
        router_losses = []
        T = data['img'].size(1)
        policy_annotation_data = policy_annotation_data or {}
        for i in range(T):
            frame_data = dict(data)
            frame_data['img'] = data['img'][:, i]
            for key, value in policy_annotation_data.items():
                frame_data[key] = self._tm_temporal_index(value, i)
            frame_metas = img_metas[i]
            scene_token = frame_metas[0].get('scene_token') if frame_metas else None
            prev_exists = data.get('prev_exists', None)
            if prev_exists is not None:
                prev_exists = prev_exists[:, i]
                if hasattr(prev_exists, 'numel') and prev_exists.numel() == 1:
                    prev_exists = bool(prev_exists.item())

            if self._tm_depth_router_enabled():
                output, route_losses = self._tm_route_depth_batch(
                    frame_data, frame_metas, training=True)
                frame_features.append(output)
                router_losses.append(route_losses)
                continue

            def extract_full():
                full = self._profile_extract_full_features(
                    frame_data, len_queue=1, training_mode=True)
                return self._squeeze_temporal_feature_dim(full)

            if (
                scene_token != getattr(self.temporal_memory_controller, 'scene_token', None)
                or not self.temporal_memory_controller._prev_exists_all(prev_exists)
            ):
                self._tm_previous_full_teacher = None

            partial_current = None
            l8_features = None
            l12_features = None
            policy_features = self._temporal_memory_policy_features(
                frame_metas, frame_data, scene_token=scene_token)
            action = self._temporal_memory_next_action(
                scene_token=scene_token,
                prev_exists=prev_exists,
                policy_features=policy_features)
            if (
                self._temporal_memory_policy_needs_l8_probe()
                and getattr(self.temporal_memory_controller, 'memory', None) is not None
            ):
                l8_features = self._profile_extract_partial_features(frame_data, 'block8')
                policy_features.update(self._temporal_memory_l8_similarity_features(
                    getattr(self, '_tm_policy_prev_l8_features', None), l8_features))
                action = self._temporal_memory_next_action(
                    scene_token=scene_token,
                    prev_exists=prev_exists,
                    policy_features=policy_features)
            if action != 'L24':
                layer_id = self._temporal_memory_action_layer(action)
                partial_features = l8_features if layer_id == 'block8' and l8_features is not None else self._profile_extract_partial_features(
                    frame_data,
                    layer_id)
                if layer_id == 'block8':
                    l8_features = partial_features
                elif layer_id == 'block12':
                    l12_features = partial_features
                partial_current = self._project_temporal_memory_partial(
                    partial_features,
                    layer_id=layer_id,
                    action=action)
            if self._temporal_memory_policy_needs_previous_l8_drift() and l8_features is None:
                l8_features = self._profile_extract_partial_features(frame_data, 'block8')
            if self._temporal_memory_policy_needs_previous_l12_drift() and l12_features is None:
                l12_features = self._extract_temporal_memory_policy_probe_features(frame_data, 'block12')
            policy_teacher = None
            policy_supervision_active = self._temporal_memory_policy_supervision_active()
            policy_requires_teacher = self._temporal_memory_policy_supervision_requires_teacher()
            if policy_supervision_active and policy_requires_teacher:
                with torch.no_grad():
                    policy_teacher = extract_full()
            previous_teacher = getattr(self, '_tm_previous_full_teacher', None)
            if (
                policy_supervision_active
                and not policy_features['_state']['is_scene_start']
                and (not policy_requires_teacher or previous_teacher is not None)
            ):
                policy_loss = self._temporal_memory_policy_supervision_loss(
                    policy_features, previous_teacher, policy_teacher)
                if policy_loss is not None:
                    policy_losses.append(policy_loss)
            output = self.temporal_memory_controller.get_or_update(
                extract_full,
                scene_token=scene_token,
                prev_exists=prev_exists,
                partial_current=partial_current,
                extract_teacher=extract_full,
                training=True,
                policy_features=policy_features)
            self._temporal_memory_commit_policy_state(
                policy_features, l8_features=l8_features, l12_features=l12_features)
            frame_features.append(output)
            if self.temporal_memory_controller.last_distillation_loss is not None:
                distill_losses.append(self.temporal_memory_controller.last_distillation_loss)
            if policy_teacher is not None:
                self._tm_previous_full_teacher = [
                    feature.detach() for feature in policy_teacher
                ]

        features = [
            torch.stack([frame[level] for frame in frame_features], dim=1)
            for level in range(len(frame_features[0]))
        ]
        losses = self._temporal_memory_aux_losses(
            features,
            distill_losses=distill_losses,
            policy_losses=policy_losses,
            router_losses=router_losses)
        self.temporal_memory_controller.memory = self._temporal_memory_detach_features(
            getattr(self.temporal_memory_controller, 'memory', None))
        self._tm_train_step = getattr(self, '_tm_train_step', 0) + 1
        return features, losses

    def _extract_img_feat_temporal_memory_train_batched(self, img_metas, policy_annotation_data=None, **data):
        frame_features = []
        distill_losses = []
        policy_losses = []
        batch_size = self._temporal_memory_batch_size_from_data(data)
        self._temporal_memory_ensure_batch_state(batch_size)
        T = data['img'].size(1)
        policy_annotation_data = policy_annotation_data or {}
        for i in range(T):
            frame_data = dict(data)
            frame_data['img'] = data['img'][:, i]
            for key, value in policy_annotation_data.items():
                frame_data[key] = self._tm_temporal_index(value, i)
            frame_metas = img_metas[i]
            prev_exists = data.get('prev_exists', None)
            if prev_exists is not None:
                prev_exists = prev_exists[:, i]

            actions = []
            is_reset_frames = []
            policy_features = []
            for batch_index in range(batch_size):
                sample_meta = frame_metas[batch_index] if batch_index < len(frame_metas) else {}
                sample_scene = sample_meta.get('scene_token') if isinstance(sample_meta, dict) else None
                indices = torch.as_tensor([batch_index], dtype=torch.long)
                sample_data = self._temporal_memory_slice_batch_data(frame_data, indices)
                state = self._tm_batch_states[batch_index]
                prev_ok = self._temporal_memory_sample_prev_exists(
                    prev_exists, batch_index)
                is_reset_frame = (
                    state.get('scene_token') != sample_scene
                    or not prev_ok
                    or state.get('memory') is None
                )
                features = self._temporal_memory_policy_features(
                    [sample_meta],
                    sample_data,
                    scene_token=sample_scene,
                    frame_number=(
                        0 if is_reset_frame else state.get('frame_index', 0)))
                policy_features.append(features)
                is_reset_frames.append(is_reset_frame)
                if is_reset_frame or self._temporal_memory_sample_force_full_for_age(state, features):
                    action = 'L24'
                else:
                    action = self.temporal_memory_controller.policy.decide(
                        frame_index=state.get('frame_index', 0),
                        has_memory=state.get('memory') is not None,
                        is_reset_frame=False,
                        scene_token=sample_scene,
                        prev_exists=prev_ok,
                        policy_features=features)
                actions.append(action)

            policy_teacher = None
            policy_supervision_active = self._temporal_memory_policy_supervision_active()
            policy_requires_teacher = self._temporal_memory_policy_supervision_requires_teacher()
            if (
                (policy_supervision_active and policy_requires_teacher)
                or getattr(self.temporal_memory_controller, 'train_with_teacher', False)
            ):
                with torch.no_grad():
                    policy_teacher = self._squeeze_temporal_feature_dim(
                        self._profile_extract_full_features(
                            frame_data, len_queue=1, training_mode=True))
            if policy_supervision_active:
                previous_teachers = getattr(self, '_tm_batch_previous_full_teacher', None)
                if previous_teachers is None or len(previous_teachers) != batch_size:
                    previous_teachers = [None for _ in range(batch_size)]
                for batch_index in range(batch_size):
                    previous_teacher = previous_teachers[batch_index]
                    if (
                        is_reset_frames[batch_index]
                        or (policy_requires_teacher and previous_teacher is None)
                    ):
                        continue
                    current_teacher = None
                    if policy_requires_teacher:
                        current_teacher = self._temporal_memory_select_feature_batch(
                            policy_teacher,
                            torch.as_tensor([batch_index], dtype=torch.long))
                    policy_loss = self._temporal_memory_policy_supervision_loss(
                        policy_features[batch_index],
                        previous_teacher,
                        current_teacher)
                    if policy_loss is not None:
                        policy_losses.append(policy_loss)

            output_groups = []
            full_indices = torch.as_tensor(
                [index for index, action in enumerate(actions) if action == 'L24'],
                dtype=torch.long)
            if full_indices.numel() > 0:
                full_data = self._temporal_memory_slice_batch_data(frame_data, full_indices)
                full_features = self._squeeze_temporal_feature_dim(
                    self._profile_extract_full_features(
                        full_data, len_queue=1, training_mode=True))
                output_groups.append((full_indices, full_features))

            partial_current = self._temporal_memory_project_mixed_batch_partials(frame_data, actions)
            for action in ('L8', 'L12', 'L16'):
                indices = torch.as_tensor(
                    [index for index, value in enumerate(actions) if value == action],
                    dtype=torch.long)
                if indices.numel() == 0:
                    continue
                previous = self._temporal_memory_gather_sample_memories(indices)
                partial = self._temporal_memory_select_feature_batch(partial_current, indices)
                updated = self.temporal_memory_controller._call_updater(
                    previous,
                    partial_current=partial,
                    action=action)
                output_groups.append((indices, updated))

            output = self._temporal_memory_scatter_feature_group_list(output_groups, batch_size)
            if (
                policy_teacher is not None
                and getattr(self.temporal_memory_controller, 'train_with_teacher', False)
            ):
                distill_losses.append(self.temporal_memory_controller.distillation_loss(
                    output,
                    policy_teacher))
            frame_features.append(output)
            current_teachers = []
            for batch_index, action in enumerate(actions):
                sample_meta = frame_metas[batch_index] if batch_index < len(frame_metas) else {}
                sample_scene = sample_meta.get('scene_token') if isinstance(sample_meta, dict) else None
                state = self._tm_batch_states[batch_index]
                state['scene_token'] = sample_scene
                state['memory'] = self._temporal_memory_select_feature_batch(
                    output,
                    torch.as_tensor([batch_index], dtype=torch.long))
                state['frame_index'] = int(
                    policy_features[batch_index]['_state']['frame_number']) + 1
                state['frames_since_full'] = 0 if action == 'L24' else int(state.get('frames_since_full', 0)) + 1
                self.temporal_memory_controller._policy_stats[action] = (
                    self.temporal_memory_controller._policy_stats.get(action, 0) + 1)
                self._temporal_memory_commit_policy_state(policy_features[batch_index])
                if policy_teacher is not None:
                    current_teachers.append([
                        feature.detach()
                        for feature in self._temporal_memory_select_feature_batch(
                            policy_teacher,
                            torch.as_tensor([batch_index], dtype=torch.long))
                    ])
            if current_teachers:
                self._tm_batch_previous_full_teacher = current_teachers

        features = [
            torch.stack([frame[level] for frame in frame_features], dim=1)
            for level in range(len(frame_features[0]))
        ]
        losses = self._temporal_memory_aux_losses(
            features,
            distill_losses=distill_losses,
            policy_losses=policy_losses)
        for state in getattr(self, '_tm_batch_states', []):
            state['memory'] = self._temporal_memory_detach_features(
                state.get('memory'))
        self._tm_train_step = getattr(self, '_tm_train_step', 0) + 1
        return features, losses

    def _temporal_memory_aux_losses(
        self, features, distill_losses=None, policy_losses=None, router_losses=None,
    ):
        distill_losses = distill_losses or []
        policy_losses = policy_losses or []
        router_losses = router_losses or []
        losses = {}
        if distill_losses:
            losses['loss_tm_distill'] = sum(distill_losses) / len(distill_losses)
        else:
            controller = getattr(self, 'temporal_memory_controller', None)
            if getattr(controller, 'train_with_teacher', False):
                losses['loss_tm_distill'] = features[0].new_zeros(())
        if policy_losses:
            losses['loss_tm_policy'] = sum(policy_losses) / len(policy_losses)
        elif self._temporal_memory_policy_supervision_active():
            losses['loss_tm_policy'] = features[0].new_zeros(())
        if router_losses:
            for key in ('loss_tm_budget', 'loss_tm_balance'):
                values = [item[key] for item in router_losses if key in item]
                if values:
                    losses[key] = sum(values) / len(values)
        return losses

    def _temporal_memory_next_action(self, scene_token=None, prev_exists=None, policy_features=None):
        controller = self.temporal_memory_controller
        if hasattr(controller, 'peek_action'):
            return controller.peek_action(
                scene_token=scene_token,
                prev_exists=prev_exists,
                policy_features=policy_features)
        if controller.will_use_keyframe(scene_token=scene_token, prev_exists=prev_exists):
            return 'L24'
        from projects.mmdet3d_plugin.models.temporal_memory import layer_to_action
        return layer_to_action(self.temporal_memory.get('partial_layer', 'block8')) or 'L8'

    def _temporal_memory_policy_needs_l8_probe(self):
        policy = getattr(self.temporal_memory_controller, 'policy', None)
        if bool(getattr(policy, 'requires_l8_similarity', False)):
            return True
        spec = self.temporal_memory.get('policy', {})
        return isinstance(spec, dict) and bool(spec.get('requires_l8_similarity', False))

    def _temporal_memory_policy_needs_previous_l8_drift(self):
        policy = getattr(self.temporal_memory_controller, 'policy', None)
        if bool(getattr(policy, 'requires_previous_l8_similarity', False)):
            return True
        spec = self.temporal_memory.get('policy', {})
        return isinstance(spec, dict) and bool(spec.get('requires_previous_l8_similarity', False))

    def _temporal_memory_policy_needs_previous_l12_drift(self):
        policy = getattr(self.temporal_memory_controller, 'policy', None)
        if bool(getattr(policy, 'requires_previous_l12_similarity', False)):
            return True
        feature_names = getattr(policy, 'feature_names', ()) or ()
        if 'previous_l12_similarity' in feature_names:
            return True
        spec = self.temporal_memory.get('policy', {})
        if isinstance(spec, dict):
            feature_names = spec.get('feature_names', ()) or ()
            return bool(spec.get('requires_previous_l12_similarity', False)) or 'previous_l12_similarity' in feature_names
        return False

    def _temporal_memory_l8_similarity_features(self, previous_features, current_features):
        if previous_features is None or current_features is None:
            return {}
        previous = previous_features[0] if isinstance(previous_features, (list, tuple)) else previous_features
        current = current_features[0] if isinstance(current_features, (list, tuple)) else current_features
        if previous.shape != current.shape or previous.dim() != 5:
            return {}
        cosine = F.cosine_similarity(previous.float(), current.float(), dim=2)
        values = cosine.mean(dim=(0, 2, 3)).detach().cpu().tolist()
        names = ('CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT')
        features = {f'similarity_L8_{name}': float(value) for name, value in zip(names, values)}
        if values:
            features['similarity_L8_mean'] = float(sum(values) / len(values))
            features['min_L8_similarity'] = float(min(values))
        return features

    def _temporal_memory_policy_target_index(self, value):
        spec = self.temporal_memory.get('policy_supervision', {}) or {}
        thresholds = tuple(spec.get('thresholds', (0.85, 0.75, 0.65)))
        if len(thresholds) != 3:
            raise ValueError('policy_supervision.thresholds must contain three values')
        if value >= thresholds[0]:
            return 0
        if value >= thresholds[1]:
            return 1
        if value >= thresholds[2]:
            return 2
        return 3

    def _temporal_memory_feature_similarity(self, previous, current):
        values = []
        for prev, cur in zip(previous, current):
            prev = prev.detach().float()
            cur = cur.detach().float()
            if prev.shape != cur.shape:
                raise ValueError('policy supervision features must have matching shapes')
            if prev.dim() == 5:
                cosine = F.cosine_similarity(prev, cur, dim=2).mean()
            else:
                cosine = F.cosine_similarity(prev.flatten(1), cur.flatten(1), dim=1).mean()
            values.append(cosine)
        return torch.stack(values).mean()

    @staticmethod
    def _temporal_memory_linear_cka(x, y, eps=1e-12):
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        xy = torch.matmul(x.transpose(0, 1), y)
        xx = torch.matmul(x.transpose(0, 1), x)
        yy = torch.matmul(y.transpose(0, 1), y)
        return xy.pow(2).sum() / (xx.pow(2).sum().sqrt() * yy.pow(2).sum().sqrt() + eps)

    def _temporal_memory_feature_cka(self, previous, current):
        values = []
        for prev, cur in zip(previous, current):
            prev = prev.detach().float()
            cur = cur.detach().float()
            if prev.shape != cur.shape:
                raise ValueError('policy supervision features must have matching shapes')
            if prev.dim() == 5:
                prev = prev.permute(0, 1, 3, 4, 2).reshape(-1, prev.shape[2])
                cur = cur.permute(0, 1, 3, 4, 2).reshape(-1, cur.shape[2])
            else:
                prev = prev.flatten(1)
                cur = cur.flatten(1)
            values.append(self._temporal_memory_linear_cka(prev, cur))
        return torch.stack(values).mean()

    def _temporal_memory_policy_target_score(self, previous_full, current_full):
        spec = self.temporal_memory.get('policy_supervision', {}) or {}
        target = spec.get('target', 'similarity')
        if target in ('similarity', 'similarity_L24_mean'):
            return self._temporal_memory_feature_similarity(previous_full, current_full)
        if target in ('cka', 'cka_L24_mean'):
            return self._temporal_memory_feature_cka(previous_full, current_full)
        raise ValueError(f'Unsupported temporal memory policy target: {target}')

    def _temporal_memory_policy_supervision_loss(self, policy_features, previous_full, current_full):
        policy = getattr(self, 'temporal_memory_policy', None)
        if not isinstance(policy, torch.nn.Module) or not hasattr(policy, 'logits'):
            return None
        spec = self.temporal_memory.get('policy_supervision', {}) or {}
        if spec.get('target') == 'counterfactual_action':
            state = policy_features.get('_state', {}) or {}
            token = state.get('sample_token')
            if not token and (
                state.get('scene_token') is None
                or state.get('frame_number') is None
            ):
                raise KeyError(
                    'counterfactual policy supervision requires a sample token '
                    'or scene/frame identity')
            targets = getattr(self, '_tm_counterfactual_targets', None)
            if targets is None:
                raise RuntimeError('counterfactual policy targets were not loaded')
            target_index = targets.label_index(
                token,
                policy.labels,
                scene_token=state.get('scene_token'),
                frame_number=state.get('frame_number'),
            )
        else:
            score = self._temporal_memory_policy_target_score(previous_full, current_full)
            target_noise_std = float(spec.get('target_noise_std', 0.0) or 0.0)
            if self.training and target_noise_std > 0.0:
                score = score + torch.randn_like(score) * target_noise_std
                clamp = spec.get('target_noise_clamp', (0.0, 1.0))
                if clamp is not None:
                    low, high = clamp
                    score = score.clamp(float(low), float(high))
            target_index = self._temporal_memory_policy_target_index(float(score.detach().cpu()))
        logits = policy.logits(self._temporal_memory_noisy_policy_features(policy_features))
        target = torch.as_tensor([target_index], dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, target)
        class_weights = spec.get('class_weights', None)
        if class_weights is not None:
            class_weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
            if class_weights.numel() != logits.shape[-1]:
                raise ValueError('policy_supervision.class_weights must match policy class count')
            loss = loss * class_weights[target_index]
        weight = float(spec.get('weight', 1.0))
        return loss * weight

    def _temporal_memory_noisy_policy_features(self, policy_features):
        spec = self.temporal_memory.get('policy_supervision', {}) or {}
        residual_noise_multiplier = float(
            spec.get('residual_noise_multiplier', 0.0) or 0.0)
        policy = getattr(self, 'temporal_memory_policy', None)
        feature_names = getattr(policy, 'feature_names', None)
        residual_std = getattr(policy, 'teacher_forcing_residual_std', None)
        if (
            self.training
            and residual_noise_multiplier > 0.0
            and feature_names
            and residual_std is not None
        ):
            residual_std = residual_std.detach().float().cpu()
            if residual_std.numel() != len(feature_names):
                raise ValueError(
                    'teacher-forcing residual std must match policy feature count')
            noise = torch.randn(len(feature_names), dtype=residual_std.dtype)
            noise = noise * residual_noise_multiplier * residual_std
            noisy = dict(policy_features)
            for index, name in enumerate(feature_names):
                value = noisy.get(name, 0.0) or 0.0
                noisy[name] = float(value) + float(noise[index])
            return noisy
        feature_noise_std = float(spec.get('feature_noise_std', 0.0) or 0.0)
        if not self.training or feature_noise_std <= 0.0:
            return policy_features
        scale = getattr(policy, 'scale', None)
        if not feature_names or scale is None:
            return policy_features
        scale = scale.detach().float().cpu()
        noise = torch.randn(len(feature_names), dtype=scale.dtype) * feature_noise_std * scale
        noisy = dict(policy_features)
        for index, name in enumerate(feature_names):
            value = noisy.get(name, 0.0) or 0.0
            noisy[name] = float(value) + float(noise[index])
        return noisy

    def _temporal_memory_policy_supervision_requires_teacher(self):
        spec = self.temporal_memory.get('policy_supervision', {}) or {}
        return spec.get('target', 'similarity') != 'counterfactual_action'

    def _temporal_memory_policy_supervision_active(self):
        spec = self.temporal_memory.get('policy_supervision', None)
        if not spec:
            return False
        if spec.get('enabled', True) is False:
            return False
        step = getattr(self, '_tm_train_step', 0)
        max_steps = int(spec.get('max_train_steps', 0) or 0)
        if 'start_train_step' in spec:
            start_step = int(spec.get('start_train_step', 0) or 0)
            return step >= start_step and (max_steps <= 0 or step < max_steps)
        warmup_ratio = float(spec.get('warmup_ratio', 0.1))
        if max_steps <= 0:
            return True
        return step < max(1, int(max_steps * warmup_ratio))

    def _maybe_log_temporal_memory_policy_stats(self):
        if self.tm_profiler is None or self.tm_profiler.window_reports != 0:
            return
        controller = getattr(self, 'temporal_memory_controller', None)
        if controller is None or not hasattr(controller, 'policy_stats'):
            return
        stats = controller.policy_stats()
        total = sum(stats.get(action, 0) for action in ('L8', 'L12', 'L16', 'L24'))
        if total <= 0:
            return
        fields = []
        for action in ('L8', 'L12', 'L16', 'L24'):
            count = stats.get(action, 0)
            fields.append(f'{action}={count}({count / total:.3f})')
        fields.append(f'forced_l24={stats.get("forced_l24", 0)}')
        fields.append(f'forced_age_l24={stats.get("forced_age_l24", 0)}')
        fields.append(f'forced_motion_l24={stats.get("forced_motion_l24", 0)}')
        fields.append(f'learned_l24={stats.get("learned_l24", 0)}')
        fields.append(f'effective_full_rate={stats.get("L24", 0) / total:.3f}')
        print(f'[temporal_memory_policy] samples={total} ' + ' '.join(fields))

    @staticmethod
    def _tm_scalar(value, default=0.0):
        if value is None:
            return default
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            return float(value.detach().reshape(-1)[0].cpu())
        if isinstance(value, (list, tuple)):
            return RepDetr3D._tm_scalar(value[0], default=default) if value else default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _tm_tensor(value):
        if value is None:
            return None
        if hasattr(value, 'tensor'):
            value = value.tensor
        if torch.is_tensor(value):
            return value.detach()
        if isinstance(value, (list, tuple)) and value:
            return RepDetr3D._tm_tensor(value[0])
        return None

    @staticmethod
    def _tm_temporal_index(value, index):
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return value[index] if len(value) > index else None
        if torch.is_tensor(value) and value.dim() > 1:
            return value[:, index]
        return value

    def _tm_pose_xy_yaw(self, data):
        pose = self._tm_tensor(data.get('ego_pose', None))
        if pose is None or pose.numel() < 16:
            return 0.0, 0.0, 0.0, 0.0
        pose = pose.reshape(-1, 4, 4)[0].float().cpu()
        x = float(pose[0, 3])
        y = float(pose[1, 3])
        z = float(pose[2, 3])
        yaw = math.atan2(float(pose[1, 0]), float(pose[0, 0]))
        return x, y, z, yaw

    @staticmethod
    def _tm_yaw_delta(prev_yaw, cur_yaw):
        return (cur_yaw - prev_yaw + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _load_tm_policy_offline_lookup(path):
        if not path:
            return None
        lookup = {}
        try:
            with open(path, newline='') as handle:
                for row in csv.DictReader(handle):
                    token = row.get('token')
                    scene_key = None
                    if row.get('scene_token') is not None and row.get('frame_number') is not None:
                        scene_key = (row.get('scene_token'), str(int(float(row.get('frame_number')))))
                    if not token and scene_key is None:
                        continue
                    online_keys = {
                        'frame_number', 'info_frame_idx', 'is_scene_start', 'dt_s',
                        'ego_x_m', 'ego_y_m', 'ego_z_m', 'ego_yaw_deg',
                        'delta_ego_translation_m', 'ego_speed_mps', 'ego_accel_mps2',
                        'yaw_rate_deg_s', 'abs_yaw_rate_deg_s', 'yaw_accel_deg_s2',
                        'abs_yaw_accel_deg_s2',
                    }
                    values = {}
                    for key, value in row.items():
                        if key in ('index', 'token', 'scene_token', 'timestamp', 'has_previous_frame') or key in online_keys:
                            continue
                        try:
                            values[key] = float(value)
                        except (TypeError, ValueError):
                            pass
                    if token:
                        lookup[token] = values
                    if scene_key is not None:
                        lookup[scene_key] = values
        except OSError:
            return None
        return lookup

    def _tm_lookup_offline_features(self, img_metas):
        lookup = getattr(self, '_tm_policy_offline_lookup', None)
        if not lookup:
            return None
        meta = img_metas[0] if img_metas and isinstance(img_metas[0], dict) else {}
        token = meta.get('token', None)
        if token is None:
            filename = meta.get('filename', None)
            token = filename[0] if isinstance(filename, (list, tuple)) and filename else None
        row = lookup.get(token)
        if row is None and meta.get('scene_token') is not None:
            frame_number = meta.get('frame_idx', getattr(self.temporal_memory_controller, 'frame_index', 0))
            row = lookup.get((meta.get('scene_token'), str(int(frame_number))))
        if row is None:
            return None
        features = self._tm_zero_offline_features()
        features.update(row)
        return features

    @staticmethod
    def _tm_zero_offline_features():
        features = {}
        for key in (
            'num_annotations', 'avg_object_distance_m', 'median_object_distance_m',
            'min_object_distance_m', 'p10_object_distance_m', 'p90_object_distance_m',
            'avg_box_length_m', 'avg_box_width_m', 'avg_box_height_m',
            'avg_box_volume_m3', 'avg_box_area_m2', 'avg_object_speed_mps',
            'max_object_speed_mps', 'camera_annotation_count',
            'max_camera_annotation_count', 'min_camera_annotation_count',
            'avg_camera_annotation_count', 'new_object_proxy_count',
            'matched_object_proxy_count', 'disappeared_object_proxy_count',
            'new_object_proxy_fraction',
        ):
            features[key] = 0.0
        for threshold in (10, 20, 30, 50):
            features[f'near_object_count_{threshold}m'] = 0.0
        return features

    def _tm_annotation_features(self, data, img_metas=None):
        features = {}
        meta = img_metas[0] if img_metas and isinstance(img_metas[0], dict) else {}
        boxes = self._tm_tensor(data.get('gt_bboxes_3d', None))
        if boxes is None:
            boxes = self._tm_tensor(meta.get('gt_bboxes_3d', None))
        if boxes is None:
            lookup_features = self._tm_lookup_offline_features(img_metas)
            if lookup_features is not None:
                return lookup_features
        if boxes is not None and boxes.numel() > 0:
            boxes = boxes.reshape(-1, boxes.shape[-1]).float().cpu()
            features['num_annotations'] = float(len(boxes))
            distances = torch.linalg.vector_norm(boxes[:, :2], dim=1)
            features['avg_object_distance_m'] = float(distances.mean())
            features['median_object_distance_m'] = float(distances.median())
            features['min_object_distance_m'] = float(distances.min())
            features['p10_object_distance_m'] = float(torch.quantile(distances, 0.1))
            features['p90_object_distance_m'] = float(torch.quantile(distances, 0.9))
            for threshold in (10, 20, 30, 50):
                features[f'near_object_count_{threshold}m'] = float((distances < threshold).sum())
            if boxes.shape[1] >= 6:
                dims = boxes[:, 3:6].abs()
                features['avg_box_length_m'] = float(dims[:, 0].mean())
                features['avg_box_width_m'] = float(dims[:, 1].mean())
                features['avg_box_height_m'] = float(dims[:, 2].mean())
                features['avg_box_volume_m3'] = float(dims.prod(dim=1).mean())
                features['avg_box_area_m2'] = float((dims[:, 0] * dims[:, 1]).mean())
            if boxes.shape[1] >= 9:
                speeds = torch.linalg.vector_norm(boxes[:, 7:9], dim=1)
                features['avg_object_speed_mps'] = float(speeds.mean())
                features['max_object_speed_mps'] = float(speeds.max())
        else:
            features.update(self._tm_zero_offline_features())
        gt_bboxes = data.get('gt_bboxes', None)
        if isinstance(gt_bboxes, (list, tuple)):
            counts = []
            for item in gt_bboxes:
                tensor = self._tm_tensor(item)
                counts.append(0 if tensor is None or tensor.numel() == 0 else int(tensor.reshape(-1, tensor.shape[-1]).shape[0]))
            if counts:
                features['camera_annotation_count'] = float(sum(counts))
                features['max_camera_annotation_count'] = float(max(counts))
                features['min_camera_annotation_count'] = float(min(counts))
                features['avg_camera_annotation_count'] = float(sum(counts) / len(counts))
        prev = getattr(self, '_tm_policy_prev_state', None)
        prev_gt = None if prev is None else prev.get('current_gt_boxes', None)
        features.update(self._tm_new_object_proxy_features(prev_gt, boxes))
        return features

    @staticmethod
    def _tm_new_object_proxy_features(previous_boxes, current_boxes, match_distance_m=2.0):
        if previous_boxes is None or current_boxes is None:
            current_count = 0 if current_boxes is None else int(current_boxes.reshape(-1, current_boxes.shape[-1]).shape[0])
            return dict(
                new_object_proxy_count=float(current_count),
                matched_object_proxy_count=0.0,
                disappeared_object_proxy_count=0.0,
                new_object_proxy_fraction=float(1.0 if current_count else 0.0),
            )
        previous = previous_boxes.reshape(-1, previous_boxes.shape[-1]).float().cpu()
        current = current_boxes.reshape(-1, current_boxes.shape[-1]).float().cpu()
        used = set()
        matched = 0
        for cur in current:
            if previous.numel() == 0:
                break
            distances = torch.linalg.vector_norm(previous[:, :2] - cur[:2], dim=1)
            best = int(distances.argmin().item())
            if best not in used and float(distances[best]) <= match_distance_m:
                used.add(best)
                matched += 1
        new_count = int(len(current) - matched)
        disappeared = int(len(previous) - matched)
        return dict(
            new_object_proxy_count=float(new_count),
            matched_object_proxy_count=float(matched),
            disappeared_object_proxy_count=float(disappeared),
            new_object_proxy_fraction=float(new_count / len(current)) if len(current) else 0.0,
        )

    def _tm_prediction_features(self, bbox_results):
        features = self._tm_zero_offline_features()
        if not bbox_results:
            return features
        result = bbox_results[0]
        boxes = result.get('boxes_3d', None)
        if boxes is None:
            boxes = result.get('bboxes_3d', None)
        scores = result.get('scores_3d', None)
        boxes = self._tm_tensor(boxes)
        if boxes is None or boxes.numel() == 0:
            return features
        boxes = boxes.reshape(-1, boxes.shape[-1]).float().cpu()
        if scores is not None:
            scores = self._tm_tensor(scores)
            if scores is not None and scores.numel() == len(boxes):
                threshold = float(self.temporal_memory.get('policy_prediction_score_thr', 0.1))
                boxes = boxes[scores.reshape(-1).float().cpu() >= threshold]
        if boxes.numel() == 0:
            return features
        distances = torch.linalg.vector_norm(boxes[:, :2], dim=1)
        features['num_annotations'] = float(len(boxes))
        features['avg_object_distance_m'] = float(distances.mean())
        features['median_object_distance_m'] = float(distances.median())
        features['min_object_distance_m'] = float(distances.min())
        features['p10_object_distance_m'] = float(torch.quantile(distances, 0.1))
        features['p90_object_distance_m'] = float(torch.quantile(distances, 0.9))
        for threshold in (10, 20, 30, 50):
            features[f'near_object_count_{threshold}m'] = float((distances < threshold).sum())
        if boxes.shape[1] >= 6:
            dims = boxes[:, 3:6].abs()
            features['avg_box_length_m'] = float(dims[:, 0].mean())
            features['avg_box_width_m'] = float(dims[:, 1].mean())
            features['avg_box_height_m'] = float(dims[:, 2].mean())
            features['avg_box_volume_m3'] = float(dims.prod(dim=1).mean())
            features['avg_box_area_m2'] = float((dims[:, 0] * dims[:, 1]).mean())
        if boxes.shape[1] >= 9:
            speeds = torch.linalg.vector_norm(boxes[:, 7:9], dim=1)
            features['avg_object_speed_mps'] = float(speeds.mean())
            features['max_object_speed_mps'] = float(speeds.max())
        return features

    def _tm_policy_offline_mode(self):
        return self.temporal_memory.get('policy_offline_features', None)

    def _tm_apply_policy_offline_features(self, features, current_gt_features):
        mode = self._tm_policy_offline_mode()
        if not mode:
            return
        if mode == 'teacher_forcing':
            mode = 'previous_gt' if self.training else 'previous_prediction'
        prev = getattr(self, '_tm_policy_prev_state', None)
        previous_gt = None if prev is None else prev.get('offline_gt_features', None)
        if mode == 'current_gt':
            features.update(current_gt_features)
        elif mode == 'previous_gt':
            features.update(previous_gt or self._tm_zero_offline_features())
        elif mode == 'previous_prediction':
            features.update(getattr(self, '_tm_policy_prev_prediction_features', None) or self._tm_zero_offline_features())
        else:
            raise ValueError(f'Unsupported temporal memory policy_offline_features: {mode}')

    def _temporal_memory_policy_features(
        self,
        img_metas,
        data,
        scene_token=None,
        l8_similarity=None,
        frame_number=None,
    ):
        meta = img_metas[0] if img_metas and isinstance(img_metas[0], dict) else {}
        sample_token = meta.get('token', meta.get('sample_idx', None))
        if sample_token is None:
            filename = meta.get('filename', None)
            sample_token = filename[0] if isinstance(filename, (list, tuple)) and filename else filename
        if frame_number is None:
            frame_number = meta.get(
                'frame_idx', getattr(self.temporal_memory_controller, 'frame_index', 0))
        frame_number = int(frame_number)
        timestamp = self._tm_scalar(data.get('timestamp', meta.get('timestamp', 0.0)))
        x, y, z, yaw = self._tm_pose_xy_yaw(data)
        prev = getattr(self, '_tm_policy_prev_state', None)
        is_scene_start = prev is None or prev.get('scene_token') != scene_token
        dt = 0.0
        delta = 0.0
        speed = 0.0
        accel = 0.0
        yaw_rate = 0.0
        yaw_accel = 0.0
        if not is_scene_start:
            raw_dt = timestamp - prev.get('timestamp', timestamp)
            dt = raw_dt / 1e6 if abs(raw_dt) > 1000.0 else raw_dt
            if dt > 0.0:
                delta = math.hypot(x - prev.get('ego_x_m', x), y - prev.get('ego_y_m', y))
                speed = delta / dt
                yaw_rate = math.degrees(self._tm_yaw_delta(prev.get('ego_yaw_rad', yaw), yaw)) / dt
                accel = (speed - prev.get('ego_speed_mps', 0.0)) / dt
                yaw_accel = (yaw_rate - prev.get('yaw_rate_deg_s', 0.0)) / dt
        features = dict(
            frame_number=float(frame_number),
            info_frame_idx=float(meta.get('frame_idx', frame_number)),
            is_scene_start=float(is_scene_start),
            dt_s=float(dt),
            ego_x_m=float(x),
            ego_y_m=float(y),
            ego_z_m=float(z),
            ego_yaw_deg=float(math.degrees(yaw)),
            delta_ego_translation_m=float(delta),
            ego_speed_mps=float(speed),
            ego_accel_mps2=float(accel),
            yaw_rate_deg_s=float(yaw_rate),
            abs_yaw_rate_deg_s=float(abs(yaw_rate)),
            yaw_accel_deg_s2=float(yaw_accel),
            abs_yaw_accel_deg_s2=float(abs(yaw_accel)),
        )
        current_gt_features = self._tm_annotation_features(data, img_metas=img_metas)
        self._tm_apply_policy_offline_features(features, current_gt_features)
        previous_l8_similarity = getattr(self, '_tm_policy_previous_l8_similarity', None)
        if not is_scene_start and previous_l8_similarity is not None:
            features['previous_l8_similarity'] = float(previous_l8_similarity)
        previous_l12_similarity = getattr(self, '_tm_policy_previous_l12_similarity', None)
        if not is_scene_start and previous_l12_similarity is not None:
            features['previous_l12_similarity'] = float(previous_l12_similarity)
        if l8_similarity:
            features.update(l8_similarity)
        current_gt_boxes = self._tm_tensor(data.get('gt_bboxes_3d', None))
        if current_gt_boxes is None:
            current_gt_boxes = self._tm_tensor(meta.get('gt_bboxes_3d', None))
        features['_state'] = dict(
            sample_token=None if sample_token is None else str(sample_token),
            scene_token=scene_token,
            frame_number=frame_number,
            timestamp=timestamp,
            ego_x_m=x,
            ego_y_m=y,
            ego_yaw_rad=yaw,
            ego_speed_mps=speed,
            yaw_rate_deg_s=yaw_rate,
            is_scene_start=is_scene_start,
            offline_gt_features=current_gt_features,
            current_gt_boxes=current_gt_boxes.detach().cpu() if current_gt_boxes is not None else None,
        )
        return features

    def _temporal_memory_commit_policy_state(self, policy_features, l8_features=None, l12_features=None):
        state = policy_features.get('_state', None) if policy_features is not None else None
        if policy_features is not None:
            self._tm_policy_prev_state = state
        if l8_features is not None:
            feature = l8_features[0] if isinstance(l8_features, (list, tuple)) else l8_features
            previous_l8_features = getattr(self, '_tm_policy_prev_l8_features', None)
            if state is not None and state.get('is_scene_start', False):
                self._tm_policy_previous_l8_similarity = None
            elif previous_l8_features is not None:
                similarity = self._temporal_memory_l8_similarity_features(
                    previous_l8_features,
                    feature)
                self._tm_policy_previous_l8_similarity = similarity.get('similarity_L8_mean', None)
            if hasattr(feature, 'detach'):
                self._tm_policy_prev_l8_features = feature.detach()
        if l12_features is not None:
            feature = l12_features[0] if isinstance(l12_features, (list, tuple)) else l12_features
            previous_l12_features = getattr(self, '_tm_policy_prev_l12_features', None)
            if state is not None and state.get('is_scene_start', False):
                self._tm_policy_previous_l12_similarity = None
            elif previous_l12_features is not None:
                similarity = self._temporal_memory_l8_similarity_features(
                    previous_l12_features,
                    feature)
                self._tm_policy_previous_l12_similarity = similarity.get('similarity_L8_mean', None)
            if hasattr(feature, 'detach'):
                self._tm_policy_prev_l12_features = feature.detach()

    def _temporal_memory_action_layer(self, action):
        from projects.mmdet3d_plugin.models.temporal_memory import action_to_layer
        layer = action_to_layer(action)
        if layer is None:
            return self.temporal_memory.get('partial_layer', 'middle')
        return f'block{layer}'

    def _tm_depth_router_metadata_tensor(self, feature_rows, device, dtype):
        router_cfg = self.temporal_memory.get('router', {}) or {}
        feature_names = tuple(router_cfg.get('feature_names', ()))
        forbidden = {'gt_bboxes_3d', 'gt_labels_3d', 'num_annotations'}
        values = []
        for row in feature_rows:
            if any(name in row for name in forbidden):
                raise ValueError('depth router accepts online-only metadata')
            values.append([float(row.get(name, 0.0) or 0.0) for name in feature_names])
        return torch.as_tensor(values, device=device, dtype=dtype)

    def _tm_depth_router_enabled(self):
        return getattr(self, 'temporal_memory_depth_router', None) is not None

    @staticmethod
    def _tm_depth_router_tokens_to_spatial(tokens, batch_size, num_cams):
        if tokens.dim() != 4:
            raise ValueError('EVA tokens must be [B*N_cam, H, W, C]')
        if tokens.size(0) != batch_size * num_cams:
            raise ValueError('token batch must equal batch_size * num_cams')
        height, width, channels = tokens.shape[1:]
        return tokens.permute(0, 3, 1, 2).reshape(
            batch_size, num_cams, channels, height, width)

    def _tm_depth_router_grouped_partial_features(
        self, l8_tokens, batch_size, num_cams, actions, l8_projected,
    ):
        """Run only the selected L12/L16 continuations for a routed batch.

        ``actions`` is deliberately indexed by sample, not camera.  Every
        selected sample contributes all of its camera tokens to the same
        continuation group, and each group remains in its own (stable) sample
        order until the caller scatters detector features back to batch order.
        """
        if actions.dim() != 1 or actions.numel() != batch_size:
            raise ValueError('depth-router actions must be one per sample')
        if not isinstance(l8_projected, (list, tuple)):
            raise ValueError('L8 projector output must be a feature list')

        partials = {}
        l8_indices = torch.nonzero(actions == 0, as_tuple=False).flatten()
        if l8_indices.numel():
            partials[0] = [feature.index_select(0, l8_indices) for feature in l8_projected]

        tokens_by_sample = l8_tokens.reshape(
            batch_size, num_cams, *l8_tokens.shape[1:])
        for action_index, stop_at in ((1, 11), (2, 15)):
            sample_indices = torch.nonzero(actions == action_index, as_tuple=False).flatten()
            if not sample_indices.numel():
                continue
            grouped_tokens = tokens_by_sample.index_select(0, sample_indices).reshape(
                -1, *l8_tokens.shape[1:])
            continued = self.img_backbone.forward_blocks(
                grouped_tokens, start_index=8, stop_at=stop_at)
            spatial = self._tm_depth_router_tokens_to_spatial(
                continued, int(sample_indices.numel()), num_cams)
            action = ('L12', 'L16')[action_index - 1]
            partials[action_index] = self._project_temporal_memory_partial(
                spatial, layer_id=f'block{stop_at + 1}', action=action)
        return partials

    def _tm_depth_router_prepare_l8(self, data):
        """Compute the shared L8 prefix once for every sample and camera."""
        img = data['img']
        if img.dim() != 5:
            raise ValueError('depth routing expects per-frame images [B, N_cam, C, H, W]')
        batch_size, num_cams = img.shape[:2]
        flat_img = img.reshape(batch_size * num_cams, *img.shape[2:])
        if self.use_grid_mask:
            flat_img = self.grid_mask(flat_img)
        tokens = self.img_backbone.prepare_tokens(flat_img)
        l8_tokens = self.img_backbone.forward_blocks(tokens, start_index=0, stop_at=7)
        l8_spatial = self._tm_depth_router_tokens_to_spatial(
            l8_tokens, batch_size, num_cams)
        l8_projected = self._project_temporal_memory_partial(
            l8_spatial, layer_id='block8', action='L8')
        return l8_tokens, l8_projected

    @staticmethod
    def _tm_depth_router_batch_value(value, index, default=0.0):
        if value is None:
            return default
        if torch.is_tensor(value):
            if value.numel() == 0:
                return default
            value = value.reshape(-1)[min(index, value.numel() - 1)].item()
        elif isinstance(value, (list, tuple)):
            if not value:
                return default
            value = value[min(index, len(value) - 1)]
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _tm_depth_router_online_rows(self, img_metas, data):
        """Collect only values available at inference for each sample router."""
        router_cfg = self.temporal_memory.get('router', {}) or {}
        feature_names = tuple(router_cfg.get('feature_names', ()))
        forbidden = {'gt_bboxes_3d', 'gt_labels_3d', 'num_annotations'}
        if forbidden.intersection(feature_names):
            raise ValueError('depth router accepts online-only metadata')
        rows = []
        for index, meta in enumerate(img_metas):
            meta = meta or {}
            row = {}
            for name in feature_names:
                if name == 'frame_number':
                    value = meta.get('frame_idx', 0.0)
                else:
                    value = meta.get(name, None)
                    if value is None:
                        value = self._tm_depth_router_batch_value(data.get(name), index)
                row[name] = self._tm_depth_router_batch_value(value, 0)
            rows.append(row)
        return rows

    @staticmethod
    def _tm_depth_router_sample_key(meta, index):
        # Batch index distinguishes parallel clips that belong to one scene.
        return (meta.get('scene_token', None), int(index))

    def _tm_depth_router_previous_memory(self, sample_keys, current_l8):
        previous = []
        states = getattr(self.temporal_memory_controller, 'states', {})
        for index, sample_key in enumerate(sample_keys):
            state = states.get(sample_key)
            memory = None if state is None else state.get('memory')
            if memory is None:
                previous.append(torch.zeros_like(current_l8[index:index + 1]))
                continue
            feature = memory[0] if isinstance(memory, (list, tuple)) else memory
            if feature.shape[1:] != current_l8.shape[1:]:
                raise ValueError('cached memory must match the L8 router feature shape')
            previous.append(feature)
        return torch.cat(previous, dim=0)

    def _tm_depth_router_extract_full_group(self, data, indices, training):
        index_tensor = torch.as_tensor(indices, device=data['img'].device, dtype=torch.long)
        group_data = dict(data)
        group_data['img'] = data['img'].index_select(0, index_tensor)
        full = self._profile_extract_full_features(
            group_data, len_queue=1, training_mode=bool(training))
        return self._squeeze_temporal_feature_dim(full)

    @staticmethod
    def _tm_depth_router_select_group_feature(features, group_indices, sample_index):
        local_index = group_indices.index(sample_index)
        return [feature[local_index:local_index + 1] for feature in features]

    def _tm_depth_router_forced_actions(self, sample_keys, scene_tokens, prev_exists, actions):
        forced = actions.clone()
        controller = self.temporal_memory_controller
        states = getattr(controller, 'states', {})
        for index, sample_key in enumerate(sample_keys):
            state = states.get(sample_key)
            no_memory = state is None or state.get('memory') is None
            scene_changed = state is not None and state.get('scene_token') != scene_tokens[index]
            max_full = getattr(controller, 'max_frames_between_full', None)
            age = 0 if state is None else state.get('frames_since_full', 0)
            reset_due = max_full is not None and age >= int(max_full) - 1
            if no_memory or scene_changed or not bool(prev_exists[index]) or reset_due:
                forced[index] = 3
        return forced

    def _tm_route_depth_batch(self, data, img_metas, training=False):
        """Hard-route a batch by depth while retaining independent cache states."""
        from projects.mmdet3d_plugin.models.temporal_memory import depth_router_losses

        batch_size, num_cams = data['img'].shape[:2]
        if len(img_metas) != batch_size:
            raise ValueError('depth router requires one metadata dict per sample')
        l8_tokens, l8_projected = self._tm_depth_router_prepare_l8(data)
        sample_keys = [self._tm_depth_router_sample_key(meta, index)
                       for index, meta in enumerate(img_metas)]
        scene_tokens = [meta.get('scene_token') for meta in img_metas]
        prev_value = data.get('prev_exists', None)
        prev_exists = [bool(self._tm_depth_router_batch_value(prev_value, index, 1.0))
                       for index in range(batch_size)]
        metadata = self._tm_depth_router_metadata_tensor(
            self._tm_depth_router_online_rows(img_metas, data),
            device=l8_projected[0].device,
            dtype=l8_projected[0].dtype)
        route = self.temporal_memory_depth_router(
            current_l8=l8_projected[0],
            previous_memory=self._tm_depth_router_previous_memory(sample_keys, l8_projected[0]),
            metadata=metadata,
            training=training)
        actions = self._tm_depth_router_forced_actions(
            sample_keys, scene_tokens, prev_exists, route.hard_actions)
        partial_groups = self._tm_depth_router_grouped_partial_features(
            l8_tokens=l8_tokens,
            batch_size=batch_size,
            num_cams=num_cams,
            actions=actions,
            l8_projected=l8_projected)
        action_indices = {
            action_index: torch.nonzero(actions == action_index, as_tuple=False).flatten().tolist()
            for action_index in range(4)
        }
        full_features = None
        if action_indices[3]:
            full_features = self._tm_depth_router_extract_full_group(
                data, action_indices[3], training)

        names = ('L8', 'L12', 'L16', 'L24')
        requests = []
        for index, action_index in enumerate(actions.tolist()):
            action = names[action_index]
            if action == 'L24':
                current_full = self._tm_depth_router_select_group_feature(
                    full_features, action_indices[3], index)
                request = dict(extract_full=lambda value=current_full: value)
            else:
                partial = self._tm_depth_router_select_group_feature(
                    partial_groups[action_index], action_indices[action_index], index)
                # Straight-through selected-route scaling is exactly one in the
                # forward pass, but exposes the detection loss to router logits.
                selector = route.hard_one_hot[index, action_index]
                request = dict(partial_current=[feature * selector for feature in partial])
            request.update(
                sample_key=sample_keys[index],
                scene_token=scene_tokens[index],
                prev_exists=prev_exists[index],
                action=action,
            )
            requests.append(request)
        outputs = self.temporal_memory_controller.get_or_update_samples(
            requests, training=training)
        features = [torch.cat([output[level] for output in outputs], dim=0)
                    for level in range(len(outputs[0]))]
        router_cfg = self.temporal_memory.get('router', {}) or {}
        losses = depth_router_losses(
            route,
            target_cost=router_cfg.get('target_cost', 0.5),
            balance_prior=router_cfg.get('balance_prior', None),
            balance_weight=router_cfg.get('balance_weight', 0.0))
        losses.pop('expected_cost', None)
        return features, losses

    def _project_temporal_memory_partial(self, partial_features, layer_id=None, action=None):
        projector = self.temporal_memory_projector
        try:
            signature = inspect.signature(projector.forward if isinstance(projector, torch.nn.Module) else projector)
        except (TypeError, ValueError):
            return projector(partial_features)
        kwargs = {}
        if 'layer_id' in signature.parameters:
            kwargs['layer_id'] = layer_id
        if 'action' in signature.parameters:
            kwargs['action'] = action
        return projector(partial_features, **kwargs)

    @staticmethod
    def _temporal_memory_batch_size_from_data(data):
        img = data.get('img', None)
        if torch.is_tensor(img) and img.dim() > 0:
            return int(img.size(0))
        return 1

    @staticmethod
    def _temporal_memory_slice_batch_value(value, indices):
        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) >= int(indices.max().item()) + 1:
            return value.index_select(0, indices.to(value.device))
        if isinstance(value, list) and len(value) >= int(indices.max().item()) + 1:
            return [value[int(index)] for index in indices.cpu().tolist()]
        if isinstance(value, tuple) and len(value) >= int(indices.max().item()) + 1:
            return tuple(value[int(index)] for index in indices.cpu().tolist())
        return value

    def _temporal_memory_slice_batch_data(self, data, indices):
        return {
            key: self._temporal_memory_slice_batch_value(value, indices)
            for key, value in data.items()
        }

    @staticmethod
    def _temporal_memory_select_feature_batch(features, indices):
        if features is None:
            return None
        return [
            feature.index_select(0, indices.to(feature.device))
            for feature in features
        ]

    @staticmethod
    def _temporal_memory_detach_features(features):
        if features is None:
            return None
        if isinstance(features, list):
            return [
                feature.detach() if hasattr(feature, 'detach') else feature
                for feature in features
            ]
        if isinstance(features, tuple):
            return tuple(
                feature.detach() if hasattr(feature, 'detach') else feature
                for feature in features
            )
        return features.detach() if hasattr(features, 'detach') else features

    def _temporal_memory_ensure_batch_state(self, batch_size):
        states = getattr(self, '_tm_batch_states', None)
        if states is not None and len(states) == batch_size:
            return
        self._tm_batch_states = [
            dict(scene_token=None, memory=None, frame_index=0, frames_since_full=0)
            for _ in range(batch_size)
        ]

    def _temporal_memory_gather_sample_memories(self, indices):
        memories = []
        for index in indices.cpu().tolist():
            memory = self._tm_batch_states[int(index)].get('memory')
            if memory is None:
                raise RuntimeError('temporal memory sample is missing previous memory')
            memories.append(memory)
        return [
            torch.cat([memory[level] for memory in memories], dim=0)
            for level in range(len(memories[0]))
        ]

    @staticmethod
    def _temporal_memory_sample_prev_exists(prev_exists, batch_index):
        if prev_exists is None:
            return True
        if torch.is_tensor(prev_exists):
            if prev_exists.numel() == 0:
                return True
            if prev_exists.numel() == 1:
                return bool(prev_exists.detach().bool().item())
            return bool(prev_exists.reshape(-1)[batch_index].detach().bool().item())
        if isinstance(prev_exists, (list, tuple)):
            if len(prev_exists) == 0:
                return True
            value = prev_exists[min(batch_index, len(prev_exists) - 1)]
            return RepDetr3D._temporal_memory_sample_prev_exists(value, 0)
        return bool(prev_exists)

    def _temporal_memory_sample_force_full_for_age(self, state, policy_features=None):
        controller = getattr(self, 'temporal_memory_controller', None)
        if controller is None:
            return False
        max_frames = controller._max_frames_for_policy_features(policy_features)
        if max_frames is None:
            return False
        return int(state.get('frames_since_full', 0)) >= max(int(max_frames) - 1, 0)

    @staticmethod
    def _temporal_memory_scatter_feature_batches(group_outputs, batch_size):
        if not group_outputs:
            return None
        first_features = next(iter(group_outputs.values()))
        merged = []
        for level, reference in enumerate(first_features):
            pieces = []
            for indices, features in group_outputs.items():
                feature = features[level]
                full = feature.new_zeros((batch_size,) + tuple(feature.shape[1:]))
                full.index_copy_(0, indices.to(feature.device), feature)
                pieces.append(full)
            merged.append(sum(pieces))
        return merged

    @staticmethod
    def _temporal_memory_scatter_feature_group_list(output_groups, batch_size):
        if not output_groups:
            return None
        merged = []
        for level in range(len(output_groups[0][1])):
            pieces = []
            for indices, features in output_groups:
                feature = features[level]
                full = feature.new_zeros((batch_size,) + tuple(feature.shape[1:]))
                full.index_copy_(0, indices.to(feature.device), feature)
                pieces.append(full)
            merged.append(sum(pieces))
        return merged

    def _temporal_memory_project_mixed_batch_partials(self, frame_data, actions):
        batch_size = self._temporal_memory_batch_size_from_data(frame_data)
        grouped_outputs = {}
        for action in ('L8', 'L12', 'L16'):
            indices = torch.as_tensor(
                [index for index, value in enumerate(actions) if value == action],
                dtype=torch.long)
            if indices.numel() == 0:
                continue
            layer_id = self._temporal_memory_action_layer(action)
            group_data = self._temporal_memory_slice_batch_data(frame_data, indices)
            partial_features = self._profile_extract_partial_features(group_data, layer_id)
            grouped_outputs[tuple(indices.tolist())] = self._project_temporal_memory_partial(
                partial_features,
                layer_id=layer_id,
                action=action)
        keyed_outputs = {
            torch.as_tensor(indices, dtype=torch.long): features
            for indices, features in grouped_outputs.items()
        }
        return self._temporal_memory_scatter_feature_batches(keyed_outputs, batch_size)

    @staticmethod
    def _squeeze_temporal_feature_dim(features):
        squeezed = []
        for feature in features:
            if feature.dim() == 6 and feature.size(1) == 1:
                squeezed.append(feature[:, 0])
            else:
                squeezed.append(feature)
        return squeezed

    def _temporal_memory_parameter_anchor_loss(self):
        modules = [getattr(self, 'temporal_memory_projector', None)]
        modules.append(getattr(self, 'temporal_memory_updater', None))
        modules.append(getattr(self, 'temporal_memory_policy', None))
        modules.append(getattr(self, 'temporal_memory_depth_router', None))

        anchor = None
        for module in modules:
            if not isinstance(module, torch.nn.Module):
                continue
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                term = param.float().sum() * 0.0
                anchor = term if anchor is None else anchor + term
        return anchor

    def _freeze_temporal_memory_pretrained_detector(self):
        for param in self.parameters():
            param.requires_grad = False
        for module in (
            getattr(self, 'temporal_memory_projector', None),
            getattr(self, 'temporal_memory_updater', None),
            getattr(self, 'temporal_memory_policy', None),
            getattr(self, 'temporal_memory_depth_router', None),
        ):
            if not isinstance(module, torch.nn.Module):
                continue
            for param in module.parameters():
                param.requires_grad = True
   

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
        
        img_feats_reshaped = []

        if self.training or training_mode:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(B, len_queue, int(BN/B / len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)
        else:
            for i in self.position_level:
                BN, C, H, W = img_feats[i].size()
                img_feat_reshaped = img_feats[i].view(B, int(BN/B/len_queue), C, H, W)
                img_feats_reshaped.append(img_feat_reshaped)

        self._maybe_log_tm_shapes(img_feats_reshaped)
        return img_feats_reshaped


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
                if key == 'img_feats':
                    data_t[key] = [feat[:, i] for feat in data[key]]
                else:
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

    def forward_roi_head(self, **data):
        if (self.aux_2d_only and not self.training) or not self.with_img_roi_head:
            return {'topk_indexes':None}
        else:
            outs_roi = self.img_roi_head(**data)
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

        if not requires_grad:
            self.eval()
            with torch.no_grad():
                outs = self.pts_bbox_head(img_metas, **data)
            self.train()

        else:
            outs_roi = self.forward_roi_head(**data)
            outs = self.pts_bbox_head(img_metas, **data)

        if return_losses:
            loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
            losses = self.pts_bbox_head.loss(*loss_inputs)
            if self.with_img_roi_head:
                loss2d_inputs = [gt_bboxes, gt_labels, centers2d, outs_roi, depths, img_metas]
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
                Defaults to None.extract_feat
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

        tm_losses = {}
        if (
            self.temporal_memory_enabled
            and self.temporal_memory.get('updater', 'reuse') != 'reuse'
        ):
            policy_annotation_data = dict(
                gt_bboxes_3d=gt_bboxes_3d,
                gt_labels_3d=gt_labels_3d,
                gt_bboxes=gt_bboxes,
                gt_labels=gt_labels,
            )
            data['img_feats'], tm_losses = self._extract_img_feat_temporal_memory_train(
                img_metas, policy_annotation_data=policy_annotation_data, **data)
        else:
            T = data['img'].size(1)

            prev_img = data['img'][:, :-self.num_frame_backbone_grads]
            rec_img = data['img'][:, -self.num_frame_backbone_grads:]
            rec_img_feats = self.extract_feat(rec_img, self.num_frame_backbone_grads)

            if T-self.num_frame_backbone_grads > 0:
                self.eval()
                with torch.no_grad():
                    prev_img_feats = self.extract_feat(prev_img, T-self.num_frame_backbone_grads, True)
                self.train()
                data['img_feats'] = [torch.cat([prev_img_feats[i], rec_img_feats[i]], dim=1) for i in range(len(self.position_level))]
            else:
                data['img_feats'] = rec_img_feats

        losses = self.obtain_history_memory(gt_bboxes_3d,
                        gt_labels_3d, gt_bboxes,
                        gt_labels, img_metas, centers2d, depths, gt_bboxes_ignore, **data)
        losses.update(tm_losses)
        if (
            self.temporal_memory_enabled
            and self.temporal_memory.get('updater', 'reuse') != 'reuse'
        ):
            anchor_loss = self._temporal_memory_parameter_anchor_loss()
            if anchor_loss is not None:
                losses['loss_tm_param_anchor'] = anchor_loss

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
        outs_roi = self.forward_roi_head(**data)

        if img_metas[0]['scene_token'] != self.prev_scene_token:
            self.prev_scene_token = img_metas[0]['scene_token']
            data['prev_exists'] = data['img'].new_zeros(1)
            self.pts_bbox_head.reset_memory()
        else:
            data['prev_exists'] = data['img'].new_ones(1)

        outs = self.pts_bbox_head(img_metas, **data)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        self._tm_policy_prev_prediction_features = self._tm_prediction_features(bbox_results)
        return bbox_results
    
    def simple_test(self, img_metas, **data):
        """Test function without augmentaiton."""
        if self.tm_profiler is not None:
            self.tm_profiler.reset()
        with self.tm_profiler.measure('total') if self.tm_profiler is not None else _nullcontext():
            data['img_feats'] = self._extract_img_feat_temporal_memory(img_metas, **data)

            bbox_list = [dict() for i in range(len(img_metas))]
            with self.tm_profiler.measure('head') if self.tm_profiler is not None else _nullcontext():
                bbox_pts = self.simple_test_pts(
                    img_metas, **data)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        if self.tm_profiler is not None:
            self.tm_profiler.log()
            self._maybe_log_temporal_memory_policy_stats()
        return bbox_list

    
