import torch
import inspect

from .policies import build_frame_policy


class TemporalMemoryController:
    def __init__(
        self,
        enabled=False,
        keyframe_interval=1,
        updater='reuse',
        updater_cfg=None,
        policy=None,
        partial_layer='block8',
        train_with_teacher=False,
        distill_weight=1.0,
        inference_teacher_forbidden=True,
        max_frames_between_full=None,
        adaptive_reset=None,
        profiler=None,
        debug=False,
    ):
        if keyframe_interval < 1:
            raise ValueError('keyframe_interval must be >= 1')
        self.enabled = enabled
        self.keyframe_interval = keyframe_interval
        self.updater_name = updater.get('type', updater.get('name', 'reuse')) if isinstance(updater, dict) else updater
        self.train_with_teacher = train_with_teacher
        self.inference_teacher_forbidden = inference_teacher_forbidden
        self.debug = debug
        self.profiler = profiler
        self.frame_index = 0
        self.frames_since_full = 0
        self.scene_token = None
        self.memory = None
        self.last_action = None
        self.last_action_forced = False
        self.last_distillation_loss = None
        self.max_frames_between_full = max_frames_between_full
        self.adaptive_reset = adaptive_reset or {}
        self._policy_stats = {
            'forced_l24': 0,
            'forced_age_l24': 0,
            'forced_motion_l24': 0,
            'learned_l24': 0,
        }
        self.states = {}
        self.policy = build_frame_policy(
            policy,
            keyframe_interval=keyframe_interval,
            partial_layer=partial_layer)

        from .updaters import build_memory_updater
        self.updater = build_memory_updater(updater, **(updater_cfg or {}))
        from .distillation import DistillationLoss
        self.distillation_loss = DistillationLoss(weight=distill_weight)

    @staticmethod
    def _prev_exists_all(prev_exists):
        if prev_exists is None:
            return True
        if isinstance(prev_exists, torch.Tensor):
            if prev_exists.numel() == 0:
                return True
            return bool(prev_exists.detach().bool().all().item())
        if isinstance(prev_exists, (list, tuple)):
            return all(TemporalMemoryController._prev_exists_all(item) for item in prev_exists)
        return bool(prev_exists)

    def reset(self, reason='manual'):
        if self.debug:
            print(f'[temporal_memory] reset reason={reason}')
        self.frame_index = 0
        self.frames_since_full = 0
        self.memory = None

    def will_use_keyframe(self, scene_token=None, prev_exists=None, policy_features=None):
        return self.peek_action(scene_token=scene_token, prev_exists=prev_exists, policy_features=policy_features) == 'L24'

    def peek_action(self, scene_token=None, prev_exists=None, policy_features=None):
        if not self.enabled:
            return 'L24'
        is_reset_frame = (
            scene_token != self.scene_token
            or not self._prev_exists_all(prev_exists)
        )
        if self._force_full_for_age(is_reset_frame, policy_features=policy_features):
            return 'L24'
        return self.policy.decide(
            frame_index=self.frame_index,
            has_memory=self.memory is not None,
            is_reset_frame=is_reset_frame,
            scene_token=scene_token,
            prev_exists=prev_exists,
            policy_features=policy_features,
        )

    def _motion_reset_active(self, policy_features=None):
        cfg = self.adaptive_reset or {}
        if not cfg or cfg.get('enabled', True) is False or policy_features is None:
            return False
        checks = (
            ('ego_speed_mps', 'ego_speed_mps'),
            ('max_object_speed_mps', 'max_object_speed_mps'),
            ('avg_object_speed_mps', 'avg_object_speed_mps'),
        )
        for feature_name, threshold_name in checks:
            threshold = cfg.get(threshold_name, None)
            if threshold is None:
                continue
            try:
                value = float(policy_features.get(feature_name, 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value >= float(threshold):
                return True
        return False

    def _max_frames_for_policy_features(self, policy_features=None):
        if self._motion_reset_active(policy_features):
            return (self.adaptive_reset or {}).get(
                'max_frames_between_full', self.max_frames_between_full)
        return self.max_frames_between_full

    def _force_full_for_age(self, is_reset_frame=False, policy_features=None):
        max_frames = self._max_frames_for_policy_features(policy_features)
        if is_reset_frame or self.memory is None or max_frames is None:
            return False
        return self.frames_since_full >= max(int(max_frames) - 1, 0)

    def _force_full_reason(self, is_reset_frame=False, policy_features=None):
        if not self._force_full_for_age(
            is_reset_frame, policy_features=policy_features):
            return None
        return 'motion' if self._motion_reset_active(policy_features) else 'age'

    def policy_stats(self):
        stats = dict(self._policy_stats)
        for action, count in sorted(stats.items()):
            stats[action] = int(count)
        return stats

    def get_or_update(
        self,
        extract_full,
        scene_token=None,
        prev_exists=None,
        partial_current=None,
        ego_motion_embedding=None,
        extract_teacher=None,
        training=False,
        policy_features=None,
    ):
        self.last_distillation_loss = None
        if not self.enabled:
            return extract_full()

        if scene_token != self.scene_token:
            self.scene_token = scene_token
            self.reset(reason='scene_boundary')

        if not self._prev_exists_all(prev_exists):
            self.reset(reason='discontinuity')

        is_reset_frame = (
            scene_token != self.scene_token
            or not self._prev_exists_all(prev_exists)
        )
        force_reason = self._force_full_reason(
            is_reset_frame, policy_features=policy_features)
        forced = force_reason is not None
        action = self.peek_action(scene_token=scene_token, prev_exists=prev_exists, policy_features=policy_features)
        self.last_action = action
        self.last_action_forced = forced
        self._policy_stats[action] = self._policy_stats.get(action, 0) + 1
        is_keyframe = action == 'L24'
        if forced:
            self._policy_stats['forced_l24'] = self._policy_stats.get('forced_l24', 0) + 1
            if force_reason == 'motion':
                self._policy_stats['forced_motion_l24'] = (
                    self._policy_stats.get('forced_motion_l24', 0) + 1)
            else:
                self._policy_stats['forced_age_l24'] = (
                    self._policy_stats.get('forced_age_l24', 0) + 1)
        elif is_keyframe and self.memory is not None:
            self._policy_stats['learned_l24'] = self._policy_stats.get('learned_l24', 0) + 1

        if is_keyframe:
            if self.debug:
                print(f'[temporal_memory] frame={self.frame_index} decision=keyframe')
            self.memory = extract_full()
            output = self.memory
        else:
            if self.debug:
                print(f'[temporal_memory] frame={self.frame_index} decision=non_keyframe updater={self.updater_name}')
            if self.profiler is None:
                output = self._call_updater(
                    self.memory,
                    partial_current=partial_current,
                    ego_motion_embedding=ego_motion_embedding,
                    action=action)
            else:
                with self.profiler.measure('updater'):
                    output = self._call_updater(
                        self.memory,
                        partial_current=partial_current,
                        ego_motion_embedding=ego_motion_embedding,
                        action=action)
            if not training and extract_teacher is not None and self.inference_teacher_forbidden:
                raise AssertionError('Full-feature teacher is forbidden on non-keyframe inference.')
            if training and self.train_with_teacher and extract_teacher is not None:
                with torch.no_grad():
                    teacher = extract_teacher()
                self.last_distillation_loss = self.distillation_loss(output, teacher)

        self.frame_index += 1
        self.frames_since_full = 0 if is_keyframe else self.frames_since_full + 1
        return output

    def _call_updater(self, previous_memory, **kwargs):
        if self._updater_accepts_action():
            return self.updater(previous_memory, **kwargs)
        kwargs.pop('action', None)
        return self.updater(previous_memory, **kwargs)

    def _updater_accepts_action(self):
        target = self.updater.forward if isinstance(self.updater, torch.nn.Module) else self.updater
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            return False
        if 'action' in signature.parameters:
            return True
        return any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in signature.parameters.values()
        )

    def get_or_update_samples(self, requests, training=False):
        """Update independent temporal-memory states in request order.

        Each request contains a stable ``sample_key``, an already-selected
        ``action``, and a single-sample feature tensor. This is deliberately
        separate from policy selection: a detector can select and group routes
        for a batch, while this controller owns only per-sequence cache state.
        """
        outputs = []
        for request in requests:
            sample_key = request['sample_key']
            state = self.states.get(sample_key)
            if state is None:
                state = dict(scene_token=None, frame_index=0, frames_since_full=0, memory=None)
                self.states[sample_key] = state

            scene_token = request.get('scene_token')
            prev_exists = request.get('prev_exists', True)
            if scene_token != state['scene_token'] or not self._prev_exists_all(prev_exists):
                state.update(scene_token=scene_token, frame_index=0, frames_since_full=0, memory=None)

            action = request.get('action', 'L24')
            forced = (
                state['memory'] is None
                or (self.max_frames_between_full is not None
                    and state['frames_since_full'] >= int(self.max_frames_between_full) - 1)
            )
            if not self.enabled or forced:
                action = 'L24'
            if action == 'L24':
                output = request['extract_full']()
                state['frames_since_full'] = 0
            else:
                output = self._call_updater(
                    state['memory'],
                    partial_current=request.get('partial_current'),
                    ego_motion_embedding=request.get('ego_motion_embedding'),
                    action=action,
                )
                state['frames_since_full'] += 1
            # The cache survives minibatches, whereas an autograd graph does
            # not.  Keep the current output connected to its detection loss,
            # but truncate history before it can be read by another frame.
            state['memory'] = self._detach_memory(output) if training else output
            state['frame_index'] += 1
            self.last_action = action
            self.last_action_forced = forced
            self._policy_stats[action] = self._policy_stats.get(action, 0) + 1
            if forced:
                self._policy_stats['forced_l24'] = self._policy_stats.get('forced_l24', 0) + 1
            outputs.append(output)
        return outputs

    @staticmethod
    def _detach_memory(memory):
        if torch.is_tensor(memory):
            return memory.detach()
        if isinstance(memory, tuple):
            return tuple(TemporalMemoryController._detach_memory(value) for value in memory)
        if isinstance(memory, list):
            return [TemporalMemoryController._detach_memory(value) for value in memory]
        if isinstance(memory, dict):
            return {key: TemporalMemoryController._detach_memory(value)
                    for key, value in memory.items()}
        return memory
