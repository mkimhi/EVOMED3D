import torch

def normalize_policy_layer(value):
    if value is None:
        return None
    if isinstance(value, int):
        return str(value)
    text = str(value).lower()
    for prefix in ('block', 'layer', 'l'):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return text


def layer_to_action(layer):
    normalized = normalize_policy_layer(layer)
    if normalized is None:
        return None
    return f'L{normalized}'


def action_to_layer(action):
    if action is None or action == 'L24':
        return None
    return normalize_policy_layer(action)


class FixedIntervalFramePolicy:
    def __init__(self, keyframe_interval=1, partial_action='L8'):
        if keyframe_interval < 1:
            raise ValueError('keyframe_interval must be >= 1')
        self.keyframe_interval = keyframe_interval
        self.partial_action = layer_to_action(partial_action) or 'L8'

    def decide(self, frame_index, has_memory, is_reset_frame=False, **kwargs):
        if is_reset_frame or not has_memory or self.keyframe_interval == 1:
            return 'L24'
        if frame_index % self.keyframe_interval == 0:
            return 'L24'
        return self.partial_action


class CycleFramePolicy:
    def __init__(self, actions):
        if not actions:
            raise ValueError('cycle policy requires at least one action')
        self.actions = tuple(layer_to_action(action) if action != 'L24' else 'L24'
                             for action in actions)

    def decide(self, frame_index, has_memory, is_reset_frame=False, **kwargs):
        if is_reset_frame or not has_memory:
            return 'L24'
        return self.actions[frame_index % len(self.actions)]


class PreviousL8DriftFramePolicy:
    def __init__(
        self,
        reset_interval=4,
        thresholds=(0.8, 0.65, 0.55),
        feature_name='previous_l8_similarity',
        fallback_action='L24',
    ):
        if reset_interval < 1:
            raise ValueError('reset_interval must be >= 1')
        if len(thresholds) != 3:
            raise ValueError('thresholds must contain three values')
        self.reset_interval = int(reset_interval)
        self.thresholds = tuple(float(value) for value in thresholds)
        self.feature_name = feature_name
        self.fallback_action = fallback_action
        self.requires_previous_l8_similarity = True

    def decide(self, frame_index, has_memory, is_reset_frame=False, policy_features=None, **kwargs):
        if is_reset_frame or not has_memory:
            return 'L24'
        if frame_index % self.reset_interval == 0:
            return 'L24'
        if policy_features is None or self.feature_name not in policy_features:
            return self.fallback_action
        similarity = float(policy_features[self.feature_name])
        high, mid, low = self.thresholds
        if similarity > high:
            return 'L8'
        if similarity > mid:
            return 'L12'
        if similarity > low:
            return 'L16'
        return 'L24'


class MlpFramePolicy(torch.nn.Module):
    def __init__(
        self,
        checkpoint=None,
        fallback_action='L24',
        device='cpu',
        load_weights=True,
        feature_names=None,
        hidden_dim=32,
        labels=None,
        actions=None,
        mean=None,
        scale=None,
        teacher_forcing_residual_std=None,
        requires_l8_similarity=False,
        requires_previous_l12_similarity=False,
    ):
        super().__init__()
        if load_weights:
            if checkpoint is None:
                raise ValueError('MlpFramePolicy requires checkpoint unless load_weights=False')
            payload = torch.load(checkpoint, map_location=device)
            feature_names = payload['feature_names']
            mean = payload.get('mean', mean)
            scale = payload.get('scale', scale)
            teacher_forcing_residual_std = payload.get(
                'teacher_forcing_residual_std', teacher_forcing_residual_std)
            hidden_dim = payload.get('hidden_dim', hidden_dim)
            labels = payload.get('labels', labels)
            actions = payload.get('actions', actions)
            requires_l8_similarity = payload.get('requires_l8_similarity', requires_l8_similarity)
            requires_previous_l12_similarity = payload.get(
                'requires_previous_l12_similarity', requires_previous_l12_similarity)
        else:
            payload = {}
            if feature_names is None:
                raise ValueError('MlpFramePolicy requires feature_names when load_weights=False')

        self.feature_names = list(feature_names)
        mean = torch.as_tensor(mean if mean is not None else [0.0] * len(self.feature_names), dtype=torch.float32)
        scale = torch.as_tensor(scale if scale is not None else [1.0] * len(self.feature_names), dtype=torch.float32)
        self.register_buffer('mean', mean)
        self.register_buffer('scale', torch.where(scale == 0, torch.ones_like(scale), scale))
        residual_std = torch.as_tensor(
            teacher_forcing_residual_std
            if teacher_forcing_residual_std is not None
            else [0.0] * len(self.feature_names),
            dtype=torch.float32,
        )
        if residual_std.numel() != len(self.feature_names):
            raise ValueError(
                'teacher_forcing_residual_std must match policy feature count')
        self.register_buffer('teacher_forcing_residual_std', residual_std)
        labels = labels or ['easy', 'medium', 'hard']
        self.labels = [str(label) for label in labels]
        self.actions = dict(actions or {'easy': 'L8', 'medium': 'L12', 'hard': 'L24'})
        hidden_dim = int(hidden_dim)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(len(self.feature_names), hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, len(self.labels)),
        )
        if load_weights:
            self.model.load_state_dict(payload['state_dict'])
        self.model.eval()
        self.fallback_action = fallback_action
        self.requires_l8_similarity = bool(requires_l8_similarity)
        self.requires_previous_l12_similarity = (
            bool(requires_previous_l12_similarity)
            or 'previous_l12_similarity' in self.feature_names
        )

    def _tensor_from_features(self, policy_features):
        values = [float(policy_features.get(name, 0.0) or 0.0) for name in self.feature_names]
        x = torch.as_tensor(values, dtype=torch.float32, device=self.mean.device)
        return (x - self.mean.to(x.device)) / self.scale.to(x.device)

    def logits(self, policy_features):
        return self.model(self._tensor_from_features(policy_features).unsqueeze(0))

    def decide(self, frame_index, has_memory, is_reset_frame=False, policy_features=None, **kwargs):
        if is_reset_frame or not has_memory:
            return 'L24'
        if policy_features is None:
            return self.fallback_action
        with torch.no_grad():
            index = int(self.logits(policy_features).argmax(dim=1).item())
        label = self.labels[index]
        return self.actions.get(label, self.fallback_action)


def build_frame_policy(spec=None, keyframe_interval=1, partial_layer='block8'):
    if spec is None:
        return FixedIntervalFramePolicy(
            keyframe_interval=keyframe_interval,
            partial_action=layer_to_action(partial_layer) or 'L8',
        )
    if callable(spec) and not isinstance(spec, (str, dict)):
        return spec
    if isinstance(spec, str):
        spec = dict(type=spec)
    spec = dict(spec)
    name = spec.pop('type', spec.pop('name', 'fixed_interval'))
    if name == 'fixed_interval':
        spec.setdefault('keyframe_interval', keyframe_interval)
        spec.setdefault('partial_action', layer_to_action(partial_layer) or 'L8')
        return FixedIntervalFramePolicy(**spec)
    if name == 'cycle':
        return CycleFramePolicy(**spec)
    if name in ('previous_l8_drift', 'previous_l8_similarity'):
        return PreviousL8DriftFramePolicy(**spec)
    if name in ('mlp', 'learned_mlp'):
        return MlpFramePolicy(**spec)
    raise KeyError(f'Unsupported temporal memory policy: {name}')
