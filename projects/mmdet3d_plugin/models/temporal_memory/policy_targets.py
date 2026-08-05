import csv
from pathlib import Path


class CounterfactualActionTargets:
    ACTIONS = ('L8', 'L12', 'L16', 'L24')

    def __init__(
        self,
        csv_path,
        token_column='token',
        action_column='oracle_action',
    ):
        csv_path = Path(csv_path)
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        targets = {}
        scene_frame_targets = {}
        with csv_path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            missing = {token_column, action_column} - set(reader.fieldnames or ())
            if missing:
                raise ValueError(
                    f'counterfactual target CSV missing columns: {sorted(missing)}')
            for row in reader:
                token = str(row[token_column])
                action = str(row[action_column])
                if token in targets:
                    raise ValueError(
                        f'duplicate counterfactual policy target token: {token}')
                if action not in self.ACTIONS:
                    raise ValueError(
                        f'unsupported counterfactual policy action: {action}')
                targets[token] = action
                if 'scene_token' in row and 'frame_number' in row:
                    scene_frame = (
                        str(row['scene_token']), int(float(row['frame_number'])))
                    if scene_frame in scene_frame_targets:
                        raise ValueError(
                            'duplicate counterfactual policy target scene/frame: '
                            f'{scene_frame}')
                    scene_frame_targets[scene_frame] = action
        if not targets:
            raise ValueError('counterfactual target CSV contains no targets')
        self.targets = targets
        self.scene_frame_targets = scene_frame_targets

    def label_index(
        self,
        token,
        labels,
        scene_token=None,
        frame_number=None,
    ):
        token = str(token)
        action = self.targets.get(token)
        if action is None and scene_token is not None and frame_number is not None:
            action = self.scene_frame_targets.get(
                (str(scene_token), int(frame_number)))
        if action is None:
            raise KeyError(f'missing counterfactual policy target for token {token}')
        labels = list(labels)
        if action not in labels:
            raise ValueError(
                f'counterfactual action {action} is absent from policy labels')
        return labels.index(action)
