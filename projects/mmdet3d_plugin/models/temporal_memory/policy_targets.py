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
        if not targets:
            raise ValueError('counterfactual target CSV contains no targets')
        self.targets = targets

    def label_index(self, token, labels):
        token = str(token)
        if token not in self.targets:
            raise KeyError(f'missing counterfactual policy target for token {token}')
        action = self.targets[token]
        labels = list(labels)
        if action not in labels:
            raise ValueError(
                f'counterfactual action {action} is absent from policy labels')
        return labels.index(action)
