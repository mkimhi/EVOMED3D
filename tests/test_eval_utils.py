import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.eval_utils import format_timing_summary, limit_dataset


class DummyDataset:
    def __init__(self, size):
        self.data_infos = list(range(size))
        self.flag = np.arange(size)

    def __len__(self):
        return len(self.data_infos)


def test_limit_dataset_truncates_data_infos_and_flag():
    dataset = DummyDataset(5)
    dataset.data_infos = [{'token': str(i)} for i in range(5)]

    limit_dataset(dataset, 3)

    assert dataset.data_infos == [{'token': '0'}, {'token': '1'}, {'token': '2'}]
    assert dataset.flag.tolist() == [0, 1, 2]
    assert dataset._eval_sample_tokens == {'0', '1', '2'}


def test_limit_dataset_ignores_none_and_large_limits():
    dataset = DummyDataset(2)

    limit_dataset(dataset, None)
    limit_dataset(dataset, 5)

    assert dataset.data_infos == [0, 1]
    assert dataset.flag.tolist() == [0, 1]


def test_limit_dataset_rejects_non_positive_limits():
    with pytest.raises(ValueError, match='positive'):
        limit_dataset(DummyDataset(2), 0)


def test_format_timing_summary_reports_msec_and_fps():
    summary = format_timing_summary(2.0, 4)

    assert summary == 'Inference timing: 500.00 ms/sample, 2.00 FPS over 4 samples'
