import sys
import importlib.util
from pathlib import Path

import torch
from mmcv import ConfigDict


class DummyDataset:
    CLASSES = ('car',)

    def __init__(self):
        self.evaluated = False

    def evaluate(self, outputs, **kwargs):
        self.evaluated = True
        assert outputs == ['prediction']
        assert kwargs['metric'] == ['bbox']
        return {'ok': True}


class DummyModel:
    pass


class DictResultModel:
    def eval(self):
        self.eval_called = True

    def __call__(self, return_loss=False, rescale=True, **data):
        assert return_loss is False
        assert rescale is True
        return {'bbox_results': ['prediction']}


class DummyLoader:
    def __init__(self):
        self.dataset = [0]

    def __iter__(self):
        yield {'img': torch.tensor([1])}


def test_direct_test_uses_single_gpu_path(monkeypatch):
    test_path = Path(__file__).resolve().parents[1] / 'tools' / 'test.py'
    monkeypatch.syspath_prepend(str(test_path.parents[1]))
    spec = importlib.util.spec_from_file_location('stream_petr_tools_test', test_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    dataset = DummyDataset()

    cfg = ConfigDict(
        model=ConfigDict(pretrained='unused'),
        data=ConfigDict(
            test=ConfigDict(test_mode=False),
            workers_per_gpu=0,
            nonshuffler_sampler=ConfigDict(type='DistributedSampler'),
        ),
        dist_params=ConfigDict(backend='nccl'),
        evaluation=ConfigDict(interval=1),
    )

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'tools/test.py',
            'dummy.py',
            'dummy.pth',
            '--eval',
            'bbox',
        ],
    )
    monkeypatch.setattr(test_module.Config, 'fromfile', lambda _: cfg)
    monkeypatch.setattr(test_module, 'build_dataset', lambda _: dataset)
    monkeypatch.setattr(test_module, 'limit_dataset', lambda ds, _: ds)
    monkeypatch.setattr(test_module, 'build_dataloader', lambda *a, **kw: object())
    monkeypatch.setattr(test_module, 'build_model', lambda *a, **kw: DummyModel())
    monkeypatch.setattr(test_module, 'load_checkpoint', lambda *a, **kw: {})
    monkeypatch.setattr(
        test_module,
        'MMDataParallel',
        lambda model, device_ids: ('wrapped', model, device_ids),
    )
    monkeypatch.setattr(
        test_module,
        'custom_single_gpu_test',
        lambda model, loader, show, show_dir, timing_warmup: ['prediction'],
    )
    monkeypatch.setattr(test_module, 'get_dist_info', lambda: (0, 1))

    test_module.main()

    assert dataset.evaluated


def test_temporal_memory_eval_forces_one_sample_per_gpu(monkeypatch):
    test_path = Path(__file__).resolve().parents[1] / 'tools' / 'test.py'
    monkeypatch.syspath_prepend(str(test_path.parents[1]))
    spec = importlib.util.spec_from_file_location('stream_petr_tools_test_tm', test_path)
    test_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(test_module)

    dataset = DummyDataset()
    loader_kwargs = {}

    cfg = ConfigDict(
        model=ConfigDict(
            pretrained='unused',
            temporal_memory=ConfigDict(enabled=True),
        ),
        data=ConfigDict(
            test=ConfigDict(test_mode=False, samples_per_gpu=2),
            workers_per_gpu=0,
            nonshuffler_sampler=ConfigDict(type='DistributedSampler'),
        ),
        dist_params=ConfigDict(backend='nccl'),
        evaluation=ConfigDict(interval=1),
    )

    monkeypatch.setattr(
        sys,
        'argv',
        [
            'tools/test.py',
            'dummy.py',
            'dummy.pth',
            '--eval',
            'bbox',
        ],
    )
    monkeypatch.setattr(test_module.Config, 'fromfile', lambda _: cfg)
    monkeypatch.setattr(test_module, 'build_dataset', lambda _: dataset)
    monkeypatch.setattr(test_module, 'limit_dataset', lambda ds, _: ds)

    def build_loader(*args, **kwargs):
        loader_kwargs.update(kwargs)
        return object()

    monkeypatch.setattr(test_module, 'build_dataloader', build_loader)
    monkeypatch.setattr(test_module, 'build_model', lambda *a, **kw: DummyModel())
    monkeypatch.setattr(test_module, 'load_checkpoint', lambda *a, **kw: {})
    monkeypatch.setattr(
        test_module,
        'MMDataParallel',
        lambda model, device_ids: ('wrapped', model, device_ids),
    )
    monkeypatch.setattr(
        test_module,
        'custom_single_gpu_test',
        lambda model, loader, show, show_dir, timing_warmup: ['prediction'],
    )
    monkeypatch.setattr(test_module, 'get_dist_info', lambda: (0, 1))

    test_module.main()

    assert loader_kwargs['samples_per_gpu'] == 1


def test_custom_single_gpu_test_flattens_bbox_result_dict():
    from projects.mmdet3d_plugin.core.apis.test import custom_single_gpu_test

    outputs = custom_single_gpu_test(DictResultModel(), DummyLoader(), timing_warmup=0)

    assert outputs == ['prediction']
