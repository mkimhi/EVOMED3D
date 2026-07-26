import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def test_profiler_disabled_does_not_record_or_synchronize(monkeypatch):
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryProfiler

    calls = []
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.synchronize", lambda: calls.append("sync"))

    profiler = TemporalMemoryProfiler(enabled=False)
    with profiler.measure("backbone_full"):
        pass

    assert calls == []
    assert profiler.report() == {}


def test_profiler_records_latency_and_gpu_memory_with_cuda_sync(monkeypatch):
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryProfiler

    calls = []
    clock = iter([1.0, 1.025])
    monkeypatch.setattr("time.perf_counter", lambda: next(clock))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr("torch.cuda.memory_allocated", lambda: 1024 * 1024)
    monkeypatch.setattr("torch.cuda.max_memory_allocated", lambda: 3 * 1024 * 1024)

    profiler = TemporalMemoryProfiler(enabled=True)
    with profiler.measure("updater"):
        pass

    report = profiler.report()
    assert calls == ["sync", "sync"]
    assert report["updater_latency_ms"] == 25.0
    assert report["gpu_memory_allocated_mb"] == 1.0
    assert report["gpu_memory_max_allocated_mb"] == 3.0


def test_profiler_prints_requested_latency_fields_at_interval(capsys):
    from projects.mmdet3d_plugin.models.temporal_memory import TemporalMemoryProfiler

    profiler = TemporalMemoryProfiler(enabled=True, log_interval=2)
    profiler.records.update({
        "backbone_full": 1.0,
        "backbone_partial": 2.0,
        "updater": 3.0,
        "head": 4.0,
        "total": 10.0,
    })

    profiler.log()
    assert capsys.readouterr().out == ""

    profiler.records.update({
        "backbone_full": 3.0,
        "backbone_partial": 4.0,
        "updater": 5.0,
        "head": 6.0,
        "total": 20.0,
    })
    profiler.log()
    out = capsys.readouterr().out

    assert "samples=2 window=2 avg" in out
    assert "backbone_full_latency_ms=2.000" in out
    assert "backbone_partial_latency_ms=3.000" in out
    assert "updater_latency_ms=4.000" in out
    assert "head_latency_ms=5.000" in out
    assert "total_latency_ms=15.000" in out
