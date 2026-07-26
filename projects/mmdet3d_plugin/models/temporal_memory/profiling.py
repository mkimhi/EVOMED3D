import time
from contextlib import contextmanager

import torch


class TemporalMemoryProfiler:
    def __init__(self, enabled=False, prefix='temporal_memory', log_interval=200):
        self.enabled = enabled
        self.prefix = prefix
        self.log_interval = log_interval
        self.records = {}
        self.window_records = {}
        self.num_reports = 0
        self.window_reports = 0
        self.last_average_report = {}
        self.gpu_memory_allocated_mb = 0.0
        self.gpu_memory_max_allocated_mb = 0.0

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            self.records[name] = self.records.get(name, 0.0) + elapsed_ms
            self._capture_memory()

    def report(self):
        if not self.enabled:
            return {}
        report = {
            f'{name}_latency_ms': round(value, 3)
            for name, value in self.records.items()
        }
        if self.records.get('total', 0.0) > 0.0:
            report['fps'] = round(1000.0 / self.records['total'], 3)
        report['gpu_memory_allocated_mb'] = round(self.gpu_memory_allocated_mb, 3)
        report['gpu_memory_max_allocated_mb'] = round(self.gpu_memory_max_allocated_mb, 3)
        return report

    def log(self):
        report = self.report()
        if not report:
            return
        for key, value in report.items():
            self.window_records[key] = self.window_records.get(key, 0.0) + value
        self.num_reports += 1
        self.window_reports += 1
        if self.window_reports < self.log_interval:
            return
        average = {
            key: value / self.window_reports
            for key, value in self.window_records.items()
        }
        self.last_average_report = average
        fields = ' '.join(f'{key}={value:.3f}' for key, value in average.items())
        print(f'[{self.prefix}] samples={self.num_reports} window={self.window_reports} avg {fields}')
        self.window_records.clear()
        self.window_reports = 0

    def reset(self):
        self.records.clear()
        self.gpu_memory_allocated_mb = 0.0
        self.gpu_memory_max_allocated_mb = 0.0

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _capture_memory(self):
        if not torch.cuda.is_available():
            return
        mb = 1024.0 * 1024.0
        self.gpu_memory_allocated_mb = torch.cuda.memory_allocated() / mb
        self.gpu_memory_max_allocated_mb = torch.cuda.max_memory_allocated() / mb
