from __future__ import annotations

import math
import threading
from collections import defaultdict
from time import perf_counter
from typing import Iterable


_DEFAULT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60)


class MetricsRegistry:
    def __init__(self, service: str):
        self.service = service
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, object]] = {}

    def inc(self, name: str, value: float = 1, **labels: object) -> None:
        with self._lock:
            self._counters[(name, self._labels(labels))] += value

    def gauge(self, name: str, value: float, **labels: object) -> None:
        with self._lock:
            self._gauges[(name, self._labels(labels))] = value

    def add_gauge(self, name: str, value: float, **labels: object) -> None:
        with self._lock:
            self._gauges[(name, self._labels(labels))] += value

    def observe(
        self,
        name: str,
        value: float,
        buckets: Iterable[float] = _DEFAULT_BUCKETS,
        **labels: object,
    ) -> None:
        key = (name, self._labels(labels))
        bucket_values = tuple(float(bucket) for bucket in buckets)
        with self._lock:
            hist = self._histograms.get(key)
            if hist is None:
                hist = {
                    "buckets": bucket_values,
                    "counts": [0 for _ in bucket_values],
                    "inf": 0,
                    "sum": 0.0,
                    "count": 0,
                }
                self._histograms[key] = hist
            for idx, bucket in enumerate(hist["buckets"]):  # type: ignore[index]
                if value <= bucket:
                    hist["counts"][idx] += 1  # type: ignore[index]
            hist["inf"] = int(hist["inf"]) + 1
            hist["sum"] = float(hist["sum"]) + value
            hist["count"] = int(hist["count"]) + 1

    def render(self) -> str:
        lines = [
            f'# HELP callbot_service_info Service metadata for {self.service}.',
            "# TYPE callbot_service_info gauge",
            f'callbot_service_info{{service="{self.service}"}} 1',
        ]
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{self._format_labels(labels)} {self._format_float(value)}")
            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{self._format_labels(labels)} {self._format_float(value)}")
            for (name, labels), hist in sorted(self._histograms.items()):
                lines.append(f"# TYPE {name} histogram")
                buckets = hist["buckets"]  # type: ignore[index]
                counts = hist["counts"]  # type: ignore[index]
                for bucket, count in zip(buckets, counts):
                    bucket_labels = labels + (("le", self._format_float(bucket)),)
                    lines.append(f"{name}_bucket{self._format_labels(bucket_labels)} {count}")
                inf_labels = labels + (("le", "+Inf"),)
                lines.append(f"{name}_bucket{self._format_labels(inf_labels)} {hist['inf']}")
                lines.append(f"{name}_sum{self._format_labels(labels)} {self._format_float(float(hist['sum']))}")
                lines.append(f"{name}_count{self._format_labels(labels)} {hist['count']}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _labels(labels: dict[str, object]) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((key, str(value)) for key, value in labels.items()))

    @staticmethod
    def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        body = ",".join(f'{key}="{value}"' for key, value in labels)
        return f"{{{body}}}"

    @staticmethod
    def _format_float(value: float) -> str:
        if math.isinf(value):
            return "+Inf" if value > 0 else "-Inf"
        return f"{value:.6g}"


class ActiveRequest:
    def __init__(self, registry: MetricsRegistry, metric: str, **labels: object):
        self.registry = registry
        self.metric = metric
        self.labels = labels
        self.started = perf_counter()

    def __enter__(self) -> "ActiveRequest":
        self.registry.add_gauge(self.metric, 1, **self.labels)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.registry.add_gauge(self.metric, -1, **self.labels)

    @property
    def elapsed_seconds(self) -> float:
        return perf_counter() - self.started
