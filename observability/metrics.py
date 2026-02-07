"""Metrics client abstraction and implementations.

This module provides:
- MetricsClient: Abstract base class for metrics emission
- NullMetricsClient: No-op implementation (default)
- StdoutMetricsClient: JSON output for debugging
- RegistryMetricsClient: In-process registry with Prometheus text export

To enable Prometheus-compatible metrics:
    from observability.metrics import set_metrics_client, RegistryMetricsClient
    set_metrics_client(RegistryMetricsClient())

Then expose via the /metrics endpoint (see modules/api/routes/metrics.py).
"""

from __future__ import annotations

import json
import os
import sys
import threading
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


class MetricsClient(ABC):
    """Abstract base class for metrics emission."""

    @abstractmethod
    def incr(self, name: str, tags: dict[str, str] | None = None, value: int = 1) -> None:
        """Increment a counter metric."""
        ...

    @abstractmethod
    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record an observation (histogram/gauge)."""
        ...

    @abstractmethod
    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        """Record a timing value in milliseconds."""
        ...


class StdoutMetricsClient(MetricsClient):
    """Emit metrics as JSON to stdout for development/debugging."""

    def __init__(self, prefix: str = "proc_suite"):
        self.prefix = prefix

    def _emit(self, metric_type: str, name: str, value: Any, tags: dict[str, str] | None) -> None:
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": metric_type,
            "metric": f"{self.prefix}.{name}",
            "value": value,
            "tags": tags or {},
        }
        print(json.dumps(record), file=sys.stderr)

    def incr(self, name: str, tags: dict[str, str] | None = None, value: int = 1) -> None:
        self._emit("counter", name, value, tags)

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        self._emit("gauge", name, value, tags)

    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        self._emit("timing", name, value_ms, tags)


class NullMetricsClient(MetricsClient):
    """No-op metrics client for when metrics are disabled."""

    def incr(self, name: str, tags: dict[str, str] | None = None, value: int = 1) -> None:
        pass

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        pass

    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        pass


# ============================================================================
# In-Process Registry Metrics Client (Prometheus-compatible)
# ============================================================================

# Default histogram buckets for timing metrics (in milliseconds)
DEFAULT_TIMING_BUCKETS = (10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


def _tags_to_labels(tags: dict[str, str] | None) -> str:
    """Convert tags dict to Prometheus label string."""
    if not tags:
        return ""
    label_pairs = [f'{k}="{v}"' for k, v in sorted(tags.items())]
    return "{" + ",".join(label_pairs) + "}"


def _sanitize_metric_name(name: str) -> str:
    """Sanitize metric name for Prometheus (replace dots with underscores)."""
    return name.replace(".", "_").replace("-", "_")


@dataclass
class CounterMetric:
    """Thread-safe counter metric."""

    name: str
    values: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def incr(self, tags: dict[str, str] | None, value: int = 1) -> None:
        labels = _tags_to_labels(tags)
        with self.lock:
            self.values[labels] += value

    def export(self, prefix: str = "proc_suite") -> list[str]:
        """Export as Prometheus text format lines."""
        lines = []
        metric_name = f"{prefix}_{_sanitize_metric_name(self.name)}_total"
        lines.append(f"# TYPE {metric_name} counter")
        with self.lock:
            for labels, value in sorted(self.values.items()):
                lines.append(f"{metric_name}{labels} {value}")
        return lines


@dataclass
class GaugeMetric:
    """Thread-safe gauge metric (stores last observation)."""

    name: str
    values: dict[str, float] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, tags: dict[str, str] | None) -> None:
        labels = _tags_to_labels(tags)
        with self.lock:
            self.values[labels] = value

    def export(self, prefix: str = "proc_suite") -> list[str]:
        """Export as Prometheus text format lines."""
        lines = []
        metric_name = f"{prefix}_{_sanitize_metric_name(self.name)}"
        lines.append(f"# TYPE {metric_name} gauge")
        with self.lock:
            for labels, value in sorted(self.values.items()):
                lines.append(f"{metric_name}{labels} {value}")
        return lines


@dataclass
class HistogramMetric:
    """Thread-safe histogram metric for timing data."""

    name: str
    buckets: tuple[float, ...] = DEFAULT_TIMING_BUCKETS
    # Key: labels string, Value: dict of bucket -> count, plus _sum, _count
    data: dict[str, dict[str, float]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def observe(self, value: float, tags: dict[str, str] | None) -> None:
        labels = _tags_to_labels(tags)
        with self.lock:
            bucket_data = self.data[labels]
            bucket_data["_sum"] += value
            bucket_data["_count"] += 1
            # Update bucket counts
            for bucket in self.buckets:
                if value <= bucket:
                    bucket_data[f"le_{bucket}"] += 1
            bucket_data["le_+Inf"] += 1  # Always increment +Inf

    def export(self, prefix: str = "proc_suite") -> list[str]:
        """Export as Prometheus text format lines."""
        lines = []
        metric_name = f"{prefix}_{_sanitize_metric_name(self.name)}"
        lines.append(f"# TYPE {metric_name} histogram")
        with self.lock:
            for labels, bucket_data in sorted(self.data.items()):
                # Export bucket counts
                for bucket in self.buckets:
                    bucket_key = f"le_{bucket}"
                    count = bucket_data.get(bucket_key, 0)
                    # Insert le="..." into labels
                    if labels:
                        bucket_labels = labels[:-1] + f',le="{bucket}"' + "}"
                    else:
                        bucket_labels = f'{{le="{bucket}"}}'
                    lines.append(f"{metric_name}_bucket{bucket_labels} {count}")
                # +Inf bucket
                inf_count = bucket_data.get("le_+Inf", 0)
                if labels:
                    inf_labels = labels[:-1] + ',le="+Inf"}'
                else:
                    inf_labels = '{le="+Inf"}'
                lines.append(f"{metric_name}_bucket{inf_labels} {inf_count}")
                # Sum and count
                lines.append(f"{metric_name}_sum{labels} {bucket_data.get('_sum', 0)}")
                lines.append(f"{metric_name}_count{labels} {bucket_data.get('_count', 0)}")
        return lines


class RegistryMetricsClient(MetricsClient):
    """In-process metrics registry with Prometheus text export.

    This client accumulates metrics in memory and can export them
    in Prometheus text format for scraping.

    Usage:
        client = RegistryMetricsClient()
        client.incr("requests_total", {"method": "POST"})
        client.timing("request_latency_ms", 150.5, {"endpoint": "/api/code"})

        # Export for Prometheus
        text = client.export_prometheus()
    """

    def __init__(self, prefix: str = "proc_suite"):
        self.prefix = prefix
        self._lock = threading.Lock()
        self._counters: dict[str, CounterMetric] = {}
        self._gauges: dict[str, GaugeMetric] = {}
        self._histograms: dict[str, HistogramMetric] = {}

    def _get_counter(self, name: str) -> CounterMetric:
        """Get or create a counter metric."""
        with self._lock:
            if name not in self._counters:
                self._counters[name] = CounterMetric(name=name)
            return self._counters[name]

    def _get_gauge(self, name: str) -> GaugeMetric:
        """Get or create a gauge metric."""
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = GaugeMetric(name=name)
            return self._gauges[name]

    def _get_histogram(self, name: str) -> HistogramMetric:
        """Get or create a histogram metric."""
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = HistogramMetric(name=name)
            return self._histograms[name]

    def incr(self, name: str, tags: dict[str, str] | None = None, value: int = 1) -> None:
        """Increment a counter metric."""
        counter = self._get_counter(name)
        counter.incr(tags, value)

    def observe(self, name: str, value: float, tags: dict[str, str] | None = None) -> None:
        """Record an observation (gauge)."""
        gauge = self._get_gauge(name)
        gauge.observe(value, tags)

    def timing(self, name: str, value_ms: float, tags: dict[str, str] | None = None) -> None:
        """Record a timing value as histogram."""
        histogram = self._get_histogram(name)
        histogram.observe(value_ms, tags)

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text format.

        Returns:
            String in Prometheus exposition format.
        """
        lines = []

        # Export counters
        with self._lock:
            counters = list(self._counters.values())
        for counter in counters:
            lines.extend(counter.export(self.prefix))

        # Export gauges
        with self._lock:
            gauges = list(self._gauges.values())
        for gauge in gauges:
            lines.extend(gauge.export(self.prefix))

        # Export histograms
        with self._lock:
            histograms = list(self._histograms.values())
        for histogram in histograms:
            lines.extend(histogram.export(self.prefix))

        return "\n".join(lines) + "\n"

    def export_json(self) -> dict[str, Any]:
        """Export metrics as JSON for debugging.

        Returns:
            Dict with counters, gauges, and histograms.
        """
        result: dict[str, Any] = {"counters": {}, "gauges": {}, "histograms": {}}

        with self._lock:
            for name, counter in self._counters.items():
                with counter.lock:
                    result["counters"][name] = dict(counter.values)

            for name, gauge in self._gauges.items():
                with gauge.lock:
                    result["gauges"][name] = dict(gauge.values)

            for name, histogram in self._histograms.items():
                with histogram.lock:
                    result["histograms"][name] = {
                        labels: dict(data) for labels, data in histogram.data.items()
                    }

        return result

    def reset(self) -> None:
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


# Global singleton
_metrics_client: MetricsClient | None = None


def get_metrics_client() -> MetricsClient:
    """Get the global metrics client, initializing if needed.

    The client type is determined by the METRICS_BACKEND environment variable:
    - "registry" or "prometheus": RegistryMetricsClient (for Prometheus export)
    - "stdout": StdoutMetricsClient (for debugging)
    - "null" or not set: NullMetricsClient (no-op, default)

    Returns:
        The configured MetricsClient instance.
    """
    global _metrics_client
    if _metrics_client is None:
        backend = os.getenv("METRICS_BACKEND", "null").lower()
        if backend in ("registry", "prometheus"):
            _metrics_client = RegistryMetricsClient()
        elif backend == "stdout":
            _metrics_client = StdoutMetricsClient()
        else:
            _metrics_client = NullMetricsClient()
    return _metrics_client


def set_metrics_client(client: MetricsClient) -> None:
    """Set the global metrics client."""
    global _metrics_client
    _metrics_client = client


def reset_metrics_client() -> None:
    """Reset the global metrics client to None.

    The next call to get_metrics_client() will re-initialize based on env vars.
    Useful for testing.
    """
    global _metrics_client
    _metrics_client = None
