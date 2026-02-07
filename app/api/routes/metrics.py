"""Metrics endpoint for Prometheus scraping.

This module provides a /metrics endpoint that exports application metrics
in Prometheus text format (when METRICS_BACKEND=registry or prometheus).

Configuration:
    METRICS_BACKEND: Set to "registry" or "prometheus" to enable metrics collection
    METRICS_ENABLED: Set to "true" to expose the /metrics endpoint (default: true if registry)

Usage:
    # Enable metrics collection
    export METRICS_BACKEND=prometheus

    # Scrape metrics
    curl http://localhost:8000/metrics
"""

from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Response
from fastapi.responses import PlainTextResponse

from observability.logging_config import get_logger
from observability.metrics import RegistryMetricsClient, get_metrics_client

router = APIRouter()
logger = get_logger("metrics_api")

# Content type for Prometheus text format
PROMETHEUS_CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"


def _is_metrics_enabled() -> bool:
    """Check if metrics endpoint should be enabled."""
    # Enabled if METRICS_BACKEND is registry/prometheus, or if METRICS_ENABLED=true
    backend = os.getenv("METRICS_BACKEND", "null").lower()
    explicit = os.getenv("METRICS_ENABLED", "").lower()
    if explicit in ("true", "1", "yes"):
        return True
    if explicit in ("false", "0", "no"):
        return False
    # Default: enable if using registry backend
    return backend in ("registry", "prometheus")


@router.get(
    "/metrics",
    summary="Prometheus metrics endpoint",
    description="Export application metrics in Prometheus text format.",
    response_class=PlainTextResponse,
)
def get_metrics() -> Response:
    """Export metrics for Prometheus scraping.

    Returns:
        - 200 with Prometheus text format if metrics are enabled and using registry backend
        - 200 with JSON summary if metrics are enabled but not using registry backend
        - 404 if metrics endpoint is disabled

    The response format depends on the configured METRICS_BACKEND:
    - registry/prometheus: Prometheus text exposition format
    - stdout: JSON summary of last values
    - null: 404 with disabled message
    """
    if not _is_metrics_enabled():
        return PlainTextResponse(
            content="# Metrics endpoint disabled. Set METRICS_BACKEND=prometheus to enable.\n",
            status_code=404,
            media_type="text/plain",
        )

    client = get_metrics_client()

    if isinstance(client, RegistryMetricsClient):
        # Export in Prometheus text format
        content = client.export_prometheus()
        return Response(
            content=content,
            media_type=PROMETHEUS_CONTENT_TYPE,
        )
    else:
        # For other clients, return a helpful message
        return PlainTextResponse(
            content=(
                "# Metrics collection is not using registry backend.\n"
                "# Set METRICS_BACKEND=prometheus for Prometheus-compatible output.\n"
                f"# Current backend: {type(client).__name__}\n"
            ),
            media_type="text/plain",
        )


@router.get(
    "/metrics/json",
    summary="Metrics as JSON",
    description="Export application metrics as JSON (for debugging).",
)
def get_metrics_json() -> dict[str, Any]:
    """Export metrics as JSON for debugging.

    Returns:
        JSON object with counters, gauges, and histograms.
        Empty dicts if not using registry backend.
    """
    if not _is_metrics_enabled():
        return {"error": "Metrics disabled", "enabled": False}

    client = get_metrics_client()

    if isinstance(client, RegistryMetricsClient):
        return {
            "enabled": True,
            "backend": "registry",
            **client.export_json(),
        }
    else:
        return {
            "enabled": True,
            "backend": type(client).__name__,
            "counters": {},
            "gauges": {},
            "histograms": {},
            "note": "JSON export only available with registry backend",
        }


@router.get(
    "/metrics/status",
    summary="Metrics status",
    description="Check metrics collection status.",
)
def get_metrics_status() -> dict[str, Any]:
    """Get metrics collection status.

    Returns:
        Status information about metrics collection.
    """
    backend = os.getenv("METRICS_BACKEND", "null")
    client = get_metrics_client()

    status: dict[str, Any] = {
        "enabled": _is_metrics_enabled(),
        "backend": backend,
        "client_type": type(client).__name__,
    }

    if isinstance(client, RegistryMetricsClient):
        with client._lock:
            status["counters_count"] = len(client._counters)
            status["gauges_count"] = len(client._gauges)
            status["histograms_count"] = len(client._histograms)

    return status


@router.get(
    "/metrics/llm_drift",
    summary="LLM drift monitoring",
    description="Get LLM suggestion acceptance metrics for drift detection.",
)
def get_llm_drift_metrics() -> dict[str, Any]:
    """Get LLM drift monitoring metrics.

    Returns aggregated acceptance metrics from the in-process registry,
    useful for quick drift detection without querying Prometheus.

    Response includes:
    - Overall acceptance rate
    - Acceptance breakdown by procedure_type
    - Raw counts for reviewed and accepted suggestions

    Note: This endpoint uses the in-process metrics registry.
    For historical data, query Prometheus directly.
    """
    from observability.coding_metrics import CodingMetrics

    if not _is_metrics_enabled():
        return {
            "error": "Metrics disabled",
            "enabled": False,
        }

    client = get_metrics_client()

    if not isinstance(client, RegistryMetricsClient):
        return {
            "enabled": True,
            "backend": type(client).__name__,
            "note": "LLM drift metrics only available with registry backend",
            "by_procedure_type": {},
        }

    # Extract LLM acceptance metrics from the registry
    json_data = client.export_json()
    counters = json_data.get("counters", {})
    # Get the acceptance counters
    reviewed_data = counters.get(CodingMetrics.LLM_SUGGESTIONS_REVIEWED, {})
    accepted_data = counters.get(CodingMetrics.LLM_SUGGESTIONS_ACCEPTED, {})

    # Parse and aggregate by procedure_type
    by_procedure_type: dict[str, dict[str, Any]] = {}

    # Parse label strings to extract procedure_type
    for labels_str, reviewed_count in reviewed_data.items():
        proc_type = _extract_label(labels_str, "procedure_type") or "unknown"
        source = _extract_label(labels_str, "source") or "unknown"

        key = f"{proc_type}:{source}"
        if key not in by_procedure_type:
            by_procedure_type[key] = {
                "procedure_type": proc_type,
                "source": source,
                "reviewed": 0,
                "accepted": 0,
                "acceptance_rate": 0.0,
            }
        by_procedure_type[key]["reviewed"] += reviewed_count

    for labels_str, accepted_count in accepted_data.items():
        proc_type = _extract_label(labels_str, "procedure_type") or "unknown"
        source = _extract_label(labels_str, "source") or "unknown"

        key = f"{proc_type}:{source}"
        if key not in by_procedure_type:
            by_procedure_type[key] = {
                "procedure_type": proc_type,
                "source": source,
                "reviewed": 0,
                "accepted": 0,
                "acceptance_rate": 0.0,
            }
        by_procedure_type[key]["accepted"] += accepted_count

    # Calculate acceptance rates
    for _key, data in by_procedure_type.items():
        if data["reviewed"] > 0:
            data["acceptance_rate"] = round(data["accepted"] / data["reviewed"], 4)

    # Calculate overall totals
    total_reviewed = sum(d["reviewed"] for d in by_procedure_type.values())
    total_accepted = sum(d["accepted"] for d in by_procedure_type.values())
    overall_rate = round(total_accepted / total_reviewed, 4) if total_reviewed > 0 else 0.0

    return {
        "enabled": True,
        "backend": "registry",
        "total_reviewed": total_reviewed,
        "total_accepted": total_accepted,
        "overall_acceptance_rate": overall_rate,
        "by_procedure_type": list(by_procedure_type.values()),
    }


def _extract_label(labels_str: str, label_name: str) -> str | None:
    """Extract a label value from a Prometheus label string.

    Args:
        labels_str: String like '{procedure_type="bronch_ebus",source="llm"}'
        label_name: Name of the label to extract

    Returns:
        The label value or None if not found
    """
    if not labels_str:
        return None

    # Parse format: {key="value",key2="value2"}
    import re
    pattern = rf'{label_name}="([^"]*)"'
    match = re.search(pattern, labels_str)
    return match.group(1) if match else None
