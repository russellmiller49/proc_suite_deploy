from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.common.path_redaction import repo_relative_path


def datetime_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_numeric(value: Any, *, prefix: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(value, bool):
        return out
    if isinstance(value, int | float):
        out[prefix] = float(value)
        return out
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric(value[key], prefix=child_prefix))
    return out


def extract_numeric_metrics(payload: dict[str, Any]) -> dict[str, float]:
    if not isinstance(payload, dict):
        return {}

    metrics: dict[str, float] = {}
    if isinstance(payload.get("summary"), dict):
        metrics.update(_flatten_numeric(payload["summary"], prefix="summary"))
    if isinstance(payload.get("counts"), dict):
        metrics.update(_flatten_numeric(payload["counts"], prefix="counts"))
    if not metrics:
        metrics.update(_flatten_numeric(payload, prefix="root"))
    return metrics


def build_report_delta(
    *,
    current_path: Path,
    baseline_path: Path,
) -> dict[str, Any]:
    current_payload = load_json(current_path)
    baseline_payload = load_json(baseline_path)
    current_metrics = extract_numeric_metrics(current_payload if isinstance(current_payload, dict) else {})
    baseline_metrics = extract_numeric_metrics(baseline_payload if isinstance(baseline_payload, dict) else {})

    common_keys = sorted(set(current_metrics) & set(baseline_metrics))
    deltas: list[dict[str, Any]] = []
    changed_count = 0
    for key in common_keys:
        current_value = float(current_metrics[key])
        baseline_value = float(baseline_metrics[key])
        delta = round(current_value - baseline_value, 6)
        if delta != 0:
            changed_count += 1
        deltas.append(
            {
                "metric": key,
                "baseline": round(baseline_value, 6),
                "current": round(current_value, 6),
                "delta": delta,
            }
        )

    added_keys = sorted(set(current_metrics) - set(baseline_metrics))
    removed_keys = sorted(set(baseline_metrics) - set(current_metrics))
    return {
        "kind": "report_delta",
        "created_at": datetime_now_iso(),
        "current_path": repo_relative_path(current_path),
        "baseline_path": repo_relative_path(baseline_path),
        "comparable_metric_count": len(common_keys),
        "changed_metric_count": changed_count,
        "added_metrics": added_keys,
        "removed_metrics": removed_keys,
        "metrics": deltas,
    }


def render_delta_markdown(delta: dict[str, Any], *, title: str) -> str:
    lines = [f"### {title}"]
    lines.append(f"- Baseline: `{delta.get('baseline_path')}`")
    lines.append(f"- Current: `{delta.get('current_path')}`")
    lines.append(
        f"- Comparable metrics: `{delta.get('comparable_metric_count', 0)}`; changed: `{delta.get('changed_metric_count', 0)}`"
    )

    interesting = [
        item
        for item in (delta.get("metrics") or [])
        if isinstance(item, dict) and float(item.get("delta", 0.0)) != 0.0
    ]
    interesting.sort(key=lambda item: abs(float(item.get("delta", 0.0))), reverse=True)
    if not interesting:
        lines.append("- No metric deltas.")
        return "\n".join(lines)

    for item in interesting[:10]:
        lines.append(
            f"- `{item['metric']}`: baseline `{item['baseline']}` -> current `{item['current']}` (delta `{item['delta']}`)"
        )
    return "\n".join(lines)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


__all__ = [
    "build_report_delta",
    "datetime_now_iso",
    "extract_numeric_metrics",
    "load_json",
    "render_delta_markdown",
    "write_json",
    "write_text",
]
