#!/usr/bin/env python3
"""Strict reporter smoke and source-slot evaluation for short robotic/EBUS prompts.

Run from the repository root. Uses the real registry seed path with the same
offline defaults as the standard baseline evaluator. This supplements, rather
than replaces, the shared extraction/CPT quality gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def observed_slots(bundle: dict) -> dict:
    def data(kind):
        return [p["data"] for p in bundle["procedures"] if p["proc_type"] == kind]

    return {
        "no_stent": not data("airway_stent_placement"),
        "generic_peripheral_biopsy": "Peripheral lung biopsy"
        in (bundle.get("findings_text") or ""),
        "cbct_guidance": "CBCT guidance:" in (bundle.get("findings_text") or ""),
        "tbna_counts": [
            p.get("samples_collected") for p in data("transbronchial_needle_aspiration")
        ],
        "cryo_counts": [p.get("num_samples") for p in data("transbronchial_cryobiopsy")],
        "brush_counts": [p.get("samples_collected") for p in data("bronchial_brushings")],
        "node_passes": {
            s["station_name"].upper(): s.get("passes")
            for p in data("ebus_tbna")
            for s in p["stations"]
        },
        "survey_only": not data("ebus_tbna")
        and "survey without nodal sampling" in (bundle.get("findings_text") or ""),
        "bal": [
            [p.get("lung_segment"), p.get("instilled_volume_cc"), p.get("returned_volume_cc")]
            for p in data("bal")
        ],
    }


def check_slots(bundle: dict, expected: dict) -> list[dict]:
    actual = observed_slots(bundle)
    return [
        {"slot": key, "expected": value, "actual": actual[key]}
        for key, value in expected.items()
        if value != actual[key]
    ]


def check_ebus_warning_review(flags: list[dict], review: dict) -> list[str]:
    if not review:
        return []
    warnings = [f for f in flags if f["code"] == "EBUS_DETAILS_UNSPECIFIED"]
    failures = []
    if bool(warnings) != review["remaining_warning"]:
        failures.append("EBUS warning presence differs from source review")
    if warnings and warnings[0]["metadata"].get("missing_details") != review.get("missing_details"):
        failures.append("EBUS warning reasons differ from source review")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "tests/fixtures/reporting/robotic_ebus_minimal_prompts.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifacts-dir", type=Path)
    parser.add_argument(
        "--existing-artifacts",
        type=Path,
        help="Score saved bundles/summary without rerunning inference",
    )
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    if not args.existing_artifacts:
        from ops.tools.eval_reporter_prompt_baseline import configure_eval_env

        configure_eval_env()
        from app.registry.application.registry_service import RegistryService
        from app.reporting.seed_pipeline import (
            run_reporter_seed_pipeline,
            seed_outcome_from_registry_result,
        )

        service = RegistryService()
    else:
        saved = {
            r["id"]: r for r in json.loads((args.existing_artifacts / "summary.json").read_text())
        }
    rows = []
    for row in payload["cases"]:
        try:
            if args.existing_artifacts:
                bundle = json.loads((args.existing_artifacts / (row["id"] + ".json")).read_text())
                fallback = saved[row["id"]]["fallback"]
                flags = saved[row["id"]]["flags"]
            else:
                # Only prompt_text enters the pipeline; metadata and expectations stay here.
                text = row["prompt_text"]
                result = service.extract_fields(text)
                out = run_reporter_seed_pipeline(
                    seed_outcome_from_registry_result(result, masked_seed_text=text),
                    note_text=text,
                    strict=True,
                    debug_enabled=False,
                )
                bundle = out.bundle.model_dump(mode="json")
                fallback = out.render_fallback_used
                flags = [
                    f
                    for f in out.quality_flags
                    if str(f.get("source", "")).startswith("quality_gate")
                ]
                if args.artifacts_dir:
                    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
                    (args.artifacts_dir / (row["id"] + ".json")).write_text(
                        json.dumps(bundle, indent=2) + "\n"
                    )
                    (args.artifacts_dir / (row["id"] + ".md")).write_text(out.markdown)
            failures = check_slots(bundle, row.get("expected", {}))
            rows.append(
                {
                    "id": row["id"],
                    "render_fallback_used": fallback,
                    "slot_failures": failures,
                    "warning_review_failures": check_ebus_warning_review(
                        flags, row.get("ebus_warning_review", {})
                    ),
                    "observed": observed_slots(bundle),
                    "quality_flags": flags,
                }
            )
        except Exception as exc:
            rows.append({"id": row["id"], "error": str(exc)})
    summary = {
        "cases": len(rows),
        "errors": sum("error" in r for r in rows),
        "strict_render_fallbacks": sum(r.get("render_fallback_used", False) for r in rows),
        "slot_checks": sum(len(r.get("expected", {})) for r in payload["cases"]),
        "slot_failures": sum(len(r.get("slot_failures", [])) for r in rows),
        "warning_reviews": sum("ebus_warning_review" in r for r in payload["cases"]),
        "warning_review_failures": sum(len(r.get("warning_review_failures", [])) for r in rows),
        "cases_with_slot_failures": sum(bool(r.get("slot_failures")) for r in rows),
        "quality_flag_counts": dict(
            Counter(f["code"] for r in rows for f in r.get("quality_flags", []))
        ),
    }
    report = {
        "schema_version": "reporter_robotic_minimal_fidelity.v1",
        "source_sha256": payload["source_sha256"],
        "seed_strategy": "registry_extract_fields",
        "summary": summary,
        "cases": rows,
        "limitations": [
            "Source-slot checks cover curated facts; "
            "successful rendering is not complete clinical validation.",
            "Extraction may fall back when model artifacts are unavailable; "
            "inspect the runtime log.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return int(
        bool(
            summary["errors"]
            or summary["strict_render_fallbacks"]
            or summary["slot_failures"]
            or summary["warning_review_failures"]
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
