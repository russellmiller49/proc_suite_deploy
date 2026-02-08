"""Navigation target heuristic extraction."""

from __future__ import annotations

import re
from typing import Any

from app.registry.constants.warnings import (
    cryobiopsy_site_added,
    nav_target_added,
    nav_target_parsed,
)
from app.registry.schema import RegistryRecord


def apply_navigation_target_heuristics(
    note_text: str,
    record_in: RegistryRecord,
) -> tuple[RegistryRecord, list[str]]:
    if record_in is None:
        return RegistryRecord(), []

    text = note_text or ""
    if ("\n" not in text and "\r" not in text) and ("\\n" in text or "\\r" in text):
        text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    nav_hint = re.search(
        r"(?i)\b(navigational bronchoscopy|robotic bronchoscopy|electromagnetic navigation|\benb\b|\bion\b|monarch|galaxy|planning station)\b",
        text,
    )
    if not nav_hint:
        return record_in, []

    from app.registry.processing.navigation_targets import (
        extract_cryobiopsy_sites,
        extract_navigation_targets,
    )

    parsed_targets = extract_navigation_targets(text)
    target_lines = [
        line
        for line in text.splitlines()
        if re.search(r"(?i)target lesion", line)
        or re.search(r"(?i)^\s*target\s*\d{1,2}\s*[:\\-]", line)
    ]
    if not parsed_targets and not target_lines:
        return record_in, []

    record_data = record_in.model_dump()
    granular = record_data.get("granular_data")
    if granular is None or not isinstance(granular, dict):
        granular = {}

    evidence = record_data.get("evidence")
    if not isinstance(evidence, dict):
        evidence = {}
        record_data["evidence"] = evidence

    targets_raw = granular.get("navigation_targets")
    if isinstance(targets_raw, list):
        targets = [dict(target) for target in targets_raw if isinstance(target, dict)]
    else:
        targets = []

    def _add_nav_evidence(idx: int, field: str, ev: object) -> None:
        nonlocal updated
        if not isinstance(idx, int) or idx < 0:
            return
        if not isinstance(field, str) or not field:
            return
        if not isinstance(ev, dict):
            return
        start = ev.get("start")
        end = ev.get("end")
        text_snippet = ev.get("text")
        if start is None or end is None or not text_snippet:
            return
        try:
            start_i = int(start)
            end_i = int(end)
        except Exception:  # noqa: BLE001
            return
        if start_i < 0 or end_i <= start_i:
            return
        key = f"granular_data.navigation_targets.{idx}.{field}"
        if evidence.get(key):
            return
        try:
            from app.common.spans import Span

            evidence.setdefault(key, []).append(
                Span(text=str(text_snippet), start=start_i, end=end_i, confidence=0.9)
            )
            updated = True
        except Exception:  # noqa: BLE001
            return

    def _is_placeholder_location(value: object) -> bool:
        if value is None:
            return True
        normalized = str(value).strip().lower()
        if not normalized:
            return True
        if "robotic navigation bronchoscopy was performed" in normalized:
            return True
        if "partial registration" in normalized and "target lesion" in normalized:
            return True
        if "||" in normalized:
            return True
        if re.search(r"(?i)^(?:pt|patient)\\s*:", normalized):
            return True
        if re.search(r"(?i)\\b(?:mrn|dob)\\b\\s*:", normalized):
            return True
        if re.search(r"(?i)\\b(?:attending|fellow)\\b\\s*:", normalized):
            return True
        return normalized in {
            "unknown",
            "unknown target",
            "target",
            "target lesion",
            "target lesion 1",
            "target lesion 2",
            "target lesion 3",
        } or normalized.startswith("target lesion")

    def _is_more_specific_location(existing_value: object, candidate_value: object) -> bool:
        existing = str(existing_value or "").strip()
        candidate = str(candidate_value or "").strip()
        if not candidate:
            return False
        if not existing:
            return True
        existing_lower = existing.lower()
        candidate_lower = candidate.lower()
        existing_has_bronchus = bool(re.search(r"\\b[LR]B\\d{1,2}\\b", existing, re.IGNORECASE))
        candidate_has_bronchus = bool(re.search(r"\\b[LR]B\\d{1,2}\\b", candidate, re.IGNORECASE))
        if candidate_has_bronchus and not existing_has_bronchus:
            return True
        if "segment" in candidate_lower and "segment" not in existing_lower:
            return True
        if ("(" in candidate and ")" in candidate) and ("(" not in existing or ")" not in existing):
            return True
        if ("nodule" in candidate_lower or "#" in candidate_lower) and (
            "nodule" not in existing_lower and "#" not in existing_lower
        ):
            return True
        return False

    def _sanitize_target_location(value: str) -> str:
        raw = (value or "").strip()
        if not raw:
            return ""
        for stop_word in ("PROCEDURE", "INDICATION", "TECHNIQUE", "DESCRIPTION"):
            match = re.search(rf"(?i)\b{re.escape(stop_word)}\b", raw)
            if match:
                raw = raw[: match.start()].strip()
        if len(raw) > 100:
            clipped = raw[:100].rsplit(" ", 1)[0].strip()
            raw = clipped or raw[:100].strip()
        return raw

    warnings: list[str] = []
    updated = False

    if parsed_targets:
        max_len = max(len(targets), len(parsed_targets))
        merged: list[dict[str, Any]] = []
        for idx in range(max_len):
            base = targets[idx] if idx < len(targets) else {}
            parsed = parsed_targets[idx] if idx < len(parsed_targets) else {}
            out = dict(base)
            parsed_evidence = (
                parsed.get("_evidence")
                if isinstance(parsed, dict) and isinstance(parsed.get("_evidence"), dict)
                else None
            )

            out["target_number"] = idx + 1

            for key, value in parsed.items():
                if value in (None, "", [], {}):
                    continue
                if key == "_evidence":
                    continue
                if key == "target_location_text":
                    if _is_placeholder_location(out.get(key)) or _is_more_specific_location(
                        out.get(key), value
                    ):
                        out[key] = value
                        updated = True
                    continue
                if key == "fiducial_marker_placed":
                    if value is True and out.get(key) is not True:
                        out[key] = True
                        updated = True
                    continue
                if out.get(key) in (None, "", [], {}):
                    out[key] = value
                    updated = True
                    if parsed_evidence and key in parsed_evidence:
                        _add_nav_evidence(idx, key, parsed_evidence.get(key))

            merged.append(out)

        targets = merged
        warnings.append(nav_target_parsed(len(parsed_targets)))
    else:
        existing_count = len(targets)
        needed_count = len(target_lines)
        if existing_count >= needed_count:
            return record_in, []

        for idx in range(existing_count, needed_count):
            line = _sanitize_target_location(target_lines[idx])
            targets.append(
                {
                    "target_number": idx + 1,
                    "target_location_text": line or f"Target lesion {idx + 1}",
                }
            )
        warnings.append(nav_target_added(needed_count - existing_count))
        updated = True

    granular["navigation_targets"] = targets

    try:
        procedures = record_data.get("procedures_performed")
        if not isinstance(procedures, dict):
            procedures = {}
            record_data["procedures_performed"] = procedures

        nav_proc = procedures.get("navigational_bronchoscopy")
        if nav_proc is None or not isinstance(nav_proc, dict):
            nav_proc = {}
            procedures["navigational_bronchoscopy"] = nav_proc

        def _copy_first_span(src_key: str, dest_key: str) -> bool:
            if evidence.get(dest_key):
                return False
            spans = evidence.get(src_key)
            if not isinstance(spans, list) or not spans:
                return False
            evidence[dest_key] = [spans[0]]
            return True

        if nav_proc.get("tool_in_lesion_confirmed") not in (True, False):
            chosen_idx: int | None = None
            chosen_value: bool | None = None

            for idx, item in enumerate(targets):
                if not isinstance(item, dict):
                    continue
                if item.get("tool_in_lesion_confirmed") is True and evidence.get(
                    f"granular_data.navigation_targets.{idx}.tool_in_lesion_confirmed"
                ):
                    chosen_idx = idx
                    chosen_value = True
                    break

            if chosen_value is None:
                false_indices: list[int] = []
                for idx, item in enumerate(targets):
                    if not isinstance(item, dict):
                        continue
                    if item.get("tool_in_lesion_confirmed") is False and evidence.get(
                        f"granular_data.navigation_targets.{idx}.tool_in_lesion_confirmed"
                    ):
                        false_indices.append(idx)

                if false_indices:
                    any_true = any(
                        isinstance(item, dict) and item.get("tool_in_lesion_confirmed") is True
                        for item in targets
                    )
                    if not any_true:
                        chosen_idx = false_indices[0]
                        chosen_value = False

            if chosen_idx is not None and chosen_value is not None:
                nav_proc["tool_in_lesion_confirmed"] = chosen_value
                if _copy_first_span(
                    f"granular_data.navigation_targets.{chosen_idx}.tool_in_lesion_confirmed",
                    "procedures_performed.navigational_bronchoscopy.tool_in_lesion_confirmed",
                ):
                    updated = True
                updated = True

        if (
            nav_proc.get("tool_in_lesion_confirmed") is True
            and nav_proc.get("confirmation_method") in (None, "", [], {})
        ):
            for idx, item in enumerate(targets):
                if not isinstance(item, dict):
                    continue
                method = item.get("confirmation_method")
                if not method:
                    continue
                if not evidence.get(f"granular_data.navigation_targets.{idx}.confirmation_method"):
                    continue

                method_norm = str(method)
                if method_norm == "Augmented fluoroscopy":
                    method_norm = "Augmented Fluoroscopy"

                nav_proc["confirmation_method"] = method_norm
                if _copy_first_span(
                    f"granular_data.navigation_targets.{idx}.confirmation_method",
                    "procedures_performed.navigational_bronchoscopy.confirmation_method",
                ):
                    updated = True
                updated = True
                break
    except Exception:  # noqa: BLE001
        pass

    existing_sites = granular.get("cryobiopsy_sites")
    if not existing_sites:
        sites = extract_cryobiopsy_sites(text)
        if sites:
            granular["cryobiopsy_sites"] = sites
            warnings.append(cryobiopsy_site_added(len(sites)))
            updated = True

    record_data["granular_data"] = granular
    record_out = RegistryRecord(**record_data)
    if not updated:
        return record_in, []
    return record_out, warnings


class NavigationTargetHeuristic:
    def apply(self, note_text: str, record: RegistryRecord) -> tuple[RegistryRecord, list[str]]:
        return apply_navigation_target_heuristics(note_text, record)


__all__ = ["NavigationTargetHeuristic", "apply_navigation_target_heuristics"]
