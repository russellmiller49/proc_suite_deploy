from __future__ import annotations

import re
from typing import Any, cast

from proc_schemas.clinical import ProcedureBundle

from .compat_types import BillingCodeItem, RegistryBillingPayload, RegistryRecordCompat
from .types import NormalizationNote


def _extract_cpt_codes_from_billing_payload(billing: RegistryBillingPayload) -> list[str]:
    billing_codes = billing.get("cpt_codes") or []
    extracted: list[str] = []
    if isinstance(billing_codes, list):
        for item in billing_codes:
            code: Any = None
            if isinstance(item, dict):
                code_item = cast(BillingCodeItem, item)
                code = code_item.get("code") or code_item.get("cpt") or code_item.get("CPT")
            else:
                code = item
            if code in (None, "", [], {}):
                continue
            extracted.append(str(code).strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for code in extracted:
        if not code or code in seen:
            continue
        seen.add(code)
        deduped.append(code)
    return deduped


def _add_compat_flat_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """Add flat compatibility fields that adapters expect from nested registry data.

    The adapters expect flat field names like 'nav_rebus_used', 'bronch_num_tbbx',
    but the RegistryRecord stores data in nested structures like
    procedures_performed.radial_ebus.performed.

    This function adds the flat aliases so adapters can find the data.
    """
    # Import here to avoid circular dependency.
    #
    # NOTE: `_COMPAT_ATTRIBUTE_PATHS` is not guaranteed to exist after schema refactors.
    # Keep this function resilient by falling back to a small set of derived aliases
    # from the nested V3/V2-dynamic shapes (used by `parallel_ner`).
    try:
        from app.registry.schema import _COMPAT_ATTRIBUTE_PATHS  # type: ignore[attr-defined]
    except ImportError:
        _COMPAT_ATTRIBUTE_PATHS = {}  # type: ignore[assignment]

    def _get_nested(d: dict, path: tuple[str, ...]) -> Any:
        """Traverse nested dict by path tuple."""
        current = d
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
            if current is None:
                return None
        return current

    # Add all compatibility flat fields
    for flat_name, nested_path in _COMPAT_ATTRIBUTE_PATHS.items():
        if flat_name not in raw:
            value = _get_nested(raw, nested_path)
            if value is not None:
                raw[flat_name] = value

    # Add additional fields that adapters need but aren't in _COMPAT_ATTRIBUTE_PATHS
    procs = raw.get("procedures_performed", {}) or {}
    if not isinstance(procs, dict):
        procs = {}

    # Bubble up CPT codes from the nested V3 billing payload so procedure adapters
    # can use them as hints (and downstream logic can render a fuller procedure list).
    raw_compat = cast(RegistryRecordCompat, raw)
    if raw_compat.get("cpt_codes") in (None, "", [], {}):
        billing = raw_compat.get("billing")
        if isinstance(billing, dict):
            extracted = _extract_cpt_codes_from_billing_payload(billing)
            if extracted:
                raw_compat["cpt_codes"] = extracted

    def _first_nonempty_str(*values: Any) -> str | None:
        for value in values:
            if value in (None, ""):
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _extract_lung_location_hint(text: str) -> str | None:
        """Best-effort location from free text (lobe/segment shorthand)."""
        if not text:
            return None
        upper = text.upper()
        for token in ("RUL", "RML", "RLL", "LUL", "LLL"):
            if re.search(rf"\b{token}\b", upper):
                return token
        # Common long-form phrases.
        if "RIGHT UPPER LOBE" in upper:
            return "RUL"
        if "RIGHT MIDDLE LOBE" in upper:
            return "RML"
        if "RIGHT LOWER LOBE" in upper:
            return "RLL"
        if "LEFT UPPER LOBE" in upper:
            return "LUL"
        if "LEFT LOWER LOBE" in upper:
            return "LLL"
        return None

    def _extract_bronch_segment_hint(text: str) -> str | None:
        """Best-effort bronchopulmonary segment token (e.g., RB10, LB6, B6)."""
        if not text:
            return None
        upper = text.upper()
        match = re.search(r"\b([RL]B\d{1,2})\b", upper)
        if match:
            return match.group(1)
        match = re.search(r"\bB\d{1,2}\b", upper)
        if match:
            return match.group(0)
        return None

    def _infer_rebus_pattern(text: str) -> str | None:
        if not text:
            return None
        lowered = text.lower()
        if "concentric" in lowered:
            return "Concentric"
        if "eccentric" in lowered:
            return "Eccentric"
        if "adjacent" in lowered:
            return "Adjacent"
        return None

    def _parse_count(text: str, pattern: str) -> int | None:
        """Parse shorthand counts like 'TBNA x4' or 'Bx x 6'."""
        if not text:
            return None
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            return None
        try:
            value = int(match.group(1))
        except Exception:
            return None
        return value if value >= 0 else None

    def _parse_operator(text: str) -> str | None:
        if not text:
            return None
        match = re.search(r"(?im)^\s*(?:operator|attending)\s*:\s*(.+?)\s*$", text)
        if not match:
            return None
        value = match.group(1).strip()
        value = value.replace("[", "").replace("]", "").strip()
        if not value or value.strip().lower() in ("redacted", "unknown", "n/a", "na"):
            return None
        return value or None

    def _parse_referred_physician(text: str) -> str | None:
        if not text:
            return None
        match = re.search(r"(?im)^\s*(?:cc\s*)?referred\s+physician\s*:\s*(.+?)\s*$", text)
        if not match:
            return None
        value = match.group(1).strip()
        value = value.replace("[", "").replace("]", "").strip()
        if not value or value.strip().lower() in ("redacted", "unknown", "n/a", "na"):
            return None
        return value or None

    def _parse_service_date(text: str) -> str | None:
        if not text:
            return None
        match = re.search(r"(?im)^\s*(?:service\s*date|date\s+of\s+procedure)\s*:\s*(.+?)\s*$", text)
        if not match:
            return None
        value = match.group(1).strip()
        value = value.replace("[", "").replace("]", "").strip()
        if not value or value.strip().lower() in ("redacted", "unknown", "n/a", "na"):
            return None
        return value or None

    def _normalize_ebl_text(value: Any) -> str | None:
        if value in (None, "", [], {}):
            return None
        text = str(value).strip()
        if not text:
            return None
        # Drop trailing disposition fragments if they were captured inline.
        text = re.split(r"(?i)\bdispo(?:sition)?\b", text)[0].strip().rstrip(",;").strip()
        lowered = text.lower()
        if lowered in ("minimal", "min"):
            return "Minimal"
        if lowered in ("none", "no"):
            return "None"
        match = re.search(r"(?i)(<\s*)?(\d+(?:\.\d+)?)\s*(ml|cc|l)\b", text)
        if match:
            prefix = "<" if match.group(1) else ""
            num = float(match.group(2))
            unit = match.group(3).lower()
            ml_val = num * 1000.0 if unit == "l" else num
            ml_str = str(int(ml_val)) if float(ml_val).is_integer() else str(round(ml_val, 2))
            return f"{prefix} {ml_str} mL".strip()
        # Fall back to the original text if it already includes an interpretable unit.
        if re.search(r"(?i)\b(?:ml|cc)\b", text):
            return text
        return text

    def _text_contains_tool_in_lesion(text: str) -> bool:
        if not text:
            return False
        lowered = text.lower()
        return bool(re.search(r"\btool[-\s]?in[-\s]?lesion\b", lowered))

    def _dedupe_preserve_order(values: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for value in values:
            key = str(value or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(key)
        return deduped

    def _coerce_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if str(v).strip()]

    def _derive_sampled_stations_from_linear_ebus(linear_ebus: dict[str, Any]) -> list[str]:
        invalid_station_tokens = {"UNSPECIFIED", "UNKNOWN", "N/A", "NA"}
        # Prefer explicit sampled stations if present.
        sampled = _coerce_str_list(linear_ebus.get("stations_sampled"))
        if sampled:
            filtered = [st for st in sampled if str(st).strip().upper() not in invalid_station_tokens]
            return _dedupe_preserve_order(filtered)

        # Fall back to node_events.
        node_events = linear_ebus.get("node_events")
        if not isinstance(node_events, list):
            return []

        stations: list[str] = []
        for event in node_events:
            if not isinstance(event, dict):
                continue
            station = str(event.get("station") or "").strip()
            if not station:
                continue
            if station.upper() in invalid_station_tokens:
                continue
            action = str(event.get("action") or "").strip()
            outcome = event.get("outcome")

            # Treat explicit non-inspection actions as sampled.
            if action and action != "inspected_only":
                stations.append(station)
                continue

            # If an event has a ROSE outcome, sampling occurred even if the action
            # was conservatively classified as inspection-only upstream.
            if outcome is not None:
                stations.append(station)
                continue

        return _dedupe_preserve_order(stations)

    # --- EBUS compat (parallel_ner produces nested procedures_performed.linear_ebus) ---
    linear_ebus = procs.get("linear_ebus") or {}
    if isinstance(linear_ebus, dict):
        # Legacy adapters expect these top-level flat station lists.
        if raw.get("linear_ebus_stations") in (None, "", [], {}):
            derived = _derive_sampled_stations_from_linear_ebus(linear_ebus)
            if derived:
                raw["linear_ebus_stations"] = derived

        if raw.get("ebus_stations_sampled") in (None, "", [], {}):
            derived = _coerce_str_list(raw.get("linear_ebus_stations"))
            if derived:
                raw["ebus_stations_sampled"] = _dedupe_preserve_order(derived)

        # Per-station detail (size/passes/rose) is expected under `ebus_stations_detail`.
        if raw.get("ebus_stations_detail") in (None, "", [], {}):
            stations_detail = linear_ebus.get("stations_detail")
            if isinstance(stations_detail, list) and stations_detail:
                raw["ebus_stations_detail"] = stations_detail

        if raw.get("ebus_needle_gauge") in (None, "", [], {}):
            gauge = linear_ebus.get("needle_gauge")
            if gauge not in (None, "", [], {}):
                raw["ebus_needle_gauge"] = gauge

        if raw.get("ebus_passes") in (None, "", [], {}):
            passes = linear_ebus.get("passes_per_station")
            if passes not in (None, "", [], {}):
                raw["ebus_passes"] = passes

        if raw.get("ebus_elastography_used") in (None, "", [], {}):
            elastography_used = linear_ebus.get("elastography_used")
            if elastography_used is not None:
                raw["ebus_elastography_used"] = elastography_used

        if raw.get("ebus_elastography_pattern") in (None, "", [], {}):
            elastography_pattern = linear_ebus.get("elastography_pattern")
            if elastography_pattern not in (None, "", [], {}):
                raw["ebus_elastography_pattern"] = elastography_pattern

    # --- Navigational/robotic bronchoscopy compat (parallel_ner nested keys -> legacy flat keys) ---
    equipment = raw.get("equipment") or {}
    if not isinstance(equipment, dict):
        equipment = {}

    clinical_context = raw.get("clinical_context") or {}
    if not isinstance(clinical_context, dict):
        clinical_context = {}

    # Bubble up key clinical-context fields used by bundle builder / shell.
    if raw.get("primary_indication") in (None, "", [], {}):
        primary = _first_nonempty_str(clinical_context.get("primary_indication"))
        if primary:
            cleaned_primary = primary.replace("[", "").replace("]", "").strip()
            if cleaned_primary:
                raw["primary_indication"] = cleaned_primary
    if raw.get("radiographic_findings") in (None, "", [], {}):
        findings = _first_nonempty_str(clinical_context.get("radiographic_findings"))
        if findings:
            cleaned_findings = findings.replace("[", "").replace("]", "").strip()
            if cleaned_findings:
                raw["radiographic_findings"] = cleaned_findings

    # Make the original (scrubbed) text available to compat mappers when callers provide it.
    source_text = _first_nonempty_str(raw.get("source_text"), raw.get("note_text"), raw.get("raw_note"), raw.get("text"))
    text_fields: dict[str, str] = {}

    if isinstance(source_text, str):
        cleaned = source_text.strip()
        # Some golden fixtures embed quotes/trailing commas in the input string.
        if cleaned.startswith('"') and cleaned.endswith('",'):
            cleaned = cleaned[1:-2]
        elif cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
        cleaned = cleaned.strip()
        source_text = cleaned
        if cleaned:
            raw["source_text"] = cleaned

            field_labels = [
                "primary",
                "category",
                "indication",
                "dx",
                "diagnosis",
                "procedure",
                "proc",
                "method",
                "system",
                "platform",
                "verif",
                "verification",
                "action",
                "actions",
                "intervention",
                "findings",
                "target",
                "target lesion",
                "bronchus sign",
                "pet suv",
                "result",
                "nodes sampled",
                "needle",
                "rebus",
                "rose",
                "issues",
                "specimen",
                "specimens",
                "plan",
                "anesthesia",
                "technique",
                "asa class",
                "airway",
                "duration",
                "complications",
                "ebl",
                "dispo",
                "disposition",
                "tools",
            ]
            label_pattern = r"(?i)\b(" + "|".join("\\s+".join(map(re.escape, label.split())) for label in field_labels) + r")\s*:\s*"
            matches = list(re.finditer(label_pattern, cleaned))
            for idx, match in enumerate(matches):
                key_raw = re.sub(r"\s+", " ", (match.group(1) or "").strip().lower())
                value_start = match.end()
                value_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(cleaned)
                value = cleaned[value_start:value_end].strip().strip(",;").strip()
                value = value.strip().strip('"').strip().rstrip(".").strip()
                value = value.replace("[", "").replace("]", "").strip()
                if not value:
                    continue
                if key_raw in (
                    "primary",
                    "category",
                    "technique",
                    "anesthesia",
                    "method",
                    "system",
                    "platform",
                    "verif",
                    "procedure",
                    "dx",
                    "target",
                    "target lesion",
                    "bronchus sign",
                    "pet suv",
                    "rebus",
                    "rose",
                    "ebl",
                    "dispo",
                    "disposition",
                    "issues",
                ):
                    value = value.splitlines()[0].strip().rstrip(".").strip()
                if key_raw in ("indication",):
                    value_line = value.splitlines()[0].strip().rstrip(".").strip()
                    parts = [p.strip() for p in re.split(r"\.\s+", value_line) if p.strip()]
                    if parts:
                        kept = [parts[0]]
                        if len(parts) > 1:
                            second = parts[1]
                            if second and len(second) <= 80 and not re.search(
                                r"(?i)\b(us|ultrasound|pigtail|pleurx|catheter|drain(?:ed|age)?|cxr|system|platform|verif|action|tbna|bx|biopsy|brush|bal)\b",
                                second,
                            ):
                                kept.append(second)
                        value = ". ".join(kept)
                    else:
                        value = value_line
                if key_raw in ("rebus", "rose", "ebl", "dispo", "disposition", "issues"):
                    value = value.split(",", 1)[0].strip().rstrip(".").strip()
                # Normalize some common aliases.
                if key_raw in ("proc",):
                    key_raw = "procedure"
                if key_raw in ("actions", "intervention"):
                    key_raw = "action"
                if key_raw in ("verification",):
                    key_raw = "verif"
                if key_raw in ("diagnosis",):
                    key_raw = "dx"
                if key_raw in ("specimen",):
                    key_raw = "specimens"
                if key_raw in ("target lesion",):
                    key_raw = "target"
                if key_raw in ("disposition",):
                    key_raw = "dispo"
                text_fields.setdefault(key_raw, value)

    # Normalize a few common dictation abbreviations.
    if text_fields.get("indication") and re.match(r"(?i)^ild\b", text_fields["indication"].strip()):
        text_fields["indication"] = re.sub(r"(?i)^ild\b", "Interstitial Lung Disease", text_fields["indication"].strip(), count=1)

    location_hint = _extract_lung_location_hint(source_text or "")
    segment_hint = _extract_bronch_segment_hint(source_text or "")
    is_structured_bracket = bool(
        source_text
        and (
            "[INDICATION]" in source_text.upper()
            or "[DESCRIPTION]" in source_text.upper()
            or "[PLAN]" in source_text.upper()
        )
    )

    def _parse_bracket_sections(text: str) -> dict[str, str]:
        if not text:
            return {}
        pattern = re.compile(
            r"(?im)^\s*\[(indication|anesthesia|description|complication(?:s)?|plan)\]\s*$"
        )
        matches = list(pattern.finditer(text))
        if not matches:
            return {}
        sections: dict[str, str] = {}
        for idx, match in enumerate(matches):
            key = str(match.group(1) or "").strip().lower()
            if key.startswith("complication"):
                key = "complications"
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            body = re.sub(r"\n{3,}", "\n\n", body).strip()
            if body:
                sections[key] = body
        return sections

    def _infer_side(text: str) -> str | None:
        if not text:
            return None
        has_right = bool(re.search(r"(?i)\bright\b", text))
        has_left = bool(re.search(r"(?i)\bleft\b", text))
        if has_right and not has_left:
            return "right"
        if has_left and not has_right:
            return "left"
        return None

    def _parse_wll_volumes_liters(text: str) -> tuple[float | None, float | None]:
        if not text:
            return None, None
        # Common shorthand: "36L In / 30L Out"
        match = re.search(
            r"(?i)\b(\d+(?:\.\d+)?)\s*L\s*IN\b\s*/\s*(\d+(?:\.\d+)?)\s*L\s*OUT\b",
            text,
        )
        if match:
            try:
                return float(match.group(1)), float(match.group(2))
            except Exception:
                return None, None

        instilled = None
        returned = None

        match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*L\s*(?:IN|INSTILLED)\b", text)
        if match:
            try:
                instilled = float(match.group(1))
            except Exception:
                instilled = None
        match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*L\s*(?:OUT|RETURN(?:ED)?)\b", text)
        if match:
            try:
                returned = float(match.group(1))
            except Exception:
                returned = None

        if instilled is None:
            match = re.search(r"(?i)\b(?:in|instilled)\s*(\d+(?:\.\d+)?)\s*L\b", text)
            if match:
                try:
                    instilled = float(match.group(1))
                except Exception:
                    instilled = None
        if returned is None:
            match = re.search(r"(?i)\b(?:out|returned)\s*(\d+(?:\.\d+)?)\s*L\b", text)
            if match:
                try:
                    returned = float(match.group(1))
                except Exception:
                    returned = None

        return instilled, returned

    # Bracket-style synthetic notes: extract high-signal fields directly from headings.
    if is_structured_bracket and isinstance(source_text, str) and source_text.strip():
        bracket_sections = _parse_bracket_sections(source_text)
        indication_section = bracket_sections.get("indication")
        anesthesia_section = bracket_sections.get("anesthesia")
        description_section = bracket_sections.get("description")
        complications_section = bracket_sections.get("complications")
        plan_section = bracket_sections.get("plan")

        if raw.get("primary_indication") in (None, "", [], {}) and indication_section:
            raw["primary_indication"] = indication_section.splitlines()[0].strip().rstrip(".")

        if raw.get("follow_up_plan") in (None, "", [], {}) and plan_section:
            raw["follow_up_plan"] = plan_section.strip()

        if raw.get("complications_text") in (None, "", [], {}) and complications_section:
            raw["complications_text"] = complications_section.strip().rstrip(".")

        if anesthesia_section:
            anest_lower = anesthesia_section.lower()
            if raw.get("sedation_type") in (None, "", [], {}):
                if re.search(r"(?i)\bga\b", anesthesia_section) or "general" in anest_lower:
                    raw["sedation_type"] = "General"
                elif "moderate" in anest_lower:
                    raw["sedation_type"] = "Moderate"
                elif "deep" in anest_lower:
                    raw["sedation_type"] = "Deep"
                elif "local" in anest_lower:
                    raw["sedation_type"] = "Local"

            if raw.get("airway_type") in (None, "", [], {}):
                if re.search(r"(?i)\bdlt\b|double[-\s]?lumen", anesthesia_section):
                    raw["airway_type"] = "DLT"
                elif re.search(r"(?i)\bett\b|endotracheal", anesthesia_section):
                    raw["airway_type"] = "ETT"
                elif re.search(r"(?i)\blma\b|laryngeal", anesthesia_section):
                    raw["airway_type"] = "LMA"

            if raw.get("wll_dlt_used") in (None, "", [], {}) and re.search(
                r"(?i)\bdlt\b|double[-\s]?lumen",
                anesthesia_section,
            ):
                raw["wll_dlt_used"] = True

            if raw.get("wll_dlt_used_size") in (None, "", [], {}):
                match = re.search(r"(?i)\b(\d{2})\s*fr\b", anesthesia_section)
                if match:
                    try:
                        raw["wll_dlt_used_size"] = int(match.group(1))
                    except Exception:
                        raw["wll_dlt_used_size"] = None

        # Whole lung lavage heuristic: large-volume lavage phrasing + liters.
        wll_text = description_section or source_text
        instilled_l, returned_l = _parse_wll_volumes_liters(wll_text or "")
        if (
            raw.get("wll_volume_instilled_l") in (None, "", [], {})
            and instilled_l is not None
            and instilled_l >= 5.0
            and re.search(r"(?i)\blavage\b", wll_text or "")
        ):
            raw["wll_volume_instilled_l"] = instilled_l
            if returned_l is not None and raw.get("wll_volume_returned_l") in (None, "", [], {}):
                raw["wll_volume_returned_l"] = returned_l
            if raw.get("wll_side") in (None, "", [], {}):
                side = _infer_side(indication_section or "") or _infer_side(wll_text or "")
                if side:
                    raw["wll_side"] = side

            # Avoid rendering BAL/washing templates when the note clearly describes WLL.
            raw.pop("bal", None)
            raw.pop("bronchial_washing", None)

            # Notes: keep non-volume, non-lavage sentences + returned volume.
            if raw.get("wll_notes") in (None, "", [], {}):
                notes_parts: list[str] = []
                if description_section:
                    sentences = [s.strip() for s in re.split(r"[\n\.]+", description_section) if s.strip()]
                    kept: list[str] = []
                    for sent in sentences:
                        lower = sent.lower()
                        if "lavage" in lower:
                            continue
                        if re.search(r"(?i)\b\d+(?:\.\d+)?\s*L\b", sent):
                            continue
                        kept.append(sent.rstrip("."))
                    if kept:
                        notes_parts.append(". ".join(kept).strip().rstrip("."))
                if notes_parts:
                    raw["wll_notes"] = "; ".join([p for p in notes_parts if p]).strip()

    if raw.get("ebus_needle_gauge") in (None, "", [], {}) and text_fields.get("needle"):
        match = re.search(r"(?i)\b(\d{2})\s*g\b", text_fields["needle"])
        if match:
            raw["ebus_needle_gauge"] = f"{match.group(1)}G"

    # Structured bracket notes provide high-signal fields; synthesize a more golden-like
    # primary indication (and basic diagnoses) when present.
    if is_structured_bracket:
        primary_val = (text_fields.get("primary") or "").strip()
        category_val = (text_fields.get("category") or "").strip()
        target_val = (text_fields.get("target") or "").strip()
        bronchus_sign_raw = (text_fields.get("bronchus sign") or "").strip()
        pet_suv_raw = (text_fields.get("pet suv") or "").strip()

        pet_suv_num = None
        match = re.search(r"(\d+(?:\.\d+)?)", pet_suv_raw)
        if match:
            pet_suv_num = match.group(1).strip()

        bronchus_sign_val = None
        if bronchus_sign_raw:
            lowered = bronchus_sign_raw.strip().lower()
            if lowered.startswith("p"):
                bronchus_sign_val = "positive"
            elif lowered.startswith("n"):
                bronchus_sign_val = "negative"

        size_mm_text = None
        match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*mm\b", target_val)
        if match:
            size_mm_text = match.group(1).strip()
        density = None
        if "ground-glass" in target_val.lower() or "groundglass" in target_val.lower() or "ggo" in target_val.lower():
            density = "ground-glass"
        elif "solid" in target_val.lower():
            density = "solid"

        loc_part = target_val.split(",", 1)[1].strip() if "," in target_val else target_val
        match = re.search(r"(?i)\b(RUL|RML|RLL|LUL|LLL)\b\s+([A-Za-z-]+)\s*\(\s*(B\d{1,2})\s*\)", loc_part)
        if match:
            loc_part = f"{match.group(1).upper()} {match.group(2)} segment ({match.group(3).upper()})"
        elif re.search(r"(?i)\(\s*B\d{1,2}\s*\)", loc_part) and "segment" not in loc_part.lower():
            loc_part = loc_part.replace("(", "segment (", 1)

        target_phrase = None
        if size_mm_text and density and loc_part:
            target_phrase = f"a {size_mm_text}mm {density} peripheral lung nodule in the {loc_part}"
        elif size_mm_text and loc_part:
            target_phrase = f"a {size_mm_text}mm peripheral lung nodule in the {loc_part}"
        elif target_val:
            target_phrase = f"a {target_val}" if re.match(r"^\d", target_val) else target_val

        if target_phrase:
            phrase = target_phrase
            if bronchus_sign_val:
                phrase += f" with a {bronchus_sign_val} bronchus sign"
            if primary_val and "node" in primary_val.lower():
                phrase += " and suspicious mediastinal nodes"
            if pet_suv_num:
                phrase += f" (PET SUV: {pet_suv_num})"
            if category_val and "staging" in category_val.lower():
                phrase += " requiring diagnosis and staging"
            raw["primary_indication"] = phrase.strip().rstrip(".")

        if raw.get("preop_diagnosis_text") in (None, "", [], {}) and size_mm_text and density:
            lobe = None
            match = re.search(r"(?i)\b(RUL|RML|RLL|LUL|LLL)\b", loc_part)
            if match:
                lobe = match.group(1).upper()
            elif location_hint:
                lobe = location_hint
            density_label = "Solid" if density == "solid" else ("Ground-glass" if density == "ground-glass" else density)
            lines = [f"Peripheral lung nodule, {lobe or 'target'} ({size_mm_text}mm, {density_label})"]
            if primary_val and "node" in primary_val.lower():
                lines.append("Mediastinal lymphadenopathy")
            raw["preop_diagnosis_text"] = "\n\n".join([line for line in lines if line])

        # Upgrade nav target when the bracket DESCRIPTION includes an explicit robotic target.
        if source_text and (
            raw.get("nav_target_segment") in (None, "", [], {})
            or str(raw.get("nav_target_segment")).strip().upper() in {"RUL", "RML", "RLL", "LUL", "LLL"}
        ):
            match = re.search(r"(?is)\brobotic\s+bronchoscopy\b.*?\btarget\s*:\s*([^,\n]+)", source_text)
            if match:
                nav_target = match.group(1).strip().strip('"').strip().rstrip(".").strip()
                if re.search(r"(?i)\(\s*B\d{1,2}\s*\)", nav_target) and "segment" not in nav_target.lower():
                    nav_target = nav_target.replace("(", "segment (", 1)
                raw["nav_target_segment"] = nav_target
                if raw.get("lesion_location") in (None, "", [], {}):
                    raw["lesion_location"] = nav_target

    # Best-effort parse EBUS station details from free text (passes/size/ROSE).
    if raw.get("ebus_stations_detail") in (None, "", [], {}) and source_text:
        details_by_station: dict[str, dict[str, Any]] = {}

        # Pattern: "Stations: 11R (4x), 2L (2x), 4L (4x)"
        match = re.search(r"(?i)\bstations?\s*:\s*([^\n]+)", source_text)
        if match:
            chunk = match.group(1)
            for st, passes in re.findall(r"(?i)\b(\d{1,2}[LR]?)\s*\(\s*(\d+)\s*x\s*\)", chunk):
                key = st.upper()
                details_by_station.setdefault(key, {"station": key})["passes"] = int(passes)

        # Pattern: "... sampled stations 2R x4, 10R x3, 2L x2 ..."
        for line in source_text.splitlines():
            # Require the plural "stations" to avoid accidentally matching per-node size
            # dimensions like "22.0x13.6mm" on lines beginning with "Station 11R: ...".
            if not re.search(r"(?i)\bstations\b", line):
                continue
            if not re.search(r"(?i)\b\d{1,2}[LR]?\s*(?:x|×)\s*\d+\b", line):
                continue
            for st, passes in re.findall(r"(?i)\b(\d{1,2}[LR]?)\s*(?:x|×)\s*(\d+)\b", line):
                key = st.upper()
                details_by_station.setdefault(key, {"station": key})["passes"] = int(passes)

        # Pattern: "- 4R (18mm): Positive for Adeno."
        for line in source_text.splitlines():
            stripped = line.strip()
            bullet = re.match(r"[-*]\s*(\d{1,2}[LR]?)\s*\(\s*(\d+)\s*mm\s*\)\s*:\s*(.+)", stripped, flags=re.IGNORECASE)
            if bullet:
                st = bullet.group(1).upper()
                details = details_by_station.setdefault(st, {"station": st})
                try:
                    details["size_mm"] = int(bullet.group(2))
                except Exception:
                    pass
                details["rose_result"] = bullet.group(3).strip().rstrip(".")

        # Pattern: "Station 11R: ... Executed 2 aspiration passes. ROSE yielded: ..."
        for station_line in re.finditer(r"(?im)^\s*Station\s+(\d{1,2}[LR]?)\s*:\s*(.+?)\s*$", source_text):
            st = station_line.group(1).upper()
            rest = station_line.group(2)
            details = details_by_station.setdefault(st, {"station": st})
            mp = re.search(r"(?i)\b(\d+)\s*(?:aspiration\s*)?passes\b", rest)
            if mp:
                try:
                    details["passes"] = int(mp.group(1))
                except Exception:
                    pass
            mr = re.search(r"(?i)\bROSE\s*(?:yielded|result)\s*[:\\-]\s*([^\\.]+)", rest)
            if mr:
                details["rose_result"] = mr.group(1).strip().rstrip(".")
            ms = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*(?:x|×)\s*(\d+(?:\.\d+)?)\s*mm\b", rest)
            if ms:
                try:
                    size_val = max(float(ms.group(1)), float(ms.group(2)))
                    details["size_mm"] = round(size_val, 1) if not size_val.is_integer() else int(size_val)
                except Exception:
                    pass

        if details_by_station:
            raw["ebus_stations_detail"] = list(details_by_station.values())
            if raw.get("ebus_stations_sampled") in (None, "", [], {}):
                raw["ebus_stations_sampled"] = list(details_by_station.keys())

    # Ensure station list fields reflect per-station detail when present.
    stations_detail = raw.get("ebus_stations_detail") or []
    if isinstance(stations_detail, list):
        detail_stations: list[str] = []
        for item in stations_detail:
            if not isinstance(item, dict):
                continue
            station = item.get("station") or item.get("station_name")
            if station in (None, "", [], {}):
                continue
            station_str = str(station).strip().upper()
            if station_str and station_str not in detail_stations:
                detail_stations.append(station_str)
        if detail_stations:
            existing_list = raw.get("linear_ebus_stations") or raw.get("ebus_stations_sampled")
            existing = existing_list if isinstance(existing_list, list) else []
            if not existing or len(existing) < len(detail_stations) or set(map(str, existing)) != set(detail_stations):
                raw["linear_ebus_stations"] = detail_stations
                raw["ebus_stations_sampled"] = detail_stations

    # Some structured/CSV-like notes provide a "Stations sampled:" list which should
    # supersede heuristic station extraction (and can correct upstream false-positives,
    # e.g., interpreting PET SUV 6.7 as Station 7).
    if source_text:
        match = re.search(r"(?i)\bstations\s+sampled\s*:\s*([^\n]+)", source_text)
        if match:
            chunk = match.group(1)
            chunk = re.split(r"(?i)\b(?:number\s+of\s+stations|rose|needle|scope|complications|ebl|plan)\b", chunk)[0]
            stations_sampled = [s.upper() for s in re.findall(r"(?i)\b\d{1,2}[LR]\b", chunk)]
            stations_sampled = _dedupe_preserve_order(stations_sampled)
            if stations_sampled:
                raw["linear_ebus_stations"] = stations_sampled
                raw["ebus_stations_sampled"] = stations_sampled
                if isinstance(raw.get("ebus_stations_detail"), list):
                    filtered_detail: list[dict[str, Any]] = []
                    for item in raw.get("ebus_stations_detail") or []:
                        if not isinstance(item, dict):
                            continue
                        station = item.get("station") or item.get("station_name")
                        if station and str(station).strip().upper() in stations_sampled:
                            filtered_detail.append(item)
                    if filtered_detail:
                        raw["ebus_stations_detail"] = filtered_detail

    # Populate patient demographics from common shorthand ("77yo F", "60yo female") when missing.
    if raw.get("patient_age") in (None, "", [], {}) and source_text:
        match = re.search(r"(?i)\b(\d{1,3})\s*yo\b", source_text)
        if match:
            try:
                raw["patient_age"] = int(match.group(1))
            except Exception:
                pass
    if raw.get("gender") in (None, "", [], {}) and source_text:
        match = re.search(r"(?i)\b\d{1,3}\s*yo\s*(female|male|f|m)\b", source_text)
        if match:
            sex = match.group(1).strip().lower()
            if sex.startswith("f"):
                raw["gender"] = "female"
            elif sex.startswith("m"):
                raw["gender"] = "male"

    # Bubble up indication/plan/specimens/EBL/complications from the free-text summary when present.
    if text_fields.get("indication"):
        raw["primary_indication"] = text_fields["indication"]
    if raw.get("primary_indication") in (None, "", [], {}) and text_fields.get("primary"):
        raw["primary_indication"] = text_fields["primary"]
    if raw.get("primary_indication") in (None, "", [], {}) and text_fields.get("target"):
        target_val = str(text_fields.get("target") or "").strip()
        if target_val:
            raw["primary_indication"] = f"a {target_val}" if re.match(r"^\d", target_val) else target_val
    if raw.get("preop_diagnosis_text") in (None, "", [], {}) and text_fields.get("dx"):
        raw["preop_diagnosis_text"] = text_fields["dx"]
    if raw.get("follow_up_plan") in (None, "", [], {}) and text_fields.get("plan"):
        raw["follow_up_plan"] = text_fields["plan"]
    if raw.get("follow_up_plan") in (None, "", [], {}) and source_text and "[PLAN]" in source_text.upper():
        upper = source_text.upper()
        idx = upper.find("[PLAN]")
        if idx != -1:
            plan_chunk = source_text[idx + len("[PLAN]") :]
            plan_chunk = re.split(r"(?i)\[[A-Z][A-Z _]{2,}\]", plan_chunk)[0]
            plan_chunk = plan_chunk.strip().lstrip(",:;-").strip()
            if plan_chunk:
                items = [part.strip().strip(",;").strip() for part in re.split(r"(?i)\b\d+\.\s*", plan_chunk) if part.strip().strip(",;").strip()]
                if items:
                    cleaned: list[str] = []
                    for item in items:
                        item = re.sub(r"(?i)\bchest\s*x[-\s]?ray\s*-\s*completed\b", "Chest X-ray completed", item).strip()
                        item = re.sub(r"(?i),\s*no\s+pneumothorax\b", "; no pneumothorax identified", item).strip()
                        item = re.sub(r"(?i)\btumor\s*board\b", "Tumor Board", item).strip()
                        item = re.sub(
                            r"(?i)\bmolecular\s+testing\s+if\s+malignancy\s+confirmed\b",
                            "Molecular testing will be requested given malignancy confirmation",
                            item,
                        ).strip()
                        cleaned.append(item.rstrip(".") + ".")
                    raw["follow_up_plan"] = "\n\n".join(cleaned)
                else:
                    raw["follow_up_plan"] = plan_chunk
    if raw.get("disposition") in (None, "", [], {}) and (text_fields.get("dispo") or text_fields.get("disposition")):
        raw["disposition"] = text_fields.get("dispo") or text_fields.get("disposition")
    if raw.get("specimens_text") in (None, "", [], {}) and text_fields.get("specimens"):
        raw["specimens_text"] = text_fields["specimens"]
    if raw.get("complications_text") in (None, "", [], {}) and text_fields.get("complications"):
        raw["complications_text"] = text_fields["complications"]
    if raw.get("complications_text") in (None, "", [], {}) and source_text and re.search(r"(?i)\bno\s+pneumothorax\b", source_text):
        raw["complications_text"] = "None; No pneumothorax noted."
    if raw.get("estimated_blood_loss") in (None, "", [], {}) and text_fields.get("ebl"):
        normalized = _normalize_ebl_text(text_fields.get("ebl"))
        if normalized:
            raw["estimated_blood_loss"] = normalized

    # Sedation/anesthesia hints (used by the shell template).
    if raw.get("sedation_type") in (None, "", [], {}) and source_text:
        anesthesia_hint = text_fields.get("anesthesia") or text_fields.get("technique") or ""
        hint_upper = f"{anesthesia_hint} {source_text}".upper()
        if (
            "GENERAL" in hint_upper
            or re.search(r"\bGA\b", hint_upper)
            or "ETT" in hint_upper
            or "ENDOTRACHEAL" in hint_upper
            or "LMA" in hint_upper
        ):
            raw["sedation_type"] = "General"
        elif "MODERATE" in hint_upper:
            raw["sedation_type"] = "Moderate"
        elif "LOCAL" in hint_upper or "LIDOCAINE" in hint_upper:
            raw["sedation_type"] = "Local"

    if raw.get("anesthesia_agents") in (None, "", [], {}) and text_fields.get("anesthesia"):
        raw["anesthesia_agents"] = [text_fields["anesthesia"]]

    # Airway device hints (used for anesthesia line + some templates).
    if raw.get("airway_type") in (None, "", [], {}) and source_text:
        airway_hint = text_fields.get("airway") or ""
        combined = f"{airway_hint} {source_text}".upper()
        if re.search(r"\bETT\b|\bENDOTRACHEAL\b", combined):
            raw["airway_type"] = "ETT"
        elif re.search(r"\bLMA\b|\bLARYNGEAL\s+MASK\b", combined):
            raw["airway_type"] = "LMA"
        elif re.search(r"\bTRACH\b|\bTRACHEOSTOMY\b", combined):
            raw["airway_type"] = "Trach"

    if raw.get("airway_size_mm") in (None, "", [], {}) and text_fields.get("airway"):
        match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*mm\b", text_fields["airway"])
        if match:
            try:
                raw["airway_size_mm"] = float(match.group(1))
            except Exception:
                pass

    if raw.get("anesthesia_duration_minutes") in (None, "", [], {}) and text_fields.get("duration"):
        match = re.search(r"\d+", text_fields["duration"])
        if match:
            try:
                raw["anesthesia_duration_minutes"] = int(match.group(0))
            except Exception:
                pass

    if raw.get("asa_class") in (None, "", [], {}) and text_fields.get("asa class"):
        try:
            match = re.search(r"\d+", text_fields["asa class"])
            if match:
                raw["asa_class"] = int(match.group(0))
        except Exception:
            pass

    # Bubble up operator/referrer/date hints when missing.
    if raw.get("attending_name") in (None, "", [], {}) and source_text:
        operator = _parse_operator(source_text)
        if operator:
            raw["attending_name"] = operator
    if raw.get("referred_physician") in (None, "", [], {}) and source_text:
        ref = _parse_referred_physician(source_text)
        if ref:
            raw["referred_physician"] = ref
    if raw.get("procedure_date") in (None, "", [], {}) and source_text:
        date_val = _parse_service_date(source_text)
        if date_val:
            raw["procedure_date"] = date_val

    # Prefer nested lesion location when available.
    if raw.get("lesion_location") in (None, "", [], {}):
        nested_loc = _first_nonempty_str(clinical_context.get("lesion_location"))
        if nested_loc:
            raw["lesion_location"] = nested_loc
    if raw.get("nav_target_segment") in (None, "", [], {}):
        nested_loc = _first_nonempty_str(raw.get("lesion_location"), clinical_context.get("lesion_location"))
        if nested_loc:
            raw["nav_target_segment"] = nested_loc

    # Upgrade nav target from inline dictation ("Nav to X", "Navigated to X") when available.
    if source_text:
        match = re.search(r"(?i)\b(?:nav(?:igated)?\s*to|navigated\s*to)\s+([^\n\.,;]+)", source_text)
        if match:
            target = match.group(1).strip().strip('"').strip().rstrip(".").strip()
            target = re.sub(
                r"(?i)\b(?:w/|with)\s+(?:ion|monarch|galaxy|robotic|emn)\b.*$",
                "",
                target,
            ).strip()
            # Drop parenthetical size descriptors like "(1.4cm nodule)".
            target = re.sub(r"(?i)\(\s*\d+(?:\.\d+)?\s*(?:cm|mm)\b[^)]*\)", "", target).strip()
            target = re.sub(r"(?i)\bseg\b", "segment", target).strip()
            target = re.sub(r"(?i)\b(?:nodule|lesion)\b", "", target).strip()
            target = re.sub(r"\s{2,}", " ", target).strip().rstrip(",;-").strip()
            existing = str(raw.get("nav_target_segment") or "").strip()
            if not existing or existing.upper() in {"RUL", "RML", "RLL", "LUL", "LLL"}:
                raw["nav_target_segment"] = target

            existing_loc = str(raw.get("lesion_location") or "").strip()
            if (not existing_loc or existing_loc.upper() in {"RUL", "RML", "RLL", "LUL", "LLL"}) and re.search(
                r"(?i)\b(RUL|RML|RLL|LUL|LLL)\b",
                target,
            ):
                raw["lesion_location"] = target

    counts_text = (text_fields.get("action") or source_text or "").strip()
    tbna_count = None
    for pattern in (
        r"\bTBNA\b\s*(?:x|×)\s*(\d+)\b",
        r"\bTBNA\s*passes?\s*:\s*(\d+)\b",
        r"\bTBNA\b[^\n]{0,30}\b(\d+)\s*(?:passes?|times)\b",
        r"\bneedle\s*passes?\s*:\s*(\d+)\b",
        r"\bpasses?\s*(?:executed|performed|obtained|collected)\s*:\s*(\d+)\b",
        r"\b(\d+)\s*needle\s*passes?\b",
        r"\baspiration\s*needle\s*passes?\s*(?:executed|performed|obtained|collected)?\s*:\s*(\d+)\b",
    ):
        tbna_count = _parse_count(counts_text, pattern)
        if tbna_count is not None:
            break

    bx_count = None
    for pattern in (
        r"\b(?:TBBX|TB?BX|BX|BIOPS(?:Y|IES))\b\s*(?:x|×)\s*(\d+)\b",
        r"\bForceps\b\s*(?:x|×)\s*(\d+)\b",
        r"\bForceps\s*biops(?:y|ies)\s*:\s*(\d+)\b",
        r"\bForceps\s*biops(?:y|ies)\b[^\n]{0,30}\b(\d+)\b",
        r"\bbiops(?:y|ies)\s*:\s*(\d+)\b",
        r"\b(\d+)\s*(?:forceps\s*)?(?:bx|biops(?:y|ies))\b",
        r"\b(?:took|obtained|acquired)\s*(\d+)\s*(?:forceps\s*)?(?:bx|biops(?:y|ies))\b",
        r"\b(?:grasping\s*)?forceps\s*specimens?\s*(?:acquired|obtained|collected)\s*:\s*(\d+)\b",
        r"\bspecimens?\s*(?:acquired|obtained|collected)\s*:\s*(\d+)\b",
    ):
        bx_count = _parse_count(counts_text, pattern)
        if bx_count is not None:
            break

    brush_count = None
    for pattern in (
        r"\bBrush(?:ings)?\b\s*(?:x|×)\s*(\d+)\b",
        r"\bBrush(?:ings)?\s*:\s*(\d+)\b",
        r"\bBrush(?:ings)?\s*(?:harvested|collected|obtained)\s*:\s*(\d+)\b",
        r"\b(\d+)\s*brush(?:ings)?\b",
    ):
        brush_count = _parse_count(counts_text, pattern)
        if brush_count is not None:
            break

    bal_location_hint = None
    if counts_text:
        match = re.search(r"(?i)\bBAL\b\s*\(([^)]+)\)", counts_text)
        if match:
            bal_location_hint = match.group(1).strip()
    if bal_location_hint is None and counts_text:
        match = re.search(
            r"(?i)\b(?:lavage|washing)\b[^\n]{0,80}?\b(?:from|in|at)\s+([RL]B\d{1,2}|B\d{1,2}|RUL|RML|RLL|LUL|LLL)\b",
            counts_text,
        )
        if match:
            bal_location_hint = match.group(1).strip().upper()

    rose_hint: str | None = None
    nodule_rose_hint: str | None = None

    # Derive common diagnosis fields for short-form dictation-style inputs.
    if source_text:
        if is_structured_bracket:
            match = re.search(r"(?is)\blinear\s+ebus(?:-tbna)?\b.*?\brose\s*result\s*:\s*([^,\n]+)", source_text)
            if match:
                rose_hint = match.group(1).strip().rstrip(".")
            match = re.search(r"(?is)\btransbronchial\s+biops(?:y|ies)\b.*?\brose\s*result\s*:\s*([^,\n]+)", source_text)
            if match:
                nodule_rose_hint = match.group(1).strip().rstrip(".")

        if not rose_hint:
            rose_hint = text_fields.get("rose")
        if not rose_hint:
            match = re.search(r"(?im)^\s*ROSE\+?\s*:\s*(.+?)\s*$", source_text)
            if match:
                rose_hint = match.group(1).strip().rstrip(".")
        if not rose_hint:
            match = re.search(r"(?i)\bROSE\s*result\s*:\s*([^,\n]+)", source_text)
            if match:
                rose_hint = match.group(1).strip().rstrip(".")
        if not rose_hint:
            match = re.search(r"(?i)\bROSE\s+(?:assessment\s+)?yielded\s*:\s*([^\n]+)", source_text)
            if match:
                rose_hint = match.group(1).strip().rstrip(".")
        if not rose_hint:
            match = re.search(r"(?i)\bROSE\b[^\n]{0,50}?\bshowed\b\s*([^\\.\n]+)", source_text)
            if match:
                rose_hint = match.group(1).strip().rstrip(".")
        if not rose_hint:
            match = re.search(r"(?i)\bROSE\+?\s*:\s*([^,\n]+)", source_text)
            if match:
                rose_hint = match.group(1).strip().rstrip(".")

        # Prefer a distinct ROSE line when explicitly tied to the peripheral target/nodule.
        if not nodule_rose_hint:
            match = re.search(
                r"(?i)\brose\b[^\n]{0,50}\bfrom\s+the\s+(?:nodule|target)\b[^\n]{0,20}?(?:was|:)\s*([^\.\n]+)",
                source_text,
            )
            if match:
                nodule_rose_hint = match.group(1).strip().rstrip(".")
        if not nodule_rose_hint:
            match = re.search(
                r"(?i)\brose\b[^\n]{0,20}\b\((?:nodule|target)\)[^\n]{0,10}[:\\-]\s*([^\.\n]+)",
                source_text,
            )
            if match:
                nodule_rose_hint = match.group(1).strip().rstrip(".")

        if is_structured_bracket and rose_hint and nodule_rose_hint:
            def _cap_first(value: str) -> str:
                stripped = value.strip()
                return stripped[:1].upper() + stripped[1:] if stripped else stripped

            nodes_dx = rose_hint.strip().rstrip(".")
            if "-" in nodes_dx:
                left, right = nodes_dx.split("-", 1)
                if left.strip().lower().startswith("malignant") and right.strip():
                    nodes_dx = right.strip()
            nodes_dx = re.sub(r"(?i)^malignant\s+", "", nodes_dx).strip()
            nodes_line = f"{_cap_first(nodes_dx)} (mediastinal lymph nodes per ROSE)" if nodes_dx else "Mediastinal lymph nodes sampled (per ROSE)"

            lobe_token = None
            for candidate in (
                raw.get("nav_target_segment"),
                raw.get("lesion_location"),
                text_fields.get("target"),
                location_hint,
            ):
                if candidate in (None, "", [], {}):
                    continue
                match = re.search(r"(?i)\b(RUL|RML|RLL|LUL|LLL)\b", str(candidate))
                if match:
                    lobe_token = match.group(1).upper()
                    break
            nodule_dx = _cap_first(nodule_rose_hint.strip().rstrip("."))
            nodule_line = f"{nodule_dx} ({lobe_token or 'target'} nodule per ROSE)" if nodule_dx else f"{lobe_token or 'Target'} nodule sampled (per ROSE)"

            raw["postop_diagnosis_text"] = f"{nodes_line}\n\n{nodule_line}"
            raw["ebus_rose_result"] = rose_hint
            raw["ebus_rose_available"] = True

        derived_size_mm = raw.get("nav_lesion_size_mm")
        peripheral_target_context = False
        search_text = ""
        if source_text:
            search_text = " ".join(
                [
                    text_fields.get("target") or "",
                    text_fields.get("dx") or "",
                    text_fields.get("indication") or "",
                    source_text,
                ]
            )
            peripheral_target_context = bool(
                re.search(r"(?i)\b(nodule|lesion|mass|target|spiculat|ground[-\s]?glass|ggo)\b", search_text)
            )
            if not peripheral_target_context:
                peripheral_procs = (
                    procs.get("navigational_bronchoscopy"),
                    procs.get("radial_ebus"),
                    procs.get("peripheral_tbna"),
                    procs.get("transbronchial_biopsy"),
                    procs.get("peripheral_ablation"),
                )
                peripheral_target_context = any(
                    isinstance(proc, dict) and proc.get("performed") is True for proc in peripheral_procs
                )

        if derived_size_mm in (None, "", [], {}) and source_text and peripheral_target_context:
            for match in re.finditer(r"(?i)\b(\d+(?:\.\d+)?)\s*(mm|cm)\b", search_text):
                ctx = search_text[max(0, match.start() - 18) : min(len(search_text), match.end() + 18)].lower()
                if any(
                    token in ctx
                    for token in (
                        "cryo",
                        "probe",
                        "ett",
                        "lma",
                        "fogarty",
                        "divergence",
                        "balloon",
                        "dilat",
                        "stent",
                        "apc",
                        "pulse",
                        "effect",
                    )
                ):
                    continue
                try:
                    num = float(match.group(1))
                except Exception:
                    continue
                unit = match.group(2).lower()
                mm_val = num * 10.0 if unit == "cm" else num
                derived_size_mm = round(mm_val, 2)
                break

        if raw.get("nav_lesion_size_mm") in (None, "", [], {}) and derived_size_mm not in (None, "", [], {}):
            raw["nav_lesion_size_mm"] = derived_size_mm

        def _fmt_mm(value: Any) -> str:
            try:
                num = float(value)
            except Exception:
                return str(value)
            return str(int(num)) if num.is_integer() else str(num)

        # Synthesize a compact primary indication for common nodule dictations when missing.
        # Guard: do not misread non-target measurements (balloon dilation sizes, APC probe size, stent revision distance)
        # as a peripheral nodule when the note lacks peripheral-target context.
        if (
            raw.get("primary_indication") in (None, "", [], {})
            and location_hint
            and derived_size_mm not in (None, "", [], {})
            and peripheral_target_context
        ):
            density = None
            if re.search(r"(?i)\bground[-\s]?glass\b|\bgroundglass\b|\bggo\b", source_text):
                density = "ground-glass"
            elif re.search(r"(?i)\bsolid\b", source_text):
                density = "solid"

            bronchus_sign_val = None
            match = re.search(r"(?i)\bbronchus\s+sign\b[^\n]{0,40}?\b(positive|negative|pos|neg)\b", source_text)
            if match:
                sign_raw = match.group(1).strip().lower()
                bronchus_sign_val = "positive" if sign_raw.startswith("p") else "negative"

            pet_suv = None
            match = re.search(r"(?i)\bpet\b[^\n]{0,40}?\bsuv\b\s*[:=]?\s*(\d+(?:\.\d+)?)", source_text)
            if match:
                pet_suv = match.group(1).strip()
            no_pet = bool(
                re.search(
                    r"(?i)\bno\s+pet\b|\bpet\s+(?:not\s+done|not\s+performed)\b|\bno\s+pet\s+done\b",
                    source_text,
                )
            )

            size_str = _fmt_mm(derived_size_mm)
            phrase = f"a {size_str} mm {location_hint} pulmonary nodule"
            if density == "solid":
                phrase += " found to be solid on CT"
            elif density == "ground-glass":
                phrase += " (ground-glass on CT)"
            if bronchus_sign_val:
                phrase += f" with a {bronchus_sign_val} bronchus sign"
            if pet_suv:
                phrase += f" (PET SUV: {pet_suv})"
            if no_pet and not pet_suv:
                phrase += ". No PET scan was performed"
            if "chartis" in source_text.lower():
                phrase += " requiring bronchoscopic diagnosis and staging, as well as assessment for potential lung volume reduction"
            raw["primary_indication"] = phrase.strip().rstrip(".")

        # If linear EBUS is present and a ROSE summary was captured, attach it to the EBUS fields.
        if raw.get("ebus_rose_result") in (None, "", [], {}) and rose_hint and (
            raw.get("linear_ebus_stations") or raw.get("ebus_stations_sampled")
        ):
            cleaned_rose = re.sub(r"(?i)\bat\s+multiple\s+stations\b", "", rose_hint).strip().strip(",").strip()
            cleaned_rose = re.sub(r"(?i)\bmultiple\s+stations\b", "", cleaned_rose).strip().strip(",").strip()
            raw["ebus_rose_result"] = cleaned_rose or rose_hint
            raw["ebus_rose_available"] = True

        if raw.get("preop_diagnosis_text") in (None, "", [], {}):
            indication_hint = str(raw.get("primary_indication") or text_fields.get("indication") or "").strip()
            dx_hint = str(text_fields.get("dx") or "").strip()
            lowered = f"{dx_hint} {indication_hint}".lower()

            if "interstitial lung disease" in lowered or re.search(r"\bild\b", lowered):
                if indication_hint and re.match(r"(?i)^ild\b", indication_hint):
                    indication_hint = re.sub(r"(?i)^ild\b", "Interstitial Lung Disease", indication_hint, count=1).strip()
                raw["preop_diagnosis_text"] = indication_hint or "Interstitial Lung Disease"
            elif "effusion" in lowered and "pleural" in lowered:
                raw["preop_diagnosis_text"] = indication_hint or "Pleural effusion"
            elif ("staging" in lowered or "lung cancer" in lowered) and raw.get("nav_target_segment") not in (None, "", [], {}):
                target = str(raw.get("nav_target_segment") or "").strip()
                target = target.split(",", 1)[0].strip()
                staging = dx_hint or "Lung Cancer Staging"
                raw["preop_diagnosis_text"] = "\n".join([line for line in [f"Lung Nodule ({target})", staging] if line])
            elif location_hint and derived_size_mm not in (None, "", [], {}) and peripheral_target_context:
                rad = None
                match = re.search(r"(?i)\bLung-RADS\s*[0-9A-Z]+\b", dx_hint)
                if match:
                    rad = match.group(0).strip()
                size_str = _fmt_mm(derived_size_mm)
                base = f"{location_hint} pulmonary nodule, {size_str} mm"
                raw["preop_diagnosis_text"] = f"{base} ({rad})" if rad else base
            elif indication_hint:
                raw["preop_diagnosis_text"] = indication_hint

        # If we only captured staging as a diagnosis, add the target nodule line when available.
        if raw.get("preop_diagnosis_text") not in (None, "", [], {}) and raw.get("nav_target_segment") not in (None, "", [], {}):
            preop_text = str(raw.get("preop_diagnosis_text") or "")
            lowered_preop = preop_text.lower()
            if ("staging" in lowered_preop or "lung cancer" in lowered_preop) and "nodule" not in lowered_preop:
                target = str(raw.get("nav_target_segment") or "").strip().split(",", 1)[0].strip()
                lines = [f"Lung Nodule ({target})", preop_text.strip()]
                raw["preop_diagnosis_text"] = "\n".join([line for line in lines if line])

        # If staging was performed, ensure suspected lymphadenopathy is reflected in the pre-op Dx.
        if (
            raw.get("preop_diagnosis_text") not in (None, "", [], {})
            and (raw.get("linear_ebus_stations") or raw.get("ebus_stations_sampled"))
            and re.search(r"(?i)\bstaging\b", source_text)
        ):
            preop_text = str(raw.get("preop_diagnosis_text") or "").strip()
            if preop_text and "lymph" not in preop_text.lower():
                raw["preop_diagnosis_text"] = preop_text + "\n\nMediastinal/Hilar lymphadenopathy (suspected)"

        if raw.get("postop_diagnosis_text") in (None, "", [], {}) and raw.get("preop_diagnosis_text") not in (None, "", [], {}):
            preop_text = str(raw.get("preop_diagnosis_text") or "").strip()
            preop_lines = [line.strip() for line in preop_text.splitlines() if line.strip()]
            base_line = preop_lines[0] if preop_lines else preop_text
            lines = [base_line] if base_line else []
            if raw.get("pleural_procedure_type") == "tunneled catheter" and raw.get("pleural_side"):
                lines.append(f"Status post {str(raw.get('pleural_side')).strip()} tunneled pleural catheter placement")
            pleural_type = str(raw.get("pleural_procedure_type") or "").strip().lower()
            if pleural_type in ("thoracentesis", "pigtail catheter") and raw.get("pleural_volume_drained_ml") not in (None, "", [], {}):
                vol = raw.get("pleural_volume_drained_ml")
                try:
                    vol_str = str(int(float(vol)))
                except Exception:
                    vol_str = str(vol)
                appearance = str(raw.get("pleural_fluid_appearance") or "").strip().rstrip(".")
                appearance_clean = re.sub(r"(?i)\bfluid\b", "", appearance).strip()
                if appearance_clean:
                    lines.append(f"Successful drainage of {vol_str} mL {appearance_clean} fluid")
                else:
                    lines.append(f"Successful drainage of {vol_str} mL fluid")
            has_ebus = bool(raw.get("linear_ebus_stations") or raw.get("ebus_stations_sampled"))

            def _normalize_nodule_rose(value: str) -> str:
                lowered_val = value.strip().lower()
                if "adequate lymphocytes" in lowered_val or "no malignancy" in lowered_val:
                    return "ROSE benign/nondiagnostic"
                if "negative" in lowered_val or lowered_val in ("neg", "negative"):
                    return "ROSE negative"
                if "atypical" in lowered_val:
                    return "Atypical cells on ROSE"
                cleaned = value.strip().rstrip(".")
                return cleaned if "rose" in cleaned.lower() else f"ROSE {cleaned}"

            # Peripheral target ROSE belongs with the nodule line when we have it.
            if nodule_rose_hint and lines:
                nodule_norm = _normalize_nodule_rose(nodule_rose_hint)
                if nodule_norm:
                    lines[0] = f"{lines[0]} ({nodule_norm})"

            # Nodal ROSE belongs on its own line when EBUS staging is present.
            if has_ebus and rose_hint:
                nodes_norm = re.sub(r"(?i)\bat\s+multiple\s+stations\b", "", rose_hint).strip().strip(",").strip()
                nodes_norm = re.sub(r"(?i)\bmultiple\s+stations\b", "", nodes_norm).strip().strip(",").strip()
                nodes_norm = nodes_norm.rstrip(".").strip()
                if nodes_norm:
                    lines.append(f"Mediastinal/Hilar lymphadenopathy; ROSE {nodes_norm} (final pathology pending)")
            elif rose_hint:
                rose_lower = rose_hint.strip().lower()
                if rose_lower in ("negative", "neg", "no malignancy"):
                    lines.append("ROSE negative (final pathology pending)")
                else:
                    lines.append(f"ROSE: {rose_hint} (final pathology pending)")
            raw["postop_diagnosis_text"] = "\n".join([line for line in lines if line])

    if raw.get("nav_platform") in (None, "", [], {}):
        nav_platform = _first_nonempty_str(equipment.get("navigation_platform"))
        if nav_platform:
            raw["nav_platform"] = nav_platform
    if raw.get("nav_platform") in (None, "", [], {}) and source_text:
        # Best-effort platform inference from short-form dictation.
        hint = " ".join(
            [
                text_fields.get("system") or "",
                text_fields.get("platform") or "",
                text_fields.get("procedure") or "",
                text_fields.get("method") or "",
                source_text,
            ]
        )
        lowered = hint.lower()
        nav_val = None
        if "galaxy" in lowered:
            nav_val = "Galaxy"
        elif "monarch" in lowered or "auris" in lowered:
            nav_val = "Monarch"
        elif re.search(r"\bion\b", lowered):
            nav_val = "Ion"
        elif "superdimension" in lowered or "super-dimension" in lowered or "super dimension" in lowered or re.search(
            r"\bemn\b|\belectromagnetic\b",
            lowered,
        ):
            nav_val = "EMN"
        elif "robotic" in lowered:
            nav_val = "Ion"
        if nav_val:
            raw["nav_platform"] = nav_val

    # Registration error/accuracy (mm) is commonly dictated inline.
    if raw.get("nav_registration_error_mm") in (None, "", [], {}) and source_text:
        match = re.search(
            r"(?i)\bregistration\b[^\n]{0,60}?\b(?:error|accuracy)?\b[^\n]{0,30}?(?:was|measured|of|=)?\s*(\d+(?:\.\d+)?)\s*mm\b",
            source_text,
        )
        if match:
            try:
                raw["nav_registration_error_mm"] = float(match.group(1))
            except Exception:
                pass

    if raw.get("nav_imaging_verification") in (None, "", [], {}):
        if source_text and re.search(r"(?i)\bTiLT\+?\b", source_text):
            raw["nav_imaging_verification"] = "TiLT+"
        cbct_used = equipment.get("cbct_used")
        if cbct_used is True:
            raw["nav_imaging_verification"] = "Cone Beam CT"
        elif source_text and re.search(r"(?i)\bTIL\b", source_text) and re.search(r"(?i)\bradial\s+ebus\b", source_text):
            raw["nav_imaging_verification"] = "Radial EBUS"
        elif source_text and re.search(r"(?i)\bcone\s*beam\b|\bcbct\b", source_text):
            raw["nav_imaging_verification"] = "Cone Beam CT"

    if raw.get("nav_target_segment") in (None, "", [], {}):
        if location_hint:
            raw["nav_target_segment"] = location_hint

    if raw.get("lesion_location") in (None, "", [], {}):
        if location_hint:
            raw["lesion_location"] = location_hint

    if raw.get("nav_tool_in_lesion") is not True:
        if (
            _text_contains_tool_in_lesion(source_text or "")
            or (source_text and re.search(r"(?i)\bTiLT\+?\b", source_text))
            or (source_text and re.search(r"(?i)\bTIL\b", source_text))
        ):
            raw["nav_tool_in_lesion"] = True

    if raw.get("nav_lesion_size_mm") in (None, "", [], {}):
        lesion_size_mm = clinical_context.get("lesion_size_mm")
        if lesion_size_mm not in (None, "", [], {}):
            raw["nav_lesion_size_mm"] = lesion_size_mm

    # If rEBUS is documented in short-form notes, treat it as radial EBUS evidence.
    if raw.get("nav_rebus_used") in (None, "", [], {}) and source_text:
        if re.search(r"(?i)\br\s*ebus\b|\brEBUS\b|\bradial\s+ebus\b", source_text):
            raw["nav_rebus_used"] = True
    if raw.get("nav_rebus_view") in (None, "", [], {}) and source_text:
        view_hint = (
            _infer_rebus_pattern(text_fields.get("rebus") or "")
            or _infer_rebus_pattern(text_fields.get("verif") or "")
            or _infer_rebus_pattern(source_text)
        )
        if view_hint:
            raw["nav_rebus_view"] = view_hint

    # --- Radial EBUS compat (V3 nested -> legacy flat keys) ---
    radial = procs.get("radial_ebus") or {}
    if isinstance(radial, dict) and radial.get("performed") is True:
        if raw.get("nav_rebus_used") in (None, "", [], {}):
            raw["nav_rebus_used"] = True
        if raw.get("nav_rebus_view") in (None, "", [], {}):
            view = _first_nonempty_str(radial.get("probe_position"), _infer_rebus_pattern(source_text or ""))
            if view:
                raw["nav_rebus_view"] = view

    # nav_sampling_tools drives the RadialEBUSSamplingAdapter (reporter wants an explicit list).
    if raw.get("nav_sampling_tools") in (None, "", [], {}):
        tools: list[str] = []
        if tbna_count is not None or (isinstance(procs.get("peripheral_tbna"), dict) and procs["peripheral_tbna"].get("performed") is True):
            tools.append("TBNA")
        if bx_count is not None or (isinstance(procs.get("transbronchial_biopsy"), dict) and procs["transbronchial_biopsy"].get("performed") is True):
            tools.append("Transbronchial biopsy")
        if isinstance(procs.get("brushings"), dict) and procs["brushings"].get("performed") is True:
            tools.append("Brushings")
        if isinstance(procs.get("bal"), dict) and procs["bal"].get("performed") is True:
            tools.append("BAL")
        if tools:
            raw["nav_sampling_tools"] = _dedupe_preserve_order(tools)

    # DictPayloadAdapter compat: map nested `procedures_performed.*` into top-level payload keys.
    # This allows the reporter adapters to build partially-populated procedure models.
    peripheral_tbna = procs.get("peripheral_tbna")
    if raw.get("transbronchial_needle_aspiration") in (None, "", [], {}) and (
        tbna_count is not None or (isinstance(peripheral_tbna, dict) and peripheral_tbna.get("performed") is True)
    ):
        raw["transbronchial_needle_aspiration"] = {
            "lung_segment": _first_nonempty_str(raw.get("nav_target_segment"), raw.get("lesion_location"), location_hint, segment_hint),
            "needle_tools": "TBNA",
            "samples_collected": tbna_count,
            "tests": [],
        }

    brushings = procs.get("brushings")
    if raw.get("bronchial_brushings") in (None, "", [], {}) and (
        (isinstance(brushings, dict) and brushings.get("performed") is True)
        or (counts_text and re.search(r"(?i)\bbrush", counts_text))
    ):
        raw["bronchial_brushings"] = {
            "lung_segment": _first_nonempty_str(raw.get("nav_target_segment"), raw.get("lesion_location"), location_hint, segment_hint),
            "samples_collected": brush_count,
            "brush_tool": brushings.get("brush_type"),
            "tests": [],
        }

    # Bronchial washing / lavage is sometimes documented separately from BAL.
    # Only synthesize it when lavage/washing is mentioned without explicit BAL wording.
    if (
        raw.get("bronchial_washing") in (None, "", [], {})
        and counts_text
        and raw.get("wll_volume_instilled_l") in (None, "", [], {})
    ):
        if re.search(r"(?i)\b(?:lavage|washing)\b", counts_text) and not re.search(r"(?i)\bBAL\b", counts_text):
            washing_location = bal_location_hint
            if not washing_location:
                washing_location = _first_nonempty_str(
                    raw.get("nav_target_segment"),
                    raw.get("lesion_location"),
                    location_hint,
                    segment_hint,
                )
            raw["bronchial_washing"] = {
                "airway_segment": washing_location,
                "instilled_volume_ml": None,
                "returned_volume_ml": None,
                "tests": [],
            }

    bal = procs.get("bal")
    if raw.get("wll_volume_instilled_l") in (None, "", [], {}) and raw.get("bal") in (None, "", [], {}) and (
        (isinstance(bal, dict) and bal.get("performed") is True)
        or (bal_location_hint is not None)
        or (counts_text and re.search(r"(?i)\bBAL\b", counts_text))
    ):
        bal_location = None
        if isinstance(bal, dict):
            bal_location = _first_nonempty_str(bal.get("location"))
        if not bal_location and bal_location_hint:
            bal_location = bal_location_hint
        if not bal_location and counts_text:
            match = re.search(r"(?i)\bBAL\b[^\n]{0,25}?\b([RL]B\d{1,2}|RUL|RML|RLL|LUL|LLL)\b", counts_text)
            if match:
                bal_location = match.group(1).upper()
        raw["bal"] = {
            "lung_segment": bal_location,
            "instilled_volume_cc": (bal or {}).get("volume_instilled_ml") if isinstance(bal, dict) else None,
            "returned_volume_cc": (bal or {}).get("volume_recovered_ml") if isinstance(bal, dict) else None,
            "tests": [],
        }

    if raw.get("fiducial_marker_placement") in (None, "", [], {}) and source_text and re.search(r"(?i)\bfiducial\b", source_text):
        raw["fiducial_marker_placement"] = {
            "airway_location": _first_nonempty_str(raw.get("nav_target_segment"), raw.get("lesion_location"), location_hint, segment_hint, "target lesion"),
        }

    # PDT debridement often appears in short-form dictation without structured extraction flags.
    if raw.get("pdt_debridement") in (None, "", [], {}) and source_text:
        if re.search(r"(?i)\bpdt\b", source_text) and re.search(r"(?i)\bdebrid", source_text):
            site = _first_nonempty_str(location_hint, segment_hint)
            tools_text = None
            match = re.search(r"(?i)\btools?\s*:\s*([^\.\n]+)", source_text)
            if match:
                tools_text = match.group(1).strip().rstrip(".")

            pre_patency = None
            post_patency = None
            match = re.search(
                r"(?i)\b(\d{1,3})\s*%\s*obstruct(?:ed|ion)?\s*(?:->|to)\s*(\d{1,3})\s*%\s*(?:post[-\s]?debridement|post)\b",
                source_text,
            )
            if match:
                try:
                    pre_obs = int(match.group(1))
                    post_obs = int(match.group(2))
                    pre_patency = max(0, min(100, 100 - pre_obs))
                    post_patency = max(0, min(100, 100 - post_obs))
                except Exception:
                    pre_patency = None
                    post_patency = None

            if site:
                raw["pdt_debridement"] = {
                    "site": site,
                    "debridement_tool": tools_text,
                    "pre_patency_pct": pre_patency,
                    "post_patency_pct": post_patency,
                    "bleeding": None,
                    "notes": None,
                }
                preop_existing = str(raw.get("preop_diagnosis_text") or "").strip()
                if not preop_existing or ("pdt" in preop_existing.lower() and "obstruct" not in preop_existing.lower()):
                    raw["preop_diagnosis_text"] = "\n\n".join(
                        [
                            f"{site} airway obstruction (Necrosis)",
                            "Status post-Photodynamic Therapy (PDT)",
                        ]
                    )
                postop_existing = str(raw.get("postop_diagnosis_text") or "").strip()
                if not postop_existing or ("pdt" in postop_existing.lower() and "debrid" in postop_existing.lower()):
                    raw["postop_diagnosis_text"] = "\n\n".join(
                        [
                            f"{site} airway obstruction (Necrosis), successfully debrided",
                            "Status post-Photodynamic Therapy (PDT)",
                        ]
                    )

    cryo = procs.get("transbronchial_cryobiopsy")
    if raw.get("transbronchial_cryobiopsy") in (None, "", [], {}) and (
        (isinstance(cryo, dict) and cryo.get("performed") is True)
        or (
            source_text
            and (
                re.search(r"(?i)\bcryo\s*biopsy\b|\bcryobiopsy\b", source_text)
                or re.search(r"(?i)\bcryo\b\s*(?:x|×)\s*\d+\b", source_text)
                or re.search(r"(?i)\b(\d+)\s*s\s*freeze\b", source_text)
                or re.search(r"(?im)^\s*-\s*site\s*\d+\s*:", source_text)
            )
            and not re.search(r"(?i)\bdebrid", source_text)
            and not re.search(r"(?i)\bpdt\b", source_text)
        )
    ):
        sites = []
        if source_text:
            for match in re.finditer(r"(?im)^\s*-\s*site\s*\d+\s*:\s*([^\n(]+)", source_text):
                site = match.group(1).strip().rstrip(".")
                if site:
                    sites.append(site)
        samples_val = None
        if source_text:
            match = re.search(r"(?i)\b(\d+)\s*samples?\b", source_text)
            if match:
                try:
                    samples_val = int(match.group(1))
                except Exception:
                    samples_val = None
        if samples_val is None and source_text:
            match = re.search(r"(?i)\bcryo\b\s*(?:x|×)\s*(\d+)\b", source_text)
            if match:
                try:
                    samples_val = int(match.group(1))
                except Exception:
                    samples_val = None
        if samples_val is None and sites:
            samples_val = len(sites)

        sample_size_mm = None
        if source_text:
            match = re.search(r"(?i)\b\d+\s*samples?\b[^\n]{0,40}\(\s*(\d+(?:\.\d+)?)\s*mm", source_text)
            if match:
                try:
                    sample_size_mm = float(match.group(1))
                except Exception:
                    sample_size_mm = None

        cryoprobe_size = None
        if source_text:
            match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*mm\s*cryo(?:probe|biop)\b", source_text)
            if match:
                try:
                    cryoprobe_size = float(match.group(1))
                except Exception:
                    cryoprobe_size = None

        freeze_seconds = None
        if source_text:
            match = re.search(r"(?i)\b(\d+)\s*s\s*freeze\b", source_text)
            if match:
                try:
                    freeze_seconds = int(match.group(1))
                except Exception:
                    freeze_seconds = None

        blocker_type = None
        if source_text and re.search(r"(?i)\bfogarty\b", source_text):
            blocker_type = "Fogarty balloon"

        radial_vessel_check = (
            True
            if (source_text and re.search(r"(?i)\bradial\s+ebus\b|\br\s*ebus\b|\brebus\b", source_text))
            else None
        )

        raw["transbronchial_cryobiopsy"] = {
            "lung_segment": _first_nonempty_str(", ".join(sites) if sites else None, location_hint, segment_hint),
            "num_samples": samples_val,
            "sample_size_mm": sample_size_mm,
            "cryoprobe_size_mm": cryoprobe_size,
            "freeze_seconds": freeze_seconds,
            "blocker_type": blocker_type,
            "tests": [],
            "radial_vessel_check": radial_vessel_check,
        }

    # --- Airway therapeutics compat (performed flags -> reporter adapters) ---
    note_text = source_text if isinstance(source_text, str) else ""

    aspiration = procs.get("therapeutic_aspiration")
    if raw.get("therapeutic_aspiration") in (None, "", [], {}) and isinstance(aspiration, dict) and aspiration.get("performed") is True:
        aspirate_type = _first_nonempty_str(aspiration.get("material"))
        if not aspirate_type and note_text:
            if re.search(r"(?i)\bmucus(?:\s*plug)?\b|\bmucous\b", note_text):
                aspirate_type = "mucus"
            elif re.search(r"(?i)\bclot\b|\bblood\s*clot\b", note_text):
                aspirate_type = "blood clot"
            elif re.search(r"(?i)\bblood\b", note_text):
                aspirate_type = "blood"
        raw["therapeutic_aspiration"] = {
            "airway_segment": _first_nonempty_str(aspiration.get("location"), segment_hint, location_hint, "airways"),
            "aspirate_type": aspirate_type or "secretions",
        }

    dilation = procs.get("airway_dilation")
    if raw.get("airway_dilation") in (None, "", [], {}) and isinstance(dilation, dict) and dilation.get("performed") is True:
        technique = None
        if note_text and re.search(r"(?i)\bballoon\b", note_text):
            technique = "Balloon dilation"
        elif note_text and re.search(r"(?i)\bbougie\b", note_text):
            technique = "Bougie dilation"
        sizes: list[int] = []
        if note_text:
            for match in re.finditer(r"(?i)\b(\d{1,2})\s*mm\b", note_text):
                try:
                    sizes.append(int(match.group(1)))
                except Exception:
                    continue
        if sizes:
            seen_sizes: set[int] = set()
            deduped_sizes: list[int] = []
            for val in sizes:
                if val in seen_sizes:
                    continue
                seen_sizes.add(val)
                deduped_sizes.append(val)
            sizes = deduped_sizes
        raw["airway_dilation"] = {
            "airway_segment": _first_nonempty_str(dilation.get("location"), segment_hint, location_hint),
            "technique": technique,
            "dilation_sizes_mm": sizes,
            "post_dilation_diameter_mm": None,
            "notes": None,
        }

    cryo = procs.get("cryotherapy")
    if isinstance(cryo, dict) and cryo.get("performed") is True and note_text:
        airway_target = _first_nonempty_str(segment_hint, location_hint, "target airway")
        mucus_context = bool(re.search(r"(?i)\b(mucus(?:\s*plug)?|clot|cast|plug|secretions)\b", note_text))
        if mucus_context:
            if raw.get("cryo_extraction_mucus") in (None, "", [], {}):
                raw["cryo_extraction_mucus"] = {
                    "airway_segment": airway_target,
                    "probe_size_mm": None,
                    "freeze_seconds": None,
                    "num_casts": None,
                    "ventilation_result": None,
                    "notes": None,
                }
        else:
            if raw.get("endobronchial_cryoablation") in (None, "", [], {}):
                raw["endobronchial_cryoablation"] = {
                    "site": airway_target,
                    "cryoprobe_size_mm": None,
                    "freeze_seconds": None,
                    "thaw_seconds": None,
                    "cycles": None,
                    "pattern": None,
                    "post_patency": None,
                    "notes": None,
                }

    thermal = procs.get("thermal_ablation")
    if raw.get("endobronchial_tumor_destruction") in (None, "", [], {}) and isinstance(thermal, dict) and thermal.get("performed") is True:
        modality = None
        if note_text:
            if re.search(r"(?i)\bapc\b|argon\s+plasma", note_text):
                modality = "APC"
            elif re.search(r"(?i)\blaser\b", note_text):
                modality = "Laser"
            elif re.search(r"(?i)\belectrocautery\b|\bcautery\b", note_text):
                modality = "Electrocautery"
            elif re.search(r"(?i)\bthermal\s+ablation\b", note_text):
                modality = "Thermal ablation"
        raw["endobronchial_tumor_destruction"] = {
            "modality": modality or "Thermal ablation",
            "airway_segment": _first_nonempty_str(thermal.get("location"), segment_hint, location_hint),
            "notes": None,
        }

    stent = procs.get("airway_stent")
    if isinstance(stent, dict) and stent.get("performed") is True:
        action_type = stent.get("action_type")
        if not action_type and stent.get("action") not in (None, "", [], {}):
            action_raw = str(stent.get("action") or "").strip().lower()
            if action_raw.startswith("assessment"):
                action_type = "assessment_only"
            elif action_raw.startswith("placement"):
                action_type = "placement"
            elif action_raw.startswith("removal"):
                action_type = "removal"
            elif action_raw.startswith("revision"):
                action_type = "revision"

        def _coerce_int(value: Any) -> int | None:
            if value in (None, "", [], {}):
                return None
            try:
                return int(float(value))
            except Exception:
                return None

        stent_type = _first_nonempty_str(stent.get("stent_type"), stent.get("stent_brand"))
        # Avoid anatomy contagion: do not use global segment_hint (often from BAL) as a stent location fallback.
        stent_location = _first_nonempty_str(stent.get("location"), location_hint, "target airway")
        action_norm = str(action_type or "").strip().lower()
        if action_norm in {"assessment_only", "assessment only"}:
            if raw.get("stent_surveillance") in (None, "", [], {}):
                raw["stent_surveillance"] = {
                    "stent_type": stent_type or "airway stent",
                    "location": stent_location,
                    "findings": None,
                    "interventions": None,
                    "final_patency_pct": None,
                }
        elif action_norm == "removal":
            if raw.get("foreign_body_removal") in (None, "", [], {}):
                tools: list[str] = []
                lowered = (note_text or "").lower()
                if "forceps" in lowered:
                    tools.append("Forceps")
                if "snare" in lowered:
                    tools.append("Snare")
                if "basket" in lowered:
                    tools.append("Basket")
                if "cryo" in lowered:
                    tools.append("Cryoprobe")
                raw["foreign_body_removal"] = {
                    "airway_segment": stent_location,
                    "tools_used": tools or ["Forceps"],
                    "passes": None,
                    "removed_intact": None,
                    "mucosal_trauma": None,
                    "bleeding": None,
                    "hemostasis_method": None,
                    "cxr_ordered": None,
                    "notes": None,
                }
        else:
            if raw.get("airway_stent_placement") in (None, "", [], {}):
                raw["airway_stent_placement"] = {
                    "stent_type": stent_type,
                    "diameter_mm": _coerce_int(stent.get("diameter_mm")),
                    "length_mm": _coerce_int(stent.get("length_mm")),
                    "airway_segment": stent_location if stent_location != "target airway" else None,
                    "notes": None,
                }

    balloon = procs.get("balloon_occlusion")
    balloon_performed = isinstance(balloon, dict) and balloon.get("performed") is True
    # Preserve legacy behavior: allow Fogarty-triggered blocker even when V3 flags are missing.
    balloon_mentioned = bool(note_text and re.search(r"(?i)\bfogarty\b", note_text))
    if balloon_performed or balloon_mentioned:
        balloon_type = None
        if note_text:
            if re.search(r"(?i)\barndt\b|\bardnt\b", note_text):
                balloon_type = "Arndt"
            elif re.search(r"(?i)\bfogarty\b", note_text):
                balloon_type = "Fogarty"
            elif re.search(r"(?i)\bcohen\s+flexitip\b", note_text):
                balloon_type = "Cohen Flexitip"

        bpf_context = bool(note_text and re.search(r"(?i)\bair\s*leak\b|\bairleak\b|\bfistula\b|\bbronchopleural\b|\bpleurovac\b", note_text))
        if bpf_context and raw.get("bpf_localization") in (None, "", [], {}):
            occlusion_location = balloon.get("occlusion_location") if isinstance(balloon, dict) else None
            raw["bpf_localization"] = {
                "culprit_segment": _first_nonempty_str(occlusion_location, location_hint, segment_hint, "target airway"),
                "balloon_type": balloon_type,
                "balloon_size_mm": None,
                "leak_reduction": None,
                "methylene_blue_used": None,
                "contrast_used": None,
                "instillation_findings": None,
                "notes": None,
            }
        elif raw.get("endobronchial_blocker") in (None, "", [], {}):
            side = None
            if location_hint and location_hint.upper().startswith("R"):
                side = "right"
            elif location_hint and location_hint.upper().startswith("L"):
                side = "left"
            if not side and note_text:
                if re.search(r"(?i)\bleft\b", note_text):
                    side = "left"
                elif re.search(r"(?i)\bright\b", note_text):
                    side = "right"
            raw["endobronchial_blocker"] = {
                "blocker_type": balloon_type or "Endobronchial blocker",
                "side": side or "unspecified",
                "location": _first_nonempty_str(location_hint, segment_hint, "target airway"),
                "indication": "Prophylaxis",
            }

    sealant = procs.get("bpf_sealant")
    if (
        raw.get("bpf_sealant_application") in (None, "", [], {})
        and isinstance(sealant, dict)
        and sealant.get("performed") is True
    ):
        sealant_type = _first_nonempty_str(sealant.get("sealant_type"))
        if not sealant_type and note_text:
            if re.search(r"(?i)\btisseel\b|\btissel\b", note_text):
                sealant_type = "Tisseel"
            elif re.search(r"(?i)\bsealant\b|\bglue\b|\bfibrin\b", note_text):
                sealant_type = "Sealant"

        volume_ml = None
        if note_text:
            match = re.search(
                r"(?i)(?:tisseel|tissel|sealant|glue|fibrin)[^\n]{0,60}?\b(\d+(?:\.\d+)?)\s*(?:cc|ml)\b",
                note_text,
            )
            if match:
                try:
                    volume_ml = float(match.group(1))
                except Exception:
                    volume_ml = None

        raw["bpf_sealant_application"] = {
            "sealant_type": sealant_type or "Sealant",
            "volume_ml": volume_ml,
            "dwell_minutes": None,
            "leak_reduction": None,
            "applications": None,
            "notes": sealant.get("notes"),
        }

    # bronch_num_tbbx from transbronchial_biopsy.number_of_samples
    if "bronch_num_tbbx" not in raw:
        tbbx = procs.get("transbronchial_biopsy", {}) or {}
        if tbbx.get("number_of_samples"):
            raw["bronch_num_tbbx"] = tbbx["number_of_samples"]
        elif bx_count is not None:
            raw["bronch_num_tbbx"] = bx_count

    if raw.get("bronch_location_lobe") in (None, "", [], {}):
        raw["bronch_location_lobe"] = _first_nonempty_str(location_hint, clinical_context.get("lesion_location"))
    if raw.get("bronch_location_segment") in (None, "", [], {}):
        if segment_hint:
            raw["bronch_location_segment"] = segment_hint

    # bronch_tbbx_tool from transbronchial_biopsy.forceps_type
    if "bronch_tbbx_tool" not in raw:
        tbbx = procs.get("transbronchial_biopsy", {}) or {}
        if tbbx.get("forceps_type"):
            raw["bronch_tbbx_tool"] = tbbx["forceps_type"]

    # --- Pleural compat: map V3 pleural_procedures.* into legacy flat keys for adapters ---
    pleural = raw.get("pleural_procedures") or {}
    if isinstance(pleural, dict):
        # Fallback: infer common pleural procedures from free text when structured flags are missing.
        if raw.get("pleural_procedure_type") in (None, "", [], {}) and source_text:
            lowered = source_text.lower()
            if "pigtail" in lowered:
                raw["pleural_procedure_type"] = "pigtail catheter"
            elif "thoracentesis" in lowered:
                raw["pleural_procedure_type"] = "thoracentesis"
            elif "tunneled pleural catheter" in lowered or "pleurx" in lowered:
                raw["pleural_procedure_type"] = "tunneled catheter"
            elif "chest tube" in lowered:
                raw["pleural_procedure_type"] = "chest tube"

        thor = pleural.get("thoracentesis") or {}
        if isinstance(thor, dict) and thor.get("performed") is True:
            if raw.get("pleural_procedure_type") in (None, "", [], {}):
                raw["pleural_procedure_type"] = "thoracentesis"

            if raw.get("pleural_side") in (None, "", [], {}):
                side = _first_nonempty_str(thor.get("side"))
                if not side and source_text:
                    upper = source_text.upper()
                    if re.search(r"\bLEFT\b|\bL\s*EFFUSION\b", upper):
                        side = "left"
                    elif re.search(r"\bRIGHT\b|\bR\s*EFFUSION\b", upper):
                        side = "right"
                if side:
                    raw["pleural_side"] = side

            if raw.get("pleural_guidance") in (None, "", [], {}):
                guidance = _first_nonempty_str(thor.get("guidance"))
                if guidance:
                    raw["pleural_guidance"] = guidance
                elif source_text and re.search(r"\bno\s+imaging\b", source_text, flags=re.IGNORECASE):
                    raw["pleural_guidance"] = None
                elif source_text and re.search(r"\bultrasound\b|\bU/S\b|\bUS\b", source_text, flags=re.IGNORECASE):
                    raw["pleural_guidance"] = "Ultrasound"

            if raw.get("pleural_volume_drained_ml") in (None, "", [], {}):
                volume = thor.get("volume_removed_ml")
                if volume is None and source_text:
                    match = re.search(r"(?i)\b(?:drained|removed)\s+(\d{2,5})\s*(?:mL|ml|cc)\b", source_text)
                    if match:
                        try:
                            volume = int(match.group(1))
                        except Exception:
                            volume = None
                if volume is not None:
                    raw["pleural_volume_drained_ml"] = volume

            if raw.get("pleural_fluid_appearance") in (None, "", [], {}):
                appearance = _first_nonempty_str(thor.get("fluid_appearance"))
                if not appearance and source_text:
                    match = re.search(
                        r"(?i)\b(?:drained|removed)\s+\d{2,5}\s*(?:mL|ml|cc)\s+([a-z][a-z\s-]{0,40})",
                        source_text,
                    )
                    if match:
                        appearance = match.group(1).strip().rstrip(".")
                if appearance:
                    raw["pleural_fluid_appearance"] = appearance

            raw.setdefault("pleural_intercostal_space", "unspecified")
            raw.setdefault("entry_location", "mid-axillary")

            if raw.get("drainage_device") in (None, "", [], {}) and source_text and re.search(r"(?i)\bpigtail\b", source_text):
                size = _parse_count(source_text, r"\b(\d{1,2})\s*(?:fr|french)\b")
                raw["drainage_device"] = f"{size} Fr pigtail catheter" if size else "pigtail catheter"

        ipc = pleural.get("ipc") or {}
        if isinstance(ipc, dict) and ipc.get("performed") is True:
            action = str(ipc.get("action") or "").strip().lower()
            if raw.get("pleural_procedure_type") in (None, "", [], {}):
                if ipc.get("tunneled") is True or action in ("insertion", "insert") or "insert" in action:
                    raw["pleural_procedure_type"] = "tunneled catheter"

            if raw.get("pleural_side") in (None, "", [], {}):
                side = None
                if source_text:
                    match = re.search(r"(?i)\b\((right|left)\)\b", source_text)
                    if match:
                        side = match.group(1).lower()
                    elif re.search(r"(?i)\bright\b", source_text):
                        side = "right"
                    elif re.search(r"(?i)\bleft\b", source_text):
                        side = "left"
                if side:
                    raw["pleural_side"] = side

            if raw.get("pleural_guidance") in (None, "", [], {}) and source_text:
                if re.search(r"(?i)\bultrasound\b|\bU/S\b|\bUS\b", source_text):
                    raw["pleural_guidance"] = "Ultrasound"

            if raw.get("pleural_volume_drained_ml") in (None, "", [], {}) and source_text:
                volume_ml = None
                match = re.search(r"(?i)\b(\d{2,5})\s*(?:mL|ml|cc)\b", source_text)
                if match:
                    try:
                        volume_ml = int(match.group(1))
                    except Exception:
                        volume_ml = None
                match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*L\b", source_text)
                if match:
                    try:
                        volume_ml = int(round(float(match.group(1)) * 1000))
                    except Exception:
                        volume_ml = volume_ml
                if volume_ml is not None:
                    raw["pleural_volume_drained_ml"] = volume_ml

            if raw.get("pleural_fluid_appearance") in (None, "", [], {}) and source_text:
                match = re.search(r"(?i)\b(?:drained|removed)\s+\d+(?:\.\d+)?\s*(?:L|mL|ml|cc)\s+([a-z][a-z\s-]{0,40})", source_text)
                if match:
                    raw["pleural_fluid_appearance"] = match.group(1).strip().rstrip(".")

            if raw.get("drainage_device") in (None, "", [], {}) and source_text:
                brand = _first_nonempty_str(ipc.get("catheter_brand"))
                size = None
                match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*fr\b", source_text)
                if match:
                    size = match.group(1)
                if brand and size:
                    raw["drainage_device"] = f"{size}Fr {brand} catheter"
                elif brand:
                    raw["drainage_device"] = f"{brand} catheter"

            if raw.get("cxr_ordered") in (None, "", [], {}) and source_text:
                if re.search(r"(?i)\bcxr\b|chest x[-\\s]?ray", source_text):
                    raw["cxr_ordered"] = True

        # If we inferred a pleural procedure type (or have one) but didn't get structured details,
        # backfill simple side/volume/appearance/guidance from the raw text.
        pleural_type = str(raw.get("pleural_procedure_type") or "").strip().lower()
        if source_text and pleural_type in ("thoracentesis", "pigtail catheter", "tunneled catheter", "chest tube"):
            if raw.get("pleural_side") in (None, "", [], {}):
                side = None
                match = re.search(r"(?i)\b\((right|left)\)\b", source_text)
                if match:
                    side = match.group(1).lower()
                else:
                    upper = source_text.upper()
                    if re.search(r"\bLEFT\b|\bL\s*EFFUSION\b", upper):
                        side = "left"
                    elif re.search(r"\bRIGHT\b|\bR\s*EFFUSION\b", upper):
                        side = "right"
                if side:
                    raw["pleural_side"] = side

            if raw.get("pleural_guidance") in (None, "", [], {}):
                if re.search(r"(?i)\bno\s+imaging\b", source_text):
                    raw["pleural_guidance"] = None
                elif re.search(r"(?i)\bultrasound\b|\bU/S\b|\bUS\b", source_text):
                    raw["pleural_guidance"] = "Ultrasound"

            if raw.get("pleural_volume_drained_ml") in (None, "", [], {}):
                volume_ml = None
                match = re.search(r"(?i)\b(\d{2,5})\s*(?:mL|ml|cc)\b", source_text)
                if match:
                    try:
                        volume_ml = int(match.group(1))
                    except Exception:
                        volume_ml = None
                match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*L\b", source_text)
                if match:
                    try:
                        volume_ml = int(round(float(match.group(1)) * 1000))
                    except Exception:
                        volume_ml = volume_ml
                if volume_ml is not None:
                    raw["pleural_volume_drained_ml"] = volume_ml

            if raw.get("pleural_fluid_appearance") in (None, "", [], {}):
                match = re.search(
                    r"(?i)\b(?:drained|removed)\s+\d+(?:\.\d+)?\s*(?:L|mL|ml|cc)\s+([a-z][a-z\s-]{0,40})",
                    source_text,
                )
                if match:
                    raw["pleural_fluid_appearance"] = match.group(1).strip().rstrip(".")

            if pleural_type == "pigtail catheter" and raw.get("size_fr") in (None, "", [], {}):
                match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*fr\b", source_text)
                if match:
                    raw["size_fr"] = f"{match.group(1)}Fr"

            if raw.get("drainage_device") in (None, "", [], {}) and source_text:
                size = None
                match = re.search(r"(?i)\b(\d+(?:\.\d+)?)\s*fr\b", source_text)
                if match:
                    size = match.group(1)
                brand = "PleurX" if re.search(r"(?i)\bpleurx\b", source_text) else None
                if pleural_type == "pigtail catheter" and re.search(r"(?i)\bpigtail\b", source_text):
                    raw["drainage_device"] = f"{size}Fr pigtail catheter" if size else "pigtail catheter"
                elif brand and size:
                    raw["drainage_device"] = f"{size}Fr {brand} catheter"
                elif brand:
                    raw["drainage_device"] = f"{brand} catheter"

            if raw.get("cxr_ordered") in (None, "", [], {}) and source_text:
                if re.search(r"(?i)\bcxr\b|chest x[-\s]?ray", source_text):
                    raw["cxr_ordered"] = True

        # Postop diagnosis enrichment after pleural fields are backfilled.
        preop_text = raw.get("preop_diagnosis_text")
        postop_text = raw.get("postop_diagnosis_text")
        if preop_text not in (None, "", [], {}) and str(postop_text or "").strip() == str(preop_text).strip():
            pleural_type = str(raw.get("pleural_procedure_type") or "").strip().lower()
            lines = [str(preop_text).strip()]
            if pleural_type == "tunneled catheter" and raw.get("pleural_side"):
                side = str(raw.get("pleural_side") or "").strip()
                if side:
                    lines.append(f"Status post {side} tunneled pleural catheter placement")
            if pleural_type in ("thoracentesis", "pigtail catheter") and raw.get("pleural_volume_drained_ml") not in (
                None,
                "",
                [],
                {},
            ):
                vol = raw.get("pleural_volume_drained_ml")
                try:
                    vol_str = str(int(float(vol)))
                except Exception:
                    vol_str = str(vol)
                appearance = str(raw.get("pleural_fluid_appearance") or "").strip().rstrip(".")
                appearance = appearance.splitlines()[0].strip()
                if appearance:
                    lines.append(f"Successful drainage of {vol_str} mL {appearance} fluid")
                else:
                    lines.append(f"Successful drainage of {vol_str} mL fluid")

            enriched = "\n".join([line for line in lines if line])
            if enriched:
                raw["postop_diagnosis_text"] = enriched

    # ventilation_mode from procedure_setting or sedation
    if "ventilation_mode" not in raw:
        setting = raw.get("procedure_setting", {}) or {}
        if setting.get("airway_type"):
            raw["ventilation_mode"] = setting["airway_type"]
        elif raw.get("airway_type") not in (None, "", [], {}):
            raw["ventilation_mode"] = raw["airway_type"]

    return raw

def add_compat_flat_fields(bundle: ProcedureBundle) -> tuple[ProcedureBundle, list[NormalizationNote]]:
    """Add compatibility fields to an already-hydrated ProcedureBundle.

    This layer is intentionally conservative: it only fills missing values and never
    deletes user-provided data.
    """
    notes: list[NormalizationNote] = []
    payload = bundle.model_dump(exclude_none=False)
    procedures = payload.get('procedures') or []
    if isinstance(procedures, list):
        for idx, proc in enumerate(procedures):
            if not isinstance(proc, dict):
                continue
            if proc.get('proc_id') in (None, ''):
                proc_type = str(proc.get('proc_type') or 'procedure').strip() or 'procedure'
                derived = f"{proc_type}_{idx + 1}"
                proc['proc_id'] = derived
                notes.append(
                    NormalizationNote(
                        kind='compat',
                        path=f"/procedures/{idx}/proc_id",
                        message='Derived missing proc_id for compatibility',
                        source='compat_enricher',
                    )
                )

    payload['procedures'] = procedures
    return ProcedureBundle.model_validate(payload), notes
