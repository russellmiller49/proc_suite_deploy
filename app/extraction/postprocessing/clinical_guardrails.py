from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.registry.schema import RegistryRecord


_NAV_FAILURE_PATTERN = re.compile(
    r"(mis-?regist|unable|aborted|fail|suboptimal).{0,50}(navigat|registration)",
    re.IGNORECASE | re.DOTALL,
)
_RADIAL_MARKER_PATTERN = re.compile(
    r"\b(?:"
    r"(?:concentric|eccentric)\b[^.\n]{0,15}\bview\b"
    r"|"
    r"\bview\b[^.\n]{0,15}\b(?:concentric|eccentric)\b"
    r"|"
    r"\b(?:radial|r-?ebus|rp-?ebus|miniprobe)\b[^.\n]{0,50}\b(?:concentric|eccentric)\b"
    r"|"
    r"\b(?:concentric|eccentric)\b[^.\n]{0,50}\b(?:radial|r-?ebus|rp-?ebus|miniprobe)\b"
    r")\b",
    re.IGNORECASE,
)
_LINEAR_MARKER_PATTERN = re.compile(
    r"\b(convex|station\s*\d{1,2}[A-Za-z]?)\b",
    re.IGNORECASE,
)
_DILATION_CONTEXT_PATTERN = re.compile(
    r"(skin|subcutaneous|chest wall|tract)(?:\W+\w+){0,10}\W+dilat"
    r"|dilat(?:\W+\w+){0,10}\W+(skin|subcutaneous|chest wall|tract)",
    re.IGNORECASE,
)

_STENT_NEGATION_PATTERNS = [
    re.compile(
        r"\bdecision\b[^.\n]{0,80}\bnot\b[^.\n]{0,40}\b(?:place|insert|perform|deploy)\w*\b"
        r"[^.\n]{0,80}\bstent\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bno\s+additional\s+stents?\b[^.\n]{0,40}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,40}\bstent(?:s)?\b"
        r"[^.\n]{0,40}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:no|not|without|declined|deferred)\b[^.\n]{0,80}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\w*\b"
        r"[^.\n]{0,80}\bstent(?:s)?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:refus(?:ed|al)|reluctan(?:t|ce)|hesitan(?:t|cy)|did\s+not\s+want)\b"
        r"[^.\n]{0,80}\bstent(?:s)?\b[^.\n]{0,80}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\b",
        re.IGNORECASE,
    ),
]

_STENT_PLACEMENT_CONTEXT_RE = re.compile(
    r"\b(?:(?:stent|bonostent)\b[^.\n]{0,30}\b(place|placed|deploy|deployed|insert|inserted|advance|advanced|seat|seated|expand|expanded|expanding)\b"
    r"|(place|placed|deploy|deployed|insert|inserted|advance|advanced|seat|seated|expand|expanded|expanding)\b[^.\n]{0,30}\b(?:stent|bonostent)\b)\b",
    re.IGNORECASE,
)
_STENT_PLACEMENT_ACTION_CONTEXT_RE = re.compile(
    r"\b(?:(?:stent|bonostent)\b[^.\n]{0,30}\b(place|placed|deploy|deployed|insert|inserted|advance|advanced|expand|expanded|expanding)\b"
    r"|(place|placed|deploy|deployed|insert|inserted|advance|advanced|expand|expanded|expanding)\b[^.\n]{0,30}\b(?:stent|bonostent)\b)\b",
    re.IGNORECASE,
)
_STENT_STRONG_PLACEMENT_RE = re.compile(
    r"\b(?:(?:stent|bonostent)\b[^.\n]{0,30}\b(deploy|deployed|insert|inserted|advance|advanced|seat|seated|expand|expanded|expanding)\b"
    r"|(deploy|deployed|insert|inserted|advance|advanced|seat|seated|expand|expanded|expanding)\b[^.\n]{0,30}\b(?:stent|bonostent)\b)\b",
    re.IGNORECASE,
)
_STENT_REMOVAL_CONTEXT_RE = re.compile(
    r"\b(?:"
    r"stent\b[^.\n]{0,60}\b(remov|retriev|extract|explant|grasp|pull|peel|exchang|replac)\w*\b"
    r"|"
    r"(remov|retriev|extract|explant|grasp|pull|peel|exchang|replac)\w*\b[^.\n]{0,60}\bstent\b"
    r")\b",
    re.IGNORECASE,
)
_STENT_INSPECTION_RE = re.compile(
    r"\b(?:"
    r"(?:stent|bms)\b[^.\n]{0,80}\b(evaluat|inspect|inspection|patent|intact|visible|stent\s+check|in\s+(?:good|adequate)\s+position|adequately\s+positioned|in\s+place|well[- ]seated|well\s+positioned|reassess(?:ment)?|position\s+(?:confirmed|stable)|defect\s+(?:seen|noted))\b"
    r"|"
    r"(evaluat|inspect|inspection|patent|intact|visible|stent\s+check|in\s+(?:good|adequate)\s+position|adequately\s+positioned|in\s+place|well[- ]seated|well\s+positioned|reassess(?:ment)?|position\s+(?:confirmed|stable)|defect\s+(?:seen|noted))\b[^.\n]{0,80}\b(?:stent|bms)\b"
    r")\b",
    re.IGNORECASE,
)
_STENT_OBSTRUCTION_RE = re.compile(
    r"\b(?:stent|bms)\b[^.\n]{0,120}\b(?:obstruct|occlud|impacted|plugged|mucous|mucus)\b",
    re.IGNORECASE,
)
_STENT_CLEANING_RE = re.compile(
    r"\b(?:"
    r"(?:stent|bms)\b[^.\n]{0,120}\b(?:clean(?:ed|ing)?|debrid(?:ed|ement)?|suction(?:ed|ing)?|clear(?:ed|ing)?)\b"
    r"|"
    r"(?:clean(?:ed|ing)?|debrid(?:ed|ement)?|suction(?:ed|ing)?|clear(?:ed|ing)?)\b[^.\n]{0,120}\b(?:stent|bms)\b"
    r")\b",
    re.IGNORECASE,
)

_IPC_TERMS = (
    "pleurx",
    "aspira",
    "tunneled",
    "tunnelled",
    "tunnel pleural catheter",
    "tunneling device",
    "indwelling pleural catheter",
    "ipc",
    "tpc",
)
_CHEST_TUBE_TERMS = ("pigtail", "wayne", "pleur-evac", "pleur evac", "tube thoracostomy", "chest tube")
_INSERT_TERMS = ("insert", "inserted", "placed", "place", "deploy", "deployed", "introduced", "positioned")
_REMOVE_TERMS = (
    "remove",
    "removed",
    "removal",
    "exchanged",
    "exchange",
    "explant",
    "withdrawn",
    "discontinue",
    "discontinued",
    "discontinuation",
    "d/c",
    "dc",
)

_CHEST_TUBE_DATE_OF_INSERTION_RE = re.compile(
    r"\bdate\s+of\s+(?:the\s+)?(?:chest\s+tube(?:\s*/\s*(?:tpc|ipc))?|tpc|ipc)\s+insertion\b",
    re.IGNORECASE,
)

_CHECKBOX_TOKEN_RE = re.compile(
    r"(?im)(?<!\d)(?P<val>[01])\s*[^\w\n]{0,6}\s*(?P<label>[A-Za-z][A-Za-z /()_-]{0,80})"
)

# Split on newlines (notes are line-oriented; avoid breaking on abbreviations like "Dr.").
_SENTENCE_SPLIT_RE = re.compile(r"\n+")

_PLEURODESIS_CUE_RE = re.compile(
    r"(?i)\b(?:pleurodesis|chemical\s+pleurodesis|32560|32650|talc|doxycycline|poudrage|slurry|sclerosing\s+agent)\b"
)
_ATTRIBUTED_NOTE_PREFIX_RE = re.compile(
    r"(?i)\b(?:see|refer(?:\s+to)?|referred\s+to|per|as\s+per)\b[^.\n]{0,120}\b(?:dr\.?|doctor|note|op\s*note|operative\s+note|surgery\s+note|report)\b"
)


@dataclass
class GuardrailOutcome:
    record: RegistryRecord | None
    warnings: list[str]
    needs_review: bool
    changed: bool


class ClinicalGuardrails:
    """Postprocess guardrails for common extraction failure modes."""

    def apply_record_guardrails(self, note_text: str, record: RegistryRecord) -> GuardrailOutcome:
        warnings: list[str] = []
        needs_review = False
        changed = False

        from app.registry.postprocess.template_checkbox_negation import (
            apply_template_checkbox_negation,
        )

        record, checkbox_warnings = apply_template_checkbox_negation(note_text or "", record)
        if checkbox_warnings:
            warnings.extend(checkbox_warnings)
            changed = True

        record_data = record.model_dump()
        text_lower = (note_text or "").lower()
        chest_tube_insertion_date_line = bool(_CHEST_TUBE_DATE_OF_INSERTION_RE.search(text_lower))

        # Non-IP GI endoscopy / PEG notes occasionally leak into the pipeline and can trip
        # bronchoscopy template cues ("Initial Airway Inspection Findings"). If PEG/EGD
        # language is present, suppress bronchoscopy/pleural procedures entirely.
        peg_like = bool(
            re.search(
                r"(?i)\b(?:peg\b|percutaneous\s+endoscopic\s+gastrostomy|gastrostomy)\b",
                note_text or "",
            )
        )
        gi_anatomy = bool(
            re.search(
                r"(?i)\b(?:stomach|gastric|duodenum|esophagus|pylorus|transillumination)\b",
                note_text or "",
            )
        )
        if peg_like and gi_anatomy:
            if record_data.get("procedures_performed") not in (None, {}, []):
                record_data["procedures_performed"] = {}
                changed = True
            if record_data.get("pleural_procedures") not in (None, {}, []):
                record_data["pleural_procedures"] = {}
                changed = True
            if record_data.get("procedure_families") not in (None, [], ""):
                record_data["procedure_families"] = []
                changed = True
            warnings.append("Non-IP PEG/EGD note detected; suppressing bronchoscopy/pleural procedures.")
            updated = RegistryRecord(**record_data) if changed else record
            return GuardrailOutcome(
                record=updated,
                warnings=warnings,
                needs_review=True,
                changed=changed,
            )

        # BLVR checkbox/table corrections: fix valve type, lobe selection, Chartis result, and count.
        blvr_warnings, blvr_changed = self._apply_blvr_guardrails(note_text or "", record_data)
        if blvr_changed:
            warnings.extend(blvr_warnings)
            changed = True

        # Thoracoscopy backstop: ensure thoracoscopy fields populate from narrative/checkboxes.
        thor_warnings, thor_changed = self._apply_thoracoscopy_guardrails(note_text or "", record_data)
        if thor_changed:
            warnings.extend(thor_warnings)
            changed = True

        # Airway dilation false positives (skin/subcutaneous/chest wall/tract context).
        if _DILATION_CONTEXT_PATTERN.search(text_lower):
            if self._set_procedure_performed(record_data, "airway_dilation", False):
                warnings.append("Airway dilation excluded due to chest wall/skin context.")
                changed = True

        # Rigid bronchoscopy header/body conflict.
        if self._rigid_header_conflict(note_text):
            if self._set_procedure_performed(record_data, "rigid_bronchoscopy", False):
                warnings.append("Rigid bronchoscopy header/body conflict; treating as not performed.")
                changed = True

        sedation = record_data.get("sedation")
        if not isinstance(sedation, dict):
            sedation = {}
        proceduralist_moderate_context = bool(
            re.search(r"(?i)\bmoderate\s+sedation\b", note_text or "")
            and (
                re.search(r"(?i)\bno\s+anesthesiologist\s+present\b|\bwithout\s+anesthesiologist\b", note_text or "")
                or re.search(
                    r"(?i)\b(?:attending|proceduralist|operator|physician)\b[^.\n]{0,80}\bperformed\s+(?:own\s+)?sedation\b",
                    note_text or "",
                )
            )
        )
        sedation_changed = False
        if proceduralist_moderate_context:
            if sedation.get("type") != "Moderate":
                sedation["type"] = "Moderate"
                changed = True
                sedation_changed = True
            if sedation.get("anesthesia_provider") != "Proceduralist":
                sedation["anesthesia_provider"] = "Proceduralist"
                changed = True
                sedation_changed = True
            if sedation_changed:
                record_data["sedation"] = sedation
                warnings.append(
                    "Sedation corrected to Moderate/Proceduralist from explicit bedside proceduralist-sedation narrative."
                )
        elif sedation.get("anesthesia_provider") == "Anesthesiologist" and re.search(
            r"(?i)\bno\s+anesthesiologist\s+present\b|\bwithout\s+anesthesiologist\b", note_text or ""
        ):
            sedation.pop("anesthesia_provider", None)
            record_data["sedation"] = sedation
            warnings.append("Sedation anesthesia_provider cleared: anesthesiologist explicitly negated in note.")
            changed = True

        if record_data.get("established_tracheostomy_route") is True and re.search(
            r"(?i)\b(?:immature|early|fresh)\s+tract\b"
            r"|\bnot\s+yet\s+epithelialized\b"
            r"|\baccidental\s+decannulat\w*\b"
            r"|\b(?:day|pod)\s*(?:0?[1-9]|1[0-4])\b[^.\n]{0,80}\btrach(?:eostomy)?\b"
            r"|\bpartially\s+closed\b[^.\n]{0,60}\btract\b",
            note_text or "",
        ):
            record_data["established_tracheostomy_route"] = False
            warnings.append("Established tracheostomy route cleared: note supports immature-tract trach reinsertion/change.")
            changed = True

        # Radial vs linear EBUS disambiguation.
        radial_marker_match = _RADIAL_MARKER_PATTERN.search(text_lower)
        radial_marker = False
        if radial_marker_match:
            # Block common false positives like "concentric stenosis" or "eccentric narrowing".
            window_start = max(0, radial_marker_match.start() - 160)
            window_end = min(len(text_lower), radial_marker_match.end() + 160)
            local_window = text_lower[window_start:window_end]
            if not re.search(r"\b(?:stenosis|stricture|narrowing)\b", local_window):
                radial_marker = True
        explicit_radial = bool(
            re.search(
                r"\b(?:radial\s+ebus|radial\s+probe|r-?ebus|rp-?ebus|miniprobe)\b",
                text_lower,
                re.IGNORECASE,
            )
        )
        radial_present = radial_marker or explicit_radial
        linear_marker = bool(_LINEAR_MARKER_PATTERN.search(text_lower))
        station_data_present = self._linear_station_data_present(record_data)
        peripheral_context_present = bool(
            re.search(
                r"\b(?:"
                r"peripheral\s+bronchoscopy|"
                r"fluoro(?:scop\w*)?|"
                r"guide\s+sheath|sheath\s+catheter|large\s+sheath|"
                r"nodule|target\s+lesion|pulmonary\s+nodule|lung\s+nodule|"
                r"transbronchial\s+(?:lung\s+)?biops|tbbx|tblb|"
                r"brush(?:ings?)?\b"
                r")\b",
                text_lower,
            )
        )

        if radial_present:
            changed |= self._set_procedure_performed(record_data, "radial_ebus", True)
            if radial_marker:
                warnings.append("Radial EBUS inferred from concentric/eccentric view.")
            else:
                warnings.append("Radial EBUS inferred from radial probe keywords.")

        if linear_marker:
            changed |= self._set_procedure_performed(record_data, "linear_ebus", True)
            warnings.append("Linear EBUS inferred from convex/mediastinal/station sampling.")

        if station_data_present:
            if self._set_procedure_performed(record_data, "linear_ebus", True):
                warnings.append("Linear EBUS inferred from sampled station data.")

        if radial_present and not linear_marker and not station_data_present:
            changed |= self._set_procedure_performed(record_data, "linear_ebus", False)
            # Avoid confusing "ghost" attributes (e.g., needle gauge) when radial EBUS is
            # present but linear EBUS-TBNA is not supported by station/staging evidence.
            procedures = record_data.get("procedures_performed")
            if isinstance(procedures, dict):
                procedures["linear_ebus"] = {"performed": False}
                record_data["procedures_performed"] = procedures

        # Combined linear (staging) + radial (peripheral localization) EBUS is common.
        # Only force manual review when there is no peripheral-context evidence that
        # explains why both marker families appear.
        if radial_present and (linear_marker or station_data_present) and not peripheral_context_present:
            needs_review = True
            warnings.append("Radial vs linear EBUS markers both present; review required.")

        # Cull hollow linear EBUS claims: if linear EBUS is marked performed but there
        # are no stations/node_events and no explicit EBUS-TBNA/linear EBUS narrative
        # marker, treat as not performed.
        procedures = record_data.get("procedures_performed")
        linear = procedures.get("linear_ebus") if isinstance(procedures, dict) else None
        if isinstance(linear, dict) and linear.get("performed") is True:
            explicit_linear_marker = bool(
                re.search(
                    r"(?i)\b(?:linear\s+ebus|ebus[-\s]?tbna|convex\s+probe\s+ebus|cp-?ebus|endobronchial\s+ultrasound)\b",
                    text_lower,
                )
            )
            has_station_payload = bool(
                linear.get("stations_sampled")
                or linear.get("stations_detail")
                or linear.get("station_count_bucket")
            )
            has_node_events = bool(linear.get("node_events"))
            if not (explicit_linear_marker or has_station_payload or has_node_events):
                if isinstance(procedures, dict):
                    procedures["linear_ebus"] = {"performed": False}
                    record_data["procedures_performed"] = procedures
                warnings.append(
                    "AUTO_CORRECTED: linear_ebus.performed=true without station/node evidence; treating as not performed."
                )
                changed = True

        # Stent negation and inspection-only guardrails.
        procedures = record_data.get("procedures_performed")
        stent = procedures.get("airway_stent") if isinstance(procedures, dict) else None
        if isinstance(stent, dict) and stent.get("performed") is True:
            negated = any(p.search(text_lower) for p in _STENT_NEGATION_PATTERNS)
            removal_text_present = bool(_STENT_REMOVAL_CONTEXT_RE.search(text_lower))
            if removal_text_present and re.search(
                r"(?i)\b(?:bronchoscop(?:e)?|scope)\b[^.\n]{0,120}\bremoved\b[^.\n]{0,180}\bstent\b"
                r"[^.\n]{0,120}\b(?:advance|insert|deploy|place|deliver|seat|implant)\w*\b",
                note_text or "",
            ):
                explicit_stent_removed = bool(
                    re.search(
                        r"(?i)\bstent(?:s)?\b[^.\n]{0,120}\b(?:remov(?:e|ed|al)|retriev(?:e|ed|al)|extract(?:ed|ion)?|explant(?:ed|ation)?|exchange(?:d)?|replac(?:ed|ement)?|pull(?:ed)?\s+out)\b",
                        note_text or "",
                    )
                )
                if not explicit_stent_removed:
                    removal_text_present = False
            removal_flag = stent.get("airway_stent_removal") is True
            removal_present = removal_text_present or removal_flag
            placement_present = bool(_STENT_PLACEMENT_CONTEXT_RE.search(text_lower))
            placement_action_present = bool(_STENT_PLACEMENT_ACTION_CONTEXT_RE.search(text_lower))
            strong_placement = bool(_STENT_STRONG_PLACEMENT_RE.search(text_lower))
            inspection_only = bool(
                _STENT_INSPECTION_RE.search(text_lower)
                or _STENT_OBSTRUCTION_RE.search(text_lower)
                or _STENT_CLEANING_RE.search(text_lower)
            )
            hypothetical_only = bool(
                re.search(
                    r"\b(?:consider(?:ed|ation)|discuss(?:ed|ion|ing)|advocat(?:e|ed|ing)|recommend(?:ed|ation)?|plan(?:ned)?|would\b|if\b)\b"
                    r"[^.\n]{0,220}\b(?:airway\s+)?stent(?:s)?\b",
                    text_lower,
                    re.IGNORECASE,
                )
                and re.search(r"\b(?:placement|insertion)\b", text_lower, re.IGNORECASE)
            )

            # Best-effort stent type enrichment (helps note_352-style "Y stent" mentions).
            if not stent.get("stent_type") and re.search(r"\by-?\s*stent\b", text_lower, re.IGNORECASE):
                stent["stent_type"] = "Y-Stent"
                changed = True

            # Disambiguate bronchoscope brands from stent brands (e.g., "Dumon ... bronchoscope").
            brand_raw = stent.get("stent_brand")
            if isinstance(brand_raw, str) and brand_raw.strip():
                brand_lower = brand_raw.strip().lower()
                if brand_lower == "dumon":
                    dumon_broncho = bool(
                        re.search(r"(?i)\bdumon\b[^.\n]{0,80}\bbronchoscop", note_text or "")
                        or re.search(r"(?i)\bbronchoscop[^.\n]{0,80}\bdumon\b", note_text or "")
                    )
                    dumon_stent = bool(
                        re.search(r"(?i)\bdumon\b[^.\n]{0,80}\bstent\b", note_text or "")
                        or re.search(r"(?i)\bstent\b[^.\n]{0,80}\bdumon\b", note_text or "")
                    )
                    if dumon_broncho and not dumon_stent:
                        stent.pop("stent_brand", None)
                        if stent.get("stent_type") == "Silicone - Dumon":
                            stent.pop("stent_type", None)
                        warnings.append(
                            "AUTO_CORRECTED: cleared Dumon stent brand/type (bronchoscope context only)."
                        )
                        changed = True

            # Revision semantics: don't label revision/repositioning as "removal" without explicit removal language.
            action_text = str(stent.get("action") or "").strip().lower()
            revision_action = "revision" in action_text or "reposition" in action_text
            if revision_action and removal_flag and not removal_text_present:
                stent["airway_stent_removal"] = False
                warnings.append("Stent revision/repositioning without removal language; clearing airway_stent_removal.")
                changed = True
                removal_flag = False
                removal_present = removal_text_present

            # Symmetric guardrail: if the extracted action is "revision/repositioning" but
            # the narrative supports removal-only (and no exchange/replace language exists),
            # treat as removal.
            if (
                revision_action
                and removal_text_present
                and not placement_present
                and not strong_placement
                and not placement_action_present
            ):
                exchange_or_replace = bool(re.search(r"(?i)\b(?:exchang|replac)\w*\b", text_lower))
                if not exchange_or_replace:
                    if self._set_stent_action(record_data, "Removal"):
                        warnings.append("Stent action revision contradicted by removal-only narrative; treating as removal.")
                        changed = True
                    revision_action = False
                    removal_flag = True
                    removal_present = True

            if negated and not strong_placement:
                if removal_present:
                    if self._set_stent_action(record_data, "Removal"):
                        warnings.append("Stent placement negated; treating as stent removal.")
                        changed = True
                else:
                    if self._clear_stent(record_data):
                        warnings.append("Stent placement negated; treating as not performed.")
                        changed = True
            elif (
                hypothetical_only
                and not strong_placement
                and not placement_present
                and not placement_action_present
                and not removal_present
                and not inspection_only
            ):
                if self._clear_stent(record_data):
                    warnings.append("Stent placement discussed/considered only; treating as not performed.")
                    changed = True
            elif (
                removal_text_present
                and not placement_present
                and not strong_placement
                and not placement_action_present
                and not revision_action
            ):
                if self._set_stent_action(record_data, "Removal"):
                    warnings.append("Stent removal language; treating as removal only.")
                    changed = True
            elif removal_flag and not removal_text_present and inspection_only and not placement_present:
                if self._set_stent_assessment_only(record_data):
                    warnings.append("Stent removal/revision not supported by text; treating as assessment only.")
                    changed = True
            elif inspection_only and not placement_action_present and not removal_text_present:
                if self._set_stent_assessment_only(record_data):
                    warnings.append("Stent inspection-only language; treating as assessment only.")
                    changed = True
            elif placement_present and not removal_present and not inspection_only:
                if self._set_stent_action(record_data, "Placement"):
                    warnings.append("Stent placement language; treating as placement.")
                    changed = True
        elif isinstance(stent, dict) and stent.get("performed") is False:
            # Keep false procedures compact (avoid "ghost" action_type/revision fields).
            if set(stent.keys()) - {"performed"}:
                if isinstance(procedures, dict):
                    procedures["airway_stent"] = {"performed": False}
                    record_data["procedures_performed"] = procedures
                    warnings.append("AUTO_CORRECTED: airway_stent.performed=false; cleared sub-fields.")
                    changed = True

        # Therapeutic aspiration material: require explicit purulence language for "Purulent secretions".
        aspiration = procedures.get("therapeutic_aspiration") if isinstance(procedures, dict) else None
        if isinstance(aspiration, dict) and aspiration.get("performed") is True:
            material = aspiration.get("material")
            if isinstance(material, str) and material.strip() == "Purulent secretions":
                if not re.search(r"(?i)\b(?:purulent|mucopurulent|pus|suppurat)\w*\b", note_text or ""):
                    aspiration["material"] = "Mucus"
                    if isinstance(procedures, dict):
                        procedures["therapeutic_aspiration"] = aspiration
                        record_data["procedures_performed"] = procedures
                    warnings.append(
                        "AUTO_CORRECTED: therapeutic_aspiration.material='Purulent secretions' without purulence language; downgraded to 'Mucus'."
                    )
                    changed = True

        # Therapeutic injection medication cleanup: strip volume prefixes like "mL of Kenalog".
        injection = procedures.get("therapeutic_injection") if isinstance(procedures, dict) else None
        if isinstance(injection, dict) and injection.get("performed") is True:
            medication = injection.get("medication")
            if isinstance(medication, str) and medication.strip():
                cleaned = re.sub(r"(?i)^(?:\d+(?:\.\d+)?\s*)?(?:ml|cc)\s+of\s+", "", medication).strip()
                if cleaned and cleaned != medication:
                    injection["medication"] = cleaned
                    if isinstance(procedures, dict):
                        procedures["therapeutic_injection"] = injection
                        record_data["procedures_performed"] = procedures
                    warnings.append("AUTO_CORRECTED: therapeutic_injection.medication stripped volume prefix.")
                    changed = True

        # TBNA conventional vs peripheral (lung lesion) guardrails.
        tbna = procedures.get("tbna_conventional") if isinstance(procedures, dict) else None
        if isinstance(tbna, dict) and tbna.get("performed") is True:
            stations = tbna.get("stations_sampled") or []
            stations_present = bool(stations)
            has_station_token = bool(
                re.search(r"\b(?:2R|2L|4R|4L|7|10R|10L|11R(?:S|I)?|11L(?:S|I)?|12R|12L)\b", text_lower)
            )
            has_nodal_context = bool(
                re.search(
                    r"\b(?:mediastinal|hilar|lymph\s+node\s+survey|systematic\b[^.\n]{0,80}\bstag|station\s*\d{1,2})\b",
                    text_lower,
                    re.IGNORECASE,
                )
            )
            has_peripheral_context = bool(
                re.search(
                    r"\b(?:"
                    r"pulmonary\s+nodule|lung\s+nodule|pulmonary\s+lesion|lung\s+lesion|nodule|lesion|mass|"
                    r"transbronchial\s+needle\s+aspiration|tbna\b|"
                    r"rb\d{1,2}\b|lb\d{1,2}\b|segment|subsegment|"
                    r"navigation|navigational|robotic|electromagnetic|\benb\b|\bion\b|monarch|"
                    r"radial\s+(?:ebus|probe|ultrasound)|rebus|miniprobe"
                    r")\b",
                    text_lower,
                    re.IGNORECASE,
                )
            )

            if not stations_present and has_peripheral_context and not has_station_token and not has_nodal_context:
                if self._set_procedure_performed(record_data, "tbna_conventional", False):
                    warnings.append(
                        "TBNA conventional marked without stations in a peripheral-lesion context; treating as not performed."
                    )
                    changed = True
                if self._set_procedure_performed(record_data, "peripheral_tbna", True):
                    warnings.append("Peripheral TBNA inferred from lung lesion TBNA context.")
                    changed = True

        # Routine anesthesia intubation should not trigger emergency intubation (31500).
        intubation = procedures.get("intubation") if isinstance(procedures, dict) else None
        if isinstance(intubation, dict) and intubation.get("performed") is True:
            selective_only_context = bool(
                re.search(
                    r"\b(?:selective\b[^.\n]{0,80}\bintubat|intubat(?:ion|ed|ing)\b[^.\n]{0,80}\bmain(?:\s*|-)?stem|into\s+the\s+(?:right|left)\s+main(?:\s*|-)?stem)\b",
                    text_lower,
                    re.IGNORECASE,
                )
            )
            explicit_tube_or_placement = bool(
                re.search(
                    r"\b(?:ett|endotracheal\s+tube|endotracheal\s+intubat(?:ion|ed|ing)|fiberoptic\s+intubat(?:ion|ed|ing)|fiber\s*optic\s+intubat(?:ion|ed|ing))\b",
                    text_lower,
                    re.IGNORECASE,
                )
                or re.search(
                    r"\bintubat(?:ion|ed|ing)\b[^.\n]{0,80}\b(?:perform(?:ed)?|place(?:d)?|insert(?:ed)?|advance(?:d)?)\b"
                    r"|\b(?:perform(?:ed)?|place(?:d)?|insert(?:ed)?|advance(?:d)?)\b[^.\n]{0,80}\bintubat(?:ion|ed|ing)\b",
                    text_lower,
                    re.IGNORECASE,
                )
            )
            special_intubation_context = bool(
                re.search(
                    r"\b(?:"
                    r"fiber\s*optic\s+intubat|fiberoptic\s+intubat|"
                    r"selective\b[^.\n]{0,80}\bintubat|"
                    r"into\s+the\s+(?:right|left)\s+main(?:\s*|-)?stem|"
                    r"difficult\s+airway|failed\s+intubation|multiple\s+attempts|"
                    r"emergent|emergency|code\s+blue|crash"
                    r")\b",
                    text_lower,
                    re.IGNORECASE,
                )
            )
            if selective_only_context and not explicit_tube_or_placement:
                if self._set_procedure_performed(record_data, "intubation", False):
                    warnings.append(
                        "Selective/mainstem isolation language without explicit tube-placement evidence; suppressing emergency intubation flag (31500)."
                    )
                    changed = True
            elif not special_intubation_context:
                if self._set_procedure_performed(record_data, "intubation", False):
                    warnings.append(
                        "Intubation appears routine/anesthesia-only; suppressing emergency intubation flag (31500)."
                    )
                    changed = True

        # Endobronchial biopsy false positives in peripheral cases.
        endobronchial_biopsy = (
            procedures.get("endobronchial_biopsy") if isinstance(procedures, dict) else None
        )
        if isinstance(endobronchial_biopsy, dict) and endobronchial_biopsy.get("performed") is True:
            no_endobronchial_lesions = bool(
                re.search(r"\bno\s+endobronchial\s+lesions?\b", text_lower, re.IGNORECASE)
            )
            explicit_endobronchial_biopsy = bool(
                re.search(r"\bendobronchial\s+biops(?:y|ies)\b|\bebbx\b", text_lower, re.IGNORECASE)
            )
            explicit_airway_biopsy_context = bool(
                re.search(
                    r"(?i)\b(?:trachea|carina|main(?:\s*|-)?stem|bronchus\s+intermedius|airway|endobronch(?:ial)?)\b[^.\n]{0,120}\bbiops(?:y|ies|ied)\b"
                    r"|\bbiops(?:y|ies|ied)\b[^.\n]{0,120}\b(?:trachea|carina|main(?:\s*|-)?stem|bronchus\s+intermedius|airway|endobronch(?:ial)?)\b",
                    note_text or "",
                )
            )
            peripheral_case = bool(
                re.search(
                    r"\b(?:"
                    r"peripheral|nodule|lung\s+(?:nodule|lesion|mass)|pulmonary\s+(?:nodule|lesion|mass)|"
                    r"navigation|navigational|electromagnetic\s+navigation|\benb\b|ion\b|robotic|"
                    r"radial\s+(?:ebus|probe|ultrasound)|rebus|miniprobe|"
                    r"transbronchial\s+(?:lung\s+)?biops|tbbx|tblb"
                    r")\b",
                    text_lower,
                    re.IGNORECASE,
                )
            )

            if no_endobronchial_lesions and peripheral_case and not explicit_endobronchial_biopsy:
                if self._set_procedure_performed(record_data, "endobronchial_biopsy", False):
                    warnings.append(
                        "Endobronchial biopsy excluded due to peripheral case + 'no endobronchial lesions' context."
                    )
                    changed = True
            elif peripheral_case and not explicit_endobronchial_biopsy and not explicit_airway_biopsy_context:
                if self._set_procedure_performed(record_data, "endobronchial_biopsy", False):
                    warnings.append(
                        "Endobronchial biopsy excluded due to peripheral biopsy workflow without airway-biopsy context."
                    )
                    changed = True

        diagnostic_bronchoscopy = (
            procedures.get("diagnostic_bronchoscopy") if isinstance(procedures, dict) else None
        )
        if isinstance(diagnostic_bronchoscopy, dict):
            no_endobronchial_disease = bool(
                re.search(
                    r"\bno\s+endobronchial\s+(?:lesions?|tumou?rs?|mass(?:es)?)\b",
                    note_text or "",
                    re.IGNORECASE,
                )
            )
            explicit_positive_endobronchial_disease = bool(
                re.search(
                    r"(?i)\b(?:there\s+(?:was|were)|found|noted|seen|visualized)\b[^.\n]{0,120}\bendobronchial\b[^.\n]{0,120}\b(?:lesion|tumou?r|mass)\b"
                    r"|\bendobronchial\b[^.\n]{0,120}\b(?:lesion|tumou?r|mass)\b[^.\n]{0,120}\b(?:was|were)\b",
                    note_text or "",
                )
            )
            if no_endobronchial_disease and not explicit_positive_endobronchial_disease:
                abnormalities = diagnostic_bronchoscopy.get("airway_abnormalities")
                if isinstance(abnormalities, list) and "Endobronchial lesion" in abnormalities:
                    diagnostic_bronchoscopy["airway_abnormalities"] = [
                        item for item in abnormalities if item != "Endobronchial lesion"
                    ]
                    warnings.append(
                        "Diagnostic bronchoscopy findings corrected: cleared false endobronchial lesion from negated narrative."
                    )
                    changed = True

                findings = diagnostic_bronchoscopy.get("inspection_findings")
                if isinstance(findings, str) and findings.strip():
                    parts: list[str] = []
                    for raw_part in re.split(r"(?i)(?:\.\s+|;\s+|\n+)", findings):
                        part = (raw_part or "").strip(" .;:-")
                        if not part:
                            continue
                        part_lower = part.lower()
                        if (
                            "endobronch" in part_lower
                            or part_lower in {"lesion", "lesions", "mass", "tumor", "tumour"}
                            or re.fullmatch(r"(?i)(?:endobronchial\s+)?(?:lesion|mass|tumou?r)s?", part)
                        ):
                            continue
                        parts.append(part)
                    cleaned_findings = ". ".join(parts).strip()
                    if cleaned_findings != findings.strip():
                        if cleaned_findings:
                            diagnostic_bronchoscopy["inspection_findings"] = cleaned_findings
                        else:
                            diagnostic_bronchoscopy.pop("inspection_findings", None)
                        warnings.append(
                            "Diagnostic bronchoscopy inspection_findings sanitized for negated endobronchial disease."
                        )
                        changed = True
                procedures["diagnostic_bronchoscopy"] = diagnostic_bronchoscopy
                record_data["procedures_performed"] = procedures

        bal = procedures.get("bal") if isinstance(procedures, dict) else None
        if isinstance(bal, dict) and bal.get("performed") is True:
            whole_lung_lavage_context = bool(
                re.search(r"(?i)\bwhole\s+lung\s+lavage\b|\bwll\b", note_text or "")
                or (
                    re.search(r"(?i)\blavage\b", note_text or "")
                    and re.search(r"(?i)\b(?:pap|pulmonary\s+alveolar\s+proteinosis|double[-\s]+lumen|lung\s+isolation)\b", note_text or "")
                )
            )
            explicit_bal = bool(
                re.search(
                    r"(?i)\b(?:broncho[-\s]?alveolar\s+lavage|bronchial\s+alveolar\s+lavage|BAL)\b",
                    note_text or "",
                )
            )
            optional_bal_only_context = bool(
                re.search(
                    r"(?i)\b(?:broncho[-\s]?alveolar\s+lavage|BAL)\b[^.\n]{0,30}\boptional\b"
                    r"|\boptional\b[^.\n]{0,30}\b(?:broncho[-\s]?alveolar\s+lavage|BAL)\b",
                    note_text or "",
                )
                and not re.search(
                    r"(?i)\b(?:broncho[-\s]?alveolar\s+lavage|BAL)\b[^.\n]{0,80}\b(?:performed|obtained|sent|returned|aliquot|micro|path)\b"
                    r"|\binstill(?:ed)?\s*\d{1,4}\s*(?:ml|cc)\b"
                    r"|\b\d{1,4}\s*(?:ml|cc)\s+(?:returned|recovered)\b",
                    note_text or "",
                )
            )
            if whole_lung_lavage_context:
                if self._set_procedure_performed(record_data, "bal", False):
                    warnings.append("BAL cleared: note supports whole-lung lavage rather than bronchoalveolar lavage.")
                    changed = True
                bal = None
            airway_toilet_lavage = bool(
                re.search(r"(?i)\b(?:lavage|saline\s+lavage)\b", note_text or "")
                and re.search(
                    r"(?i)\b(?:mucus|secretions?|plugging|plug|toilet|toileting|suction|cleared?)\b",
                    note_text or "",
                )
            )
            non_bal_specimen_context = bool(
                re.search(
                    r"(?i)\bspecimens?\b[^.\n]{0,160}\b(?:granulation\s+tissue|tissue\s+only|biopsy|formalin)\b",
                    note_text or "",
                )
                and not re.search(
                    r"(?i)\bspecimens?\b[^.\n]{0,160}\b(?:broncho[-\s]?alveolar\s+lavage|BAL)\b",
                    note_text or "",
                )
            )
            washings_only_context = bool(
                re.search(r"(?i)\b(?:bronchial\s+wash(?:ings?)?|washings?)\b", note_text or "")
                and re.search(r"(?i)\b(?:therapeutic\s+aspiration|secretions?|mucus\s+plug|saline\s+lavage)\b", note_text or "")
            )
            if optional_bal_only_context:
                if self._set_procedure_performed(record_data, "bal", False):
                    warnings.append("BAL cleared: note lists BAL as optional/header-only without performed lavage evidence.")
                    changed = True
            if not explicit_bal and (washings_only_context or (airway_toilet_lavage and non_bal_specimen_context)):
                if self._set_procedure_performed(record_data, "bal", False):
                    warnings.append(
                        "BAL cleared: note supports washings/toileting with saline lavage but not bronchoalveolar lavage."
                    )
                    changed = True

        peripheral_tbna = procedures.get("peripheral_tbna") if isinstance(procedures, dict) else None
        transthoracic_core_context = bool(
            re.search(
                r"(?i)\b(?:transthoracic|percutaneous|coaxial|core\s+needle\s+biops(?:y|ies)|pleural[-\s]?based|chest\s+wall)\b",
                note_text or "",
            )
            and re.search(r"(?i)\b(?:ultrasound|u/s|ct|computed\s+tomography)\b", note_text or "")
            and not re.search(r"(?i)\bbronchoscop|bronchoscope|ebus|radial|navigation|robotic\b", note_text or "")
        )
        if isinstance(peripheral_tbna, dict) and peripheral_tbna.get("performed") is True and transthoracic_core_context:
            if self._set_procedure_performed(record_data, "peripheral_tbna", False):
                warnings.append(
                    "Peripheral TBNA cleared: note describes a percutaneous transthoracic/core biopsy rather than bronchoscopic TBNA."
                )
                changed = True

            pleural = record_data.get("pleural_procedures")
            if not isinstance(pleural, dict):
                pleural = {}
            pleural_biopsy = pleural.get("pleural_biopsy")
            if not isinstance(pleural_biopsy, dict):
                pleural_biopsy = {}
            if pleural_biopsy.get("performed") is not True:
                pleural_biopsy["performed"] = True
                changed = True
            if "ultrasound" in text_lower and pleural_biopsy.get("guidance") != "Ultrasound":
                pleural_biopsy["guidance"] = "Ultrasound"
                changed = True
            elif re.search(r"(?i)\bct\b|\bcomputed\s+tomography\b", note_text or "") and pleural_biopsy.get("guidance") != "CT":
                pleural_biopsy["guidance"] = "CT"
                changed = True
            if not pleural_biopsy.get("needle_type"):
                if re.search(r"(?i)\babrams\b", note_text or ""):
                    pleural_biopsy["needle_type"] = "Abrams needle"
                elif re.search(r"(?i)\btru[-\s]?cut\b", note_text or ""):
                    pleural_biopsy["needle_type"] = "Tru-cut"
                else:
                    pleural_biopsy["needle_type"] = "Cutting needle"
                changed = True
            if pleural_biopsy.get("number_of_samples") in (None, 0):
                match = re.search(
                    r"(?i)\bobtain(?:ed)?\s+(?P<count>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:core\s+)?samples?\b"
                    r"|\b(?P<count2>\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(?:core\s+)?(?:samples?|cores?)\b(?:[^.\n]{0,40}\b(?:were\s+)?obtained\b)?",
                    note_text or "",
                )
                if match:
                    raw_count = (match.group("count") or match.group("count2") or "").strip().lower()
                    try:
                        pleural_biopsy["number_of_samples"] = int(raw_count)
                    except Exception:
                        word_to_int = {
                            "one": 1,
                            "two": 2,
                            "three": 3,
                            "four": 4,
                            "five": 5,
                            "six": 6,
                            "seven": 7,
                            "eight": 8,
                            "nine": 9,
                            "ten": 10,
                        }
                        parsed = word_to_int.get(raw_count)
                        if parsed is not None:
                            pleural_biopsy["number_of_samples"] = parsed
                    if pleural_biopsy.get("number_of_samples") not in (None, 0):
                        changed = True
            pleural["pleural_biopsy"] = pleural_biopsy
            record_data["pleural_procedures"] = pleural

        explicit_washings = re.search(
            r"(?i)\b(?:bronchial\s+wash(?:ings?)?|washings?)\b",
            note_text or "",
        )
        if explicit_washings:
            procedures = record_data.get("procedures_performed")
            if not isinstance(procedures, dict):
                procedures = {}
            bronchial_wash = procedures.get("bronchial_wash")
            if not isinstance(bronchial_wash, dict):
                bronchial_wash = {}
            if bronchial_wash.get("performed") is not True:
                bronchial_wash["performed"] = True
                changed = True
            if not bronchial_wash.get("location"):
                try:
                    from app.registry.deterministic_extractors import _extract_lung_locations_from_text
                except Exception:  # pragma: no cover
                    _extract_lung_locations_from_text = None  # type: ignore[assignment]
                if _extract_lung_locations_from_text is not None:
                    line_start = (note_text or "").rfind("\n", 0, explicit_washings.start()) + 1
                    line_end = (note_text or "").find("\n", explicit_washings.end())
                    if line_end == -1:
                        line_end = len(note_text or "")
                    line_text = (note_text or "")[line_start:line_end]
                    locations = _extract_lung_locations_from_text(line_text)
                    if not locations:
                        window = (note_text or "")[
                            max(0, explicit_washings.start() - 120) : min(len(note_text or ""), explicit_washings.end() + 120)
                        ]
                        locations = _extract_lung_locations_from_text(window)
                    if locations:
                        bronchial_wash["location"] = locations[0]
                        changed = True
            procedures["bronchial_wash"] = bronchial_wash
            record_data["procedures_performed"] = procedures

        foreign_body = procedures.get("foreign_body_removal") if isinstance(procedures, dict) else None
        airway_stent = procedures.get("airway_stent") if isinstance(procedures, dict) else None
        if isinstance(foreign_body, dict) and foreign_body.get("performed") is True:
            stent_removal_context = bool(
                re.search(
                    r"(?i)\bstent\b[^.\n]{0,120}\b(?:remov|retriev|extract|grasp|withdraw|en\s+bloc)\w*\b"
                    r"|\b(?:remov|retriev|extract|grasp|withdraw)\w*\b[^.\n]{0,120}\bstent\b",
                    note_text or "",
                )
            ) or (
                isinstance(airway_stent, dict)
                and airway_stent.get("performed") is True
                and str(airway_stent.get("action") or "").lower().startswith("remov")
            )
            if stent_removal_context:
                if self._set_procedure_performed(record_data, "foreign_body_removal", False):
                    warnings.append(
                        "Foreign-body removal cleared: airway stent removal is tracked under airway_stent, not foreign_body_removal."
                    )
                    changed = True

        airway_stent = procedures.get("airway_stent") if isinstance(procedures, dict) else None
        if isinstance(airway_stent, dict) and airway_stent.get("performed") is True:
            action_type = str(airway_stent.get("action_type") or "").strip().lower()
            stent_type = str(airway_stent.get("stent_type") or "").strip()
            stent_brand = str(airway_stent.get("stent_brand") or "").strip()
            if stent_type == "Silicone - Dumon" and stent_brand.lower() in {"", "ent", "stent", "airway"}:
                airway_stent["stent_brand"] = "Dumon"
                warnings.append("Airway stent brand normalized to Dumon from explicit Dumon stent narrative.")
                changed = True

            stent_location = str(airway_stent.get("location") or "").strip()
            if (
                stent_location == "Carina (Y)"
                and "y-stent" not in text_lower
                and re.search(
                    r"(?i)\btracheal\s+stent\b|\bstent\b[^.\n]{0,80}\b(?:mm|cm)\b[^.\n]{0,40}\bfrom\s+carina\b",
                    note_text or "",
                )
            ):
                airway_stent["location"] = "Trachea"
                warnings.append("Airway stent location normalized to Trachea from tracheal-stent/carina-distance narrative.")
                changed = True

            preexisting_stent_context = bool(
                re.search(
                    r"(?i)\b(?:prior|previous|existing)\b[^.\n]{0,120}\bstent\b|\bstent\b[^.\n]{0,120}\bin\s+place\b",
                    note_text or "",
                )
            )
            strong_placement = bool(
                re.search(
                    r"(?i)\b(?:stent|aero(?:stent)?|dumon|silicone\s+stent|metal(?:lic)?\s+stent)\b[^.\n]{0,80}\b(?:deployed?|insert(?:ed|ion)?|advance(?:d|ment)?|implant(?:ed|ation)?|placed|placement)\b"
                    r"|\b(?:deployed?|insert(?:ed|ion)?|advance(?:d|ment)?|implant(?:ed|ation)?|placed|placement)\b[^.\n]{0,80}\b(?:stent|aero(?:stent)?|dumon|silicone\s+stent|metal(?:lic)?\s+stent)\b",
                    note_text or "",
                )
            )
            management_only_context = bool(
                re.search(
                    r"(?i)\b(?:toilet(?:ing)?|suction(?:ed|ing)?|debrid(?:ed|ement)?|granulation|surveillance|clean(?:ed|ing)?)\b",
                    note_text or "",
                )
            )
            if action_type == "placement" and preexisting_stent_context and management_only_context and not strong_placement:
                if self._set_stent_assessment_only(record_data):
                    warnings.append(
                        "Existing-stent surveillance/toileting context detected; downgraded false airway stent placement to assessment_only."
                    )
                    changed = True

        complications = record_data.get("complications")
        if isinstance(complications, dict):
            complications_none = bool(
                re.search(r"(?i)\bcomplications?\s*:?\s*none\b|\bno\s+immediate\s+complications\b", note_text or "")
            )
            low_grade_bleeding_only = bool(
                re.search(
                    r"(?i)\b(?:minor|minimal|mild|trace|scant|contact|blood-tinged)\b(?:\s+\w+){0,2}\s+(?:bleeding|oozing|hemorrhag(?:e|ic))\b"
                    r"|\bminor\s+oozing\b|\bmild\s+oozing\b|\boo?zing\b",
                    note_text or "",
                )
            )
            high_grade_bleeding = bool(
                re.search(
                    r"(?i)\b(?:moderate|significant|severe|massive|brisk|active)\s+bleeding\b|\bhemorrhag(?:e|ic)\b",
                    note_text or "",
                )
            )
            if complications_none and low_grade_bleeding_only and not high_grade_bleeding:
                bleeding = complications.get("bleeding")
                local_changed = False
                if isinstance(bleeding, dict):
                    if bleeding.get("occurred") is not False:
                        bleeding["occurred"] = False
                        local_changed = True
                    if bleeding.get("bleeding_grade_nashville") not in (None, 0):
                        bleeding["bleeding_grade_nashville"] = 0
                        local_changed = True
                    complications["bleeding"] = bleeding

                comp_list = complications.get("complication_list")
                if isinstance(comp_list, list):
                    filtered = [item for item in comp_list if "bleeding" not in str(item).lower()]
                    if filtered != comp_list:
                        complications["complication_list"] = filtered
                        local_changed = True

                events = complications.get("events")
                if isinstance(events, list):
                    filtered_events = [
                        event
                        for event in events
                        if not (isinstance(event, dict) and str(event.get("type") or "").lower() == "bleeding")
                    ]
                    if filtered_events != events:
                        complications["events"] = filtered_events
                        local_changed = True

                remaining_complications = bool(complications.get("complication_list"))
                remaining_events = bool(complications.get("events"))
                other_flags = False
                for key in ("pneumothorax", "respiratory"):
                    value = complications.get(key)
                    if isinstance(value, dict) and value.get("occurred") is True:
                        other_flags = True
                        break
                if not (remaining_complications or remaining_events or other_flags):
                    if complications.get("any_complication") is not False:
                        complications["any_complication"] = False
                        local_changed = True

                if local_changed:
                    record_data["complications"] = complications
                    warnings.append(
                        "Low-grade bleeding complication cleared: note documents minor/minimal oozing with Complications: None."
                    )
                    changed = True

        # IPC vs chest tube disambiguation.
        ipc_checkbox = self._checkbox_state(
            note_text or "",
            (
                "tunneled pleural catheter",
                "tunnelled pleural catheter",
                "indwelling pleural catheter",
                "ipc",
                "pleurx",
                "aspira",
            ),
        )
        ipc_present = self._contains_any(text_lower, _IPC_TERMS)
        ipc_header_removal = bool(
            re.search(r"(?i)\b32552\b", note_text or "")
            or re.search(r"(?i)\bRemoval\s+of\s+indwelling\s+tunneled\s+pleural\s+catheter\b", note_text or "")
        )
        if ipc_header_removal:
            ipc_present = True
        if ipc_checkbox is False and not ipc_header_removal:
            ipc_present = False
        tube_present = self._contains_any(text_lower, _CHEST_TUBE_TERMS)
        pleural_flagged = self._pleural_procedure_flagged(record_data)
        if pleural_flagged and (ipc_present or tube_present):
            if chest_tube_insertion_date_line:
                tube_cleared = self._set_pleural_performed(record_data, "chest_tube", False)
                if tube_cleared:
                    warnings.append("Chest tube excluded due to 'Date of chest tube insertion' history line.")
                    changed = True
            has_device_flag = bool(self._pleural_device_flagged(record_data))
            ipc_insert = any(self._has_action_near(text_lower, term, _INSERT_TERMS) for term in _IPC_TERMS)
            tube_insert = any(self._has_action_near(text_lower, term, _INSERT_TERMS) for term in _CHEST_TUBE_TERMS)
            ipc_remove = any(self._has_action_near(text_lower, term, _REMOVE_TERMS) for term in _IPC_TERMS)
            tube_remove = any(self._has_action_near(text_lower, term, _REMOVE_TERMS) for term in _CHEST_TUBE_TERMS)
            if chest_tube_insertion_date_line:
                tube_insert = False
            tube_specific_insertion_detail = bool(
                re.search(
                    r"(?i)\b(?:\d+\s*fr|intercostal|pleural\s+space|seldinger|thoracostomy|pigtail|wayne|yueh|incision)\b",
                    note_text or "",
                )
            )
            existing_tube_context = bool(
                re.search(
                    r"(?i)\bchest\s+tube\b[^.\n]{0,120}\b(?:in\s+place|to\s+water\s+seal|water\s+seal\s+trial|to\s+suction|connected\s+to\s+suction|bubbling|air\s+leak)\b"
                    r"|\bcontinue\s+chest\s+tube\b"
                    r"|\bwith\s+chest\s+tube\s+in\s+place\b",
                    note_text or "",
                )
            )
            if existing_tube_context and not tube_remove and not tube_specific_insertion_detail:
                if self._set_pleural_performed(record_data, "chest_tube", False):
                    warnings.append("Chest tube mention appears historical/ongoing rather than a new insertion; clearing chest_tube.")
                    changed = True
                tube_present = False
                tube_insert = False
                tube_remove = False
                has_device_flag = bool(self._pleural_device_flagged(record_data))

            # If a pleural procedure is flagged (e.g., fibrinolytic instillation) but there is
            # no explicit device action, avoid inferring a device placement solely from mention
            # of "chest tube"/"PleurX"/etc in historical/route context.
            if not has_device_flag and not any((ipc_insert, tube_insert, ipc_remove, tube_remove)):
                pass
            else:
                removal_only = False
                if tube_present and tube_remove and not tube_insert and not ipc_insert:
                    changed |= self._set_pleural_performed(record_data, "chest_tube", False)
                    warnings.append("Chest tube discontinue/removal language; treating as not performed.")
                    removal_only = True
                if ipc_present and ipc_remove and not ipc_insert and not tube_insert:
                    pleural_local = record_data.get("pleural_procedures")
                    ipc_local = pleural_local.get("ipc") if isinstance(pleural_local, dict) else None
                    if not isinstance(ipc_local, dict):
                        ipc_local = {}
                    prior_performed = ipc_local.get("performed")
                    prior_action = ipc_local.get("action")
                    ipc_local["performed"] = True
                    ipc_local["action"] = "Removal"
                    if isinstance(pleural_local, dict):
                        pleural_local["ipc"] = ipc_local
                        record_data["pleural_procedures"] = pleural_local
                    if prior_performed is not True or prior_action != "Removal":
                        changed = True
                    warnings.append("IPC removal language detected; preserving pleural_procedures.ipc as Removal.")
                    removal_only = True
                if removal_only:
                    pass
                else:
                    preferred = self._resolve_pleural_device(text_lower, ipc_present, tube_present)
                    if preferred == "ipc":
                        changed |= self._set_pleural_performed(record_data, "ipc", True)
                        changed |= self._set_pleural_performed(record_data, "chest_tube", False)
                        warnings.append("IPC inferred from tunneled/IPC device language.")
                    elif preferred == "chest_tube":
                        changed |= self._set_pleural_performed(record_data, "chest_tube", True)
                        changed |= self._set_pleural_performed(record_data, "ipc", False)
                        warnings.append("Chest tube inferred from pigtail/Wayne/tube thoracostomy language.")
                    elif ipc_present and tube_present:
                        needs_review = True
                        warnings.append("IPC vs chest tube conflict; review required.")

        # If a pleural device was flagged but the note contains no device language at all,
        # treat it as a likely hallucination (e.g., thoracentesis notes accidentally marked as IPC).
        pleural = record_data.get("pleural_procedures")
        if isinstance(pleural, dict):
            ipc_proc = pleural.get("ipc")
            if isinstance(ipc_proc, dict) and ipc_proc.get("performed") is True and not ipc_present:
                if self._set_pleural_performed(record_data, "ipc", False):
                    warnings.append("IPC not supported by note text; clearing pleural_procedures.ipc.performed.")
                    changed = True

            tube_proc = pleural.get("chest_tube")
            if (
                isinstance(tube_proc, dict)
                and tube_proc.get("performed") is True
                and not tube_present
                and not chest_tube_insertion_date_line
            ):
                if self._set_pleural_performed(record_data, "chest_tube", False):
                    warnings.append(
                        "Chest tube not supported by note text; clearing pleural_procedures.chest_tube.performed."
                    )
                    changed = True

        # Pleurodesis attribution guardrail: suppress pleurodesis when it's only mentioned
        # in referral context (e.g., "See Dr. X's note for VATS and pleurodesis").
        pleural = record_data.get("pleural_procedures")
        if isinstance(pleural, dict):
            pleuro_proc = pleural.get("pleurodesis")
            if isinstance(pleuro_proc, dict) and pleuro_proc.get("performed") is True:
                cue_sentences: list[str] = []
                for raw_sentence in _SENTENCE_SPLIT_RE.split(note_text or ""):
                    sentence = (raw_sentence or "").strip()
                    if not sentence:
                        continue
                    if _PLEURODESIS_CUE_RE.search(sentence):
                        cue_sentences.append(sentence)

                if not cue_sentences:
                    if self._set_pleural_performed(record_data, "pleurodesis", False):
                        warnings.append(
                            "Pleurodesis not supported by note text; clearing pleural_procedures.pleurodesis.performed."
                        )
                        changed = True
                else:
                    only_attributed = True
                    for sentence in cue_sentences:
                        m = _PLEURODESIS_CUE_RE.search(sentence)
                        if not m:
                            continue
                        prefix = sentence[: m.start()]
                        if not _ATTRIBUTED_NOTE_PREFIX_RE.search(prefix):
                            only_attributed = False
                            break
                    if only_attributed:
                        if self._set_pleural_performed(record_data, "pleurodesis", False):
                            warnings.append(
                                "Pleurodesis mentioned only in attributed/referral context; treating as not performed."
                            )
                            changed = True

        updated = RegistryRecord(**record_data) if changed else record
        return GuardrailOutcome(
            record=updated,
            warnings=warnings,
            needs_review=needs_review,
            changed=changed,
        )

    def apply_code_guardrails(
        self, note_text: str, codes: list[str]
    ) -> GuardrailOutcome:
        warnings: list[str] = []
        needs_review = False

        if "31627" in codes and _NAV_FAILURE_PATTERN.search(note_text or ""):
            warnings.append("Navigation failure detected. Verify Modifier -53.")
            needs_review = True

        return GuardrailOutcome(
            record=None,
            warnings=warnings,
            needs_review=needs_review,
            changed=False,
        )

    def _contains_any(self, text: str, terms: tuple[str, ...]) -> bool:
        return any(term in text for term in terms)

    def _checkbox_state(self, note_text: str, candidates: tuple[str, ...]) -> bool | None:
        selected = False
        deselected = False
        text = note_text or ""
        for match in _CHECKBOX_TOKEN_RE.finditer(text):
            val = (match.group("val") or "").strip()
            label = (match.group("label") or "").strip().lower()
            if not label:
                continue
            if not any(re.search(rf"\b{re.escape(candidate)}\b", label) for candidate in candidates):
                continue
            if val == "1":
                selected = True
            elif val == "0":
                deselected = True
        if selected:
            return True
        if deselected:
            return False
        return None

    def _apply_thoracoscopy_guardrails(
        self, note_text: str, record_data: dict[str, Any]
    ) -> tuple[list[str], bool]:
        """Backstop thoracoscopy extraction from narrative + checkbox lines."""
        warnings: list[str] = []
        changed = False
        text_lower = (note_text or "").lower()
        if not re.search(r"\bthoracoscop|\bpleuroscopy|\bmedical\s+thoracoscopy", text_lower, re.IGNORECASE):
            return warnings, changed

        # Some notes reference a prior/concurrent pleuroscopy ("pleuroscopy insertion site") while documenting
        # a different procedure. Avoid triggering thoracoscopy solely on that phrase when it is explicitly
        # delegated to separate documentation.
        incidental_pleuroscopy_site = bool(
            re.search(r"(?i)\bpleuroscopy\b[^.\n]{0,60}\binsertion\s+site\b", note_text or "")
        )
        separate_documentation = bool(
            re.search(r"(?i)\b(?:see|per)\s+separate\s+documentation\b|\bseparate\s+documentation\b", note_text or "")
        )
        explicit_thoracoscopy = bool(
            re.search(r"(?i)\bthoracoscop|\bmedical\s+thoracoscopy\b", note_text or "")
        )
        if incidental_pleuroscopy_site and separate_documentation and not explicit_thoracoscopy:
            return warnings, changed

        pleural = record_data.get("pleural_procedures")
        if not isinstance(pleural, dict):
            pleural = {}
        thor = pleural.get("medical_thoracoscopy")
        if not isinstance(thor, dict):
            thor = {}

        if thor.get("performed") is not True:
            thor["performed"] = True
            warnings.append("Backstop: detected thoracoscopy language; setting medical_thoracoscopy.performed=true")
            changed = True

        # Biopsy Taken: 0 No / 1 Yes
        for raw_line in (note_text or "").splitlines():
            if "biopsy taken" not in raw_line.lower():
                continue
            if re.search(r"(?i)\b1\D{0,6}yes\b", raw_line):
                if thor.get("biopsies_taken") is not True:
                    thor["biopsies_taken"] = True
                    changed = True
                num_match = re.search(r"(?i)\bnumber\s*:\s*(\d{1,3})\b", raw_line)
                if num_match and thor.get("number_of_biopsies") in (None, "", 0):
                    try:
                        thor["number_of_biopsies"] = int(num_match.group(1))
                    except ValueError:
                        pass
                    else:
                        changed = True
            break

        if re.search(r"(?i)\blysis\s+of\s+adhesions\b|\badhesiolysis\b", note_text):
            if thor.get("adhesiolysis_performed") is not True:
                thor["adhesiolysis_performed"] = True
                changed = True

        pleural["medical_thoracoscopy"] = thor
        record_data["pleural_procedures"] = pleural

        # Chest tube "existing ... left in place" should not be treated as new insertion.
        if re.search(r"(?i)\bchest\s+tube/s\b[^.\n]{0,160}\bexisting\b[^.\n]{0,120}\bleft\s+in\s+place\b", note_text):
            chest_tube = pleural.get("chest_tube")
            if not isinstance(chest_tube, dict):
                chest_tube = {}
            if chest_tube.get("performed") is not True:
                chest_tube["performed"] = True
                changed = True
            if chest_tube.get("action") in (None, "", "Insertion"):
                chest_tube["action"] = "Repositioning"
                changed = True
            pleural["chest_tube"] = chest_tube
            record_data["pleural_procedures"] = pleural

        return warnings, changed

    def _apply_blvr_guardrails(self, note_text: str, record_data: dict[str, Any]) -> tuple[list[str], bool]:
        """Correct BLVR fields when checkbox/table structure is present in the note."""
        warnings: list[str] = []
        changed = False

        procedures = record_data.get("procedures_performed")
        if not isinstance(procedures, dict):
            return warnings, changed
        blvr = procedures.get("blvr")
        if not isinstance(blvr, dict) or blvr.get("performed") is not True:
            return warnings, changed

        # Manufacturer selection (checkbox-aware)
        spiration = self._checkbox_state(note_text, ("spiration",))
        zephyr = self._checkbox_state(note_text, ("zephyr",))
        desired_valve_type: str | None = None
        if spiration is True and zephyr is not True:
            desired_valve_type = "Spiration (Olympus)"
        elif zephyr is True and spiration is not True:
            desired_valve_type = "Zephyr (Pulmonx)"
        elif spiration is True and zephyr is True:
            warnings.append("NEEDS_REVIEW: Both Zephyr and Spiration selected in BLVR checkbox list.")

        if desired_valve_type and blvr.get("valve_type") != desired_valve_type:
            blvr["valve_type"] = desired_valve_type
            warnings.append("BLVR valve_type corrected from checkbox selection.")
            changed = True

        # Lobe selection (checkbox-aware)
        lobe_map = {
            "left upper": "LUL",
            "left lower": "LLL",
            "right upper": "RUL",
            "right middle": "RML",
            "right lower": "RLL",
            "lingula": "Lingula",
        }
        selected_lobes: list[str] = []
        for label, code in lobe_map.items():
            if self._checkbox_state(note_text, (label,)) is True:
                selected_lobes.append(code)
        if len(selected_lobes) == 1:
            if blvr.get("target_lobe") != selected_lobes[0]:
                blvr["target_lobe"] = selected_lobes[0]
                changed = True
        elif len(selected_lobes) > 1:
            warnings.append(f"NEEDS_REVIEW: Multiple BLVR lobes selected: {sorted(set(selected_lobes))}")

        # Chartis (checkbox-aware): capture "no/minimal collateral ventilation" as Chartis negative.
        chartis_idx = (note_text or "").lower().find("chartis system")
        chartis_window = note_text[chartis_idx : chartis_idx + 240] if chartis_idx != -1 else note_text
        chartis_yes = bool(re.search(r"(?i)\b1\D{0,6}yes\b", chartis_window))
        balloon_idx = (note_text or "").lower().find("balloon occlusion")
        balloon_window = note_text[balloon_idx : balloon_idx + 240] if balloon_idx != -1 else ""
        balloon_yes = bool(re.search(r"(?i)\b1\D{0,6}yes\b", balloon_window)) or bool(
            re.search(r"(?i)\bballoon\s+occlusion\b[^.\n]{0,80}\bperformed\b", note_text)
        )
        chartis_performed = chartis_yes or balloon_yes

        if chartis_performed and re.search(r"(?i)\bno/minimal\s+collateral\s+ventilation\b", note_text):
            if blvr.get("collateral_ventilation_assessment") != "Chartis negative":
                blvr["collateral_ventilation_assessment"] = "Chartis negative"
                changed = True

        # Promote procedure_type when valve placement details are present.
        placed_token = r"(?<!well\s)(?<!previously\s)(?<!prior\s)(?<!already\s)placed\b"
        placement_indicator = bool(
            re.search(
                r"(?i)\bvalves?\b[^.\n]{0,80}\b(?:deploy|deployed|deployment|insert|inserted|insertion|placement|placing)\w*\b",
                note_text,
            )
            or re.search(rf"(?i)\bvalves?\b[^.\n]{{0,80}}\b{placed_token}", note_text)
            or re.search(r"(?i)\bvalve\s+sizes\s+used\b", note_text)
        )
        if placement_indicator and blvr.get("procedure_type") in (None, "", "Valve assessment"):
            blvr["procedure_type"] = "Valve placement"
            changed = True

        # Guardrail: inspection-only valve wording ("previously placed", "well placed", "visualized")
        # is common in non-BLVR procedures (e.g., glue installation for fistula) and must not be
        # interpreted as new valve placement.
        inspection_only = bool(
            re.search(r"(?i)\b(?:previously|prior|already)\s+placed\b", note_text)
            or re.search(r"(?i)\bwell\s+placed\b", note_text)
            or re.search(r"(?i)\bvalves?\b[^.\n]{0,80}\bvisualiz", note_text)
        )
        removal_indicator = bool(re.search(r"(?i)\bvalve\b[^.\n]{0,80}\b(?:remov|retriev|extract|explant)\w*\b", note_text))
        action_indicator = placement_indicator or removal_indicator or bool(re.search(r"(?i)\bchartis\b", note_text))

        if inspection_only and not action_indicator:
            # Inspection-only mentions (e.g., "previously placed valves visualized") should not
            # imply a BLVR workflow was performed.
            procedures["blvr"] = {"performed": False}
            record_data["procedures_performed"] = procedures
            warnings.append("AUTO_CORRECTED: BLVR inspection-only language without action; treating as not performed.")
            return warnings, True

        # Valve table count heuristic (+ foreign body removal when a valve is removed).
        valve_block = ""
        match = re.search(r"(?i)\bvalve\s+sizes\s+used\b", note_text)
        if match:
            valve_block = note_text[match.end() : match.end() + 2000]
            stop = re.search(
                r"(?i)\n(?:bronchial\s+alveolar\s+lavage|the\s+patient\s+tolerated|specimen\\(s\\)|impression/plan|impression|plan)\b",
                valve_block,
            )
            if stop:
                valve_block = valve_block[: stop.start()]

        sizes: list[str] = []
        segments: list[str] = []
        placements: list[dict[str, object]] = []
        removed_valve = False
        pending_segment: str | None = None
        for line in valve_block.splitlines():
            clean = line.strip()
            if not clean:
                continue
            if re.search(r"(?i)^\s*airway\s*\t\s*valve\b", clean):
                continue
            if not re.search(r"(?i)\b(zephyr|spiration)\b", clean):
                continue
            size_match = re.search(r"(?i)\bsize\s*([0-9]+(?:\.[0-9]+)?)\b", clean)
            if not size_match:
                continue

            seg = None
            if "\t" in clean:
                seg = clean.split("\t", 1)[0].strip()
            else:
                marker = re.search(r"(?i)\b(?:olympus|pulmonx|zephyr|spiration)\b", clean)
                if marker:
                    seg = clean[: marker.start()].strip() or None

            is_removed = bool(re.search(r"(?i)\bremoved\b|\bretriev|\bextract|\bexplant", clean))
            if is_removed:
                removed_valve = True
                # Common BLVR template: "Segment ... size 9 placed initially then removed."
                # The next line may be a replacement valve ("... size 7 placed") without a segment token.
                if seg:
                    pending_segment = seg
                continue

            valve_size = size_match.group(1)
            sizes.append(valve_size)
            if not seg and pending_segment:
                seg = pending_segment
            pending_segment = None
            if seg:
                segments.append(seg)
            valve_type: str | None = None
            lower = clean.lower()
            if "zephyr" in lower:
                valve_type = "Zephyr (Pulmonx)"
            elif "spiration" in lower:
                valve_type = "Spiration (Olympus)"
            if valve_type is None and blvr.get("valve_type") in ("Zephyr (Pulmonx)", "Spiration (Olympus)"):
                valve_type = blvr.get("valve_type")
            placements.append(
                {
                    "segment": seg or "",
                    "valve_size": valve_size,
                    "valve_type": valve_type,
                }
            )

        count = len(sizes)
        if count:
            try:
                existing_count = int(blvr.get("number_of_valves")) if blvr.get("number_of_valves") is not None else None
            except Exception:
                existing_count = None
            if existing_count is None or existing_count < count:
                blvr["number_of_valves"] = count
                changed = True
            if not blvr.get("valve_sizes") or len(blvr.get("valve_sizes") or []) < count:
                blvr["valve_sizes"] = sizes
                changed = True
        if segments:
            existing_segments = blvr.get("segments_treated")
            existing_len = len(existing_segments) if isinstance(existing_segments, list) else 0
            if not existing_segments or existing_len < len(segments):
                blvr["segments_treated"] = segments
                changed = True

        # BLVR valve remove/replace is an adjustment (not foreign body removal).
        if removed_valve or removal_indicator:
            foreign = procedures.get("foreign_body_removal")
            if isinstance(foreign, dict) and foreign.get("performed") is True:
                procedures["foreign_body_removal"] = {"performed": False}
                record_data["procedures_performed"] = procedures
                warnings.append(
                    "AUTO_CORRECTED: BLVR valve removal/exchange context; cleared foreign_body_removal (do not derive 31635)."
                )
                changed = True
            if removed_valve:
                warnings.append("BLVR valve removal/exchange noted; do not treat as foreign body removal.")

        # Promote BLVR valve table rows into granular_data.blvr_valve_placements for accurate lobe counting.
        def _infer_target_lobe(segment: str) -> str | None:
            upper = (segment or "").upper()
            if "RUL" in upper or "RIGHT UPPER" in upper:
                return "RUL"
            if "RML" in upper or "RIGHT MIDDLE" in upper:
                return "RML"
            if "RLL" in upper or "RIGHT LOWER" in upper:
                return "RLL"
            if "LUL" in upper or "LEFT UPPER" in upper:
                return "LUL"
            if "LLL" in upper or "LEFT LOWER" in upper:
                return "LLL"
            if "LING" in upper:
                return "Lingula"
            return None

        granular = record_data.get("granular_data")
        if not isinstance(granular, dict):
            granular = {}

        existing_valves = granular.get("blvr_valve_placements")
        if not isinstance(existing_valves, list):
            existing_valves = []

        valve_number = len(existing_valves) + 1
        added_valves = 0
        for placement in placements:
            segment = str(placement.get("segment") or "").strip()
            if not segment:
                continue
            target_lobe = _infer_target_lobe(segment)
            if target_lobe is None:
                continue
            valve_type = placement.get("valve_type")
            if valve_type not in ("Zephyr (Pulmonx)", "Spiration (Olympus)"):
                continue
            valve_size = str(placement.get("valve_size") or "").strip()
            if not valve_size:
                continue

            existing_valves.append(
                {
                    "valve_number": valve_number,
                    "target_lobe": target_lobe,
                    "segment": segment,
                    "valve_size": valve_size,
                    "valve_type": valve_type,
                    "deployment_successful": True,
                }
            )
            valve_number += 1
            added_valves += 1

        if added_valves:
            granular["blvr_valve_placements"] = existing_valves
            record_data["granular_data"] = granular
            changed = True

        # Capture Chartis/balloon occlusion as granular measurements so CPT logic can bundle 31634 correctly.
        if chartis_performed:
            cv_negative = bool(
                re.search(r"(?i)\bno/minimal\s+collateral\s+ventilation\b", note_text)
                or re.search(r"(?i)\bno\s+collateral\s+ventilation\b", note_text)
            )
            cv_positive = bool(re.search(r"(?i)\bcollateral\s+ventilation\b[^.\n]{0,40}\bpresent\b", note_text))
            cv_result = "Indeterminate"
            if cv_negative:
                cv_result = "CV Negative"
            elif cv_positive:
                cv_result = "CV Positive"

            valve_lobes: set[str] = set()
            for valve in existing_valves:
                lobe = valve.get("target_lobe")
                if lobe in {"RUL", "RML", "RLL", "LUL", "LLL", "Lingula"}:
                    valve_lobes.add(lobe)
            if not valve_lobes and isinstance(blvr.get("target_lobe"), str):
                inferred = _infer_target_lobe(blvr.get("target_lobe") or "")
                if inferred:
                    valve_lobes.add(inferred)

            existing_meas = granular.get("blvr_chartis_measurements")
            if not isinstance(existing_meas, list):
                existing_meas = []
            existing_lobes = {
                m.get("lobe_assessed")
                for m in existing_meas
                if isinstance(m, dict) and m.get("lobe_assessed") in {"RUL", "RML", "RLL", "LUL", "LLL", "Lingula"}
            }
            added_meas = 0
            for lobe in sorted(valve_lobes):
                if lobe in existing_lobes:
                    continue
                existing_meas.append({"lobe_assessed": lobe, "cv_result": cv_result})
                added_meas += 1

            if added_meas:
                granular["blvr_chartis_measurements"] = existing_meas
                record_data["granular_data"] = granular
                changed = True

        procedures["blvr"] = blvr
        record_data["procedures_performed"] = procedures
        return warnings, changed

    def _has_action_near(self, text: str, term: str, actions: tuple[str, ...], window: int = 80) -> bool:
        start = 0
        while True:
            idx = text.find(term, start)
            if idx == -1:
                return False
            window_start = max(0, idx - window)
            window_end = min(len(text), idx + len(term) + window)
            window_text = text[window_start:window_end]
            if any(action in window_text for action in actions):
                return True
            start = idx + len(term)

    def _resolve_pleural_device(
        self,
        text: str,
        ipc_present: bool,
        tube_present: bool,
    ) -> str | None:
        if ipc_present and not tube_present:
            return "ipc"
        if tube_present and not ipc_present:
            return "chest_tube"

        ipc_insert = any(self._has_action_near(text, term, _INSERT_TERMS) for term in _IPC_TERMS)
        tube_insert = any(self._has_action_near(text, term, _INSERT_TERMS) for term in _CHEST_TUBE_TERMS)
        ipc_remove = any(self._has_action_near(text, term, _REMOVE_TERMS) for term in _IPC_TERMS)
        tube_remove = any(self._has_action_near(text, term, _REMOVE_TERMS) for term in _CHEST_TUBE_TERMS)

        if ipc_insert and not tube_insert:
            return "ipc"
        if tube_insert and not ipc_insert:
            return "chest_tube"
        if tube_insert and ipc_remove and not ipc_insert:
            return "chest_tube"
        if ipc_insert and tube_remove and not tube_insert:
            return "ipc"

        return None

    def _rigid_header_conflict(self, note_text: str) -> bool:
        if not note_text:
            return False
        lines = note_text.splitlines()
        header = "\n".join(lines[:8]).lower()
        body = "\n".join(lines[8:]).lower()
        if "rigid" not in header:
            return False
        if "rigid" in body:
            return False
        return "flexible" in body

    def _set_procedure_performed(self, record_data: dict[str, Any], proc_name: str, value: bool) -> bool:
        procedures = record_data.get("procedures_performed")
        if not isinstance(procedures, dict):
            procedures = {}
        proc = procedures.get(proc_name)
        if not isinstance(proc, dict):
            proc = {}
        current = proc.get("performed")
        proc["performed"] = value
        procedures[proc_name] = proc
        record_data["procedures_performed"] = procedures
        return current != value

    def _set_stent_action(self, record_data: dict[str, Any], action: str) -> bool:
        procedures = record_data.get("procedures_performed")
        if not isinstance(procedures, dict):
            procedures = {}
        stent = procedures.get("airway_stent")
        if not isinstance(stent, dict):
            stent = {}
        current = stent.get("action")
        stent["performed"] = True
        stent["action"] = action
        if action == "Placement":
            stent["action_type"] = "placement"
        elif action == "Removal":
            stent["action_type"] = "removal"
        elif action == "Revision/Repositioning":
            stent["action_type"] = "revision"
        elif action == "Assessment only":
            stent["action_type"] = "assessment_only"
        if action.lower().startswith("remov"):
            stent["airway_stent_removal"] = True
        procedures["airway_stent"] = stent
        record_data["procedures_performed"] = procedures
        return current != action

    def _set_stent_assessment_only(self, record_data: dict[str, Any]) -> bool:
        procedures = record_data.get("procedures_performed")
        if not isinstance(procedures, dict):
            procedures = {}
        stent = procedures.get("airway_stent")
        if not isinstance(stent, dict):
            stent = {}
        current_action = stent.get("action")
        current_removal = stent.get("airway_stent_removal")
        stent["performed"] = True
        stent["action"] = "Assessment only"
        stent["action_type"] = "assessment_only"
        stent["airway_stent_removal"] = False
        procedures["airway_stent"] = stent
        record_data["procedures_performed"] = procedures
        return current_action != "Assessment only" or current_removal is not False

    def _clear_stent(self, record_data: dict[str, Any]) -> bool:
        procedures = record_data.get("procedures_performed")
        if not isinstance(procedures, dict):
            procedures = {}
        stent = procedures.get("airway_stent")
        if not isinstance(stent, dict):
            stent = {}
        current = stent.get("performed")
        procedures["airway_stent"] = {"performed": False}
        record_data["procedures_performed"] = procedures
        return current is not False or len(stent) > 1

    def _set_pleural_performed(self, record_data: dict[str, Any], proc_name: str, value: bool) -> bool:
        pleural = record_data.get("pleural_procedures")
        if not isinstance(pleural, dict):
            pleural = {}
        proc = pleural.get(proc_name)
        if not isinstance(proc, dict):
            proc = {}
        current = proc.get("performed")
        proc["performed"] = value
        pleural[proc_name] = proc
        record_data["pleural_procedures"] = pleural
        return current != value

    def _clear_pleural_proc(self, record_data: dict[str, Any], proc_name: str) -> bool:
        pleural = record_data.get("pleural_procedures")
        if not isinstance(pleural, dict):
            pleural = {}
        prior = pleural.get(proc_name)
        prior_performed = prior.get("performed") if isinstance(prior, dict) else None
        prior_extra = isinstance(prior, dict) and any(k != "performed" for k in prior.keys())
        pleural[proc_name] = {"performed": False}
        record_data["pleural_procedures"] = pleural
        return prior_performed is not False or prior_extra

    def _set_complication_pneumothorax_occurred(self, record_data: dict[str, Any], value: bool) -> bool:
        complications = record_data.get("complications")
        if not isinstance(complications, dict):
            complications = {}
        pneumothorax = complications.get("pneumothorax")
        if not isinstance(pneumothorax, dict):
            pneumothorax = {}
        current = pneumothorax.get("occurred")
        pneumothorax["occurred"] = value
        complications["pneumothorax"] = pneumothorax
        record_data["complications"] = complications
        return current != value

    def _pleural_procedure_flagged(self, record_data: dict[str, Any]) -> bool:
        pleural = record_data.get("pleural_procedures")
        if not isinstance(pleural, dict):
            return False
        for proc in pleural.values():
            if isinstance(proc, dict) and proc.get("performed") is True:
                return True
        return False

    def _pleural_device_flagged(self, record_data: dict[str, Any]) -> bool:
        pleural = record_data.get("pleural_procedures")
        if not isinstance(pleural, dict):
            return False
        for name in ("ipc", "chest_tube"):
            proc = pleural.get(name)
            if isinstance(proc, dict) and proc.get("performed") is True:
                return True
        return False

    def _linear_station_data_present(self, record_data: dict[str, Any]) -> bool:
        procedures = record_data.get("procedures_performed")
        if not isinstance(procedures, dict):
            return False
        linear = procedures.get("linear_ebus")
        if not isinstance(linear, dict):
            return False
        stations_sampled = linear.get("stations_sampled")
        stations_detail = linear.get("stations_detail")
        station_bucket = linear.get("station_count_bucket")
        if stations_sampled:
            return True
        if stations_detail:
            return True
        if isinstance(station_bucket, str) and station_bucket.strip():
            return True
        return False


__all__ = ["ClinicalGuardrails", "GuardrailOutcome"]
