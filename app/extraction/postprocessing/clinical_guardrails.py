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
    r"\b(convex|mediastinal|station\s*\d{1,2}[A-Za-z]?)\b",
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
        r"\b(?:refus(?:ed|al)|reluctan(?:t|ce)|hesitan(?:t|cy)|did\s+not\s+want)\b"
        r"[^.\n]{0,80}\bstent(?:s)?\b[^.\n]{0,80}\b(?:place|placed|placement|insert|inserted|deploy|deployed)\b",
        re.IGNORECASE,
    ),
]

_STENT_PLACEMENT_CONTEXT_RE = re.compile(
    r"\b(?:stent\b[^.\n]{0,30}\b(place|placed|deploy|deployed|insert|inserted|advance|advanced|seat|seated|expand|expanded|expanding)\b"
    r"|(place|placed|deploy|deployed|insert|inserted|advance|advanced|seat|seated|expand|expanded|expanding)\b[^.\n]{0,30}\bstent\b)\b",
    re.IGNORECASE,
)
_STENT_PLACEMENT_ACTION_CONTEXT_RE = re.compile(
    r"\b(?:stent\b[^.\n]{0,30}\b(place|placed|deploy|deployed|insert|inserted|advance|advanced|expand|expanded|expanding)\b"
    r"|(place|placed|deploy|deployed|insert|inserted|advance|advanced|expand|expanded|expanding)\b[^.\n]{0,30}\bstent\b)\b",
    re.IGNORECASE,
)
_STENT_STRONG_PLACEMENT_RE = re.compile(
    r"\b(?:stent\b[^.\n]{0,30}\b(deploy|deployed|insert|inserted|advance|advanced|seat|seated|expand|expanded|expanding)\b"
    r"|(deploy|deployed|insert|inserted|advance|advanced|seat|seated|expand|expanded|expanding)\b[^.\n]{0,30}\bstent\b)\b",
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

        # Stent negation and inspection-only guardrails.
        procedures = record_data.get("procedures_performed")
        stent = procedures.get("airway_stent") if isinstance(procedures, dict) else None
        if isinstance(stent, dict) and stent.get("performed") is True:
            negated = any(p.search(text_lower) for p in _STENT_NEGATION_PATTERNS)
            removal_text_present = bool(_STENT_REMOVAL_CONTEXT_RE.search(text_lower))
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

            # Revision semantics: don't label revision/repositioning as "removal" without explicit removal language.
            action_text = str(stent.get("action") or "").strip().lower()
            revision_action = "revision" in action_text or "reposition" in action_text
            if revision_action and removal_flag and not removal_text_present:
                stent["airway_stent_removal"] = False
                warnings.append("Stent revision/repositioning without removal language; clearing airway_stent_removal.")
                changed = True
                removal_flag = False
                removal_present = removal_text_present

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
            elif removal_text_present and not placement_present and not strong_placement:
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
            if not special_intubation_context:
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
        if ipc_checkbox is False:
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
                    changed |= self._set_pleural_performed(record_data, "ipc", False)
                    warnings.append("IPC discontinue/removal language; treating as not performed.")
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
            if not any(candidate in label for candidate in candidates):
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
        placement_indicator = bool(
            re.search(r"(?i)\bvalves?\b[^.\n]{0,80}\bplaced\b", note_text)
            or re.search(r"(?i)\bvalve\s+sizes\s+used\b", note_text)
        )
        if placement_indicator and blvr.get("procedure_type") in (None, "", "Valve assessment"):
            blvr["procedure_type"] = "Valve placement"
            changed = True

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
            is_removed = bool(re.search(r"(?i)\bremoved\b|\bretriev|\bextract|\bexplant", clean))
            if is_removed:
                removed_valve = True
                continue

            valve_size = size_match.group(1)
            sizes.append(valve_size)
            seg = None
            if "\t" in clean:
                seg = clean.split("\t", 1)[0].strip()
            else:
                marker = re.search(r"(?i)\b(?:olympus|pulmonx|zephyr|spiration)\b", clean)
                if marker:
                    seg = clean[: marker.start()].strip() or None
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
            if segments and not blvr.get("segments_treated"):
                blvr["segments_treated"] = segments
                changed = True

        if removed_valve:
            if self._set_procedure_performed(record_data, "foreign_body_removal", True):
                warnings.append("BLVR valve removal noted; setting foreign_body_removal.performed=true")
                changed = True

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
