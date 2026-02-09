from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel

from proc_schemas.clinical import OperativeShellInputs, ProcedureBundle, ProcedureInput

from app.reporting.metadata import ProcedureAutocodeResult, ProcedureMetadata, ReportMetadata, StructuredReport
from app.reporting.normalization.normalize import normalize_bundle

from .engine import (
    _dedupe_labels,
    _embed_metadata,
    _merge_cpt_sources,
    _merge_str_lists,
    _normalize_payload,
    _summarize_cpt_candidates,
    _truthy_env,
    _try_proc_autocode,
)

if TYPE_CHECKING:
    from .engine import ReporterEngine


@dataclass(frozen=True)
class ReporterConfig:
    autocode_result: ProcedureAutocodeResult | None = None
    assume_normalized: bool = False


@dataclass
class _PipelineState:
    include_pre_anesthesia: bool
    include_discharge: bool
    sections: dict[str, list[str]] = field(default_factory=dict)
    procedure_labels: list[str] = field(default_factory=list)
    bronchoscopy_blocks: list[str] = field(default_factory=list)
    bronchoscopy_shells: list[tuple[Any, ProcedureInput, ProcedureMetadata]] = field(default_factory=list)
    discharge_templates: dict[str, list[ProcedureMetadata]] = field(default_factory=dict)
    procedures_metadata: list[ProcedureMetadata] = field(default_factory=list)
    autocode_payload: dict[str, Any] | ProcedureAutocodeResult | None = None
    autocode_modifiers: list[str] = field(default_factory=list)
    unmatched_autocode: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class _TemplateSelection:
    sorted_procs: list[ProcedureInput]
    has_bronchoscopy_shell: bool
    paired_surveys: dict[str, ProcedureInput]
    reserved_surveys: set[str]
    first_by_type: dict[str, ProcedureInput]


class ReportPipeline:
    def __init__(self, engine: ReporterEngine, *, config: ReporterConfig | None = None):
        self.engine = engine
        self.config = config or ReporterConfig()

    def run(self, bundle: ProcedureBundle, *, strict: bool, embed_metadata: bool) -> StructuredReport:
        bundle = self._normalize(bundle)
        state = self._build_metadata(bundle, strict=strict)
        selected = self._select_templates(bundle, state)
        self._render_sections(bundle, state, selected)
        self._render_discharge(bundle, state)
        note, metadata = self._assemble(bundle, state, selected, strict=strict)
        output_text = _embed_metadata(note, metadata) if embed_metadata else note
        return StructuredReport(
            text=output_text,
            metadata=metadata,
            warnings=[],
            issues=[],
        )

    def _normalize(self, bundle: ProcedureBundle) -> ProcedureBundle:
        if self.config.assume_normalized:
            return bundle
        return normalize_bundle(bundle).bundle

    def _build_metadata(self, bundle: ProcedureBundle, *, strict: bool) -> _PipelineState:
        self.engine._strict_render = strict

        include_pre_anesthesia = _truthy_env("REPORTER_INCLUDE_PRE_ANESTHESIA_ASSESSMENT", "0")
        include_discharge = _truthy_env("REPORTER_INCLUDE_DISCHARGE_INSTRUCTIONS", "0")
        state = _PipelineState(
            include_pre_anesthesia=include_pre_anesthesia,
            include_discharge=include_discharge,
            sections={
                "HEADER": [],
                "PRE_ANESTHESIA": [],
                "PROCEDURE_DETAILS": [],
                "INSTRUCTIONS": [],
                "DISCHARGE": [],
            },
        )

        autocode_payload = self.config.autocode_result or _try_proc_autocode(bundle)
        autocode_codes = [str(code) for code in autocode_payload.get("cpt", [])] if autocode_payload else []
        autocode_modifiers = [str(mod) for mod in autocode_payload.get("modifiers", [])] if autocode_payload else []
        state.autocode_payload = autocode_payload
        state.autocode_modifiers = autocode_modifiers
        state.unmatched_autocode = set(autocode_codes)

        if include_pre_anesthesia:
            pre_meta = self.engine.templates.get("ip_pre_anesthesia_assessment")
            if pre_meta and bundle.pre_anesthesia:
                rendered = self.engine._render_payload(pre_meta, bundle.pre_anesthesia, bundle)
                if rendered:
                    state.sections["PRE_ANESTHESIA"].append(rendered)

        return state

    def _select_templates(self, bundle: ProcedureBundle, state: _PipelineState) -> _TemplateSelection:
        sorted_procs = self.engine._sorted_procedures(bundle.procedures, source_text=bundle.free_text_hint)

        has_robotic_nav = any(proc.proc_type == "robotic_navigation" for proc in sorted_procs)
        if has_robotic_nav:
            sorted_procs = [
                proc
                for proc in sorted_procs
                if proc.proc_type not in ("robotic_monarch_bronchoscopy", "robotic_ion_bronchoscopy")
            ]

        has_medical_thoracoscopy = any(proc.proc_type == "medical_thoracoscopy" for proc in sorted_procs)
        if has_medical_thoracoscopy:
            sorted_procs = [proc for proc in sorted_procs if proc.proc_type != "chest_tube"]

        has_bronchoscopy_shell = any(
            any(meta.id == "ip_general_bronchoscopy_shell" for meta in self.engine.templates.find_for_procedure(proc.proc_type, proc.cpt_candidates))
            for proc in sorted_procs
        )

        survey_procs = [p for p in sorted_procs if p.proc_type == "radial_ebus_survey"]
        sampling_procs = [p for p in sorted_procs if p.proc_type == "radial_ebus_sampling"]
        paired_surveys: dict[str, ProcedureInput] = {}
        reserved_surveys: set[str] = set()
        survey_iter = iter(survey_procs)
        for sampling in sampling_procs:
            survey = next(survey_iter, None)
            if not survey:
                break
            key = sampling.proc_id or sampling.schema_id
            paired_surveys[key] = survey
            reserved_surveys.add(survey.proc_id or survey.schema_id)

        first_by_type: dict[str, ProcedureInput] = {}
        for proc in sorted_procs:
            first_by_type.setdefault(proc.proc_type, proc)

        return _TemplateSelection(
            sorted_procs=sorted_procs,
            has_bronchoscopy_shell=has_bronchoscopy_shell,
            paired_surveys=paired_surveys,
            reserved_surveys=reserved_surveys,
            first_by_type=first_by_type,
        )

    def _render_sections(self, bundle: ProcedureBundle, state: _PipelineState, selected: _TemplateSelection) -> None:
        include_discharge = state.include_discharge
        autocode_payload = state.autocode_payload
        autocode_modifiers = state.autocode_modifiers
        unmatched_autocode = state.unmatched_autocode

        def _validated(proc_input: ProcedureInput) -> Any:
            if isinstance(proc_input.data, BaseModel):
                return proc_input.data
            try:
                model_cls = self.engine.schemas.get(proc_input.schema_id)
                return model_cls.model_validate(proc_input.data or {})
            except Exception:
                return proc_input.data

        for proc in selected.sorted_procs:
            if proc.proc_type == "radial_ebus_survey" and (proc.proc_id or proc.schema_id) in selected.reserved_surveys:
                continue
            metas = self.engine.templates.find_for_procedure(proc.proc_type, proc.cpt_candidates)
            label = metas[0].label if metas else proc.proc_type
            section = metas[0].output_section if metas else ""
            proc_meta = ProcedureMetadata(
                proc_id=proc.proc_id or proc.schema_id,
                proc_type=proc.proc_type,
                label=label,
                cpt_candidates=[],
                icd_candidates=[],
                modifiers=[],
                section=section,
                templates_used=[],
                has_critical_missing=False,
                missing_critical_fields=[],
                extra={"data": _normalize_payload(proc.data)},
            )
            state.procedures_metadata.append(proc_meta)

            if not metas:
                proc_meta.has_critical_missing = True
                proc_meta.missing_critical_fields.append("template_missing")
                continue

            for meta in metas:
                if not proc_meta.label:
                    proc_meta.label = meta.label
                if not proc_meta.section:
                    proc_meta.section = meta.output_section

                if meta.id == "ip_general_bronchoscopy_shell":
                    cpts, modifiers = _merge_cpt_sources(proc, meta, autocode_payload)
                    proc_meta.cpt_candidates = _merge_str_lists(proc_meta.cpt_candidates, cpts)
                    proc_meta.modifiers = _merge_str_lists(proc_meta.modifiers, modifiers or autocode_modifiers)
                    proc_meta.icd_candidates = _merge_str_lists(
                        proc_meta.icd_candidates, autocode_payload.get("icd", []) if autocode_payload else []
                    )
                    proc_meta.templates_used = _merge_str_lists(proc_meta.templates_used, [meta.id])
                    for code in cpts:
                        unmatched_autocode.discard(code)
                    state.bronchoscopy_shells.append((meta, proc, proc_meta))
                    continue

                # Track discharge/instructions attachments based on procedures
                if include_discharge:
                    if meta.id in ("tunneled_pleural_catheter_insert", "ipc_insert"):
                        state.discharge_templates.setdefault("pleurx_instructions", []).append(proc_meta)
                    if meta.id == "blvr_valve_placement":
                        state.discharge_templates.setdefault("blvr_discharge_instructions", []).append(proc_meta)
                    if meta.id in ("chest_tube", "pigtail_catheter"):
                        state.discharge_templates.setdefault("chest_tube_discharge", []).append(proc_meta)
                    if meta.id in ("peg_placement",):
                        state.discharge_templates.setdefault("peg_discharge", []).append(proc_meta)

                extra_context: dict[str, Any] | None = None
                if proc.proc_type == "radial_ebus_sampling":
                    survey_proc = selected.paired_surveys.get(proc.proc_id or proc.schema_id)
                    if survey_proc:
                        extra_context = {"survey": _validated(survey_proc)}
                    nav_proc = selected.first_by_type.get("robotic_navigation")
                    if nav_proc:
                        extra_context = extra_context or {}
                        extra_context["nav"] = _validated(nav_proc)
                if proc.proc_type == "robotic_navigation":
                    extra_context = extra_context or {}
                    survey_proc = selected.first_by_type.get("radial_ebus_survey")
                    if survey_proc:
                        extra_context["survey"] = _validated(survey_proc)
                    sampling_proc = selected.first_by_type.get("radial_ebus_sampling")
                    if sampling_proc:
                        extra_context["sampling"] = _validated(sampling_proc)

                rendered = self.engine._render_procedure_template(meta, proc, bundle, extra_context=extra_context)
                proc_meta.templates_used = _merge_str_lists(proc_meta.templates_used, [meta.id])
                cpts, modifiers = _merge_cpt_sources(proc, meta, autocode_payload)
                proc_meta.cpt_candidates = _merge_str_lists(proc_meta.cpt_candidates, cpts)
                proc_meta.modifiers = _merge_str_lists(proc_meta.modifiers, modifiers or autocode_modifiers)
                proc_meta.icd_candidates = _merge_str_lists(
                    proc_meta.icd_candidates, autocode_payload.get("icd", []) if autocode_payload else []
                )
                for code in cpts:
                    unmatched_autocode.discard(code)

                if not rendered:
                    continue
                if meta.output_section == "PROCEDURE_DETAILS":
                    state.procedure_labels.append(meta.label)
                if meta.category == "bronchoscopy" and selected.has_bronchoscopy_shell:
                    state.bronchoscopy_blocks.append(rendered)
                else:
                    state.sections.setdefault(meta.output_section, []).append(rendered)

        if state.bronchoscopy_blocks:
            joined_bronch = self.engine._join_blocks(state.bronchoscopy_blocks)
            if state.bronchoscopy_shells:
                for meta, proc, proc_meta in state.bronchoscopy_shells:
                    cpts, modifiers = _merge_cpt_sources(proc, meta, autocode_payload)
                    proc_meta.cpt_candidates = _merge_str_lists(proc_meta.cpt_candidates, cpts)
                    proc_meta.modifiers = _merge_str_lists(proc_meta.modifiers, modifiers or autocode_modifiers)
                    proc_meta.icd_candidates = _merge_str_lists(
                        proc_meta.icd_candidates, autocode_payload.get("icd", []) if autocode_payload else []
                    )
                    proc_meta.templates_used = _merge_str_lists(proc_meta.templates_used, [meta.id])
                    for code in cpts:
                        unmatched_autocode.discard(code)
                    rendered = self.engine._render_procedure_template(
                        meta,
                        proc,
                        bundle,
                        extra_context={
                            "procedure_details": joined_bronch,
                            "procedures_summary": ", ".join(_dedupe_labels(state.procedure_labels)),
                        },
                    )
                    if rendered:
                        if meta.output_section == "PROCEDURE_DETAILS":
                            state.procedure_labels.append(meta.label)
                        state.sections.setdefault(meta.output_section, []).append(rendered)
            else:
                state.sections["PROCEDURE_DETAILS"].append(joined_bronch)

    def _render_discharge(self, bundle: ProcedureBundle, state: _PipelineState) -> None:
        if not state.include_discharge:
            return
        for discharge_id, owners in state.discharge_templates.items():
            discharge_meta = self.engine.templates.get(discharge_id)
            if discharge_meta:
                rendered = self.engine._render_payload(discharge_meta, {}, bundle)
                if rendered:
                    state.sections.setdefault(discharge_meta.output_section, []).append(rendered)
                    for owner in owners:
                        owner.templates_used = _merge_str_lists(owner.templates_used, [discharge_id])

    def _assemble(
        self,
        bundle: ProcedureBundle,
        state: _PipelineState,
        selected: _TemplateSelection,
        *,
        strict: bool,
    ) -> tuple[str, ReportMetadata]:
        autocode_payload = state.autocode_payload
        unmatched_autocode = state.unmatched_autocode

        shell = self.engine.templates.get(self.engine.shell_template_id) if self.engine.shell_template_id else None
        if shell:
            procedure_details_block = self.engine._join_blocks(
                state.sections.get("PRE_ANESTHESIA", [])
                + state.sections.get("PROCEDURE_DETAILS", [])
                + state.sections.get("INSTRUCTIONS", [])
                + state.sections.get("DISCHARGE", [])
            )

            sorted_procs = selected.sorted_procs
            procedure_labels = state.procedure_labels

            def _build_procedure_summary() -> str:
                # Prefer a golden-style, line-oriented summary for navigation cases.
                by_type: dict[str, ProcedureInput] = {}
                for proc in sorted_procs:
                    by_type.setdefault(proc.proc_type, proc)

                note_text = (bundle.free_text_hint or "").strip()
                note_upper = note_text.upper()

                lines: list[str] = []

                nav_target = None
                nav_platform = None

                tpc_proc = by_type.get("tunneled_pleural_catheter_insert")
                if tpc_proc is not None:
                    data = (
                        tpc_proc.data.model_dump(exclude_none=True)
                        if isinstance(tpc_proc.data, BaseModel)
                        else (tpc_proc.data or {})
                    )
                    side = str(data.get("side") or "").strip().lower()
                    side_title = side.title() if side in {"left", "right"} else ""
                    line = "Indwelling Tunneled Pleural Catheter Placement"
                    if side_title:
                        line += f" ({side_title})"
                    lines.append(line)
                    lines.append("Thoracic Ultrasound")

                def _as_text(value: Any) -> str:
                    if value is None:
                        return ""
                    if isinstance(value, str):
                        return value.strip()
                    if isinstance(value, (int, float)):
                        return str(value)
                    if isinstance(value, list):
                        parts = [str(item).strip() for item in value if item not in (None, "")]
                        return ", ".join([p for p in parts if p])
                    return str(value).strip()

                def _normalize_rebus(value: Any) -> str:
                    text = _as_text(value)
                    if not text:
                        return ""
                    lowered = text.lower()
                    if "concentric" in lowered:
                        return "Concentric"
                    if "eccentric" in lowered:
                        return "Eccentric"
                    return text

                nav_proc = by_type.get("robotic_navigation")
                if nav_proc is not None:
                    data = nav_proc.data.model_dump(exclude_none=True) if isinstance(nav_proc.data, BaseModel) else (nav_proc.data or {})
                    nav_platform = data.get("platform")
                    nav_target = data.get("lesion_location") or data.get("target_lung_segment")
                    base = "Robotic navigational bronchoscopy"
                    if nav_platform:
                        base += f" ({nav_platform})"
                    if nav_target:
                        base += f" to {nav_target} target"
                    lines.append(base)

                emn_proc = by_type.get("emn_bronchoscopy")
                if emn_proc is not None and not lines:
                    data = emn_proc.data.model_dump(exclude_none=True) if isinstance(emn_proc.data, BaseModel) else (emn_proc.data or {})
                    nav_platform = data.get("navigation_system") or "EMN"
                    nav_target = data.get("target_lung_segment")
                    base = "Electromagnetic Navigation Bronchoscopy"
                    if nav_platform:
                        base += f" ({nav_platform})"
                    if nav_target:
                        base += f" to {nav_target} target"
                    lines.append(base)

                til_proc = by_type.get("tool_in_lesion_confirmation")
                if til_proc is not None:
                    data = (
                        til_proc.data.model_dump(exclude_none=True)
                        if isinstance(til_proc.data, BaseModel)
                        else (til_proc.data or {})
                    )
                    method = _as_text(data.get("confirmation_method"))
                    if method and "tilt" in method.lower():
                        lines.append("TiLT+ (Tomosynthesis-based Tool-in-Lesion Tomography) with trajectory adjustment")
                    else:
                        lines.append("Tool-in-lesion confirmation")

                radial_survey = by_type.get("radial_ebus_survey")
                if radial_survey is not None:
                    data = (
                        radial_survey.data.model_dump(exclude_none=True)
                        if isinstance(radial_survey.data, BaseModel)
                        else (radial_survey.data or {})
                    )
                    pattern = _normalize_rebus(data.get("rebus_features"))
                    line = "rEBUS localization"
                    if pattern:
                        line += f" ({pattern})"
                    lines.append(line)

                cryo_proc = by_type.get("transbronchial_cryobiopsy")
                if cryo_proc is not None:
                    data = (
                        cryo_proc.data.model_dump(exclude_none=True)
                        if isinstance(cryo_proc.data, BaseModel)
                        else (cryo_proc.data or {})
                    )
                    seg = _as_text(data.get("lung_segment"))
                    lines.append(f"Transbronchial Cryobiopsy ({seg})" if seg else "Transbronchial Cryobiopsy")

                blocker_proc = by_type.get("endobronchial_blocker")
                if blocker_proc is not None:
                    data = (
                        blocker_proc.data.model_dump(exclude_none=True)
                        if isinstance(blocker_proc.data, BaseModel)
                        else (blocker_proc.data or {})
                    )
                    blocker = _as_text(data.get("blocker_type"))
                    if blocker:
                        lines.append(f"Prophylactic {blocker} balloon placement")
                    else:
                        lines.append("Prophylactic endobronchial blocker placement")

                radial_sampling = by_type.get("radial_ebus_sampling")
                if radial_sampling is not None:
                    data = (
                        radial_sampling.data.model_dump(exclude_none=True)
                        if isinstance(radial_sampling.data, BaseModel)
                        else (radial_sampling.data or {})
                    )
                    view = _normalize_rebus(data.get("ultrasound_pattern"))
                    line = "Radial EBUS"
                    if view:
                        line += f" ({view} view)"
                    lines.append(line)

                if "CONE BEAM" in note_upper or "CBCT" in note_upper:
                    lines.append("Cone-beam CT imaging with trajectory adjustment and confirmation")
                elif "FLUORO" in note_upper:
                    lines.append("Fluoroscopy with trajectory adjustment and confirmation")

                tbna_proc = by_type.get("transbronchial_needle_aspiration")
                if tbna_proc is not None:
                    data = (
                        tbna_proc.data.model_dump(exclude_none=True)
                        if isinstance(tbna_proc.data, BaseModel)
                        else (tbna_proc.data or {})
                    )
                    passes = data.get("samples_collected")
                    target = nav_target or data.get("lung_segment")
                    if passes:
                        line = f"TBNA of {target or 'target'} ({passes} passes)"
                    else:
                        line = "TBNA"
                    lines.append(line)

                bx_proc = by_type.get("transbronchial_biopsy")
                if bx_proc is not None:
                    data = (
                        bx_proc.data.model_dump(exclude_none=True)
                        if isinstance(bx_proc.data, BaseModel)
                        else (bx_proc.data or {})
                    )
                    count = data.get("number_of_biopsies")
                    target = nav_target or data.get("lobe") or data.get("segment")
                    if count:
                        line = f"Transbronchial biopsy of {target or 'target'} ({count} samples)"
                    else:
                        line = "Transbronchial biopsy"
                    lines.append(line)

                brush_proc = by_type.get("bronchial_brushings")
                if brush_proc is not None:
                    lines.append("Bronchial Brush")

                wll_proc = by_type.get("whole_lung_lavage")
                if wll_proc is not None:
                    data = (
                        wll_proc.data.model_dump(exclude_none=True)
                        if isinstance(wll_proc.data, BaseModel)
                        else (wll_proc.data or {})
                    )
                    side = str(data.get("side") or "").strip().lower()
                    side_title = side.title() if side in {"left", "right"} else ""
                    total_l = data.get("total_volume_l") or data.get("max_volume_l")
                    total_str = ""
                    if total_l not in (None, "", [], {}):
                        try:
                            total_f = float(total_l)
                            total_str = str(int(total_f)) if total_f.is_integer() else str(round(total_f, 1))
                        except Exception:
                            total_str = str(total_l)
                    line = "Whole Lung Lavage"
                    if side_title and total_str:
                        line += f" ({side_title}, {total_str} L)"
                    elif side_title:
                        line += f" ({side_title})"
                    elif total_str:
                        line += f" ({total_str} L)"
                    lines.append(line)

                bal_proc = by_type.get("bal")
                if bal_proc is not None:
                    data = bal_proc.data.model_dump(exclude_none=True) if isinstance(bal_proc.data, BaseModel) else (bal_proc.data or {})
                    seg = data.get("lung_segment")
                    if seg:
                        lines.append(f"Bronchoalveolar Lavage ({seg})")
                    else:
                        lines.append("Bronchoalveolar Lavage (BAL)")

                fid_proc = by_type.get("fiducial_marker_placement")
                if fid_proc is not None:
                    lines.append("Fiducial marker placement")

                ebus_proc = by_type.get("ebus_tbna")
                if ebus_proc is not None:
                    data = ebus_proc.data.model_dump(exclude_none=True) if isinstance(ebus_proc.data, BaseModel) else (ebus_proc.data or {})
                    stations = []
                    for st in data.get("stations") or []:
                        if isinstance(st, dict) and st.get("station_name"):
                            stations.append(str(st["station_name"]))
                    stations = _dedupe_labels([s for s in stations if s])
                    if stations:
                        lines.append(
                            "Endobronchial Ultrasound-Guided Transbronchial Needle Aspiration (EBUS-TBNA) "
                            f"(Stations {', '.join(stations)})"
                        )
                    else:
                        lines.append("Endobronchial Ultrasound-Guided Transbronchial Needle Aspiration (EBUS-TBNA)")

                ablation_proc = by_type.get("peripheral_ablation")
                if ablation_proc is not None:
                    data = (
                        ablation_proc.data.model_dump(exclude_none=True)
                        if isinstance(ablation_proc.data, BaseModel)
                        else (ablation_proc.data or {})
                    )
                    modality = data.get("modality") or "Microwave"
                    target = data.get("target") or nav_target
                    lines.append(f"{modality} Ablation of {target or 'target'} target")

                if lines:
                    return "\n".join(_dedupe_labels([str(line).strip() for line in lines if str(line).strip()]))

                # Fallback: use template labels we actually rendered.
                return "\n".join(_dedupe_labels(procedure_labels)) if procedure_labels else "See procedure details below"

            label_summary = _build_procedure_summary()
            cpt_summary = _summarize_cpt_candidates(state.procedures_metadata, unmatched_autocode)
            shell_payload = OperativeShellInputs(
                indication_text=bundle.indication_text,
                preop_diagnosis_text=bundle.preop_diagnosis_text,
                postop_diagnosis_text=bundle.postop_diagnosis_text,
                procedures_summary=label_summary,
                cpt_summary=cpt_summary,
                estimated_blood_loss=bundle.estimated_blood_loss,
                complications_text=bundle.complications_text,
                specimens_text=bundle.specimens_text,
                impression_plan=bundle.impression_plan,
            )
            shell_context = {
                "procedure_details_block": procedure_details_block,
                "procedure_types": [p.proc_type for p in sorted_procs],
                "free_text_hint": bundle.free_text_hint,
            }
            rendered = self.engine._render_payload(shell, shell_payload, bundle, extra_context=shell_context)
            if strict:
                self.engine._validate_style(rendered)
            metadata = self.engine._build_metadata(bundle, state.procedures_metadata, autocode_payload)
            return rendered, metadata

        note = self.engine._join_sections(state.sections)
        if strict:
            self.engine._validate_style(note)
        metadata = self.engine._build_metadata(bundle, state.procedures_metadata, autocode_payload)
        return note, metadata
