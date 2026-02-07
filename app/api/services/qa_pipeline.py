"""QA Pipeline Service - orchestrates registry, reporter, and coder app.

This service layer extracts the business logic from the /qa/run endpoint
into a testable, reusable service with proper error handling.
"""

from __future__ import annotations

import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from app.common.spans import Span

logger = logging.getLogger(__name__)


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


class ReporterException(RuntimeError):
    """Explicit reporter failure used to control fallback behavior."""


def _serialize_jsonable(value: Any) -> Any:
    """Best-effort JSON-serializable conversion.

    Supports:
    - dataclasses (via asdict)
    - Pydantic models (via model_dump)
    - nested dict/list structures
    """
    if value is None:
        return None
    if is_dataclass(value):
        # asdict() recursively converts nested dataclasses.
        return asdict(value)
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            # Fall through to best-effort stringification below.
            pass
    if isinstance(value, dict):
        return {k: _serialize_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_jsonable(v) for v in value]
    return value


@dataclass
class ModuleOutcome:
    """Internal representation of a module execution result.

    Attributes:
        ok: Whether the module executed successfully
        data: Module output data (dict)
        error_code: Machine-readable error code on failure
        error_message: Human-readable error description on failure
        skipped: Whether the module was skipped (not requested)
    """

    ok: bool = False
    data: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    skipped: bool = False


@dataclass
class QAPipelineResult:
    """Aggregated result from pipeline execution.

    Attributes:
        registry: Registry module outcome
        reporter: Reporter module outcome
        coder: Coder module outcome
    """

    registry: ModuleOutcome = field(default_factory=ModuleOutcome)
    reporter: ModuleOutcome = field(default_factory=ModuleOutcome)
    coder: ModuleOutcome = field(default_factory=ModuleOutcome)


class SimpleReporterStrategy:
    """Simple reporter using text-based composition.

    Falls back to compose_report_from_text when registry data is unavailable.
    """

    def render(
        self, text: str, procedure_type: str | None = None
    ) -> dict[str, Any]:
        """Generate a simple text-based report.

        Args:
            text: Raw procedure note text
            procedure_type: Optional procedure type hint

        Returns:
            Dict with markdown, procedure_core, indication, postop
        """
        from app.reporting.engine import compose_report_from_text

        hints: dict[str, Any] = {}
        if procedure_type:
            hints["procedure_type"] = procedure_type

        report, markdown = compose_report_from_text(text, hints)
        proc_core = report.procedure_core

        return {
            "markdown": markdown,
            "procedure_core": (
                proc_core.model_dump() if hasattr(proc_core, "model_dump") else {}
            ),
            "indication": report.indication,
            "postop": report.postop,
            "fallback_used": True,
        }


class ReportingStrategy:
    """Registry-aware reporting strategy.

    Uses structured registry data when available, falls back to simple
    reporter when registry extraction fails or is unavailable.
    """

    def __init__(
        self,
        reporter_engine: Any,
        inference_engine: Any,
        validation_engine: Any,
        registry_engine: Any,
        simple_strategy: SimpleReporterStrategy,
    ):
        """Initialize reporting strategy.

        Args:
            reporter_engine: ReporterEngine for structured reports
            inference_engine: InferenceEngine for bundle enrichment
            validation_engine: ValidationEngine for issue detection
            registry_engine: RegistryEngine for fallback extraction
            simple_strategy: Fallback strategy for text-only reports
        """
        self.reporter_engine = reporter_engine
        self.inference_engine = inference_engine
        self.validation_engine = validation_engine
        self.registry_engine = registry_engine
        self.simple_strategy = simple_strategy

    def render(
        self,
        text: str,
        registry_data: dict[str, Any] | None = None,
        procedure_type: str | None = None,
    ) -> dict[str, Any]:
        """Render a procedure report.

        If registry_data is provided and contains a record, generates a
        structured report with bundle, issues, and warnings.

        If registry_data is None or empty, attempts a lightweight registry
        extraction first. If that fails, falls back to simple text-based
        report generation.

        Args:
            text: Raw procedure note text
            registry_data: Optional pre-computed registry data
            procedure_type: Optional procedure type hint

        Returns:
            Dict with report data (markdown, bundle/issues/warnings or
            procedure_core/indication/postop for fallback)
        """

        allow_fallback = _truthy_env("QA_REPORTER_ALLOW_SIMPLE_FALLBACK")
        lightweight_error: str | None = None

        # Case 1: We have registry data with a record
        if registry_data and registry_data.get("record"):
            try:
                return self._render_structured(registry_data["record"], source_text=text)
            except Exception as e:
                if allow_fallback:
                    logger.warning(
                        f"Structured reporter failed, falling back to simple: {e}"
                    )
                    fallback = self.simple_strategy.render(text, procedure_type)
                    fallback.update(
                        {
                            "render_mode": "simple_fallback",
                            "fallback_reason": "structured_reporter_failed",
                            "reporter_errors": [str(e)],
                            "fallback_used": True,
                        }
                    )
                    return fallback
                raise ReporterException("Structured reporter failed") from e

        # Case 2: No registry data - try lightweight extraction
        try:
            logger.debug("Running lightweight registry extraction for reporter")
            reg_result = self.registry_engine.run(text, explain=False)
            if isinstance(reg_result, tuple):
                reg_record, _ = reg_result
            else:
                reg_record = reg_result

            reg_dict = (
                reg_record.model_dump()
                if hasattr(reg_record, "model_dump")
                else {}
            )

            if reg_dict:
                return self._render_structured(reg_dict, source_text=text)
        except Exception as e:
            lightweight_error = str(e)
            if allow_fallback:
                logger.warning(
                    f"Lightweight registry extraction failed for reporter: {e}"
                )
            else:
                raise ReporterException("Lightweight registry extraction failed") from e

        # Case 3: All structured approaches failed - use simple reporter
        if not allow_fallback:
            raise ReporterException(
                "Structured reporter unavailable and QA_REPORTER_ALLOW_SIMPLE_FALLBACK is disabled"
            )

        fallback = self.simple_strategy.render(text, procedure_type)
        fallback.update(
            {
                "render_mode": "simple_fallback",
                "fallback_reason": "structured_reporter_unavailable",
                "reporter_errors": [lightweight_error] if lightweight_error else [],
                "fallback_used": True,
            }
        )
        return fallback

    def _render_structured(self, record: dict[str, Any], *, source_text: str | None = None) -> dict[str, Any]:
        """Render structured report from registry record.

        Args:
            record: Registry extraction record dict

        Returns:
            Dict with markdown, bundle, issues, warnings
        """
        from app.reporting.engine import (
            apply_patch_result,
            build_procedure_bundle_from_extraction,
        )

        # Build bundle from registry extraction
        bundle = build_procedure_bundle_from_extraction(record, source_text=source_text)

        # Run inference to enrich the bundle
        inference_result = self.inference_engine.infer_bundle(bundle)
        bundle = apply_patch_result(bundle, inference_result)

        # Validate and get issues/warnings
        issues = self.validation_engine.list_missing_critical_fields(bundle)
        warnings = self.validation_engine.apply_warn_if_rules(bundle)

        # Generate structured report
        structured = self.reporter_engine.compose_report_with_metadata(
            bundle,
            strict=False,
            embed_metadata=False,
            validation_issues=issues,
            warnings=warnings,
        )

        return {
            "markdown": structured.text,
            "bundle": bundle.model_dump() if hasattr(bundle, "model_dump") else {},
            "issues": _serialize_jsonable(issues) if issues else [],
            "warnings": warnings,
            "fallback_used": False,
            "render_mode": "structured",
            "fallback_reason": None,
            "reporter_errors": [],
        }


class QAPipelineService:
    """Orchestrates the QA pipeline: registry -> reporter -> coder.

    This service coordinates the execution of registry extraction,
    report generation, and code suggestion app.
    """

    def __init__(
        self,
        registry_engine: Any,
        reporting_strategy: ReportingStrategy,
        coding_service: Any,
    ):
        """Initialize the QA pipeline service.

        Args:
            registry_engine: RegistryEngine for procedure extraction
            reporting_strategy: Strategy for report generation
            coding_service: CodingService for code suggestions
        """
        self.registry_engine = registry_engine
        self.reporting_strategy = reporting_strategy
        self.coding_service = coding_service

    def run_pipeline(
        self,
        text: str,
        modules: str = "all",
        procedure_type: str | None = None,
    ) -> QAPipelineResult:
        """Execute the QA pipeline on procedure note text.

        Args:
            text: Raw procedure note text
            modules: Which modules to run ("all", "registry", "reporter", "coder")
            procedure_type: Optional procedure type hint

        Returns:
            QAPipelineResult with outcomes for each module
        """
        result = QAPipelineResult()

        # Determine which modules to run
        run_registry = modules in ("registry", "all")
        run_reporter = modules in ("reporter", "all")
        run_coder = modules in ("coder", "all")

        # Mark skipped modules
        if not run_registry:
            result.registry = ModuleOutcome(skipped=True)
        if not run_reporter:
            result.reporter = ModuleOutcome(skipped=True)
        if not run_coder:
            result.coder = ModuleOutcome(skipped=True)

        registry_data: dict[str, Any] | None = None

        # Overlap independent work (registry + coder) to reduce wall-clock latency.
        registry_future = None
        coder_future = None

        with ThreadPoolExecutor(max_workers=2) as pool:
            if run_registry:
                registry_future = pool.submit(self._run_registry, text)
            if run_coder:
                coder_future = pool.submit(self._run_coder, text, procedure_type)

            # Registry result (needed for reporter)
            if registry_future is not None:
                result.registry = registry_future.result()
                if result.registry.ok and result.registry.data:
                    registry_data = result.registry.data

            # Reporter (depends on registry_data, but can run while coder is still in-flight)
            if run_reporter:
                result.reporter = self._run_reporter(text, registry_data, procedure_type)

            # Coder result (independent)
            if coder_future is not None:
                result.coder = coder_future.result()

        return result

    def _run_registry(self, text: str) -> ModuleOutcome:
        """Run registry extraction.

        Args:
            text: Raw procedure note text

        Returns:
            ModuleOutcome with registry data or error
        """
        try:
            result = self.registry_engine.run(text, explain=True)
            if isinstance(result, tuple):
                record, evidence = result
            else:
                record, evidence = result, getattr(result, "evidence", {})

            return ModuleOutcome(
                ok=True,
                data={
                    "record": (
                        record.model_dump()
                        if hasattr(record, "model_dump")
                        else dict(record)
                    ),
                    "evidence": self._serialize_evidence(evidence),
                },
            )
        except ValueError as ve:
            logger.error(f"Registry validation error: {ve}")
            return ModuleOutcome(
                ok=False,
                error_code="REGISTRY_VALIDATION_ERROR",
                error_message=f"Registry validation failed: {str(ve)}",
            )
        except Exception as e:
            logger.error(f"Registry extraction error: {e}")
            return ModuleOutcome(
                ok=False,
                error_code="REGISTRY_ERROR",
                error_message=f"Registry extraction failed: {str(e)}",
            )

    def _run_reporter(
        self,
        text: str,
        registry_data: dict[str, Any] | None,
        procedure_type: str | None,
    ) -> ModuleOutcome:
        """Run reporter module.

        Args:
            text: Raw procedure note text
            registry_data: Optional registry extraction data
            procedure_type: Optional procedure type hint

        Returns:
            ModuleOutcome with reporter data or error
        """
        try:
            data = self.reporting_strategy.render(
                text, registry_data, procedure_type
            )
            return ModuleOutcome(ok=True, data=data)
        except Exception as e:
            logger.error(f"Reporter error: {e}")
            return ModuleOutcome(
                ok=False,
                error_code="REPORTER_ERROR",
                error_message=f"Report generation failed: {str(e)}",
            )

    def _run_coder(
        self, text: str, procedure_type: str | None
    ) -> ModuleOutcome:
        """Run coder module.

        Args:
            text: Raw procedure note text
            procedure_type: Optional procedure type hint

        Returns:
            ModuleOutcome with coder data or error
        """
        try:
            procedure_id = str(uuid.uuid4())

            result = self.coding_service.generate_result(
                procedure_id=procedure_id,
                report_text=text,
                use_llm=True,
                procedure_type=procedure_type,
            )

            codes = [
                {
                    "cpt": s.code,
                    "description": s.description,
                    "confidence": s.final_confidence,
                    "source": s.source,
                    "hybrid_decision": s.hybrid_decision,
                    # QA schema expects a boolean; treat required/recommended as needing review.
                    "review_flag": str(getattr(s, "review_flag", "")).lower()
                    in ("required", "recommended"),
                }
                for s in result.suggestions
            ]

            return ModuleOutcome(
                ok=True,
                data={
                    "codes": codes,
                    "total_work_rvu": None,
                    "estimated_payment": None,
                    "bundled_codes": [],
                    "kb_version": result.kb_version,
                    "policy_version": result.policy_version,
                    "model_version": result.model_version,
                    "processing_time_ms": (
                        int(result.processing_time_ms)
                        if result.processing_time_ms is not None
                        else None
                    ),
                },
            )
        except Exception as e:
            logger.error(f"Coder error: {e}")
            return ModuleOutcome(
                ok=False,
                error_code="CODER_ERROR",
                error_message=f"Code suggestion failed: {str(e)}",
            )

    def _serialize_evidence(
        self, evidence: dict[str, list[Span]] | None
    ) -> dict[str, list[dict[str, Any]]]:
        """Serialize evidence spans to JSON-compatible structure.

        Args:
            evidence: Field-to-spans mapping

        Returns:
            JSON-serializable evidence dict
        """
        from dataclasses import asdict

        serialized: dict[str, list[dict[str, Any]]] = {}
        for field_name, spans in (evidence or {}).items():
            serialized[field_name] = [asdict(span) for span in spans]
        return serialized


__all__ = [
    "ModuleOutcome",
    "QAPipelineResult",
    "QAPipelineService",
    "ReportingStrategy",
    "SimpleReporterStrategy",
]
