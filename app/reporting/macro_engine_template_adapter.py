from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from proc_schemas.clinical import ProcedureBundle

from app.reporting.engine import ReporterEngine, SchemaRegistry, TemplateMeta, TemplateRegistry
from app.reporting.macro_registry import MacroRegistry, get_macro_registry


@dataclass(frozen=True)
class RenderContext:
    """Minimal context for rendering a single structured template."""

    free_text_hint: str = ""
    render_style: str = "clinical"


class MacroEngineTemplateAdapter:
    """Adapt structured reporter templates to a macro-like render API.

    This is a migration bridge for consolidating template systems without changing
    the underlying template bodies or rendered output.
    """

    def __init__(
        self,
        template_registry: TemplateRegistry,
        schema_registry: SchemaRegistry,
        *,
        macro_registry: MacroRegistry | None = None,
    ) -> None:
        self.templates = template_registry
        self.schemas = schema_registry
        self.macro_registry = macro_registry or get_macro_registry()

    def get_template(self, template_id: str) -> TemplateMeta:
        meta = self.templates.get(template_id)
        if not meta:
            raise KeyError(f"Unknown template_id: {template_id}")
        return meta

    def render_template(
        self,
        template_id: str,
        proc_payload: dict[str, Any],
        *,
        context: RenderContext | None = None,
        strict: bool = False,
        extra_context: dict[str, Any] | None = None,
    ) -> str:
        meta = self.get_template(template_id)
        ctx = context or RenderContext()

        bundle = ProcedureBundle.model_validate(
            {
                "patient": {},
                "encounter": {},
                "procedures": [],
                "free_text_hint": ctx.free_text_hint,
            }
        )

        engine = ReporterEngine(
            self.templates,
            self.schemas,
            procedure_order={},
            render_style=ctx.render_style,
            macro_registry=self.macro_registry,
        )
        engine._strict_render = strict
        return engine._render_payload(meta, proc_payload, bundle, extra_context=extra_context)
