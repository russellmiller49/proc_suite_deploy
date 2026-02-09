from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from jinja2 import Environment, FileSystemLoader


MacroCallable = Callable[..., str]


@dataclass(frozen=True)
class Macro:
    name: str
    category: str
    description: str
    cpt: Any
    params: list[str]
    defaults: dict[str, Any]
    required: bool
    essential: list[str]
    essential_labels: dict[str, str]
    note: str | None
    template_file: str
    callable: MacroCallable

    def metadata_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "description": self.description,
            "cpt": self.cpt,
            "params": list(self.params),
            "defaults": dict(self.defaults),
            "required": self.required,
            "essential": list(self.essential),
            "essential_labels": dict(self.essential_labels),
            "note": self.note,
            "template_file": self.template_file,
        }


def load_schema(schema_path: Path) -> dict[str, Any]:
    """Load a macro template schema JSON document."""
    if schema_path.exists():
        try:
            raw = json.loads(schema_path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else {"categories": {}}
        except Exception:
            return {"categories": {}}
    return {"categories": {}}


def build_env(template_root: Path) -> Environment:
    return Environment(
        loader=FileSystemLoader(str(template_root)),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True,
    )


def build_registry(schema: dict[str, Any], template_root: Path) -> tuple[dict[str, Macro], Environment]:
    """Build a macro registry from a schema and a template root.

    Returns a tuple of (macro_name -> Macro, jinja_env).
    """
    env = build_env(template_root)

    category_files = {
        "01_minor_trach_laryngoscopy": "01_minor_trach_laryngoscopy.j2",
        "02_core_bronchoscopy": "02_core_bronchoscopy.j2",
        "03_navigation_robotic_ebus": "03_navigation_robotic_ebus.j2",
        "04_blvr_cryo": "04_blvr_cryo.j2",
        "05_pleural": "05_pleural.j2",
        "06_other_interventions": "06_other_interventions.j2",
        "07_clinical_assessment": "07_clinical_assessment.j2",
    }

    registry: dict[str, Macro] = {}
    for category_key, category_data in (schema.get("categories", {}) or {}).items():
        if not isinstance(category_data, dict):
            continue
        template_file = category_files.get(category_key)
        if not template_file:
            continue

        try:
            template = env.get_template(template_file)
            module = template.module
        except Exception:
            continue

        description = str(category_data.get("description", "") or "")
        macros = category_data.get("macros", {}) or {}
        if not isinstance(macros, dict):
            continue

        for macro_name, macro_meta in macros.items():
            if not isinstance(macro_name, str) or not macro_name:
                continue
            if not isinstance(macro_meta, dict):
                macro_meta = {}
            if not hasattr(module, macro_name):
                continue
            registry[macro_name] = Macro(
                name=macro_name,
                category=category_key,
                description=description,
                cpt=macro_meta.get("cpt"),
                params=list(macro_meta.get("params", []) or []),
                defaults=dict(macro_meta.get("defaults", {}) or {}),
                required=bool(macro_meta.get("required", False)),
                essential=list(macro_meta.get("essential", []) or []),
                essential_labels=dict(macro_meta.get("essential_labels", {}) or {}),
                note=macro_meta.get("note"),
                template_file=template_file,
                callable=getattr(module, macro_name),
            )

    return registry, env


class MacroRegistry:
    def __init__(self, schema_path: Path, template_root: Path):
        self.schema_path = schema_path
        self.template_root = template_root
        self.schema = load_schema(schema_path)
        self.registry, self.env = build_registry(self.schema, template_root)
        self.category_macros = self._load_category_macros(self.schema)

    @staticmethod
    def _load_category_macros(schema: dict[str, Any]) -> dict[str, list[str]]:
        raw = schema.get("category_macros", {}) or {}
        if not isinstance(raw, dict):
            return {}
        parsed: dict[str, list[str]] = {}
        for key, value in raw.items():
            if not isinstance(key, str) or not key:
                continue
            if not isinstance(value, list):
                continue
            parsed[key] = [str(item) for item in value if str(item)]
        return parsed

    def get(self, name: str) -> Macro:
        return self.registry[name]

    def maybe_get(self, name: str) -> Macro | None:
        return self.registry.get(name)

    def list_macros(self) -> list[str]:
        return list(self.registry.keys())

    def list_macros_by_category(self, category: str) -> list[str]:
        return [name for name, macro in self.registry.items() if macro.category == category]

    def list_categories(self) -> list[str]:
        categories = self.schema.get("categories", {}) or {}
        return list(categories.keys()) if isinstance(categories, dict) else []

    def get_category_description(self, category: str) -> str | None:
        categories = self.schema.get("categories", {}) or {}
        if not isinstance(categories, dict):
            return None
        cat_data = categories.get(category)
        if not isinstance(cat_data, dict):
            return None
        desc = cat_data.get("description")
        return str(desc) if desc not in (None, "") else None

    def get_category_macros(self, category: str) -> list[str]:
        return list(self.category_macros.get(category, []))


DEFAULT_TEMPLATE_ROOT = Path(__file__).parent / "templates" / "macros"
DEFAULT_SCHEMA_PATH = DEFAULT_TEMPLATE_ROOT / "template_schema.json"


@lru_cache(maxsize=1)
def get_macro_registry() -> MacroRegistry:
    return MacroRegistry(schema_path=DEFAULT_SCHEMA_PATH, template_root=DEFAULT_TEMPLATE_ROOT)
