"""IP Add-on Templates Module.

Loads and serves procedural add-on templates from ip_addon_templates_parsed.json.
These templates provide standardized text snippets for various IP procedures that
can be included in synoptic reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.settings import KnowledgeSettings

def _addons_json_path() -> Path:
    return KnowledgeSettings().addon_templates_path

# Module-level cache
_ADDONS_DATA: dict[str, Any] | None = None
_ADDONS_BY_SLUG: dict[str, str] = {}
_ADDONS_BY_CATEGORY: dict[str, list[dict[str, Any]]] = {}
_ADDON_METADATA: dict[str, dict[str, Any]] = {}


def _load_addons() -> None:
    """Load addon templates from JSON file (lazy cached)."""
    global _ADDONS_DATA, _ADDONS_BY_SLUG, _ADDONS_BY_CATEGORY, _ADDON_METADATA

    if _ADDONS_DATA is not None:
        return

    addons_path = _addons_json_path()
    if not addons_path.exists():
        _ADDONS_DATA = {"templates": []}
        return

    try:
        _ADDONS_DATA = json.loads(addons_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Failed to load addon templates: {exc}") from exc

    templates = _ADDONS_DATA.get("templates", [])
    for template in templates:
        slug = template.get("slug", "")
        if not slug:
            continue

        body = template.get("body", "")
        _ADDONS_BY_SLUG[slug] = body

        # Store full metadata for each template
        _ADDON_METADATA[slug] = {
            "title": template.get("title", ""),
            "category": template.get("category", ""),
            "cpt_codes": template.get("cpt_codes", []),
            "params": template.get("params", []) or [],
            "body": body,
        }

        # Index by category
        category = template.get("category", "UNCATEGORIZED")
        if category not in _ADDONS_BY_CATEGORY:
            _ADDONS_BY_CATEGORY[category] = []
        _ADDONS_BY_CATEGORY[category].append(template)


def get_addon_body(slug: str) -> str | None:
    """Get the body text of an addon template by its slug.

    Args:
        slug: The unique identifier for the addon template
              (e.g., "control_of_minor_tracheostomy_bleeding_electrocautery")

    Returns:
        The body text of the template, or None if not found
    """
    _load_addons()
    return _ADDONS_BY_SLUG.get(slug)


def get_addon_metadata(slug: str) -> dict[str, Any] | None:
    """Get full metadata for an addon template by its slug.

    Args:
        slug: The unique identifier for the addon template

    Returns:
        Dict with title, category, cpt_codes, and body; or None if not found
    """
    _load_addons()
    return _ADDON_METADATA.get(slug)


def list_addon_slugs() -> list[str]:
    """Get a list of all available addon template slugs.

    Returns:
        List of all slug identifiers
    """
    _load_addons()
    return list(_ADDONS_BY_SLUG.keys())


def list_addons_by_category(category: str) -> list[dict[str, Any]]:
    """Get all addon templates in a specific category.

    Args:
        category: The category name (e.g., "PLEURAL PROCEDURES")

    Returns:
        List of template dicts in that category
    """
    _load_addons()
    return _ADDONS_BY_CATEGORY.get(category, [])


def list_categories() -> list[str]:
    """Get a list of all available categories.

    Returns:
        List of category names
    """
    _load_addons()
    if _ADDONS_DATA:
        return _ADDONS_DATA.get("categories", list(_ADDONS_BY_CATEGORY.keys()))
    return []


def get_addon_count() -> int:
    """Get the total number of addon templates.

    Returns:
        Number of addon templates loaded
    """
    _load_addons()
    return len(_ADDONS_BY_SLUG)


def find_addons_by_cpt(cpt_code: str) -> list[dict[str, Any]]:
    """Find addon templates that match a specific CPT code.

    Args:
        cpt_code: The CPT code to search for

    Returns:
        List of template metadata dicts that include this CPT code
    """
    _load_addons()
    results = []
    cpt_str = str(cpt_code)
    for slug, meta in _ADDON_METADATA.items():
        if cpt_str in [str(c) for c in meta.get("cpt_codes", [])]:
            results.append({"slug": slug, **meta})
    return results


def render_addon(slug: str, context: dict[str, Any] | None = None) -> str | None:
    """Render an addon template, optionally with context for placeholder substitution.

    This function returns the addon body as-is. If context is provided,
    it can be used for simple placeholder replacement (e.g., replacing [side] with "right").

    Note: Bracketed placeholders like [side], [volume mL] are left as-is by default
    so they can be manually replaced in the final note.

    Args:
        slug: The addon template slug
        context: Optional dict of placeholder replacements

    Returns:
        The rendered addon text, or None if slug not found
    """
    _load_addons()
    body = get_addon_body(slug)
    if body is None:
        return None

    if not context:
        return body

    # Simple placeholder substitution for context values
    result = body
    for key, value in context.items():
        # Replace bracketed placeholders like [side] with context values
        placeholder = f"[{key}]"
        if placeholder in result:
            result = result.replace(placeholder, str(value))

    return result


def get_addon_title(slug: str) -> str | None:
    """Get just the title of an addon template.

    Args:
        slug: The addon template slug

    Returns:
        The title string, or None if not found
    """
    _load_addons()
    meta = _ADDON_METADATA.get(slug)
    return meta.get("title") if meta else None


def validate_addons() -> dict[str, list[str]]:
    """Validate the loaded addon templates.

    Returns:
        Dict with 'errors' and 'warnings' lists
    """
    _load_addons()
    errors = []
    warnings = []
    seen_slugs = set()

    for slug, meta in _ADDON_METADATA.items():
        # Check for duplicate slugs
        if slug in seen_slugs:
            errors.append(f"Duplicate slug: {slug}")
        seen_slugs.add(slug)

        # Check for empty slugs
        if not slug or not slug.strip():
            errors.append("Empty slug found")

        # Check for empty body
        if not meta.get("body", "").strip():
            warnings.append(f"Empty body for slug: {slug}")

        # Check for missing title
        if not meta.get("title", "").strip():
            warnings.append(f"Missing title for slug: {slug}")

    return {"errors": errors, "warnings": warnings}


# Convenience alias for Jinja integration
ADDONS_BY_SLUG = _ADDONS_BY_SLUG


__all__ = [
    "get_addon_body",
    "get_addon_metadata",
    "get_addon_title",
    "list_addon_slugs",
    "list_addons_by_category",
    "list_categories",
    "get_addon_count",
    "find_addons_by_cpt",
    "render_addon",
    "validate_addons",
    "ADDONS_BY_SLUG",
]
