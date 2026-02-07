#!/usr/bin/env python3
"""Generate Jinja addon template files from ip_addon_templates_parsed.json.

This script reads the parsed addon templates JSON and generates individual
Jinja template files in the templates/addons/ directory.

Each generated file contains:
- A comment header with title, category, and CPT codes
- A render() macro that returns the template body
"""

import json
import re
from pathlib import Path


def sanitize_filename(slug: str) -> str:
    """Ensure slug is valid as a filename."""
    # Remove any characters that aren't alphanumeric, underscore, or hyphen
    return re.sub(r"[^a-zA-Z0-9_-]", "_", slug)


def escape_jinja_chars(text: str) -> str:
    """Escape any Jinja-like syntax in the body text.

    Since we're putting raw text into a Jinja template, we need to ensure
    any {{ or {% sequences don't get interpreted as Jinja.
    """
    # Replace any accidental Jinja-like syntax
    text = text.replace("{{", "{ {")
    text = text.replace("}}", "} }")
    text = text.replace("{%", "{ %")
    text = text.replace("%}", "% }")
    return text


def generate_template_file(template: dict, output_dir: Path) -> Path:
    """Generate a single Jinja template file for an addon.

    Args:
        template: Dict with slug, title, category, cpt_codes, body
        output_dir: Directory to write the template file to

    Returns:
        Path to the generated file
    """
    slug = template.get("slug", "")
    title = template.get("title", "")
    category = template.get("category", "")
    cpt_codes = template.get("cpt_codes", [])
    body = template.get("body", "")

    if not slug:
        raise ValueError("Template missing slug")

    filename = f"{sanitize_filename(slug)}.jinja"
    filepath = output_dir / filename

    # Format CPT codes for comment
    cpt_str = ", ".join(str(c) for c in cpt_codes) if cpt_codes else "None"

    # Escape the body text
    escaped_body = escape_jinja_chars(body)

    # Generate the Jinja template content using {# #} for Jinja2 comments
    content = f'''{{#
  file: templates/addons/{filename}
  Title: {title}
  Category: {category}
  CPT: {cpt_str}
#}}
{{% macro render(ctx=None) -%}}
{escaped_body}
{{%- endmacro %}}
'''

    filepath.write_text(content, encoding="utf-8")
    return filepath


def main():
    """Main entry point."""
    # Paths
    project_root = Path(__file__).resolve().parents[2]
    json_path = project_root / "data" / "knowledge" / "ip_addon_templates_parsed.json"
    output_dir = project_root / "proc_report" / "templates" / "addons"

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the JSON
    print(f"Loading templates from: {json_path}")
    if not json_path.exists():
        print(f"ERROR: JSON file not found at {json_path}")
        return 1

    data = json.loads(json_path.read_text(encoding="utf-8"))
    templates = data.get("templates", [])

    print(f"Found {len(templates)} templates")

    # Generate template files
    generated = []
    errors = []

    for template in templates:
        try:
            filepath = generate_template_file(template, output_dir)
            generated.append(filepath.name)
            print(f"  Generated: {filepath.name}")
        except Exception as e:
            slug = template.get("slug", "unknown")
            errors.append(f"{slug}: {e}")
            print(f"  ERROR generating {slug}: {e}")

    # Summary
    print(f"\nGenerated {len(generated)} template files in {output_dir}")
    if errors:
        print(f"Errors: {len(errors)}")
        for err in errors:
            print(f"  - {err}")
        return 1

    # Create an __init__.py in the addons directory for easy importing
    init_path = output_dir / "__init__.py"
    init_content = '''"""Auto-generated addon templates.

These Jinja templates were generated from ip_addon_templates_parsed.json.
Each template provides a render(ctx=None) macro for inclusion in procedure reports.
"""

# This file intentionally left mostly empty.
# Templates are loaded via the Jinja environment, not Python imports.
'''
    init_path.write_text(init_content, encoding="utf-8")
    print(f"Created {init_path}")

    return 0


if __name__ == "__main__":
    exit(main())
