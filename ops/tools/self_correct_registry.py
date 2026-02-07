import argparse
import os
from pathlib import Path

# Ensure local imports
import sys
sys.path.append(str(Path.cwd()))

from app.registry.prompts import FIELD_INSTRUCTIONS  # noqa: E402
from app.registry.self_correction import (  # noqa: E402
    get_allowed_values,
    suggest_improvements_for_field,
)


def main():
    parser = argparse.ArgumentParser(description="Generate registry self-correction suggestions.")
    parser.add_argument("--field", required=True, help="Field name to analyze (e.g., sedation_type)")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Maximum number of error examples to send to the LLM",
    )
    parser.add_argument(
        "--model",
        help="Override model for self-correction (e.g., gpt-5.1, gemini-2.5-flash-lite). Defaults to env REGISTRY_SELF_CORRECTION_MODEL.",
    )
    args = parser.parse_args()

    field_name = args.field
    if args.model:
        os.environ["REGISTRY_SELF_CORRECTION_MODEL"] = args.model

    allowed_values = get_allowed_values(field_name)
    suggestions = suggest_improvements_for_field(field_name, allowed_values, max_examples=args.max_examples)

    current_instruction = FIELD_INSTRUCTIONS.get(field_name, "No instruction available.")

    print(f"Field: {field_name}")
    print("\nCurrent instruction:")
    print(current_instruction)

    print("\nSuggested updates:")
    for key in ("updated_instruction", "python_postprocessing_rules", "comments"):
        val = suggestions.get(key, "<none>")
        print(f"{key}: {val}")

    report_path = Path(f"reports/registry_self_correction_{field_name}.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        f.write(f"# Registry self-correction for {field_name}\n\n")
        f.write("## Current instruction\n")
        f.write(f"{current_instruction}\n\n")
        f.write("## Suggested updates\n")
        for key in ("updated_instruction", "python_postprocessing_rules", "comments"):
            f.write(f"### {key}\n")
            f.write(f"{suggestions.get(key, '<none>')}\n\n")

    print(f"\nSuggestions written to {report_path}")


if __name__ == "__main__":
    main()
