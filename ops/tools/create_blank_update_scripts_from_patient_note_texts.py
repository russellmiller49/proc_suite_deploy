#!/usr/bin/env python3
"""
Create blank per-patient Python update scripts.

For each JSON file in `data/knowledge/patient_note_texts/` (e.g., 74-8829-C.json),
create a matching blank Python script in:
  `data/granular annotations/Python_update_scripts/74-8829-C.py`

This is useful for a manual workflow where you later fill in one script per patient.

Usage:
  python ops/tools/create_blank_update_scripts_from_patient_note_texts.py \
    --input-dir data/knowledge/patient_note_texts \
    --output-dir "data/granular annotations/Python_update_scripts"
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


TEMPLATE = """\
#!/usr/bin/env python3
\"\"\"Blank patient update script (auto-generated).

Source JSON: {source_json}
\"\"\"


def main() -> None:
    # TODO: implement per-patient updates here
    pass


if __name__ == \"__main__\":
    main()
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create blank .py scripts for each patient_note_texts/*.json")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/knowledge/patient_note_texts"),
        help="Directory containing per-patient JSON files.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/granular annotations/Python_update_scripts"),
        help="Directory to write per-patient blank .py scripts.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing .py files.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="If set, do not write files; only log actions.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    args = parse_args(argv)
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Missing input dir: {args.input_dir}")

    json_paths = sorted(args.input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No *.json files found in: {args.input_dir}")

    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped_exists = 0
    for jp in json_paths:
        out_path = args.output_dir / f"{jp.stem}.py"
        if out_path.exists() and not args.overwrite:
            skipped_exists += 1
            continue

        content = TEMPLATE.format(source_json=str(jp))
        if args.dry_run:
            logger.info("Would write %s", out_path)
        else:
            out_path.write_text(content, encoding="utf-8")
        created += 1

    logger.info(
        "Done. input_json=%d created=%d skipped_exists=%d output_dir=%s",
        len(json_paths),
        created,
        skipped_exists,
        args.output_dir,
    )


if __name__ == "__main__":
    main(sys.argv[1:])

