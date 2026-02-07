#!/usr/bin/env python3
"""
Test PHI Redaction on Random Sample of Golden Notes
====================================================

Randomly selects notes from golden JSON files and runs PHI redaction,
producing side-by-side comparison of original and redacted content.

Usage:
    python ops/tools/test_phi_redaction_sample.py [--count N] [--output FILE] [--no-ner]

Examples:
    python ops/tools/test_phi_redaction_sample.py                    # 10 random notes to stdout
    python ops/tools/test_phi_redaction_sample.py --count 5          # 5 random notes
    python ops/tools/test_phi_redaction_sample.py --output test.txt  # Save to file
    python ops/tools/test_phi_redaction_sample.py --no-ner             # Regex-only mode (faster)

Author: Claude Code
"""

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from app.phi.adapters.phi_redactor_hybrid import PHIRedactor, RedactionConfig


def load_golden_notes(golden_dir: Path, limit: int = None) -> List[Tuple[str, str, str]]:
    """
    Load notes from golden JSON files.

    Returns:
        List of (note_text, source_file, note_index) tuples
    """
    notes = []
    pattern = golden_dir / "golden_*.json"

    for filepath in glob.glob(str(pattern)):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            filename = os.path.basename(filepath)

            # Handle both array and single-object formats
            if isinstance(data, list):
                for i, entry in enumerate(data):
                    if isinstance(entry, dict) and 'note_text' in entry:
                        note_text = entry['note_text']
                        if note_text and len(note_text.strip()) > 50:
                            notes.append((note_text, filename, str(i)))
            elif isinstance(data, dict) and 'note_text' in data:
                note_text = data['note_text']
                if note_text and len(note_text.strip()) > 50:
                    notes.append((note_text, filename, "0"))

        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {filepath}: {e}", file=sys.stderr)
            continue

    return notes


def format_comparison(
    original: str,
    redacted: str,
    source: str,
    index: str,
    audit: Dict[str, Any],
    note_num: int
) -> str:
    """
    Format side-by-side comparison of original and redacted text.
    """
    separator = "=" * 80
    section_sep = "-" * 80

    lines = [
        separator,
        f"NOTE {note_num}: {source} [entry {index}]",
        separator,
        "",
        "ORIGINAL TEXT:",
        section_sep,
        original.strip(),
        "",
        section_sep,
        "REDACTED TEXT:",
        section_sep,
        redacted.strip(),
        "",
        section_sep,
        f"REDACTION SUMMARY: {audit.get('redaction_count', 0)} items redacted",
        section_sep,
    ]

    # Add detection details
    detections = audit.get('detections', [])
    if detections:
        lines.append("Detected PHI:")
        for det in detections:
            entity_type = det.get('type', 'UNKNOWN')
            text = det.get('text', '')[:50]  # Truncate long text
            confidence = det.get('confidence', 0)
            source_type = det.get('source', 'unknown')
            lines.append(f"  - [{entity_type}] \"{text}\" (conf={confidence:.2f}, source={source_type})")
    else:
        lines.append("No PHI detected.")

    # Add protected zones summary
    protected = audit.get('protected_zones', [])
    if protected:
        lines.append("")
        lines.append(f"Protected zones: {len(protected)}")
        # Show first few
        for pz in protected[:5]:
            reason = pz.get('reason', 'unknown')
            lines.append(f"  - {reason}")
        if len(protected) > 5:
            lines.append(f"  ... and {len(protected) - 5} more")

    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Test PHI redaction on random sample of golden notes"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=10,
        help="Number of notes to sample (default: 10)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--no-ner",
        action="store_true",
        help="Disable NER model (regex-only, faster)"
    )
    parser.add_argument(
        "--keep-dates",
        action="store_true",
        help="Do not redact procedure dates"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--golden-dir",
        type=str,
        default=None,
        help="Path to golden extractions directory"
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Determine golden directory
    if args.golden_dir:
        golden_dir = Path(args.golden_dir)
    else:
        golden_dir = PROJECT_ROOT / "data" / "knowledge" / "golden_extractions"

    if not golden_dir.exists():
        print(f"Error: Golden directory not found: {golden_dir}", file=sys.stderr)
        sys.exit(1)

    # Load notes
    print(f"Loading notes from {golden_dir}...", file=sys.stderr)
    all_notes = load_golden_notes(golden_dir)

    if not all_notes:
        print("Error: No notes found in golden files.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_notes)} notes total.", file=sys.stderr)

    # Sample random notes
    sample_size = min(args.count, len(all_notes))
    sample_notes = random.sample(all_notes, sample_size)

    print(f"Selected {sample_size} random notes.", file=sys.stderr)

    # Initialize redactor
    print(f"Initializing PHI Redactor (NER: {not args.no_ner})...", file=sys.stderr)
    config = RedactionConfig(
        redact_procedure_dates=not args.keep_dates
    )
    redactor = PHIRedactor(config=config, use_ner_model=not args.no_ner)

    # Process notes
    output_lines = []
    output_lines.append("=" * 80)
    output_lines.append("PHI REDACTION TEST REPORT")
    output_lines.append(f"Sample size: {sample_size} notes")
    output_lines.append(f"Mode: {'Regex + NER' if not args.no_ner else 'Regex-only'}")
    output_lines.append("=" * 80)
    output_lines.append("")

    total_redactions = 0

    for i, (note_text, source, index) in enumerate(sample_notes, 1):
        print(f"Processing note {i}/{sample_size}...", file=sys.stderr)

        try:
            redacted_text, audit = redactor.scrub(note_text)
            total_redactions += audit.get('redaction_count', 0)

            comparison = format_comparison(
                original=note_text,
                redacted=redacted_text,
                source=source,
                index=index,
                audit=audit,
                note_num=i
            )
            output_lines.append(comparison)

        except Exception as e:
            output_lines.append(f"ERROR processing note {i} from {source}[{index}]: {e}")
            output_lines.append("")

    # Summary
    output_lines.append("=" * 80)
    output_lines.append("SUMMARY")
    output_lines.append("=" * 80)
    output_lines.append(f"Notes processed: {sample_size}")
    output_lines.append(f"Total redactions: {total_redactions}")
    output_lines.append(f"Average redactions per note: {total_redactions / sample_size:.1f}")
    output_lines.append("")

    # Output results
    output_text = "\n".join(output_lines)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output_text)
        print(f"Results saved to {args.output}", file=sys.stderr)
    else:
        print(output_text)

    print("Done!", file=sys.stderr)


if __name__ == "__main__":
    main()
