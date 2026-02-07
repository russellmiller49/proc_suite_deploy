#!/usr/bin/env python3
"""
Run the full coding pipeline (rules + LLM advisor + smart_hybrid merge)
over a JSONL notes file and emit CodeSuggestion[] per note.

Usage:
    python ops/tools/run_coder_hybrid.py \
        --notes data/synthetic/synthetic_notes_with_registry.jsonl \
        --kb data/knowledge/ip_coding_billing_v3_0.json \
        --keyword-dir data/keyword_mappings \
        --model-version gemini-1.5-pro-002 \
        --out-json outputs/coder_suggestions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Dict, Any

from config.settings import CoderSettings
from app.coder.adapters.persistence.csv_kb_adapter import CsvKnowledgeBaseAdapter
from app.coder.adapters.nlp.keyword_mapping_loader import YamlKeywordMappingRepository
from app.coder.adapters.nlp.simple_negation_detector import SimpleNegationDetector
from app.coder.adapters.llm.gemini_advisor import GeminiAdvisorAdapter  # per update-recs doc
from app.domain.coding_rules.rule_engine import RuleEngine
from app.coder.application.coding_service import CodingService


def iter_notes(jsonl_path: Path) -> Iterator[Dict[str, Any]]:
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:  # noqa: TRY003
                print(f"Skipping bad JSON line: {exc}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run smart-hybrid coder over notes.")
    ap.add_argument("--notes", required=True, help="JSONL file with notes.")
    ap.add_argument("--kb", default="data/knowledge/ip_coding_billing_v3_0.json")
    ap.add_argument(
        "--keyword-dir",
        default="data/keyword_mappings",
        help="Directory of YAML keyword mapping files.",
    )
    ap.add_argument(
        "--model-version",
        default="gemini-1.5-pro-002",
        help="LLM model identifier for provenance.",
    )
    ap.add_argument(
        "--out-json",
        required=True,
        help="Output JSONL; one line per note with CodeSuggestion[] dump.",
    )
    args = ap.parse_args()

    notes_path = Path(args.notes)
    kb_path = Path(args.kb)
    keyword_dir = Path(args.keyword_dir)
    out_path = Path(args.out_json)

    if not notes_path.is_file():
        print(f"ERROR: notes file not found at {notes_path}", file=sys.stderr)
        return 1

    settings = CoderSettings(
        model_version=args.model_version,
        kb_path=str(kb_path),
        kb_version=kb_path.name,
        keyword_mapping_dir=str(keyword_dir),
        keyword_mapping_version="v1",
    )

    # Infra adapters
    kb_repo = CsvKnowledgeBaseAdapter(settings.kb_path)
    keyword_repo = YamlKeywordMappingRepository(settings.keyword_mapping_dir)
    negation_detector = SimpleNegationDetector()

    # Rule engine + LLM advisor
    rule_engine = RuleEngine(kb_repo=kb_repo)
    llm_advisor = GeminiAdvisorAdapter(
        model_name=settings.model_version,
        allowed_codes=list(kb_repo.get_all_codes()),
    )

    coding_service = CodingService(
        kb_repo=kb_repo,
        keyword_repo=keyword_repo,
        negation_detector=negation_detector,
        rule_engine=rule_engine,
        llm_advisor=llm_advisor,
        config=settings,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as out_f:
        for note in iter_notes(notes_path):
            procedure_id = str(note.get("procedure_id") or note.get("id"))
            report_text = note.get("report_text") or note.get("note_text")
            if not (procedure_id and report_text):
                print(f"Skipping note with missing fields: {note}", file=sys.stderr)
                continue

            suggestions = coding_service.generate_suggestions(
                procedure_id=procedure_id,
                report_text=report_text,
            )

            out_record = {
                "procedure_id": procedure_id,
                "suggestions": [s.model_dump(mode="json") for s in suggestions],
            }
            out_f.write(json.dumps(out_record, default=str) + "\n")

    print(f"Coder run complete. Wrote suggestions to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
