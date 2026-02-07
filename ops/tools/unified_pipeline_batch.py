#!/usr/bin/env python3
"""Batch unified pipeline test on random notes.

This script randomly selects N notes from data/granular annotations/notes_text,
runs the full unified pipeline (same as UI at /ui/), and saves results to a text file.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in ("1", "true", "yes")


if not _truthy_env("PROCSUITE_SKIP_DOTENV"):
    load_dotenv(override=False)

# Import after environment setup
from app.api.adapters.response_adapter import build_v3_evidence_payload  # noqa: E402
from app.api.dependencies import get_coding_service, get_registry_service  # noqa: E402
from app.api.phi_dependencies import get_phi_scrubber  # noqa: E402
from app.api.phi_redaction import apply_phi_redaction  # noqa: E402
from app.api.schemas import (  # noqa: E402
    CodeSuggestionSummary,
    UnifiedProcessRequest,
    UnifiedProcessResponse,
)
from app.coder.application.coding_service import CodingService  # noqa: E402
from app.coder.domain_rules.registry_to_cpt.coding_rules import derive_all_codes_with_meta  # noqa: E402
from app.coder.phi_gating import is_phi_review_required  # noqa: E402
from app.common.exceptions import LLMError  # noqa: E402
from app.registry.application.registry_service import (  # noqa: E402
    RegistryExtractionResult,
    RegistryService,
)
from config.settings import CoderSettings  # noqa: E402


def _load_notes_from_directory(notes_dir: Path) -> dict[str, str]:
    """Load all notes from .txt files in the directory.
    
    Returns a dict mapping note_id (filename without .txt) to note_text.
    """
    notes = {}
    
    for txt_file in sorted(notes_dir.glob("*.txt")):
        try:
            note_text = txt_file.read_text(encoding="utf-8")
            note_id = txt_file.stem  # filename without .txt extension
            notes[note_id] = note_text
        except Exception as exc:
            print(f"Warning: Failed to load {txt_file}: {exc}", file=sys.stderr)
    
    return notes


def _run_unified_pipeline(
    note_text: str,
    registry_service: RegistryService,
    coding_service: CodingService,
    phi_scrubber,
    *,
    include_financials: bool = True,
    explain: bool = True,
) -> UnifiedProcessResponse:
    """Run the unified pipeline (same as /api/v1/process endpoint).
    
    This replicates the exact logic from app/api/routes/unified_process.py
    """
    import time
    
    start_time = time.time()
    
    # PHI redaction (if not already scrubbed)
    # For batch testing, we'll treat notes as already scrubbed to match UI behavior
    # when user submits via PHI redactor
    redaction = apply_phi_redaction(note_text, phi_scrubber)
    scrubbed_text = redaction.text
    
    # Step 1: Registry extraction (synchronous call)
    try:
        extraction_result = registry_service.extract_fields(scrubbed_text)
    except Exception as exc:
        if isinstance(exc, LLMError) and "429" in str(exc):
            raise Exception("Upstream LLM rate limited") from exc
        raise
    
    # Step 2: Derive CPT codes from registry
    record = extraction_result.record
    if record is None:
        from app.registry.schema import RegistryRecord
        record = RegistryRecord.model_validate(extraction_result.mapped_fields)
    
    codes, rationales, derivation_warnings = derive_all_codes_with_meta(record)
    
    # Build suggestions with confidence and rationale
    suggestions = []
    base_confidence = 0.95 if extraction_result.coder_difficulty == "HIGH_CONF" else 0.80
    
    for code in codes:
        proc_info = coding_service.kb_repo.get_procedure_info(code)
        description = proc_info.description if proc_info else ""
        rationale = rationales.get(code, "")
        
        # Determine review flag
        if extraction_result.needs_manual_review:
            review_flag = "required"
        elif extraction_result.audit_warnings:
            review_flag = "recommended"
        else:
            review_flag = "optional"
        
        suggestions.append(
            CodeSuggestionSummary(
                code=code,
                description=description,
                confidence=base_confidence,
                rationale=rationale,
                review_flag=review_flag,
            )
        )
    
    # Step 3: Calculate financials if requested
    total_work_rvu = None
    estimated_payment = None
    per_code_billing = []
    
    if include_financials and codes:
        settings = CoderSettings()
        conversion_factor = settings.cms_conversion_factor
        total_work = 0.0
        total_payment = 0.0
        
        for code in codes:
            proc_info = coding_service.kb_repo.get_procedure_info(code)
            if proc_info:
                work_rvu = proc_info.work_rvu
                total_rvu = proc_info.total_facility_rvu
                payment = total_rvu * conversion_factor
                
                total_work += work_rvu
                total_payment += payment
                
                per_code_billing.append({
                    "cpt_code": code,
                    "description": proc_info.description,
                    "work_rvu": work_rvu,
                    "total_facility_rvu": total_rvu,
                    "facility_payment": round(payment, 2),
                })
        
        total_work_rvu = round(total_work, 2)
        estimated_payment = round(total_payment, 2)
    
    # Combine audit warnings
    all_warnings: list[str] = []
    all_warnings.extend(extraction_result.warnings or [])
    all_warnings.extend(extraction_result.audit_warnings or [])
    all_warnings.extend(derivation_warnings)
    
    # Deduplicate warnings
    deduped_warnings: list[str] = []
    seen_warnings: set[str] = set()
    for warning in all_warnings:
        if warning in seen_warnings:
            continue
        seen_warnings.add(warning)
        deduped_warnings.append(warning)
    all_warnings = deduped_warnings
    
    # Build evidence payload
    evidence_payload = build_v3_evidence_payload(record=record, codes=codes)
    if not explain and not evidence_payload:
        evidence_payload = {}
    
    # Determine review status
    needs_manual_review = extraction_result.needs_manual_review
    if is_phi_review_required():
        review_status = "pending_phi_review"
        needs_manual_review = True
    elif needs_manual_review:
        review_status = "unverified"
    else:
        review_status = "finalized"
    
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Build response
    registry_payload = record.model_dump(exclude_none=True)
    
    return UnifiedProcessResponse(
        registry=registry_payload,
        evidence=evidence_payload,
        cpt_codes=codes,
        suggestions=suggestions,
        total_work_rvu=total_work_rvu,
        estimated_payment=estimated_payment,
        per_code_billing=per_code_billing,
        pipeline_mode="extraction_first",
        coder_difficulty=extraction_result.coder_difficulty or "",
        needs_manual_review=needs_manual_review,
        audit_warnings=all_warnings,
        validation_errors=extraction_result.validation_errors or [],
        kb_version=coding_service.kb_repo.version,
        policy_version="extraction_first_v1",
        processing_time_ms=round(processing_time_ms, 2),
        review_status=review_status,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch unified pipeline test on random notes from notes_text directory."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of random notes to test (default: 10)",
    )
    parser.add_argument(
        "--notes-dir",
        type=Path,
        default=ROOT / "data" / "granular annotations" / "notes_text",
        help="Directory containing note .txt files (default: data/granular annotations/notes_text)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: unified_pipeline_batch_<timestamp>.txt)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--include-financials",
        action="store_true",
        default=True,
        help="Include RVU and payment information (default: True)",
    )
    parser.add_argument(
        "--no-financials",
        dest="include_financials",
        action="store_false",
        help="Exclude RVU and payment information",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        default=True,
        help="Include evidence/explanation data (default: True)",
    )
    parser.add_argument(
        "--no-explain",
        dest="explain",
        action="store_false",
        help="Exclude evidence/explanation data",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Allow real LLM calls (disables stub/offline defaults).",
    )
    args = parser.parse_args()
    
    # Set up environment
    if args.real_llm:
        os.environ.setdefault("REGISTRY_USE_STUB_LLM", "0")
        os.environ.setdefault("GEMINI_OFFLINE", "0")
    else:
        # Use stub LLM for offline testing
        if os.getenv("REGISTRY_USE_STUB_LLM") is None:
            os.environ["REGISTRY_USE_STUB_LLM"] = "1"
        if os.getenv("GEMINI_OFFLINE") is None:
            os.environ["GEMINI_OFFLINE"] = "1"
    
    # Load notes
    if not args.notes_dir.exists():
        print(f"ERROR: Notes directory not found: {args.notes_dir}", file=sys.stderr)
        return 1
    
    print(f"Loading notes from {args.notes_dir}...", file=sys.stderr)
    all_notes = _load_notes_from_directory(args.notes_dir)
    
    if not all_notes:
        print(f"ERROR: No notes found in {args.notes_dir}", file=sys.stderr)
        return 1
    
    print(f"Loaded {len(all_notes)} notes", file=sys.stderr)
    
    # Select random notes
    if args.seed is not None:
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}", file=sys.stderr)
    
    count = min(args.count, len(all_notes))
    selected_notes = random.sample(list(all_notes.items()), count)
    
    print(f"Selected {count} random notes for testing", file=sys.stderr)
    
    # Determine output file
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = ROOT / f"unified_pipeline_batch_{timestamp}.txt"
    
    # Initialize services
    print("Initializing services...", file=sys.stderr)
    registry_service = get_registry_service()
    coding_service = get_coding_service()
    phi_scrubber = get_phi_scrubber()
    
    # Run unified pipeline on each note
    print(f"Running unified pipeline on {count} notes...", file=sys.stderr)
    print(f"Output will be saved to: {output_path}", file=sys.stderr)
    
    all_results = []
    for i, (note_id, note_text) in enumerate(selected_notes, 1):
        print(f"[{i}/{count}] Processing {note_id}...", file=sys.stderr)
        try:
            result = _run_unified_pipeline(
                note_text,
                registry_service,
                coding_service,
                phi_scrubber,
                include_financials=args.include_financials,
                explain=args.explain,
            )
            all_results.append((note_id, note_text, result, None))
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            all_results.append((note_id, note_text, None, str(exc)))
    
    # Write output file
    with open(output_path, "w", encoding="utf-8") as f:
        # Write header
        f.write("=" * 80 + "\n")
        f.write("UNIFIED PIPELINE BATCH TEST RESULTS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Notes tested: {count}\n")
        f.write(f"Notes directory: {args.notes_dir}\n")
        if args.seed is not None:
            f.write(f"Random seed: {args.seed}\n")
        f.write(f"Include financials: {args.include_financials}\n")
        f.write(f"Include explain: {args.explain}\n")
        f.write(f"Real LLM enabled: {args.real_llm}\n")
        f.write("=" * 80 + "\n")
        f.write("\n")
        
        # Write results for each note
        success_count = 0
        failed_count = 0
        
        for note_id, note_text, result, error in all_results:
            f.write("=" * 80 + "\n")
            f.write(f"NOTE: {note_id}\n")
            f.write("=" * 80 + "\n")
            f.write("\n")
            
            # Write note text
            f.write("NOTE TEXT:\n")
            f.write("-" * 80 + "\n")
            f.write(note_text)
            f.write("\n")
            f.write("-" * 80 + "\n")
            f.write("\n")
            
            # Write results
            if error:
                f.write("ERROR:\n")
                f.write(f"{error}\n")
                f.write("\n")
                f.write("STATUS: FAILED\n")
                failed_count += 1
            else:
                f.write("RESULTS (JSON):\n")
                f.write("-" * 80 + "\n")
                # Convert Pydantic model to dict and serialize
                result_dict = result.model_dump(exclude_none=True)
                f.write(json.dumps(result_dict, indent=2, ensure_ascii=False))
                f.write("\n")
                f.write("-" * 80 + "\n")
                f.write("\n")
                f.write("STATUS: SUCCESS\n")
                success_count += 1
            
            f.write("\n")
        
        # Write summary
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total notes tested: {count}\n")
        f.write(f"Successful: {success_count}\n")
        f.write(f"Failed: {failed_count}\n")
        if count > 0:
            f.write(f"Success rate: {success_count/count*100:.1f}%\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nCompleted! Results saved to: {output_path}", file=sys.stderr)
    print(f"Summary: {success_count} successful, {failed_count} failed", file=sys.stderr)
    
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
