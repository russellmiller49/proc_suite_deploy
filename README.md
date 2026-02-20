# Procedure Suite

**Automated CPT Coding, Registry Extraction, and Synoptic Reporting for Interventional Pulmonology.**

This toolkit enables:
1.  **Predict CPT Codes**: Analyze procedure notes using ML + LLM hybrid pipeline to generate billing codes with RVU calculations.
2.  **Extract Registry Data**: Use deterministic extractors and LLMs to extract structured clinical data (EBUS stations, complications, demographics) into a validated schema.
3.  **Generate Reports**: Create standardized, human-readable procedure reports from structured data (Reporter Builder + deterministic templates).

## Documentation

- **[Docs Home](docs/README.md)**: Start here — reading order and documentation map.
- **[Repo Guide](docs/REPO_GUIDE.md)**: End-to-end explanation of how the repo functions.
- **[Installation & Setup](docs/INSTALLATION.md)**: Setup guide for Python, spaCy models, and API keys.
- **[Repo Index](docs/REPO_INDEX.md)**: One-page map of the repo (entrypoints, key folders, knowledge assets).
- **[User Guide](docs/USER_GUIDE.md)**: How to use the CLI tools and API endpoints.
- **[Registry Prodigy Workflow](docs/REGISTRY_PRODIGY_WORKFLOW.md)**: Human-in-the-loop “Diamond Loop” for the registry procedure classifier.
- **[Development Guide](docs/DEVELOPMENT.md)**: **CRITICAL** for contributors and AI Agents. Defines the system architecture and coding standards.
- **[Architecture](docs/ARCHITECTURE.md)**: System design, module breakdown, and data flow.
- **[Agents](docs/AGENTS.md)**: Multi-agent pipeline documentation for Parser, Summarizer, and Structurer.
- **[Registry API](docs/Registry_API.md)**: Registry extraction service API documentation.
- **[CPT Reference](docs/REFERENCES.md)**: List of supported codes.

## Quick Start

1.  **Install**:
    ```bash
    micromamba activate medparse-py311
    make install
    make preflight
    ```

2.  **Configure**:
    Create `.env` with your `GEMINI_API_KEY`.

3.  **Run**:
    ```bash
    # Start the API/Dev Server
    ./ops/devserver.sh
    ```

    Then open:
    - UI (Clinical Dashboard / PHI Redactor): `http://localhost:8000/ui/`
    - Workflow overview: `http://localhost:8000/ui/workflow.html`

    The UI flow is: paste note -> run PHI detection -> apply redactions -> submit scrubbed note -> review results.
    Optional: edit values in **Flattened Tables (Editable)** (generates **Edited JSON (Training)**) and export JSON/tables.

### Client-Side PDF Extraction (UI)

PDF upload/extraction is browser-local and worker-based (`ui/static/phi_redactor/pdf_local/`):

- Native parsing/layout runs in `workers/pdf.worker.js` (pdf.js text layer + text/image region analysis).
- Per-page native density is computed as `nativeTextDensity = charCount / pageArea`.
  - High-density digital pages short-circuit OCR (`NATIVE_DENSE_TEXT`) and stay native.
- OCR pages run in `workers/ocr.worker.js` with:
  - Image masking modes: `auto`, `on`, `off`
  - Left-column/body crop logic
  - Header zonal OCR: top 25% split into left/right columns, OCRed independently, then recombined in zone order.
- OCR postprocessing applies figure-overlap suppression and clinical hardening heuristics:
  - `Lidocaine 49%` -> `Lidocaine 4%`
  - `Atropine 9.5 mg` -> `Atropine 0.5 mg`
  - `lyrnphadenopathy` -> `lymphadenopathy`
  - `hytnph` -> `lymph`
  - Lightweight Levenshtein correction for long clinical terms (e.g., `tracheobronchial`).
- Native/OCR fusion prefers native text unless OCR adds clear missing content.

Security/ops constraints:

- Raw PDF bytes and unredacted extracted text do not leave the browser.
- OCR/model assets are self-hosted same-origin (vendored under `ui/static/phi_redactor/vendor/`).
- Debug output is metrics-only (no raw clinical text).

### Reporter Builder Quick Notes

- Reporter Builder UI: `http://localhost:8000/ui/reporter_builder.html`
- Client-side PHI gate is enforced before seeding: `Run Detection` -> `Apply Redactions` -> `Seed Bundle`
- `POST /report/seed_from_text` supports:
  - `REPORTER_SEED_STRATEGY=registry_extract_fields` (default)
  - `REPORTER_SEED_STRATEGY=llm_findings` (reporter-only findings path)
  - `REPORTER_SEED_LLM_STRICT=1` to disable fallback

Reporter findings mode uses existing OpenAI-compatible settings:

- `LLM_PROVIDER=openai_compat`
- `OPENAI_MODEL_STRUCTURER=gpt-5-mini`
- `OPENAI_API_KEY=...`
- `OPENAI_OFFLINE=0` (or fallback/strict behavior applies)

### Reporter Prompt Sampling Tool

Generate random prompt/output reporter runs from training JSONL files:

```bash
python ops/tools/run_reporter_random_seeds.py \
  --input-dir /home/rjm/projects/proc_suite_notes/reporter_training/reporter_training \
  --count 20 \
  --seed 42 \
  --output reporter_tests.txt \
  --include-metadata-json
```

Outputs:
- `reporter_tests.txt`
- `reporter_tests.json` (or custom path via `--metadata-output`)

### Reporter LLM Findings Evaluator

```bash
PROCSUITE_ALLOW_ONLINE=1 \
LLM_PROVIDER=openai_compat \
OPENAI_MODEL_STRUCTURER=gpt-5-mini \
python ops/tools/eval_reporter_prompt_llm_findings.py
```

## Recent Updates (2026-01-25)

- **Schema refactor:** shared EBUS node-event types now live in `proc_schemas/shared/ebus_events.py` and are re-exported via `app/registry/schema/ebus_events.py`.
- **Granular split:** models moved to `app/registry/schema/granular_models.py` and logic to `app/registry/schema/granular_logic.py`; `app/registry/schema_granular.py` is a compat shim.
- **V2 dynamic builder:** moved to `app/registry/schema/v2_dynamic.py`; `app/registry/schema.py` is now a thin entrypoint preserving the `__path__` hack.
- **V3 extraction schema:** renamed to `app/registry/schema/ip_v3_extraction.py` with a compatibility re-export at `app/registry/schema/ip_v3.py`; the rich registry entry schema remains at `proc_schemas/registry/ip_v3.py`.
- **V3→V2 adapter:** now in `app/registry/schema/adapters/v3_to_v2.py` with a compat shim at `app/registry/adapters/v3_to_v2.py`.
- **Refactor notes/tests:** see `NOTES_SCHEMA_REFACTOR.md` and `tests/registry/test_schema_refactor_smoke.py`.

## Recent Updates (2026-01-24)

- **BLVR CPT derivation:** valve placement uses `31647` (initial lobe) + `31651` (each additional lobe); valve removal uses `31648` (initial lobe) + `31649` (each additional lobe).
- **Chartis bundling:** `31634` is derived only when Chartis is documented; suppressed when Chartis is in the same lobe as valve placement, and flagged for modifier documentation when distinct lobes are present.
- **Moderate sedation threshold:** `99152`/`99153` are derived only when `sedation.type="Moderate"`, `anesthesia_provider="Proceduralist"`, and intraservice minutes ≥10 (computed from start/end if needed).
- **Coding support + traceability:** extraction-first now populates `registry.coding_support` (rules applied + QA flags) and enriches `registry.billing.cpt_codes[]` with `description`, `derived_from`, and evidence spans.
- **Providers normalization:** added `providers_team[]` (auto-derived from legacy `providers` when missing).
- **Registry schema:** added `pathology_results.pdl1_tps_text` to preserve values like `"<1%"` or `">50%"`.
- **KB hygiene (Phase 0–2):** added `docs/KNOWLEDGE_INVENTORY.md`, `docs/KNOWLEDGE_RELEASE_CHECKLIST.md`, and `make validate-knowledge-release` for safer knowledge/schema updates.
- **KB version gating:** loaders now enforce KB filename semantic version ↔ internal `"version"` (override: `PSUITE_KNOWLEDGE_ALLOW_VERSION_MISMATCH=1`).
- **Single source of truth:** runtime code metadata/RVUs come from `master_code_index`, and synonym phrase lists are centralized in KB `synonyms`.

## Recent Updates (2026-02-13)

- **Reporter seed strategy switch:** `POST /report/seed_from_text` now supports `REPORTER_SEED_STRATEGY=llm_findings` (default remains `registry_extract_fields`).
- **LLM findings reporter path:** masked prompt -> evidence-cited findings -> synthetic NER -> `NERToRegistryMapper` -> `ClinicalGuardrails` -> deterministic CPT derivation -> existing Jinja templates.
- **Client-side PHI in Reporter Builder:** seeding is gated until local PHI detection + redaction are applied; seeded requests send `already_scrubbed=true`.
- **Reporter prompt tooling:** added `ops/tools/run_reporter_random_seeds.py` and `ops/tools/eval_reporter_prompt_llm_findings.py`.

## Key Modules

| Module | Description |
|--------|-------------|
| **`app/api/fastapi_app.py`** | Main FastAPI backend |
| **`app/coder/`** | CPT coding engine with CodingService (8-step pipeline) |
| **`ml/lib/ml_coder/`** | ML-based code predictor and training pipeline |
| **`app/registry/`** | Registry extraction with RegistryService and RegistryEngine |
| **`app/agents/`** | 3-agent pipeline: Parser → Summarizer → Structurer |
| **`app/reporter/`** | Template-based synoptic report generator |
| **`ui/static/phi_redactor/`** | Main UI (served at `/ui/`): client-side PHI scrubbing + clinical dashboard |

## System Architecture

> **Note (Current as of 2026-01):** The server enforces `PROCSUITE_PIPELINE_MODE=extraction_first`
> at startup. The **authoritative production endpoint** is `POST /api/v1/process`, and its
> primary pipeline is **Extraction‑First**: **Registry extraction → deterministic Registry→CPT rules**.
> The older **CPT-first (ML-first) hybrid** flows still exist in code for legacy endpoints and
> tooling, but are expected to be gated/disabled in production.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Procedure Note                               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Layer (app/api/)                      │
│  • /api/v1/process - Unified extraction-first endpoint (prod)       │
│  • /v1/coder/run - Legacy CPT coding endpoint (gated)               │
│  • /v1/registry/run - Legacy registry extraction endpoint (gated)   │
│  • /v1/report/render - Report generation endpoint                   │
└─────────────────────────────────────────────────────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│ RegistryService │    │    Reporter     │
│ (Extraction-    │    │ (Jinja temps)   │
│  First)         │    └─────────────────┘
└─────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ RegistryRecord (V3-shaped)   │
└──────────────────────────────┘
         │
         ▼
┌──────────────────────────────┐
│ Deterministic Registry→CPT   │
│ (no note parsing)            │
└──────────────────────────────┘
```

### Extraction-First Pipeline (Current: `/api/v1/process`)

The production pipeline (as exercised by the UI at `/ui/` and `POST /api/v1/process`) is:

1. **(Optional) PHI redaction** (skipped when `already_scrubbed=true`)
2. **Registry extraction** from note text (engine selected by `REGISTRY_EXTRACTION_ENGINE`, production requires `parallel_ner`)
3. **Deterministic Registry→CPT derivation** from the extracted `RegistryRecord` (no raw note parsing)
4. **RAW-ML auditing** (and optional self-correction) to detect omissions/mismatches
5. **UI-ready response** with evidence spans + review flags

### Legacy CPT-First / Hybrid-First Flows (kept for tooling, gated in prod)

Some older endpoints and internal tools still use a CPT-first hybrid approach:

1. **CPT Coding** → Get codes from SmartHybridOrchestrator
2. **CPT Mapping** → Map CPT codes to registry boolean flags
3. **LLM Extraction** → Extract additional fields via RegistryEngine
4. **Reconciliation** → Merge CPT-derived and LLM-extracted fields
5. **Validation** → Validate against IP_Registry.json schema

## Data & Schemas

| File | Purpose |
|------|---------|
| `data/knowledge/ip_coding_billing_v3_0.json` | CPT codes, RVUs, bundling rules |
| `data/knowledge/IP_Registry.json` | Registry schema definition |
| `data/knowledge/golden_extractions/` | Training data for ML models |
| `schemas/IP_Registry.json` | JSON Schema for validation |

## Testing

```bash
# Run all tests
make test

# Run specific test suites
pytest tests/coder/ -v          # Coder tests
pytest tests/registry/ -v       # Registry tests
pytest tests/ml_coder/ -v       # ML coder tests

# Validate registry extraction
make validate-registry

# Run preflight checks
make preflight
```

## Note for AI Assistants

**Please read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) before making changes.**

- Always edit `app/api/fastapi_app.py` (not `api/app.py` - deprecated)
- Use `CodingService` from `app/coder/application/coding_service.py`
- Use `RegistryService` from `app/registry/application/registry_service.py`
- Knowledge base is at `data/knowledge/ip_coding_billing_v3_0.json`
- Run `make test` before committing

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_PROVIDER` | LLM backend: `gemini` or `openai_compat` | `gemini` |
| `GEMINI_API_KEY` | API key for Gemini LLM | Required for LLM features |
| `GEMINI_OFFLINE` | Disable LLM calls (use stubs) | `1` |
| `REGISTRY_USE_STUB_LLM` | Use stub LLM for registry tests | `1` |
| `OPENAI_API_KEY` | API key for OpenAI-protocol backend (openai_compat) | Required unless `OPENAI_OFFLINE=1` |
| `OPENAI_BASE_URL` | Base URL for OpenAI-protocol backend (no `/v1`) | `https://api.openai.com` |
| `OPENAI_MODEL` | Default model name for openai_compat | Required unless `OPENAI_OFFLINE=1` |
| `OPENAI_MODEL_SUMMARIZER` | Model override for summarizer/focusing tasks (openai_compat only) | `OPENAI_MODEL` |
| `OPENAI_MODEL_STRUCTURER` | Model override for structurer tasks (openai_compat only) | `OPENAI_MODEL` |
| `OPENAI_MODEL_JUDGE` | Model override for self-correction judge (openai_compat only) | `OPENAI_MODEL` |
| `OPENAI_OFFLINE` | Disable openai_compat network calls (use stubs) | `0` |
| `REPORTER_SEED_STRATEGY` | Reporter seed mode: `registry_extract_fields` or `llm_findings` | `registry_extract_fields` |
| `REPORTER_SEED_LLM_STRICT` | In `llm_findings` mode, fail instead of fallback when LLM seeding errors | `0` |
| `OPENAI_PRIMARY_API` | Primary API: `responses` or `chat` | `responses` |
| `OPENAI_RESPONSES_FALLBACK_TO_CHAT` | Fall back to Chat Completions on 404 | `1` |
| `OPENAI_TIMEOUT_READ_REGISTRY_SECONDS` | Read timeout for registry tasks (seconds) | `180` |
| `OPENAI_TIMEOUT_READ_DEFAULT_SECONDS` | Read timeout for default tasks (seconds) | `60` |
| `PROCSUITE_SKIP_WARMUP` | Skip NLP model loading at startup | `false` |
| `PROCSUITE_PIPELINE_MODE` | Pipeline mode (startup-enforced): `extraction_first` | `extraction_first` |
| `REGISTRY_EXTRACTION_ENGINE` | Registry extraction engine: `engine`, `agents_focus_then_engine`, or `agents_structurer` | `engine` |
| `REGISTRY_AUDITOR_SOURCE` | Registry auditor source (extraction-first): `raw_ml` or `disabled` | `raw_ml` |
| `REGISTRY_ML_AUDIT_USE_BUCKETS` | Audit set = `high_conf + gray_zone` when `1`; else use `top_k + min_prob` | `1` |
| `REGISTRY_ML_AUDIT_TOP_K` | Audit top-k predictions when buckets disabled | `25` |
| `REGISTRY_ML_AUDIT_MIN_PROB` | Audit minimum probability when buckets disabled | `0.50` |
| `REGISTRY_ML_SELF_CORRECT_MIN_PROB` | Min prob for self-correction trigger candidates | `0.95` |
| `REGISTRY_SELF_CORRECT_ENABLED` | Enable guarded self-correction loop | `0` |
| `REGISTRY_SELF_CORRECT_ALLOWLIST` | Comma-separated JSON Pointer allowlist for self-correction patch paths (default: `app/registry/self_correction/validation.py` `ALLOWED_PATHS`) | `builtin` |
| `REGISTRY_SELF_CORRECT_MAX_ATTEMPTS` | Max successful auto-corrections per case | `1` |
| `REGISTRY_SELF_CORRECT_MAX_PATCH_OPS` | Max JSON Patch ops per proposal | `5` |

---

*Last updated: January 2026*
