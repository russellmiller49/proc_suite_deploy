## PHI data model (Phase 0.1)

Requirements captured from `docs/Multi_agent_collaboration/V8_MIGRATION_PLAN_UPDATED.md`:

- Encrypt raw PHI in a dedicated vault table; only the vault holds unsanitized data.
- Store scrubbed text plus entity placeholder map separately to feed CodingService/LLMs.
- Track processing status for the PHI→coding workflow (pending_review → phi_confirmed → processing → completed/failed).
- Maintain a HIPAA-style audit trail for all PHI access/decrypt actions.
- Capture scrubbing feedback (false positives/negatives, precision/recall) for ML tuning.
- Design for portability (Supabase/Postgres today, HIPAA-compliant vault later) and avoid logging raw PHI.

Proposed SQLAlchemy models (aligning with the migration plan):

- `PHIVault`: id (UUID PK), `encrypted_data` (bytes), `data_hash`, `encryption_algorithm`, `key_version`, `created_at`, `is_deleted`; one-to-one with `ProcedureData`; raw PHI lives only here.
- `ProcedureData`: id (UUID PK), `phi_vault_id` FK, `scrubbed_text`, `original_text_hash`, `entity_map` (JSON list), `status` (enum), optional `coding_results`, metadata (`document_type`, `specialty`), submit/review user fields, timestamps; no raw PHI.
- `AuditLog`: id (UUID PK); optional `phi_vault_id` / `procedure_data_id`; actor info; `action` enum; optional `action_detail`; request metadata (ip, agent, request_id, metadata JSON); timestamp. Used for PHI access audit.
- `ScrubbingFeedback`: id (UUID PK); `procedure_data_id` FK; `presidio_entities`, `confirmed_entities`, `false_positives`, `false_negatives`, `true_positives`, metrics (precision/recall/f1), metadata (document_type, specialty), `created_at`. Stores physician corrections; no raw PHI text.

Entity boundaries:

- Raw PHI: `PHIVault.encrypted_data` only (never logged).
- De-identified pipeline/LLM inputs: `ProcedureData.scrubbed_text` and `entity_map`.
- Audit + feedback: structured metadata only; no plaintext PHI stored or logged.
