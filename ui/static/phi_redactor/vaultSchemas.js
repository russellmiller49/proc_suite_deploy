import { z } from "./vendor/zod.bundle.mjs";

const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;

export const VaultPatientSchema = z.object({
  schema_version: z.number().int(),
  patient_label: z.string().min(1),
  index_date: z.string().regex(ISO_DATE_RE).nullable(),
  local_meta: z.record(z.any()),
  registry_uuid: z.string().min(1),
  saved_at: z.string().min(1),
});

export const CaseSnapshotSchema = z
  .object({
    registry_uuid: z.string().min(1),
    registry: z.record(z.any()),
  })
  .passthrough();

export function validateVaultPatient(value) {
  return VaultPatientSchema.parse(value);
}

export function validateCaseSnapshot(value, expectedRegistryUuid = null) {
  const parsed = CaseSnapshotSchema.parse(value);
  const expected = String(expectedRegistryUuid || "").trim();
  if (expected && String(parsed.registry_uuid || "").trim() && String(parsed.registry_uuid).trim() !== expected) {
    throw new Error("Snapshot registry_uuid mismatch");
  }
  return parsed;
}

