import { decryptVaultJson, encryptVaultJson } from "./vaultClient.js";
import { validateCaseSnapshot } from "./vaultSchemas.js";

const SNAPSHOT_INDEX_STORAGE_PREFIX = "procsuite.vault.snapshots.v1";
const SNAPSHOT_RECORD_STORAGE_PREFIX = "procsuite.vault.snapshot.v1";
const SNAPSHOT_SCHEMA_VERSION = 1;

// Keep well under common per-item localStorage limits (~5MB) and leave room for overhead.
const MAX_CIPHERTEXT_CHUNK_CHARS = 900_000;

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asTrimmedString(value) {
  return String(value ?? "").trim();
}

function canUseLocalStorage() {
  try {
    if (!globalThis.localStorage) return false;
    const probeKey = "__ps_snapshot_probe__";
    globalThis.localStorage.setItem(probeKey, "1");
    globalThis.localStorage.removeItem(probeKey);
    return true;
  } catch {
    return false;
  }
}

function snapshotIndexStorageKey(userId) {
  return `${SNAPSHOT_INDEX_STORAGE_PREFIX}:${asTrimmedString(userId)}`;
}

function snapshotRecordStorageKey(userId, registryUuid) {
  return `${SNAPSHOT_RECORD_STORAGE_PREFIX}:${asTrimmedString(userId)}:${asTrimmedString(registryUuid)}`;
}

function snapshotChunkStorageKey(userId, registryUuid, chunkIndex) {
  return `${SNAPSHOT_RECORD_STORAGE_PREFIX}:${asTrimmedString(userId)}:${asTrimmedString(registryUuid)}:chunk:${chunkIndex}`;
}

function safeJsonParse(raw) {
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

function readSnapshotIndex(userId) {
  if (!canUseLocalStorage()) return { schema_version: SNAPSHOT_SCHEMA_VERSION, snapshots: {} };
  const raw = globalThis.localStorage.getItem(snapshotIndexStorageKey(userId));
  const parsed = safeJsonParse(raw);
  if (!isRecord(parsed)) return { schema_version: SNAPSHOT_SCHEMA_VERSION, snapshots: {} };
  const snapshots = isRecord(parsed.snapshots) ? parsed.snapshots : {};
  return {
    schema_version: SNAPSHOT_SCHEMA_VERSION,
    snapshots,
  };
}

function writeSnapshotIndex(userId, index) {
  if (!canUseLocalStorage()) return;
  const payload = isRecord(index) ? index : { schema_version: SNAPSHOT_SCHEMA_VERSION, snapshots: {} };
  try {
    globalThis.localStorage.setItem(snapshotIndexStorageKey(userId), JSON.stringify(payload));
  } catch {
    // Ignore quota/write errors.
  }
}

function normalizeEncryptedSnapshotRecordShape(value) {
  const record = isRecord(value) ? value : {};
  const registryUuid = asTrimmedString(record.registry_uuid);
  const iv = asTrimmedString(record.iv_b64);
  const version = Number(record.crypto_version);
  const chunkCount = record.chunk_count == null ? null : Number(record.chunk_count);
  const ciphertext = asTrimmedString(record.ciphertext_b64);

  if (!registryUuid || !iv || !Number.isFinite(version)) return null;

  if (Number.isFinite(chunkCount) && chunkCount > 1) {
    return {
      registry_uuid: registryUuid,
      iv_b64: iv,
      crypto_version: Math.trunc(version),
      chunk_count: Math.trunc(chunkCount),
    };
  }

  if (!ciphertext) return null;
  return {
    registry_uuid: registryUuid,
    iv_b64: iv,
    crypto_version: Math.trunc(version),
    ciphertext_b64: ciphertext,
    chunk_count: 1,
  };
}

function cleanupSnapshotChunks(userId, registryUuid, chunkCount) {
  if (!canUseLocalStorage()) return;
  const safeCount = Number.isFinite(chunkCount) ? Math.trunc(chunkCount) : 0;
  if (safeCount <= 1) return;
  for (let i = 0; i < safeCount; i += 1) {
    try {
      globalThis.localStorage.removeItem(snapshotChunkStorageKey(userId, registryUuid, i));
    } catch {
      // ignore
    }
  }
}

function deriveSnapshotMeta(snapshotData, registryUuid, encryptedRecord, savedAtIso) {
  const safeUuid = asTrimmedString(registryUuid);
  const perCodeBilling = Array.isArray(snapshotData?.per_code_billing) ? snapshotData.per_code_billing : [];
  const cptCodes = Array.isArray(snapshotData?.cpt_codes) ? snapshotData.cpt_codes : [];
  const registryCpt =
    Array.isArray(snapshotData?.registry?.billing?.cpt_codes) ? snapshotData.registry.billing.cpt_codes : [];
  const cptCount = perCodeBilling.length || cptCodes.length || registryCpt.length || 0;
  const totalWorkRvu = Number.isFinite(snapshotData?.total_work_rvu) ? Number(snapshotData.total_work_rvu) : null;

  return {
    registry_uuid: safeUuid,
    saved_at: savedAtIso,
    schema_version: SNAPSHOT_SCHEMA_VERSION,
    cpt_count: cptCount,
    total_work_rvu: totalWorkRvu,
    estimated_payment: Number.isFinite(snapshotData?.estimated_payment) ? Number(snapshotData.estimated_payment) : null,
    chunk_count: Number.isFinite(encryptedRecord?.chunk_count) ? Number(encryptedRecord.chunk_count) : 1,
    ciphertext_chars: asTrimmedString(encryptedRecord?.ciphertext_b64).length || 0,
  };
}

export async function persistCaseSnapshot(vmk, userId, registryUuid, snapshotData) {
  const safeUserId = asTrimmedString(userId);
  const safeRegistryUuid = asTrimmedString(registryUuid);
  if (!vmk || !safeUserId || !safeRegistryUuid) return null;
  if (!canUseLocalStorage()) return null;

  const encrypted = await encryptVaultJson(vmk, safeUserId, safeRegistryUuid, snapshotData);
  const ciphertext = asTrimmedString(encrypted?.ciphertext_b64);
  const iv = asTrimmedString(encrypted?.iv_b64);
  if (!ciphertext || !iv) return null;

  const recordKey = snapshotRecordStorageKey(safeUserId, safeRegistryUuid);
  const savedAtIso = new Date().toISOString();

  const priorRaw = safeJsonParse(globalThis.localStorage.getItem(recordKey));
  const prior = normalizeEncryptedSnapshotRecordShape(priorRaw);
  if (prior) cleanupSnapshotChunks(safeUserId, safeRegistryUuid, prior.chunk_count);

  const needsChunking = ciphertext.length > MAX_CIPHERTEXT_CHUNK_CHARS;
  let storedRecord = null;

  if (!needsChunking) {
    storedRecord = {
      registry_uuid: safeRegistryUuid,
      ciphertext_b64: ciphertext,
      iv_b64: iv,
      crypto_version: encrypted.crypto_version,
      chunk_count: 1,
    };
    try {
      globalThis.localStorage.setItem(recordKey, JSON.stringify(storedRecord));
    } catch {
      return null;
    }
  } else {
    const chunks = [];
    for (let i = 0; i < ciphertext.length; i += MAX_CIPHERTEXT_CHUNK_CHARS) {
      chunks.push(ciphertext.slice(i, i + MAX_CIPHERTEXT_CHUNK_CHARS));
    }

    storedRecord = {
      registry_uuid: safeRegistryUuid,
      iv_b64: iv,
      crypto_version: encrypted.crypto_version,
      chunk_count: chunks.length,
    };

    try {
      globalThis.localStorage.setItem(recordKey, JSON.stringify(storedRecord));
      for (let i = 0; i < chunks.length; i += 1) {
        globalThis.localStorage.setItem(snapshotChunkStorageKey(safeUserId, safeRegistryUuid, i), chunks[i]);
      }
    } catch {
      // Best-effort cleanup if we failed midway.
      cleanupSnapshotChunks(safeUserId, safeRegistryUuid, storedRecord.chunk_count);
      try {
        globalThis.localStorage.removeItem(recordKey);
      } catch {
        // ignore
      }
      return null;
    }
  }

  const index = readSnapshotIndex(safeUserId);
  const meta = deriveSnapshotMeta(snapshotData, safeRegistryUuid, storedRecord, savedAtIso);
  index.snapshots[safeRegistryUuid] = meta;
  writeSnapshotIndex(safeUserId, index);
  return meta;
}

export async function loadCaseSnapshot(vmk, userId, registryUuid) {
  const safeUserId = asTrimmedString(userId);
  const safeRegistryUuid = asTrimmedString(registryUuid);
  if (!vmk || !safeUserId || !safeRegistryUuid) return null;
  if (!canUseLocalStorage()) return null;

  const recordKey = snapshotRecordStorageKey(safeUserId, safeRegistryUuid);
  const raw = safeJsonParse(globalThis.localStorage.getItem(recordKey));
  const record = normalizeEncryptedSnapshotRecordShape(raw);
  if (!record) return null;

  let ciphertext_b64 = record.ciphertext_b64;
  if (!ciphertext_b64 && record.chunk_count > 1) {
    const parts = [];
    for (let i = 0; i < record.chunk_count; i += 1) {
      const chunk = globalThis.localStorage.getItem(snapshotChunkStorageKey(safeUserId, safeRegistryUuid, i));
      if (typeof chunk !== "string" || !chunk) return null;
      parts.push(chunk);
    }
    ciphertext_b64 = parts.join("");
  }
  if (!ciphertext_b64) return null;

  const decrypted = await decryptVaultJson(vmk, safeUserId, safeRegistryUuid, {
    ciphertext_b64,
    iv_b64: record.iv_b64,
    crypto_version: record.crypto_version,
  });
  return validateCaseSnapshot(decrypted, safeRegistryUuid);
}

export function listCaseSnapshotMeta(userId) {
  const safeUserId = asTrimmedString(userId);
  if (!safeUserId) return [];
  const index = readSnapshotIndex(safeUserId);
  const rows = [];
  for (const meta of Object.values(index.snapshots || {})) {
    if (!isRecord(meta)) continue;
    const registryUuid = asTrimmedString(meta.registry_uuid);
    const savedAt = asTrimmedString(meta.saved_at);
    if (!registryUuid || !savedAt) continue;
    rows.push({
      registry_uuid: registryUuid,
      saved_at: savedAt,
      cpt_count: Number.isFinite(meta.cpt_count) ? Number(meta.cpt_count) : 0,
      total_work_rvu: Number.isFinite(meta.total_work_rvu) ? Number(meta.total_work_rvu) : null,
      estimated_payment: Number.isFinite(meta.estimated_payment) ? Number(meta.estimated_payment) : null,
      chunk_count: Number.isFinite(meta.chunk_count) ? Number(meta.chunk_count) : 1,
    });
  }
  rows.sort((a, b) => String(b.saved_at).localeCompare(String(a.saved_at)));
  return rows;
}
