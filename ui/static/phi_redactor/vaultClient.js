import { validateVaultPatient } from "./vaultSchemas.js";

const CRYPTO_VERSION = 1;
const DEFAULT_PBKDF2_ITERATIONS = 210000;
const PBKDF2_HASH = "PBKDF2-SHA256";

const WRAP_AAD_SCOPE = "vault-wrap";
const RECORD_AAD_SCOPE = "vault-record";
const JSON_AAD_SCOPE = "vault-json";
const encoder = new TextEncoder();
const decoder = new TextDecoder();
const ISO_DATE_RE = /^\d{4}-\d{2}-\d{2}$/;
const LOCAL_RECORDS_STORAGE_PREFIX = "procsuite.vault.records.v1";

function asBuffer(bytes) {
  const copy = new Uint8Array(bytes.byteLength);
  copy.set(bytes);
  return copy.buffer;
}

function getSubtle() {
  const subtle = globalThis.crypto?.subtle;
  if (!subtle) throw new Error("WebCrypto SubtleCrypto is unavailable");
  return subtle;
}

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asTrimmedString(value) {
  return String(value ?? "").trim();
}

function normalizeIndexDate(value) {
  const text = asTrimmedString(value);
  if (!text) return null;
  return ISO_DATE_RE.test(text) ? text : null;
}

function normalizeSavedAt(value) {
  const text = asTrimmedString(value);
  if (text) return text;
  return new Date().toISOString();
}

function normalizePatientLabel(raw, registryUuid) {
  const direct = asTrimmedString(raw.patient_label);
  if (direct) return direct;
  const fallback = asTrimmedString(raw.display_name || raw.name || raw.mrn);
  if (fallback) return fallback;
  const prefix = asTrimmedString(registryUuid).slice(0, 8) || "unknown";
  return `Case ${prefix}`;
}

export function normalizeVaultPatientData(patientJson, registryUuid) {
  const raw = isRecord(patientJson) ? patientJson : {};
  const localMeta = isRecord(raw.local_meta) ? { ...raw.local_meta } : {};
  const legacyMrn = asTrimmedString(raw.mrn);
  if (legacyMrn && localMeta.mrn === undefined) localMeta.mrn = legacyMrn;
  const legacyCustomNotes = asTrimmedString(raw.custom_notes);
  if (legacyCustomNotes && localMeta.custom_notes === undefined) localMeta.custom_notes = legacyCustomNotes;
  const normalizedRegistryUuid = asTrimmedString(raw.registry_uuid) || asTrimmedString(registryUuid);
  return {
    schema_version: 2,
    patient_label: normalizePatientLabel(raw, normalizedRegistryUuid),
    index_date: normalizeIndexDate(raw.index_date),
    local_meta: localMeta,
    registry_uuid: normalizedRegistryUuid,
    saved_at: normalizeSavedAt(raw.saved_at),
  };
}

function buildWrapAad(userId, cryptoVersion) {
  return asBuffer(encoder.encode(`${WRAP_AAD_SCOPE}|v${cryptoVersion}|u:${userId}`));
}

function buildRecordAad(userId, registryUuid, cryptoVersion) {
  return asBuffer(
    encoder.encode(`${RECORD_AAD_SCOPE}|v${cryptoVersion}|u:${userId}|r:${String(registryUuid || "").toLowerCase()}`)
  );
}

function buildJsonAad(userId, registryUuid, cryptoVersion) {
  return asBuffer(
    encoder.encode(`${JSON_AAD_SCOPE}|v${cryptoVersion}|u:${userId}|r:${String(registryUuid || "").toLowerCase()}`)
  );
}

function randomBytes(size) {
  const bytes = new Uint8Array(size);
  globalThis.crypto.getRandomValues(bytes);
  return bytes;
}

export function bytesToBase64(bytes) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  if (typeof globalThis.btoa === "function") return globalThis.btoa(binary);
  throw new Error("No base64 encoder available");
}

export function base64ToBytes(base64) {
  if (typeof base64 !== "string") throw new Error("Invalid base64 input");
  if (typeof globalThis.atob !== "function") throw new Error("No base64 decoder available");
  const binary = globalThis.atob(base64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) out[i] = binary.charCodeAt(i);
  return out;
}

async function deriveKekFromPassword(password, saltBytes, iterations) {
  const subtle = getSubtle();
  const keyMaterial = await subtle.importKey(
    "raw",
    asBuffer(encoder.encode(String(password || ""))),
    { name: "PBKDF2" },
    false,
    ["deriveKey"]
  );
  return subtle.deriveKey(
    {
      name: "PBKDF2",
      salt: asBuffer(saltBytes),
      iterations,
      hash: "SHA-256",
    },
    keyMaterial,
    { name: "AES-GCM", length: 256 },
    false,
    ["wrapKey", "unwrapKey"]
  );
}

export async function initNewVault(password, userId) {
  const subtle = getSubtle();
  const vmk = await subtle.generateKey({ name: "AES-GCM", length: 256 }, true, ["encrypt", "decrypt"]);
  const salt = randomBytes(16);
  const iv = randomBytes(12);
  const kek = await deriveKekFromPassword(password, salt, DEFAULT_PBKDF2_ITERATIONS);
  const wrapped = await subtle.wrapKey("raw", vmk, kek, {
    name: "AES-GCM",
    iv: asBuffer(iv),
    additionalData: buildWrapAad(userId, CRYPTO_VERSION),
    tagLength: 128,
  });

  return {
    settingsPayload: {
      wrapped_vmk_b64: bytesToBase64(new Uint8Array(wrapped)),
      wrap_iv_b64: bytesToBase64(iv),
      kdf_salt_b64: bytesToBase64(salt),
      kdf_iterations: DEFAULT_PBKDF2_ITERATIONS,
      kdf_hash: PBKDF2_HASH,
      crypto_version: CRYPTO_VERSION,
    },
    vmk,
  };
}

export async function unlockVault(password, userId, settings) {
  if (!settings || Number(settings.crypto_version) !== CRYPTO_VERSION) {
    throw new Error("Unsupported crypto version");
  }
  const subtle = getSubtle();
  const salt = base64ToBytes(settings.kdf_salt_b64);
  const iv = base64ToBytes(settings.wrap_iv_b64);
  const wrapped = base64ToBytes(settings.wrapped_vmk_b64);
  const kek = await deriveKekFromPassword(password, salt, Number(settings.kdf_iterations || 0));

  return subtle.unwrapKey(
    "raw",
    asBuffer(wrapped),
    kek,
    {
      name: "AES-GCM",
      iv: asBuffer(iv),
      additionalData: buildWrapAad(userId, CRYPTO_VERSION),
      tagLength: 128,
    },
    { name: "AES-GCM", length: 256 },
    true,
    ["encrypt", "decrypt"]
  );
}

export async function encryptPatientData(vmk, userId, registryUuid, patientJson) {
  const subtle = getSubtle();
  const iv = randomBytes(12);
  const normalized = normalizeVaultPatientData(patientJson, registryUuid);
  const plaintext = encoder.encode(JSON.stringify(normalized));
  const ciphertext = await subtle.encrypt(
    {
      name: "AES-GCM",
      iv: asBuffer(iv),
      additionalData: buildRecordAad(userId, registryUuid, CRYPTO_VERSION),
      tagLength: 128,
    },
    vmk,
    asBuffer(plaintext)
  );

  return {
    registry_uuid: registryUuid,
    ciphertext_b64: bytesToBase64(new Uint8Array(ciphertext)),
    iv_b64: bytesToBase64(iv),
    crypto_version: CRYPTO_VERSION,
  };
}

export async function decryptPatientData(vmk, userId, registryUuid, record) {
  if (!record || Number(record.crypto_version) !== CRYPTO_VERSION) {
    throw new Error("Unsupported crypto version");
  }
  const subtle = getSubtle();
  const ciphertext = base64ToBytes(record.ciphertext_b64);
  const iv = base64ToBytes(record.iv_b64);
  const plaintext = await subtle.decrypt(
    {
      name: "AES-GCM",
      iv: asBuffer(iv),
      additionalData: buildRecordAad(userId, registryUuid, CRYPTO_VERSION),
      tagLength: 128,
    },
    vmk,
    asBuffer(ciphertext)
  );
  const parsed = JSON.parse(decoder.decode(plaintext));
  return validateVaultPatient(normalizeVaultPatientData(parsed, registryUuid));
}

export async function encryptVaultJson(vmk, userId, registryUuid, jsonObj) {
  const subtle = getSubtle();
  const iv = randomBytes(12);
  const plaintext = encoder.encode(JSON.stringify(jsonObj ?? null));
  const ciphertext = await subtle.encrypt(
    {
      name: "AES-GCM",
      iv: asBuffer(iv),
      additionalData: buildJsonAad(userId, registryUuid, CRYPTO_VERSION),
      tagLength: 128,
    },
    vmk,
    asBuffer(plaintext)
  );

  return {
    registry_uuid: registryUuid,
    ciphertext_b64: bytesToBase64(new Uint8Array(ciphertext)),
    iv_b64: bytesToBase64(iv),
    crypto_version: CRYPTO_VERSION,
  };
}

export async function decryptVaultJson(vmk, userId, registryUuid, record) {
  if (!record || Number(record.crypto_version) !== CRYPTO_VERSION) {
    throw new Error("Unsupported crypto version");
  }
  const subtle = getSubtle();
  const ciphertext = base64ToBytes(record.ciphertext_b64);
  const iv = base64ToBytes(record.iv_b64);
  const plaintext = await subtle.decrypt(
    {
      name: "AES-GCM",
      iv: asBuffer(iv),
      additionalData: buildJsonAad(userId, registryUuid, CRYPTO_VERSION),
      tagLength: 128,
    },
    vmk,
    asBuffer(ciphertext)
  );
  return JSON.parse(decoder.decode(plaintext));
}

function userHeaders(userId) {
  return {
    "Content-Type": "application/json",
    "X-User-Id": String(userId || "").trim(),
  };
}

async function safeJson(response) {
  const text = await response.text();
  if (!text) return null;
  return JSON.parse(text);
}

async function safeDetail(response, { maxLen = 280 } = {}) {
  try {
    const text = await response.text();
    if (!text) return "";
    try {
      const parsed = JSON.parse(text);
      const detail = parsed?.detail;
      if (typeof detail === "string" && detail.trim()) return detail.trim();
    } catch {
      // ignore JSON parse failures
    }
    return String(text).trim().slice(0, maxLen);
  } catch {
    return "";
  }
}

function localRecordsStorageKey(userId) {
  return `${LOCAL_RECORDS_STORAGE_PREFIX}:${String(userId || "").trim()}`;
}

function canUseLocalStorage() {
  try {
    if (!globalThis.localStorage) return false;
    const probeKey = "__ps_vault_probe__";
    globalThis.localStorage.setItem(probeKey, "1");
    globalThis.localStorage.removeItem(probeKey);
    return true;
  } catch {
    return false;
  }
}

function parseLocalRecordStore(userId) {
  if (!canUseLocalStorage()) return {};
  try {
    const raw = globalThis.localStorage.getItem(localRecordsStorageKey(userId));
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return isRecord(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function writeLocalRecordStore(userId, store) {
  if (!canUseLocalStorage()) return;
  try {
    globalThis.localStorage.setItem(localRecordsStorageKey(userId), JSON.stringify(store || {}));
  } catch {
    // Ignore quota/write errors; caller will still rely on in-memory state.
  }
}

function normalizeEncryptedRecordShape(value) {
  const record = isRecord(value) ? value : {};
  const registryUuid = asTrimmedString(record.registry_uuid);
  const ciphertext = asTrimmedString(record.ciphertext_b64);
  const iv = asTrimmedString(record.iv_b64);
  const version = Number(record.crypto_version);
  if (!registryUuid || !ciphertext || !iv || !Number.isFinite(version)) return null;
  return {
    registry_uuid: registryUuid,
    ciphertext_b64: ciphertext,
    iv_b64: iv,
    crypto_version: Math.trunc(version),
  };
}

function listLocalEncryptedRecords(userId) {
  const store = parseLocalRecordStore(userId);
  const records = [];
  for (const value of Object.values(store)) {
    const normalized = normalizeEncryptedRecordShape(value);
    if (normalized) records.push(normalized);
  }
  return records;
}

function putLocalEncryptedRecord(userId, encryptedPayload) {
  const normalized = normalizeEncryptedRecordShape(encryptedPayload);
  if (!normalized) return;
  const store = parseLocalRecordStore(userId);
  store[normalized.registry_uuid] = normalized;
  writeLocalRecordStore(userId, store);
}

function removeLocalEncryptedRecord(userId, registryUuid) {
  const key = asTrimmedString(registryUuid);
  if (!key) return;
  const store = parseLocalRecordStore(userId);
  if (!Object.prototype.hasOwnProperty.call(store, key)) return;
  delete store[key];
  writeLocalRecordStore(userId, store);
}

function buildAnchorPatient(registryUuid) {
  const prefix = asTrimmedString(registryUuid).slice(0, 8) || "unknown";
  return {
    schema_version: 2,
    patient_label: `Case ${prefix}`,
    index_date: null,
    local_meta: {},
    registry_uuid: asTrimmedString(registryUuid),
    saved_at: new Date().toISOString(),
  };
}

function hasLocalIdentityData(patientJson, registryUuid) {
  const normalized = normalizeVaultPatientData(patientJson, registryUuid);
  if (normalized.index_date) return true;
  if (isRecord(normalized.local_meta) && Object.keys(normalized.local_meta).length > 0) return true;
  const defaultLabel = buildAnchorPatient(registryUuid).patient_label;
  return asTrimmedString(normalized.patient_label) !== asTrimmedString(defaultLabel);
}

async function upsertRemoteAnchorRecord({ apiBase, userId, vmk, registryUuid }) {
  const anchorPatient = buildAnchorPatient(registryUuid);
  const anchorPayload = await encryptPatientData(vmk, userId, registryUuid, anchorPatient);
  const res = await fetch(`${apiBase}/record`, {
    method: "PUT",
    headers: userHeaders(userId),
    body: JSON.stringify(anchorPayload),
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`Failed to save vault record (${res.status}${detail ? `): ${detail}` : ")"}`);
  }
  return safeJson(res);
}

export async function unlockOrInitVault({ apiBase = "/api/v1/vault", userId, password }) {
  const settingsRes = await fetch(`${apiBase}/settings`, {
    method: "GET",
    headers: userHeaders(userId),
  });

  if (settingsRes.status === 404) {
    const { settingsPayload, vmk } = await initNewVault(password, userId);
    const putRes = await fetch(`${apiBase}/settings`, {
      method: "PUT",
      headers: userHeaders(userId),
      body: JSON.stringify(settingsPayload),
    });
    if (!putRes.ok) {
      const detail = await safeDetail(putRes);
      throw new Error(
        `Failed to save new vault settings (${putRes.status}${detail ? `): ${detail}` : ")"}`,
      );
    }
    return { vmk, created: true };
  }

  if (!settingsRes.ok) {
    const detail = await safeDetail(settingsRes);
    throw new Error(
      `Failed to load vault settings (${settingsRes.status}${detail ? `): ${detail}` : ")"}`,
    );
  }
  const settings = await safeJson(settingsRes);
  const vmk = await unlockVault(password, userId, settings);
  return { vmk, created: false };
}

export async function loadVaultPatients({ apiBase = "/api/v1/vault", userId, vmk }) {
  const map = new Map();
  const localRows = listLocalEncryptedRecords(userId);
  for (const row of localRows) {
    const registryUuid = String(row?.registry_uuid || "");
    if (!registryUuid) continue;
    try {
      const decrypted = await decryptPatientData(vmk, userId, registryUuid, row);
      map.set(registryUuid, decrypted);
    } catch {
      // Ignore stale/corrupt local entries.
    }
  }
  if (map.size > 0) {
    return map;
  }

  // Backward compatibility migration:
  // 1) Read legacy server-side ciphertext records.
  // 2) Re-save them locally (browser-only).
  // 3) Overwrite server copy with a non-identifying anchor record.
  const res = await fetch(`${apiBase}/records`, {
    method: "GET",
    headers: userHeaders(userId),
  });
  if (!res.ok) {
    const detail = await safeDetail(res);
    throw new Error(`Failed to load vault records (${res.status}${detail ? `): ${detail}` : ")"}`);
  }
  const rows = (await safeJson(res)) || [];
  for (const row of rows) {
    const registryUuid = String(row?.registry_uuid || "");
    if (!registryUuid) continue;
    const decrypted = await decryptPatientData(vmk, userId, registryUuid, row);
    map.set(registryUuid, decrypted);
    putLocalEncryptedRecord(userId, {
      registry_uuid: registryUuid,
      ciphertext_b64: row.ciphertext_b64,
      iv_b64: row.iv_b64,
      crypto_version: row.crypto_version,
    });
    if (hasLocalIdentityData(decrypted, registryUuid)) {
      try {
        await upsertRemoteAnchorRecord({ apiBase, userId, vmk, registryUuid });
      } catch (err) {
        console.warn("Vault anchor migration failed (non-fatal):", err);
      }
    }
  }
  return map;
}

export async function upsertVaultPatient({ apiBase = "/api/v1/vault", userId, vmk, registryUuid, patientJson }) {
  const normalizedPatient = normalizeVaultPatientData(patientJson, registryUuid);
  const localPayload = await encryptPatientData(vmk, userId, registryUuid, normalizedPatient);
  putLocalEncryptedRecord(userId, localPayload);
  return upsertRemoteAnchorRecord({ apiBase, userId, vmk, registryUuid });
}

export async function deleteVaultPatient({ apiBase = "/api/v1/vault", userId, registryUuid }) {
  removeLocalEncryptedRecord(userId, registryUuid);
  const res = await fetch(`${apiBase}/records/${encodeURIComponent(registryUuid)}`, {
    method: "DELETE",
    headers: userHeaders(userId),
  });
  if (!res.ok && res.status !== 404) {
    const detail = await safeDetail(res);
    throw new Error(`Failed to delete vault record (${res.status}${detail ? `): ${detail}` : ")"}`);
  }
  return (await safeJson(res)) || { ok: true, registry_uuid: registryUuid };
}
