import { repairSpeechTranscript } from "./speechTranscriptRepair.js";

const statusTextEl = document.getElementById("statusText");
const actionBannerEl = document.getElementById("actionBanner");
const seedTextEl = document.getElementById("seedText");
const speechStartBtn = document.getElementById("speechStartBtn");
const speechStopBtn = document.getElementById("speechStopBtn");
const speechDiscardBtn = document.getElementById("speechDiscardBtn");
const speechCloudFallbackBtn = document.getElementById("speechCloudFallbackBtn");
const speechStartConfirmModalEl = document.getElementById("speechStartConfirmModal");
const speechStartConfirmBtn = document.getElementById("speechStartConfirmBtn");
const speechStartCancelBtn = document.getElementById("speechStartCancelBtn");
const speechStatusTextEl = document.getElementById("speechStatusText");
const speechWarningTextEl = document.getElementById("speechWarningText");
const speechModelSelectEl = document.getElementById("speechModelSelect");
const speechModelHintTextEl = document.getElementById("speechModelHintText");
const speechSourceBadgeEl = document.getElementById("speechSourceBadge");
const strictToggleEl = document.getElementById("strictToggle");
const seedBtn = document.getElementById("seedBtn");
const refreshBtn = document.getElementById("refreshBtn");
const clearBtn = document.getElementById("clearBtn");
const applyPatchBtn = document.getElementById("applyPatchBtn");
const transferToDashboardBtn = document.getElementById("transferToDashboardBtn");
const completenessPromptsCardEl = document.getElementById("completenessPromptsCard");
const completenessPromptsBodyEl = document.getElementById("completenessPromptsBody");
const completenessInsertBtn = document.getElementById("completenessInsertBtn");
const completenessCopyBtn = document.getElementById("completenessCopyBtn");

const phiRunBtn = document.getElementById("phiRunBtn");
const phiApplyBtn = document.getElementById("phiApplyBtn");
const phiRevertBtn = document.getElementById("phiRevertBtn");
const phiRedactProvidersToggleEl = document.getElementById("phiRedactProvidersToggle");
const phiStatusTextEl = document.getElementById("phiStatusText");
const phiProgressTextEl = document.getElementById("phiProgressText");
const phiSummaryTextEl = document.getElementById("phiSummaryText");
const phiDetectionsListEl = document.getElementById("phiDetectionsList");
const phiDetectionsCountEl = document.getElementById("phiDetectionsCount");
const phiEntityTypeSelectEl = document.getElementById("phiEntityTypeSelect");
const phiAddRedactionBtn = document.getElementById("phiAddRedactionBtn");
const phiConfirmAckEl = document.getElementById("phiConfirmAck");

const summaryQuestionsEl = document.getElementById("summaryQuestions");
const summaryIssuesEl = document.getElementById("summaryIssues");
const summaryWarningsEl = document.getElementById("summaryWarnings");
const summarySuggestionsEl = document.getElementById("summarySuggestions");

const markdownOutputEl = document.getElementById("markdownOutput");
const questionsHostEl = document.getElementById("questionsHost");
const validationHostEl = document.getElementById("validationHost");
const bundleJsonEl = document.getElementById("bundleJson");
const patchJsonEl = document.getElementById("patchJson");

const DASHBOARD_TRANSFER_STORAGE_KEY = "ps.reporter_to_dashboard_note_v1";
const DASHBOARD_TO_REPORTER_STORAGE_KEY = "ps.dashboard_to_reporter_note_v1";

const COMPLETENESS_ADDENDUM_BEGIN = "[PS_COMPLETENESS_BEGIN]";
const COMPLETENESS_ADDENDUM_END = "[PS_COMPLETENESS_END]";

let completenessPrompts = [];
let completenessValuesByPath = new Map(); // key: dotted path, value: raw string or string[]

const state = {
  bundle: null,
  seed: null,
  verify: null,
  render: null,
  questions: [],
  lastPatch: [],
  busy: false,
};

const REPORTER_SPEECH_MAX_SECONDS = 60;
const REPORTER_SPEECH_SAMPLE_RATE = 16000;
const REPORTER_SPEECH_WORKER_PATH = new URL("./speech.worker.js", import.meta.url).toString();
const REPORTER_SPEECH_MODEL_STORAGE_KEY = "ps.reporter_speech_model_v1";
const REPORTER_SPEECH_DEFAULT_MODEL_KEY = "tiny";
const REPORTER_SPEECH_TRANSCRIBE_TIMEOUT_MS = Object.freeze({
  tiny: 45_000,
  base: 90_000,
});
const REPORTER_SPEECH_MODELS = Object.freeze({
  base: Object.freeze({
    key: "base",
    label: "Base",
  }),
  tiny: Object.freeze({
    key: "tiny",
    label: "Tiny",
  }),
});

let speechWorker = null;
let speechWorkerReady = false;
let speechWorkerUnavailableReason = "";
let speechPendingRequest = null;
let speechRequestCounter = 0;
let speechMediaRecorder = null;
let speechMediaStream = null;
let speechRecording = false;
let speechRequestingAccess = false;
let speechLocalPending = false;
let speechCloudPending = false;
let speechChunks = [];
let speechLastRecordingBlob = null;
let speechStopTimer = null;
let speechSource = "";
let speechFallbackUsed = false;
let speechCleaned = false;
let speechIgnoreNextStop = false;
let preservePhiStateOnNextSeedInput = false;
let speechStartConfirmedForCurrentNote = false;
let speechModelKey = loadStoredSpeechModelKey();
let speechWorkerModelKey = speechModelKey;
let speechWorkerModelLabel = getSpeechModelLabel(speechModelKey);

// --- Client-side PHI redaction state (local worker) ---

const PHI_WORKER_BASE_CONFIG = {
  aiThreshold: 0.5,
  debug: false,
  // Quantized INT8 ONNX can silently collapse to all-"O" under WASM.
  // Keep this ON until quantized inference is validated end-to-end.
  forceUnquantized: true,
  // Merge mode: "union" (default, safer) or "best_of" (legacy)
  mergeMode: "union",
  // If false, clinician/provider/staff names are treated as PHI and can be redacted.
  protectProviders: false,
};

let phiWorker = null;
let phiWorkerReady = false;
let phiWorkerUsingLegacy = false;
let phiWorkerLegacyFallbackAttempted = false;
let phiRunning = false;
let phiHasRunDetection = false;
let phiScrubbedConfirmed = false;
let phiDetections = [];
let phiOriginalText = "";
let phiExcludedDetections = new Set();
let phiSelection = null;

function setStatus(text) {
  statusTextEl.textContent = text;
}

function setPhiStatus(text) {
  if (!phiStatusTextEl) return;
  phiStatusTextEl.textContent = String(text || "");
}

function setPhiProgress(text) {
  if (!phiProgressTextEl) return;
  phiProgressTextEl.textContent = String(text || "");
}

function setPhiSummary(text) {
  if (!phiSummaryTextEl) return;
  phiSummaryTextEl.textContent = String(text || "");
}

function setPhiConfirmAck(value) {
  if (!phiConfirmAckEl) return;
  phiConfirmAckEl.checked = Boolean(value);
}

function setSpeechStatus(text) {
  if (!speechStatusTextEl) return;
  speechStatusTextEl.textContent = String(text || "");
}

function setSpeechWarningText(text) {
  if (!speechWarningTextEl) return;
  speechWarningTextEl.textContent = String(text || "");
}

function normalizeSpeechModelKey(modelKey) {
  return REPORTER_SPEECH_MODELS[String(modelKey || "").trim()] ? String(modelKey || "").trim() : REPORTER_SPEECH_DEFAULT_MODEL_KEY;
}

function getSpeechModelConfig(modelKey = speechModelKey) {
  return REPORTER_SPEECH_MODELS[normalizeSpeechModelKey(modelKey)];
}

function getSpeechModelLabel(modelKey = speechModelKey) {
  return getSpeechModelConfig(modelKey).label;
}

function loadStoredSpeechModelKey() {
  try {
    return normalizeSpeechModelKey(window.localStorage?.getItem(REPORTER_SPEECH_MODEL_STORAGE_KEY));
  } catch {
    return REPORTER_SPEECH_DEFAULT_MODEL_KEY;
  }
}

function persistSpeechModelKey(modelKey) {
  try {
    window.localStorage?.setItem(REPORTER_SPEECH_MODEL_STORAGE_KEY, normalizeSpeechModelKey(modelKey));
  } catch {
    // ignore
  }
}

function isLikelyMobileDictationDevice() {
  if (typeof navigator?.userAgentData?.mobile === "boolean") {
    return navigator.userAgentData.mobile;
  }
  const userAgent = String(navigator?.userAgent || "");
  if (/android|iphone|ipad|ipod|mobile|tablet/i.test(userAgent)) return true;
  try {
    return Boolean((navigator?.maxTouchPoints || 0) > 1 && window.matchMedia?.("(max-width: 900px)").matches);
  } catch {
    return false;
  }
}

function renderSpeechModelHint() {
  if (!speechModelHintTextEl) return;
  if (isLikelyMobileDictationDevice()) {
    if (speechModelKey === "tiny") {
      speechModelHintTextEl.textContent = "Tiny is recommended on phones and tablets, and it is selected now.";
      return;
    }
    speechModelHintTextEl.textContent =
      "Using a phone or tablet? Tiny is recommended because it usually loads and transcribes faster there.";
    return;
  }
  if (speechModelKey === "base") {
    speechModelHintTextEl.textContent =
      "Base is larger and slower in-browser, and it does not always outperform Tiny on procedure dictation.";
    return;
  }
  speechModelHintTextEl.textContent =
    "Tiny is the recommended local model right now. It usually loads faster and has been more reliable in-browser.";
}

function syncSpeechModelControl() {
  if (speechModelSelectEl) speechModelSelectEl.value = speechModelKey;
  renderSpeechModelHint();
}

function applySpeechModelSelection(nextModelKey, { persist = true, restartWorker = false } = {}) {
  const normalizedModelKey = normalizeSpeechModelKey(nextModelKey);
  speechModelKey = normalizedModelKey;
  speechWorkerModelKey = normalizedModelKey;
  speechWorkerModelLabel = getSpeechModelLabel(normalizedModelKey);
  if (persist) persistSpeechModelKey(normalizedModelKey);
  syncSpeechModelControl();
  if (restartWorker) startSpeechWorker();
}

function renderSpeechSourceBadge() {
  if (!speechSourceBadgeEl) return;

  let label = "";
  if (speechCleaned) {
    label = "cleaned";
  } else if (speechFallbackUsed) {
    label = "cloud fallback";
  } else if (speechSource === "speech_local") {
    label = "local transcript";
  }

  if (!label) {
    speechSourceBadgeEl.classList.add("hidden");
    speechSourceBadgeEl.textContent = "";
    return;
  }

  speechSourceBadgeEl.classList.remove("hidden");
  speechSourceBadgeEl.textContent = label;
}

function clearSpeechAudioBlob() {
  speechLastRecordingBlob = null;
  speechChunks = [];
}

function stopSpeechTracks() {
  if (speechMediaStream && typeof speechMediaStream.getTracks === "function") {
    speechMediaStream.getTracks().forEach((track) => {
      try {
        track.stop();
      } catch {
        // ignore
      }
    });
  }
  speechMediaStream = null;
}

function clearSpeechStopTimer() {
  if (!speechStopTimer) return;
  clearTimeout(speechStopTimer);
  speechStopTimer = null;
}

function closeSpeechStartConfirmModal() {
  if (!speechStartConfirmModalEl || typeof speechStartConfirmModalEl.close !== "function") return;
  if (speechStartConfirmModalEl.open) speechStartConfirmModalEl.close();
}

function resetSpeechStartConfirmation() {
  speechStartConfirmedForCurrentNote = false;
  closeSpeechStartConfirmModal();
}

function openSpeechStartConfirmModal() {
  if (!speechStartConfirmModalEl || typeof speechStartConfirmModalEl.showModal !== "function") return false;
  if (!speechStartConfirmModalEl.open) speechStartConfirmModalEl.showModal();
  if (speechStartConfirmBtn && typeof speechStartConfirmBtn.focus === "function") {
    window.setTimeout(() => speechStartConfirmBtn.focus(), 0);
  }
  return true;
}

function resetSpeechState({ clearText = false } = {}) {
  clearSpeechStopTimer();
  stopSpeechTracks();

  if (speechMediaRecorder && speechMediaRecorder.state !== "inactive") {
    speechIgnoreNextStop = true;
    try {
      speechMediaRecorder.stop();
    } catch {
      // ignore
    }
  }

  speechMediaRecorder = null;
  speechRecording = false;
  speechRequestingAccess = false;
  speechLocalPending = false;
  speechCloudPending = false;
  speechPendingRequest = null;
  clearSpeechAudioBlob();
  speechSource = "";
  speechFallbackUsed = false;
  speechCleaned = false;

  if (clearText && seedTextEl) {
    seedTextEl.value = "";
    seedTextEl.dispatchEvent(new Event("input", { bubbles: true }));
  }

  setSpeechWarningText("");
  renderSpeechSourceBadge();
}

function formatScore(score) {
  if (typeof score !== "number") return "—";
  return score.toFixed(2);
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function safeSnippet(text, start, end, radius = 30) {
  const source = String(text || "");
  const s = clamp(Number(start) || 0, 0, source.length);
  const e = clamp(Number(end) || 0, 0, source.length);
  if (e <= s) return "";
  const left = clamp(s - radius, 0, source.length);
  const right = clamp(e + radius, 0, source.length);
  const prefix = left > 0 ? "…" : "";
  const suffix = right < source.length ? "…" : "";
  return `${prefix}${source.slice(left, right)}${suffix}`;
}

function ensurePhiDetectionIds(list) {
  const seen = new Map();
  const input = Array.isArray(list) ? list : [];
  return input.map((d) => {
    if (d && typeof d.id === "string" && d.id) return d;
    const label = String(d?.label ?? "OTHER");
    const source = String(d?.source ?? "unknown");
    const start = Number.isFinite(d?.start) ? Number(d.start) : -1;
    const end = Number.isFinite(d?.end) ? Number(d.end) : -1;
    const base = `${label}:${source}:${start}:${end}`;
    const n = (seen.get(base) || 0) + 1;
    seen.set(base, n);
    const id = n === 1 ? base : `${base}:${n}`;
    return { ...d, id };
  });
}

function getIncludedPhiDetections() {
  return (Array.isArray(phiDetections) ? phiDetections : []).filter((d) => !phiExcludedDetections.has(d.id));
}

function renderPhiDetections() {
  if (!phiDetectionsListEl || !phiDetectionsCountEl) return;

  const sourceText = String(phiOriginalText || seedTextEl?.value || "");
  const list = Array.isArray(phiDetections) ? phiDetections : [];
  phiDetectionsCountEl.textContent = String(list.length);
  phiDetectionsListEl.innerHTML = "";

  const sorted = [...list].sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    return (b.score ?? 0) - (a.score ?? 0);
  });

  sorted.forEach((d) => {
    const row = document.createElement("div");
    row.className = "detRow";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.checked = !phiExcludedDetections.has(d.id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) phiExcludedDetections.delete(d.id);
      else phiExcludedDetections.add(d.id);
    });

    const body = document.createElement("div");
    const meta = document.createElement("div");
    meta.className = "detMeta";

    const labelPill = document.createElement("span");
    labelPill.className = "pill label";
    labelPill.textContent = String(d.label || "PHI");
    meta.appendChild(labelPill);

    const sourcePill = document.createElement("span");
    sourcePill.className = d.source === "manual" ? "pill source source-manual" : "pill source";
    sourcePill.textContent = String(d.source || "unknown");
    meta.appendChild(sourcePill);

    const scorePill = document.createElement("span");
    scorePill.className = "pill score";
    scorePill.textContent = d.source === "manual" ? "Manual" : `score ${formatScore(d.score)}`;
    meta.appendChild(scorePill);

    const rangePill = document.createElement("span");
    rangePill.className = "pill";
    rangePill.textContent = `${d.start}-${d.end}`;
    meta.appendChild(rangePill);

    const snippet = document.createElement("div");
    snippet.className = "snippet";
    snippet.textContent = safeSnippet(sourceText, d.start, d.end);

    body.appendChild(meta);
    body.appendChild(snippet);
    row.appendChild(checkbox);
    row.appendChild(body);
    phiDetectionsListEl.appendChild(row);
  });

  if (!list.length) {
    const empty = document.createElement("div");
    empty.className = "subtle";
    empty.style.padding = "1rem";
    empty.style.textAlign = "center";
    empty.textContent =
      phiHasRunDetection && !phiRunning
        ? 'No PHI detected. Click "Apply Redactions" to continue.'
        : 'Run detection to populate this panel. Use "Manual → Add" for missed spans.';
    phiDetectionsListEl.appendChild(empty);
  }
}

function updatePhiSelectionFromTextarea() {
  if (!seedTextEl) return;
  const start = seedTextEl.selectionStart;
  const end = seedTextEl.selectionEnd;
  const hasSelection = Number.isFinite(start) && Number.isFinite(end) && end > start;
  phiSelection = hasSelection ? { start: Number(start), end: Number(end) } : null;
  updateControls();
}

function showBanner(level, text) {
  actionBannerEl.classList.remove("hidden", "success", "warning", "error");
  actionBannerEl.classList.add(level);
  actionBannerEl.textContent = text;
}

function hideBanner() {
  actionBannerEl.classList.add("hidden");
  actionBannerEl.textContent = "";
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

async function copyToClipboard(text) {
  const payload = String(text || "");
  if (!payload.trim()) return false;

  if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
    try {
      await navigator.clipboard.writeText(payload);
      return true;
    } catch {
      // fall back
    }
  }

  try {
    const ta = document.createElement("textarea");
    ta.value = payload;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    ta.style.left = "-1000px";
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch {
    return false;
  }
}

function escapeRegExp(s) {
  return String(s).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildPhiWorkerConfigForRun() {
  const cfg = { ...PHI_WORKER_BASE_CONFIG };
  const redactProviders = phiRedactProvidersToggleEl ? Boolean(phiRedactProvidersToggleEl.checked) : true;
  cfg.protectProviders = !redactProviders;
  return cfg;
}

function shouldForceLegacyPhiWorker() {
  const params = new URLSearchParams(location.search);
  if (params.get("legacy_phi") === "1") return true;

  const ua = navigator.userAgent || "";
  const isIOSDevice =
    /iPad|iPhone|iPod/i.test(ua) || (ua.includes("Mac") && "ontouchend" in document);
  const isSafari =
    /Safari/i.test(ua) && !/Chrome|CriOS|FxiOS|EdgiOS|OPR/i.test(ua);
  return isIOSDevice && isSafari;
}

function buildPhiWorkerUrl(name) {
  return `/ui/${name}?v=${Date.now()}`;
}

function startPhiWorker({ forceLegacy = false } = {}) {
  if (!phiRunBtn) return;

  if (phiWorker) {
    try {
      phiWorker.terminate();
    } catch {
      // ignore
    }
  }

  phiWorkerReady = false;
  phiWorkerUsingLegacy = forceLegacy;

  let nextWorker = null;
  let nextIsLegacy = forceLegacy;

  if (!forceLegacy) {
    try {
      nextWorker = new Worker(buildPhiWorkerUrl("redactor.worker.js"), { type: "module" });
      nextIsLegacy = false;
    } catch {
      phiWorkerLegacyFallbackAttempted = true;
      nextIsLegacy = true;
      setPhiStatus("Module worker unsupported; falling back to legacy worker…");
    }
  }

  if (!nextWorker) {
    nextWorker = new Worker(buildPhiWorkerUrl("redactor.worker.legacy.js"));
    nextIsLegacy = true;
  }

  phiWorker = nextWorker;
  phiWorkerUsingLegacy = nextIsLegacy;
  attachPhiWorkerHandlers(phiWorker);
  try {
    phiWorker.postMessage({ type: "init", debug: false, config: PHI_WORKER_BASE_CONFIG });
  } catch (e) {
    setPhiStatus(`PHI worker init failed: ${e?.message || e}`);
  }
}

function attachPhiWorkerHandlers(activeWorker) {
  activeWorker.addEventListener("error", (ev) => {
    if (!phiWorkerUsingLegacy && !phiWorkerLegacyFallbackAttempted) {
      phiWorkerLegacyFallbackAttempted = true;
      setPhiStatus("Module worker failed to load; falling back to legacy worker…");
      setPhiProgress("");
      startPhiWorker({ forceLegacy: true });
      return;
    }
    phiWorkerReady = false;
    setPhiStatus(`PHI worker error: ${ev.message || "failed to load"}`);
    setPhiProgress("");
    phiRunning = false;
    updateControls();
  });

  activeWorker.addEventListener("messageerror", () => {
    phiWorkerReady = false;
    setPhiStatus("PHI worker message error (serialization failed)");
    setPhiProgress("");
    phiRunning = false;
    updateControls();
  });

  activeWorker.onmessage = (e) => {
    const msg = e.data;
    if (!msg || typeof msg.type !== "string") return;

    if (msg.type === "ready") {
      phiWorkerReady = true;
      setPhiStatus("Ready (local model loaded)");
      setPhiProgress("");
      updateControls();
      return;
    }

    if (msg.type === "progress") {
      const stage = msg.stage ? String(msg.stage) : "";
      if (stage) setPhiProgress(stage);
      return;
    }

    if (msg.type === "done") {
      phiRunning = false;
      const detections = Array.isArray(msg.detections) ? msg.detections : [];
      phiDetections = ensurePhiDetectionIds(
        detections
          .filter((d) => Number.isFinite(d?.start) && Number.isFinite(d?.end) && d.end > d.start)
          .map((d) => ({
            id: typeof d?.id === "string" ? d.id : undefined,
            start: Number(d.start),
            end: Number(d.end),
            label: String(d.label || "PHI"),
            score: typeof d?.score === "number" ? d.score : null,
            source: typeof d?.source === "string" ? d.source : "auto",
          })),
      );
      phiExcludedDetections = new Set();
      phiHasRunDetection = true;
      phiScrubbedConfirmed = false;
      setPhiConfirmAck(false);
      setPhiStatus("Detection complete. Apply redactions to confirm scrubbed text.");
      setPhiProgress("");
      setPhiSummary(`${phiDetections.length} PHI span${phiDetections.length === 1 ? "" : "s"} detected`);
      renderPhiDetections();
      updateControls();
      return;
    }

    if (msg.type === "error") {
      phiRunning = false;
      const message = String(msg.message || "unknown error");
      setPhiStatus(`PHI worker error: ${message}`);
      setPhiProgress("");
      updateControls();
    }
  };
}

function upsertCompletenessAddendum(text, block) {
  const base = String(text || "");
  const nextBlock = String(block || "").trim();
  const re = new RegExp(
    `${escapeRegExp(COMPLETENESS_ADDENDUM_BEGIN)}[\\s\\S]*?${escapeRegExp(COMPLETENESS_ADDENDUM_END)}`,
    "m",
  );

  if (!nextBlock) {
    return base.replace(re, "").trim();
  }

  if (re.test(base)) {
    return base.replace(re, nextBlock).trim();
  }

  const trimmed = base.trim();
  if (!trimmed) return nextBlock;
  return `${trimmed}\n\n${nextBlock}`.trim();
}

function getCompletenessInputSpec(promptPath) {
  const map = {
    "patient_demographics.age_years": { type: "integer", placeholder: "e.g., 67" },
    "patient.age": { type: "integer", placeholder: "e.g., 67" },
    "patient.sex": { type: "enum", options: ["M", "F", "O"] },
    "procedure.indication": { type: "string", placeholder: "Primary indication" },
    "clinical_context.asa_class": { type: "integer", placeholder: "1–6" },
    "risk_assessment.asa_class": { type: "integer", placeholder: "1–6" },
    "risk_assessment.anticoagulant_use": { type: "string", placeholder: "e.g., Apixaban" },
    "risk_assessment.mallampati_score": { type: "integer", placeholder: "1–4" },
    "clinical_context.ecog_score": { type: "ecog", placeholder: "0–4 or 0–1" },
    "clinical_context.bronchus_sign": {
      type: "enum",
      options: ["Positive", "Negative", "Not assessed"],
    },
    // Complications
    "complications.bleeding.bleeding_grade_nashville": { type: "integer", placeholder: "0–4" },
    "complications.pneumothorax.intervention": {
      type: "multiselect",
      options: ["Observation", "Aspiration", "Pigtail catheter", "Chest tube", "Heimlich valve", "Surgery"],
    },

    // Navigation / radial EBUS
    "procedures_performed.radial_ebus.probe_position": {
      type: "enum",
      options: ["Concentric", "Eccentric", "Adjacent", "Not visualized"],
    },
    "procedures_performed.navigational_bronchoscopy.tool_in_lesion_confirmed": { type: "boolean" },
    "procedures_performed.navigational_bronchoscopy.confirmation_method": {
      type: "enum",
      options: ["Radial EBUS", "CBCT", "Fluoroscopy", "Augmented Fluoroscopy", "None"],
    },
    "procedures_performed.navigational_bronchoscopy.divergence_mm": { type: "number", placeholder: "mm" },

    // Navigation per-target (wildcard)
    "granular_data.navigation_targets[*].target_location_text": { type: "string", placeholder: "e.g., RUL apical" },
    "granular_data.navigation_targets[*].lesion_size_mm": { type: "number", placeholder: "mm" },
    "granular_data.navigation_targets[*].ct_characteristics": {
      type: "enum",
      options: ["Solid", "Part-solid", "Ground-glass", "Cavitary", "Calcified"],
    },
    "granular_data.navigation_targets[*].distance_from_pleura_mm": { type: "number", placeholder: "mm" },
    "granular_data.navigation_targets[*].pet_suv_max": { type: "number", placeholder: "e.g., 4.2" },
    "granular_data.navigation_targets[*].registration_error_mm": { type: "number", placeholder: "mm" },
    "granular_data.navigation_targets[*].tool_in_lesion_confirmed": { type: "boolean" },
    "granular_data.navigation_targets[*].confirmation_method": {
      type: "enum",
      options: ["CBCT", "Augmented fluoroscopy", "Fluoroscopy", "Radial EBUS", "None"],
    },

    // Linear EBUS per-station (wildcard)
    "granular_data.linear_ebus_stations_detail[*].needle_gauge": {
      type: "enum",
      options: [19, 21, 22, 25],
    },
    "granular_data.linear_ebus_stations_detail[*].number_of_passes": { type: "integer", placeholder: "passes" },
    "granular_data.linear_ebus_stations_detail[*].short_axis_mm": { type: "number", placeholder: "mm" },
    "granular_data.linear_ebus_stations_detail[*].lymphocytes_present": { type: "boolean" },
    "pleural_procedures.chest_ultrasound.hemithorax": {
      type: "enum",
      options: ["Right", "Left", "Bilateral"],
    },
    "pleural_procedures.chest_ultrasound.image_documentation": { type: "boolean" },
    "pleural_procedures.chest_ultrasound.effusion_volume": {
      type: "enum",
      options: ["None", "Minimal", "Small", "Moderate", "Large"],
    },
    "pleural_procedures.chest_ultrasound.effusion_echogenicity": {
      type: "enum",
      options: ["Anechoic", "Hypoechoic", "Isoechoic", "Hyperechoic"],
    },
    "pleural_procedures.chest_ultrasound.effusion_loculations": {
      type: "enum",
      options: ["None", "Thin", "Thick"],
    },
    "pleural_procedures.chest_ultrasound.diaphragmatic_motion": {
      type: "enum",
      options: ["Normal", "Diminished", "Absent"],
    },
    "pleural_procedures.chest_ultrasound.lung_sliding_pre": {
      type: "enum",
      options: ["Present", "Absent"],
    },
    "pleural_procedures.chest_ultrasound.lung_sliding_post": {
      type: "enum",
      options: ["Present", "Absent"],
    },
    "pleural_procedures.chest_ultrasound.lung_consolidation_present": { type: "boolean" },
    "pleural_procedures.chest_ultrasound.pleura_characteristics": {
      type: "enum",
      options: ["Normal", "Thick", "Nodular"],
    },
    "pleural_procedures.fibrinolytic_therapy.agents": {
      type: "multiselect",
      options: ["tPA", "DNase", "Streptokinase", "Urokinase"],
    },
    "pleural_procedures.fibrinolytic_therapy.tpa_dose_mg": { type: "number", placeholder: "mg" },
    "pleural_procedures.fibrinolytic_therapy.dnase_dose_mg": { type: "number", placeholder: "mg" },
    "pleural_procedures.fibrinolytic_therapy.number_of_doses": { type: "integer", placeholder: "e.g., 2" },
    "pleural_procedures.fibrinolytic_therapy.indication": {
      type: "enum",
      options: ["Complex parapneumonic", "Empyema", "Hemothorax", "Malignant effusion"],
    },
  };

  return map[String(promptPath || "")] || { type: "string", placeholder: "Enter value" };
}

function formatCompletenessValueForNote(raw) {
  if (Array.isArray(raw)) return raw.map((v) => String(v || "").trim()).filter(Boolean).join(", ");
  const s = String(raw || "").trim();
  if (s.toLowerCase() === "true") return "Yes";
  if (s.toLowerCase() === "false") return "No";
  return s;
}

function buildCompletenessAddendumBlock() {
  const prompts = Array.isArray(completenessPrompts) ? completenessPrompts : [];
  if (!prompts.length) return "";

  const lines = [];
  prompts.forEach((p) => {
    const path = String(p?.path || "").trim();
    if (!path) return;
    const raw = completenessValuesByPath.get(path);
    const val = formatCompletenessValueForNote(raw);
    if (!val) return;
    const label = String(p?.label || path).trim();
    lines.push(`${label}: ${val}`);
  });

  const body = lines.join("\n").trim();
  if (!body) return "";
  return `${COMPLETENESS_ADDENDUM_BEGIN}\n${body}\n${COMPLETENESS_ADDENDUM_END}`.trim();
}

async function postJSON(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API Error: ${response.status} ${response.statusText} - ${text}`);
  }
  return response.json();
}

async function postFormData(url, formData) {
  const response = await fetch(url, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`API Error: ${response.status} ${response.statusText} - ${text}`);
  }
  return response.json();
}

function getSpeechRecordingUnsupportedReason() {
  if (!speechStartBtn) return "";
  if (!window.isSecureContext) {
    return "Microphone recording requires a secure local origin. Open this page on http://localhost:8000 or http://127.0.0.1:8000 instead of http://0.0.0.0:8000.";
  }
  if (!navigator?.mediaDevices || typeof navigator.mediaDevices.getUserMedia !== "function") {
    return "This browser does not expose microphone capture APIs.";
  }
  if (typeof window.MediaRecorder !== "function") {
    return "This browser cannot record microphone audio. Try current Chrome or Edge.";
  }
  return "";
}

function supportsSpeechRecording() {
  return !getSpeechRecordingUnsupportedReason();
}

function buildSpeechRepairWarnings(replacements) {
  if (!Array.isArray(replacements) || !replacements.length) return [];
  return [`REPORTER_SPEECH_LOCAL_REPAIR: applied_${replacements.length}_deterministic_fixes`];
}

function joinSpeechWarnings(warnings) {
  return (Array.isArray(warnings) ? warnings : [])
    .map((warning) => String(warning || "").trim())
    .filter(Boolean)
    .join(" | ");
}

function describeSpeechAccessError(error) {
  const name = String(error?.name || "");
  const message = String(error?.message || "").trim();
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "Microphone access was blocked. Allow microphone access for localhost in Chrome and macOS System Settings, then try again.";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "No microphone was found. Connect or enable a microphone and try again.";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "Chrome found a microphone, but another app may be using it. Close other recording apps and try again.";
  }
  if (name === "SecurityError") {
    return "Microphone access is blocked by browser security settings. Open the reporter on localhost and try again.";
  }
  if (name === "AbortError") {
    return "The microphone request was interrupted. Try again.";
  }
  return message || "Microphone access failed.";
}

function resolveSpeechIdleStatus() {
  if (!speechStartBtn) return "";
  if (speechRequestingAccess) return "Waiting for microphone permission…";
  const unsupportedReason = getSpeechRecordingUnsupportedReason();
  if (unsupportedReason) return unsupportedReason;
  const modelLabel = speechWorkerReady ? speechWorkerModelLabel : getSpeechModelLabel(speechModelKey);
  if (speechWorkerReady) return `Local speech support ready (${modelLabel}).`;
  if (speechWorkerUnavailableReason) return `Local ${modelLabel} speech unavailable. Record only if you intend to use cloud fallback.`;
  if (speechWorker) return `Loading local speech support (${modelLabel})…`;
  return "Speech support unavailable.";
}

function getSpeechTranscribeTimeoutMs(modelKey = speechWorkerModelKey) {
  return REPORTER_SPEECH_TRANSCRIBE_TIMEOUT_MS[normalizeSpeechModelKey(modelKey)] || 60_000;
}

function resetSpeechWorkerAfterFailure() {
  if (speechPendingRequest) speechPendingRequest = null;
  startSpeechWorker();
}

function renderSpeechState() {
  renderSpeechSourceBadge();
  updateControls();
}

function pickSpeechRecordingMimeType() {
  if (typeof window.MediaRecorder !== "function") return "";
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/ogg;codecs=opus",
    "audio/ogg",
    "audio/mp4",
  ];
  for (const candidate of candidates) {
    try {
      if (typeof MediaRecorder.isTypeSupported !== "function" || MediaRecorder.isTypeSupported(candidate)) {
        return candidate;
      }
    } catch {
      // ignore
    }
  }
  return "";
}

function buildSpeechUploadFilename(blob) {
  const contentType = String(blob?.type || "").toLowerCase();
  if (contentType.includes("ogg")) return "reporter_dictation.ogg";
  if (contentType.includes("mp4") || contentType.includes("m4a")) return "reporter_dictation.m4a";
  if (contentType.includes("mpeg") || contentType.includes("mp3")) return "reporter_dictation.mp3";
  if (contentType.includes("wav")) return "reporter_dictation.wav";
  return "reporter_dictation.webm";
}

function mixAudioBufferToMono(audioBuffer) {
  const channelCount = Math.max(1, Number(audioBuffer?.numberOfChannels) || 1);
  const frameCount = Math.max(0, Number(audioBuffer?.length) || 0);
  const mono = new Float32Array(frameCount);
  if (!frameCount) return mono;

  for (let channelIndex = 0; channelIndex < channelCount; channelIndex += 1) {
    const channel = audioBuffer.getChannelData(channelIndex);
    for (let i = 0; i < frameCount; i += 1) {
      mono[i] += (channel[i] || 0) / channelCount;
    }
  }
  return mono;
}

function resampleFloat32(samples, inputRate, targetRate) {
  if (!samples?.length) return new Float32Array();
  if (!Number.isFinite(inputRate) || inputRate <= 0 || inputRate === targetRate) {
    return samples instanceof Float32Array ? samples : new Float32Array(samples);
  }

  const source = samples instanceof Float32Array ? samples : new Float32Array(samples);
  const targetLength = Math.max(1, Math.round((source.length * targetRate) / inputRate));
  const output = new Float32Array(targetLength);
  const ratio = inputRate / targetRate;

  for (let i = 0; i < targetLength; i += 1) {
    const sourceIndex = i * ratio;
    const leftIndex = Math.floor(sourceIndex);
    const rightIndex = Math.min(leftIndex + 1, source.length - 1);
    const weight = sourceIndex - leftIndex;
    const left = source[leftIndex] || 0;
    const right = source[rightIndex] || 0;
    output[i] = left + (right - left) * weight;
  }

  return output;
}

async function decodeAudioBlobToMono16kFloat32(blob) {
  const arrayBuffer = await blob.arrayBuffer();
  const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
  if (!AudioContextCtor) {
    throw new Error("Audio decoding is not supported in this browser");
  }

  const audioContext = new AudioContextCtor();
  try {
    const decoded = await new Promise((resolve, reject) => {
      const result = audioContext.decodeAudioData(arrayBuffer.slice(0), resolve, reject);
      if (result && typeof result.then === "function") result.then(resolve, reject);
    });
    const mono = mixAudioBufferToMono(decoded);
    return resampleFloat32(mono, decoded.sampleRate, REPORTER_SPEECH_SAMPLE_RATE);
  } finally {
    try {
      const closeResult = audioContext.close?.();
      if (closeResult && typeof closeResult.catch === "function") closeResult.catch(() => {});
    } catch {
      // ignore
    }
  }
}

function attachSpeechWorkerHandlers(activeWorker) {
  activeWorker.addEventListener("error", (event) => {
    speechWorkerReady = false;
    speechWorkerUnavailableReason = event?.message || "Local speech worker failed to load";
    if (speechPendingRequest) {
      const pending = speechPendingRequest;
      speechPendingRequest = null;
      pending.reject(new Error(speechWorkerUnavailableReason));
    }
    setSpeechStatus(`Local ${speechWorkerModelLabel} speech unavailable.`);
    setSpeechWarningText(speechWorkerUnavailableReason);
    renderSpeechState();
  });

  activeWorker.addEventListener("messageerror", () => {
    if (speechPendingRequest) {
      const pending = speechPendingRequest;
      speechPendingRequest = null;
      pending.reject(new Error("Speech worker message error"));
    }
    speechWorkerReady = false;
    speechWorkerUnavailableReason = "Speech worker message error";
    setSpeechStatus(`Local ${speechWorkerModelLabel} speech unavailable.`);
    setSpeechWarningText("Speech worker message error");
    renderSpeechState();
  });

  activeWorker.onmessage = (event) => {
    const msg = event?.data || {};
    if (!msg || typeof msg.type !== "string") return;

    if (msg.type === "status") {
      speechWorkerModelKey = normalizeSpeechModelKey(msg.modelKey || speechModelKey);
      speechWorkerModelLabel = String(msg.modelLabel || getSpeechModelLabel(speechWorkerModelKey));
      setSpeechStatus(String(msg.message || resolveSpeechIdleStatus()));
      renderSpeechState();
      return;
    }

    if (msg.type === "ready") {
      speechWorkerModelKey = normalizeSpeechModelKey(msg.modelKey || speechModelKey);
      speechWorkerModelLabel = String(msg.modelLabel || getSpeechModelLabel(speechWorkerModelKey));
      speechWorkerReady = true;
      speechWorkerUnavailableReason = "";
      setSpeechStatus(`Local speech support ready (${speechWorkerModelLabel}).`);
      setSpeechWarningText("");
      renderSpeechState();
      return;
    }

    if (msg.type === "unavailable") {
      speechWorkerModelKey = normalizeSpeechModelKey(msg.modelKey || speechModelKey);
      speechWorkerModelLabel = String(msg.modelLabel || getSpeechModelLabel(speechWorkerModelKey));
      speechWorkerReady = false;
      speechWorkerUnavailableReason = String(msg.message || "Local speech model assets are unavailable");
      if (speechPendingRequest) {
        const pending = speechPendingRequest;
        speechPendingRequest = null;
        pending.reject(new Error(speechWorkerUnavailableReason));
      }
      setSpeechStatus(`Local ${speechWorkerModelLabel} speech unavailable.`);
      setSpeechWarningText(speechWorkerUnavailableReason);
      renderSpeechState();
      return;
    }

    if (msg.type === "transcription_result") {
      const requestId = String(msg.requestId || "");
      if (!speechPendingRequest || speechPendingRequest.requestId !== requestId) return;
      const pending = speechPendingRequest;
      speechPendingRequest = null;
      pending.resolve({
        transcript: String(msg.transcript || ""),
        warnings: Array.isArray(msg.warnings) ? msg.warnings : [],
      });
      return;
    }

    if (msg.type === "transcription_error") {
      const requestId = String(msg.requestId || "");
      if (!speechPendingRequest || speechPendingRequest.requestId !== requestId) return;
      const pending = speechPendingRequest;
      speechPendingRequest = null;
      pending.reject(new Error(String(msg.message || "Local speech transcription failed")));
    }
  };
}

function startSpeechWorker() {
  if (!speechStartBtn || typeof window.Worker !== "function") {
    setSpeechStatus(resolveSpeechIdleStatus());
    renderSpeechState();
    return;
  }

  const selectedModel = getSpeechModelConfig(speechModelKey);
  speechWorkerModelKey = selectedModel.key;
  speechWorkerModelLabel = selectedModel.label;

  if (speechWorker) {
    try {
      speechWorker.terminate();
    } catch {
      // ignore
    }
  }

  speechWorkerReady = false;
  speechWorkerUnavailableReason = "";
  speechPendingRequest = null;
  try {
    speechWorker = new Worker(REPORTER_SPEECH_WORKER_PATH, { type: "module" });
    attachSpeechWorkerHandlers(speechWorker);
    setSpeechStatus(`Loading local speech support (${selectedModel.label})…`);
    setSpeechWarningText("");
    renderSpeechState();
    speechWorker.postMessage({ type: "init", modelKey: selectedModel.key });
  } catch (error) {
    speechWorkerUnavailableReason = error?.message || "Local speech worker initialization failed";
    speechWorker = null;
    setSpeechStatus(`Local ${selectedModel.label} speech unavailable.`);
    setSpeechWarningText(speechWorkerUnavailableReason);
    renderSpeechState();
  }
}

async function requestLocalSpeechTranscript(blob) {
  if (!speechWorker || !speechWorkerReady) {
    throw new Error(speechWorkerUnavailableReason || "Local speech support is still loading");
  }
  if (speechPendingRequest) {
    throw new Error("Speech transcription is already in progress");
  }

  speechLocalPending = true;
  setSpeechStatus(`Transcribing locally with ${speechWorkerModelLabel}…`);
  setSpeechWarningText("");
  renderSpeechState();

  try {
    const audio = await decodeAudioBlobToMono16kFloat32(blob);
    if (!audio.length) throw new Error("Unable to decode recorded audio");

    const requestId = `speech-${Date.now()}-${++speechRequestCounter}`;
    const timeoutMs = getSpeechTranscribeTimeoutMs(speechWorkerModelKey);
    const response = await new Promise((resolve, reject) => {
      speechPendingRequest = { requestId, resolve, reject };
      speechWorker.postMessage({ type: "transcribe", requestId, audio }, [audio.buffer]);
      window.setTimeout(() => {
        if (!speechPendingRequest || speechPendingRequest.requestId !== requestId) return;
        speechPendingRequest = null;
        resetSpeechWorkerAfterFailure();
        reject(
          new Error(
            `Local ${speechWorkerModelLabel} transcription timed out after ${Math.round(timeoutMs / 1000)} seconds. `
            + "Try Tiny or use Cloud Fallback if the recording contains no PHI.",
          ),
        );
      }, timeoutMs);
    });
    const transcript = String(response?.transcript || "").trim();
    if (!transcript) throw new Error("Local speech transcription returned an empty transcript");
    return {
      transcript,
      warnings: Array.isArray(response?.warnings) ? response.warnings : [],
    };
  } finally {
    speechLocalPending = false;
    renderSpeechState();
  }
}

function replaceSeedTextFromSpeech(
  text,
  { source = "speech_local", fallbackUsed = false, cleaned = false, warnings = [], preservePhiState = false } = {},
) {
  seedTextEl.value = String(text || "").trim();
  speechSource = String(source || "speech_local");
  speechFallbackUsed = Boolean(fallbackUsed);
  speechCleaned = Boolean(cleaned);
  renderSpeechSourceBadge();
  setSpeechWarningText(joinSpeechWarnings(warnings));
  preservePhiStateOnNextSeedInput = Boolean(preservePhiState);
  seedTextEl.dispatchEvent(new Event("input", { bubbles: true }));
  seedTextEl.focus();
}

async function finalizeSpeechRecording(blob) {
  speechLastRecordingBlob = blob;
  speechSource = "";
  speechFallbackUsed = false;
  speechCleaned = false;
  renderSpeechState();

  if (!blob || !blob.size) {
    setSpeechStatus("Recording was empty.");
    setSpeechWarningText("Try again and speak a little louder or closer to the microphone.");
    return;
  }

  if (!speechWorkerReady) {
    if (speechWorkerUnavailableReason) {
      setSpeechStatus(`Local ${getSpeechModelLabel(speechModelKey)} speech unavailable.`);
      setSpeechWarningText(`${speechWorkerUnavailableReason} Use Cloud Fallback only if the recording contains no PHI.`);
      showBanner("warning", "Recording captured. Local speech is unavailable, so only cloud fallback can transcribe it.");
    } else {
      setSpeechStatus(`Recording captured while local ${getSpeechModelLabel(speechModelKey)} speech support was still loading.`);
      setSpeechWarningText("Wait for local speech support or use Cloud Fallback only if the recording contains no PHI.");
      showBanner("warning", "Recording captured. Local speech is still loading.");
    }
    renderSpeechState();
    return;
  }

  try {
    const localResult = await requestLocalSpeechTranscript(blob);
    replaceSeedTextFromSpeech(localResult.transcript, {
      source: "speech_local",
      fallbackUsed: false,
      cleaned: false,
      warnings: localResult.warnings,
    });
    setSpeechStatus(`Local ${speechWorkerModelLabel} transcript ready. Review and redact before seeding.`);
    showBanner("success", "Local dictation transcript inserted. Run PHI detection and apply redactions before seeding.");
  } catch (error) {
    setSpeechStatus(`Local ${speechWorkerModelLabel} transcription failed.`);
    setSpeechWarningText(`${error?.message || "Local transcription failed"} Use Cloud Fallback only if the recording contains no PHI.`);
    showBanner("warning", "Local transcription failed. You can use Cloud Fallback if the recording contains no PHI.");
  } finally {
    renderSpeechState();
  }
}

async function startSpeechDictation() {
  if (!speechStartConfirmedForCurrentNote) {
    if (openSpeechStartConfirmModal()) return;
    const confirmed = window.confirm(
      "I confirm that I will not include patient names, MRN, DOB, phone, address, or exact dates in this dictation.",
    );
    if (!confirmed) return;
    speechStartConfirmedForCurrentNote = true;
  }
  const unsupportedReason = getSpeechRecordingUnsupportedReason();
  if (unsupportedReason) {
    setSpeechStatus(unsupportedReason);
    if (!window.isSecureContext) {
      showBanner(
        "error",
        "Microphone recording requires a secure local origin. Open the reporter on http://localhost:8000 or http://127.0.0.1:8000.",
      );
    } else {
      showBanner("error", unsupportedReason);
    }
    return;
  }
  if (speechRecording || speechLocalPending || speechCloudPending || state.busy) return;

  clearSpeechStopTimer();
  clearSpeechAudioBlob();
  speechIgnoreNextStop = false;
  speechRequestingAccess = true;
  setSpeechStatus("Waiting for microphone permission…");
  setSpeechWarningText("Chrome may show a permission prompt in the address bar, and macOS may show a system prompt.");
  renderSpeechState();

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    const mimeType = pickSpeechRecordingMimeType();
    const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

    speechRequestingAccess = false;
    speechMediaStream = stream;
    speechMediaRecorder = recorder;
    speechChunks = [];
    speechLastRecordingBlob = null;

    recorder.addEventListener("dataavailable", (event) => {
      if (event?.data?.size) speechChunks.push(event.data);
    });

    recorder.addEventListener("error", () => {
      speechRecording = false;
      stopSpeechTracks();
      clearSpeechStopTimer();
      setSpeechStatus("Recording failed.");
      setSpeechWarningText("Microphone capture failed.");
      renderSpeechState();
    });

    recorder.addEventListener("stop", () => {
      const shouldIgnore = speechIgnoreNextStop;
      speechIgnoreNextStop = false;
      const recordedBlob = new Blob(speechChunks, {
        type: recorder.mimeType || mimeType || "audio/webm",
      });
      speechMediaRecorder = null;
      speechRecording = false;
      stopSpeechTracks();
      clearSpeechStopTimer();
      renderSpeechState();
      if (shouldIgnore) {
        clearSpeechAudioBlob();
        return;
      }
      void finalizeSpeechRecording(recordedBlob);
    });

    recorder.start(250);
    speechRecording = true;
    setSpeechStatus("Recording… Dictate procedure details only.");
    setSpeechWarningText("Do not dictate names, MRN, DOB, phone, address, or exact dates.");
    renderSpeechState();

    speechStopTimer = setTimeout(() => {
      if (!speechMediaRecorder || speechMediaRecorder.state !== "recording") return;
      setSpeechWarningText(`Recording stopped at ${REPORTER_SPEECH_MAX_SECONDS} seconds.`);
      try {
        speechMediaRecorder.stop();
      } catch {
        // ignore
      }
    }, REPORTER_SPEECH_MAX_SECONDS * 1000);
  } catch (error) {
    speechRequestingAccess = false;
    speechRecording = false;
    stopSpeechTracks();
    clearSpeechStopTimer();
    setSpeechStatus("Unable to access the microphone.");
    setSpeechWarningText(describeSpeechAccessError(error));
    renderSpeechState();
    showBanner("error", describeSpeechAccessError(error));
  }
}

function stopSpeechDictation() {
  if (!speechMediaRecorder || speechMediaRecorder.state !== "recording") return;
  setSpeechStatus("Finalizing recording…");
  renderSpeechState();
  try {
    speechMediaRecorder.stop();
  } catch {
    setSpeechStatus("Unable to stop recording cleanly.");
    renderSpeechState();
  }
}

function discardSpeechTranscript() {
  const shouldClearText = Boolean((speechSource || speechCleaned) && String(seedTextEl?.value || "").trim());
  resetSpeechState({ clearText: shouldClearText });
  setSpeechStatus(resolveSpeechIdleStatus());
  renderSpeechState();
  showBanner("success", "Speech recording discarded.");
}

async function runCloudSpeechFallback() {
  if (!speechLastRecordingBlob) {
    showBanner("warning", "Record audio first before using cloud fallback.");
    return;
  }

  const confirmed = window.confirm(
    "Cloud fallback will upload raw audio to the server transcription provider. Continue only if the recording contains no PHI.",
  );
  if (!confirmed) return;

  speechCloudPending = true;
  setSpeechStatus("Uploading audio for cloud fallback…");
  setSpeechWarningText("Raw audio leaves the browser during cloud fallback.");
  renderSpeechState();

  try {
    const formData = new FormData();
    formData.append("audio_file", speechLastRecordingBlob, buildSpeechUploadFilename(speechLastRecordingBlob));
    formData.append("source", "reporter_builder");
    formData.append("cloud_fallback_confirmed", "true");

    const result = await postFormData("/report/transcribe_audio", formData);
    const repaired = repairSpeechTranscript(String(result?.transcript || ""));
    const warnings = [
      ...(Array.isArray(result?.warnings) ? result.warnings : []),
      ...buildSpeechRepairWarnings(repaired.replacements),
    ];
    const transcript = String(repaired.text || "").trim();
    if (!transcript) throw new Error("Cloud fallback returned an empty transcript");

    replaceSeedTextFromSpeech(transcript, {
      source: "speech_cloud_fallback",
      fallbackUsed: true,
      cleaned: false,
      warnings,
    });
    setSpeechStatus("Cloud fallback transcript ready. Review, redact, then seed.");
    showBanner(
      "warning",
      "Cloud fallback transcript inserted. Review carefully, then run PHI detection and apply redactions before seeding.",
    );
  } finally {
    speechCloudPending = false;
    renderSpeechState();
  }
}

async function cleanSpeechTranscript({ automatic = false } = {}) {
  const text = String(seedTextEl?.value || "").trim();
  if (!text) {
    if (!automatic) showBanner("warning", "Record or paste transcript text before cleaning it.");
    return false;
  }
  if (!speechSource) {
    if (!automatic) showBanner("warning", "Cleaning is only available for speech-derived transcript text.");
    return false;
  }
  if (!phiScrubbedConfirmed) {
    if (!automatic) showBanner("warning", "Run PHI detection and apply redactions before cleaning the transcript.");
    return false;
  }

  const priorSource = speechSource;
  const priorFallbackUsed = speechFallbackUsed;
  const priorPhiSummary = String(phiSummaryTextEl?.textContent || "");
  const priorPhiStatus = String(phiStatusTextEl?.textContent || "");

  await withBusy(automatic ? "Auto-cleaning transcript..." : "Cleaning transcript...", async () => {
    setSpeechStatus(automatic ? "Auto-cleaning scrubbed transcript…" : "Cleaning scrubbed transcript…");
    setSpeechWarningText("");
    const result = await postJSON("/report/clean_seed_text", {
      text,
      already_scrubbed: true,
      source: priorSource,
      strict: true,
    });
    const warnings = Array.isArray(result?.warnings) ? result.warnings : [];
    const cleanedText = String(result?.cleaned_text || "").trim();

    if (result?.changed && cleanedText) {
      replaceSeedTextFromSpeech(cleanedText, {
        source: priorSource,
        fallbackUsed: priorFallbackUsed,
        cleaned: true,
        warnings,
        preservePhiState: automatic,
      });
      if (automatic) {
        phiScrubbedConfirmed = true;
        setPhiConfirmAck(false);
        setPhiStatus(priorPhiStatus || "Redactions applied (scrubbed text ready to seed)");
        setPhiSummary(priorPhiSummary);
        setSpeechStatus("Transcript auto-cleaned. Confirm PHI removal, then seed.");
        renderPhiDetections();
        showBanner("success", "Transcript auto-cleaned after redaction. Confirm PHI removal, then seed.");
      } else {
        setSpeechStatus("Transcript cleaned. Re-run PHI detection before seeding.");
        showBanner("success", "Transcript cleaned. Re-run PHI detection, apply redactions, and confirm before seeding.");
      }
      return true;
    }

    speechSource = priorSource;
    speechFallbackUsed = priorFallbackUsed;
    speechCleaned = true;
    setSpeechStatus(
      automatic ? "Transcript auto-clean completed. Confirm PHI removal, then seed." : "Transcript checked. No cleanup changes were needed.",
    );
    setSpeechWarningText(joinSpeechWarnings(warnings));
    renderSpeechState();
    showBanner(
      "success",
      automatic
        ? "Transcript auto-clean completed. No cleanup changes were needed."
        : "Transcript cleanup completed. No text changes were needed.",
    );
    return true;
  });
  return true;
}

async function autoCleanSpeechTranscriptAfterRedaction() {
  if (!speechSource || speechCleaned || !phiScrubbedConfirmed) return false;
  try {
    return await cleanSpeechTranscript({ automatic: true });
  } catch (error) {
    setSpeechStatus("Redactions applied (scrubbed text ready to seed)");
    setSpeechWarningText(error?.message || "Automatic transcript cleanup failed");
    renderSpeechState();
    showBanner(
      "warning",
      "Redactions were applied, but automatic transcript cleanup was unavailable. You can still confirm PHI removal and seed.",
    );
    return false;
  }
}

function currentIssues() {
  if (state.verify?.issues) return state.verify.issues;
  if (state.render?.issues) return state.render.issues;
  return state.seed?.issues || [];
}

function currentWarnings() {
  if (state.verify?.warnings) return state.verify.warnings;
  if (state.render?.warnings) return state.render.warnings;
  return state.seed?.warnings || [];
}

function currentSuggestions() {
  if (state.verify?.suggestions) return state.verify.suggestions;
  if (state.render?.suggestions) return state.render.suggestions;
  return state.seed?.suggestions || [];
}

function currentInferenceNotes() {
  if (state.verify?.inference_notes) return state.verify.inference_notes;
  if (state.render?.inference_notes) return state.render.inference_notes;
  return state.seed?.inference_notes || [];
}

function currentQualityFlags() {
  if (state.verify?.quality_flags) return state.verify.quality_flags;
  if (state.render?.quality_flags) return state.render.quality_flags;
  return state.seed?.quality_flags || [];
}

function currentNeedsManualReview() {
  if (typeof state.verify?.needs_manual_review === "boolean") return state.verify.needs_manual_review;
  if (typeof state.render?.needs_manual_review === "boolean") return state.render.needs_manual_review;
  return Boolean(state.seed?.needs_manual_review);
}

function isBlockingQualityFlag(flag) {
  const severity = String(flag?.severity || "").trim().toLowerCase();
  if (severity === "blocker") return true;
  return Boolean(flag?.metadata?.blocking);
}

function currentBlockerFlags() {
  return currentQualityFlags().filter((flag) => isBlockingQualityFlag(flag));
}

function currentMarkdown() {
  if (typeof state.render?.markdown === "string") return state.render.markdown;
  if (typeof state.seed?.markdown === "string") return state.seed.markdown;
  return "";
}

function buildTransferNoteText() {
  const renderedMarkdown = String(currentMarkdown() || "").trim();
  const base = renderedMarkdown || String(seedTextEl.value || "").trim();
  const addendum = buildCompletenessAddendumBlock();
  return upsertCompletenessAddendum(base, addendum);
}

function safeSetStorageItem(storage, key, value) {
  if (!storage || typeof storage.setItem !== "function") return false;
  try {
    storage.setItem(key, value);
    return true;
  } catch {
    return false;
  }
}

function safeGetStorageItem(storage, key) {
  if (!storage || typeof storage.getItem !== "function") return null;
  try {
    return storage.getItem(key);
  } catch {
    return null;
  }
}

function safeRemoveStorageItem(storage, key) {
  if (!storage || typeof storage.removeItem !== "function") return;
  try {
    storage.removeItem(key);
  } catch {
    // ignore
  }
}

function parseTransferPayload(raw) {
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    const note = typeof parsed?.note === "string" ? parsed.note : "";
    if (!note.trim()) return null;
    return { note };
  } catch {
    return null;
  }
}

function consumeDashboardTransferPayload() {
  const parsed =
    parseTransferPayload(safeGetStorageItem(globalThis.sessionStorage, DASHBOARD_TO_REPORTER_STORAGE_KEY)) ||
    parseTransferPayload(safeGetStorageItem(globalThis.localStorage, DASHBOARD_TO_REPORTER_STORAGE_KEY));
  safeRemoveStorageItem(globalThis.sessionStorage, DASHBOARD_TO_REPORTER_STORAGE_KEY);
  safeRemoveStorageItem(globalThis.localStorage, DASHBOARD_TO_REPORTER_STORAGE_KEY);
  return parsed;
}

function transferToDashboard() {
  const blockerCount = currentBlockerFlags().length;
  if (blockerCount) {
    showBanner(
      "warning",
      `Resolve ${blockerCount} blocker flag${blockerCount === 1 ? "" : "s"} before sending this note to the dashboard.`,
    );
    return;
  }
  const note = buildTransferNoteText();
  if (!note) {
    showBanner("warning", "No note text available to transfer.");
    return;
  }

  const payload = JSON.stringify({
    note,
    source: "reporter_builder",
    note_type: String(currentMarkdown() || "").trim() ? "rendered_markdown" : "seed_text",
    transferred_at: new Date().toISOString(),
  });

  const wroteSession = safeSetStorageItem(globalThis.sessionStorage, DASHBOARD_TRANSFER_STORAGE_KEY, payload);
  const wroteLocal = safeSetStorageItem(globalThis.localStorage, DASHBOARD_TRANSFER_STORAGE_KEY, payload);

  if (!wroteSession && !wroteLocal) {
    showBanner("error", "Browser storage unavailable. Transfer to dashboard failed.");
    return;
  }

  window.location.href = "./";
}

function updateControls() {
  const hasBundle = Boolean(state.bundle);
  const hasQuestions = Array.isArray(state.questions) && state.questions.length > 0;
  const hasTransferNote = buildTransferNoteText().length > 0;
  const hasBlockers = currentBlockerFlags().length > 0;
  const hasSpeechTranscript = Boolean(speechSource || speechCleaned);
  const speechBusy = speechRequestingAccess || speechRecording || speechLocalPending || speechCloudPending;
  const requiresPhi = Boolean(phiRunBtn && phiApplyBtn && phiStatusTextEl);
  const ackChecked = !phiConfirmAckEl || Boolean(phiConfirmAckEl.checked);
  const phiOk = !requiresPhi || (phiScrubbedConfirmed && ackChecked);
  seedBtn.disabled = state.busy || speechBusy || !phiOk;
  if (state.busy) seedBtn.title = "";
  else if (!requiresPhi || phiOk) seedBtn.title = "";
  else if (!phiScrubbedConfirmed) seedBtn.title = "Run PHI detection and apply redactions first";
  else seedBtn.title = "Confirm PHI removal before seeding";
  refreshBtn.disabled = state.busy || !hasBundle;
  clearBtn.disabled = state.busy || speechBusy;
  applyPatchBtn.disabled = state.busy || !hasBundle || !hasQuestions;
  if (transferToDashboardBtn) {
    transferToDashboardBtn.disabled = state.busy || !hasTransferNote || hasBlockers;
    transferToDashboardBtn.title = hasBlockers
      ? "Resolve reporter blocker flags before sending this note to the dashboard"
      : "";
  }
  if (completenessInsertBtn) completenessInsertBtn.disabled = state.busy || !completenessPrompts.length;
  if (completenessCopyBtn) completenessCopyBtn.disabled = state.busy || !buildCompletenessAddendumBlock();
  strictToggleEl.disabled = state.busy || speechBusy;

  if (speechStartBtn) {
    const speechUnsupportedReason = getSpeechRecordingUnsupportedReason();
    const speechWorkerLoading = Boolean(speechWorker) && !speechWorkerReady && !speechWorkerUnavailableReason;
    speechStartBtn.disabled = state.busy || speechBusy || Boolean(speechUnsupportedReason) || speechWorkerLoading;
    if (speechUnsupportedReason) {
      speechStartBtn.title = speechUnsupportedReason;
    } else if (speechWorkerLoading) {
      speechStartBtn.title = `Wait for the local ${getSpeechModelLabel(speechModelKey)} model to finish loading.`;
    } else {
      speechStartBtn.title = "";
    }
  }
  if (speechStopBtn) speechStopBtn.disabled = state.busy || !speechRecording;
  if (speechDiscardBtn) {
    speechDiscardBtn.disabled = state.busy || speechBusy || (!speechLastRecordingBlob && !hasSpeechTranscript);
  }
  if (speechCloudFallbackBtn) speechCloudFallbackBtn.disabled = state.busy || speechBusy || !speechLastRecordingBlob;
  if (speechModelSelectEl) speechModelSelectEl.disabled = state.busy || speechBusy;

  if (phiRunBtn) phiRunBtn.disabled = state.busy || phiRunning || !phiWorkerReady;
  if (phiApplyBtn) phiApplyBtn.disabled = state.busy || phiRunning || !phiHasRunDetection;
  if (phiRevertBtn) phiRevertBtn.disabled = state.busy || phiRunning || !phiOriginalText;
  if (phiAddRedactionBtn) phiAddRedactionBtn.disabled = state.busy || phiRunning || !phiSelection;
  if (phiConfirmAckEl) phiConfirmAckEl.disabled = state.busy || phiRunning || !phiScrubbedConfirmed;
}

function renderSummary() {
  summaryQuestionsEl.textContent = String(state.questions?.length || 0);
  summaryIssuesEl.textContent = String(currentIssues().length);
  summaryWarningsEl.textContent = String(currentWarnings().length);
  summarySuggestionsEl.textContent = String(currentSuggestions().length);
}

function renderMarkdown() {
  const markdown = currentMarkdown();
  markdownOutputEl.textContent = markdown || "No rendered markdown yet.";
}

function renderValidation() {
  const issues = currentIssues();
  const warnings = currentWarnings();
  const suggestions = currentSuggestions();
  const notes = currentInferenceNotes();
  const qualityFlags = currentQualityFlags();
  const blockers = currentBlockerFlags();

  if (!issues.length && !warnings.length && !suggestions.length && !notes.length && !qualityFlags.length) {
    validationHostEl.innerHTML = '<div class="empty-state">No validation output yet.</div>';
    return;
  }

  let html = "";
  if (blockers.length) {
    html += "<h4>Blockers</h4><ul class=\"builder-list blocker-list\">";
    blockers.forEach((flag) => {
      const code = String(flag?.code || "BLOCKER");
      const message = String(flag?.message || "Manual review required.");
      html += `<li><strong>${escapeHtml(code)}</strong>: ${escapeHtml(message)}</li>`;
    });
    html += "</ul>";
  }

  const nonBlockingFlags = qualityFlags.filter((flag) => !isBlockingQualityFlag(flag));
  if (nonBlockingFlags.length) {
    html += "<h4>Quality Flags</h4><ul class=\"builder-list\">";
    nonBlockingFlags.forEach((flag) => {
      const code = String(flag?.code || "FLAG");
      const message = String(flag?.message || "");
      html += `<li><strong>${escapeHtml(code)}</strong>: ${escapeHtml(message)}</li>`;
    });
    html += "</ul>";
  }

  if (issues.length) {
    html += "<h4>Issues</h4><ul class=\"builder-list\">";
    issues.forEach((issue) => {
      const label = issue?.proc_id || issue?.proc_type || "procedure";
      const message = issue?.message || issue?.field_path || "Missing field";
      const severity = issue?.severity ? ` (${issue.severity})` : "";
      html += `<li><strong>${escapeHtml(label)}</strong>: ${escapeHtml(message)}${escapeHtml(severity)}</li>`;
    });
    html += "</ul>";
  }

  if (warnings.length) {
    html += "<h4>Warnings</h4><ul class=\"builder-list\">";
    warnings.forEach((warning) => {
      html += `<li>${escapeHtml(warning)}</li>`;
    });
    html += "</ul>";
  }

  if (suggestions.length) {
    html += "<h4>Suggestions</h4><ul class=\"builder-list\">";
    suggestions.forEach((suggestion) => {
      html += `<li>${escapeHtml(suggestion)}</li>`;
    });
    html += "</ul>";
  }

  if (notes.length) {
    html += "<h4>Inference Notes</h4><ul class=\"builder-list\">";
    notes.forEach((note) => {
      html += `<li>${escapeHtml(note)}</li>`;
    });
    html += "</ul>";
  }

  validationHostEl.innerHTML = html;
}

function renderCompletenessPrompts() {
  if (!completenessPromptsCardEl || !completenessPromptsBodyEl) return;
  completenessPromptsBodyEl.innerHTML = "";

  const prompts = Array.isArray(completenessPrompts) ? completenessPrompts : [];
  if (!prompts.length) {
    completenessPromptsCardEl.classList.add("hidden");
    if (completenessInsertBtn) completenessInsertBtn.disabled = true;
    if (completenessCopyBtn) completenessCopyBtn.disabled = true;
    return;
  }

  completenessPromptsCardEl.classList.remove("hidden");
  if (completenessInsertBtn) completenessInsertBtn.disabled = false;
  if (completenessCopyBtn) completenessCopyBtn.disabled = false;

  const counts = { required: 0, recommended: 0 };
  prompts.forEach((p) => {
    if (String(p?.severity || "").toLowerCase() === "required") counts.required += 1;
    else counts.recommended += 1;
  });

  const summary = document.createElement("div");
  summary.className = "completeness-summary";
  const summaryText = document.createElement("div");
  summaryText.className = "qa-line";
  summaryText.textContent = `${counts.required} required, ${counts.recommended} recommended`;
  summary.appendChild(summaryText);

  const hint = document.createElement("div");
  hint.className = "subtle";
  hint.textContent = "Fill values below and insert a short addendum into the note so extraction can capture them next run.";
  summary.appendChild(hint);

  completenessPromptsBodyEl.appendChild(summary);

  const list = document.createElement("ul");
  list.className = "completeness-list";

  prompts.forEach((p) => {
    const li = document.createElement("li");
    li.className = "completeness-item";

    const sevRaw = String(p?.severity || "recommended").toLowerCase();
    const sev = sevRaw === "required" ? "required" : "recommended";
    const badge = document.createElement("span");
    badge.className = `status-badge severity-${sev}`;
    badge.textContent = sev === "required" ? "Required" : "Recommended";
    li.appendChild(badge);

    const main = document.createElement("div");
    main.className = "completeness-item-main";

    const label = document.createElement("div");
    label.className = "completeness-item-label";
    label.textContent = String(p?.label || "Missing field");
    main.appendChild(label);

    const message = document.createElement("div");
    message.className = "completeness-item-message";
    message.textContent = String(p?.message || "");
    main.appendChild(message);

    const path = String(p?.path || "").trim();
    if (path) {
      const meta = document.createElement("div");
      meta.className = "completeness-item-path";
      meta.textContent = path;
      main.appendChild(meta);
    }

    const controls = document.createElement("div");
    controls.className = "completeness-item-controls";

    const spec = getCompletenessInputSpec(path);
    const stored = completenessValuesByPath.get(path);

    if (spec.type === "enum" && Array.isArray(spec.options)) {
      const select = document.createElement("select");
      select.className = "flat-select";
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "—";
      select.appendChild(blank);
      spec.options.forEach((opt) => {
        const option = document.createElement("option");
        option.value = String(opt);
        option.textContent = String(opt);
        select.appendChild(option);
      });
      select.value = stored === undefined || stored === null ? "" : String(stored);
      select.addEventListener("change", () => {
        completenessValuesByPath.set(path, String(select.value || ""));
        updateControls();
      });
      controls.appendChild(select);
    } else if (spec.type === "boolean") {
      const select = document.createElement("select");
      select.className = "flat-select";
      [
        ["", "—"],
        ["true", "Yes"],
        ["false", "No"],
      ].forEach(([value, labelText]) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = labelText;
        select.appendChild(option);
      });
      select.value = stored === undefined || stored === null ? "" : String(stored);
      select.addEventListener("change", () => {
        completenessValuesByPath.set(path, String(select.value || ""));
        updateControls();
      });
      controls.appendChild(select);
    } else if (spec.type === "multiselect" && Array.isArray(spec.options)) {
      const select = document.createElement("select");
      select.className = "flat-select";
      select.multiple = true;
      select.size = Math.min(Math.max(spec.options.length, 3), 6);
      spec.options.forEach((opt) => {
        const option = document.createElement("option");
        option.value = String(opt);
        option.textContent = String(opt);
        select.appendChild(option);
      });
      const selected = Array.isArray(stored)
        ? stored.map((v) => String(v))
        : typeof stored === "string"
          ? stored.split(",").map((v) => v.trim()).filter(Boolean)
          : [];
      Array.from(select.options).forEach((opt) => {
        opt.selected = selected.includes(String(opt.value));
      });
      select.addEventListener("change", () => {
        const vals = Array.from(select.selectedOptions || [])
          .map((o) => String(o.value || "").trim())
          .filter((v) => v !== "");
        completenessValuesByPath.set(path, vals);
        updateControls();
      });
      controls.appendChild(select);
    } else {
      const input = document.createElement("input");
      input.className = "flat-input";
      input.type = spec.type === "integer" || spec.type === "number" ? "number" : "text";
      if (spec.type === "integer") input.step = "1";
      if (spec.type === "number") input.step = "any";
      input.placeholder = spec.placeholder || "Enter value";
      input.value = stored === undefined || stored === null ? "" : String(stored);
      input.addEventListener("input", () => {
        completenessValuesByPath.set(path, String(input.value || ""));
        updateControls();
      });
      controls.appendChild(input);
    }

    li.appendChild(main);
    li.appendChild(controls);
    list.appendChild(li);
  });

  completenessPromptsBodyEl.appendChild(list);
}

function createQuestionInput(question, index) {
  const inputId = `question-input-${index}`;
  const type = question?.input_type || "string";
  const options = Array.isArray(question?.options) ? question.options : [];

  if (type === "boolean") {
    const select = document.createElement("select");
    select.id = inputId;
    select.className = "question-select";
    [
      ["", "-- Select --"],
      ["true", "Yes"],
      ["false", "No"],
    ].forEach(([value, label]) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = label;
      select.appendChild(option);
    });
    return select;
  }

  if ((type === "enum" || type === "multiselect") && options.length) {
    const select = document.createElement("select");
    select.id = inputId;
    select.className = "question-select";
    if (type === "multiselect") {
      select.multiple = true;
      select.size = Math.min(Math.max(options.length, 3), 8);
    } else {
      const blank = document.createElement("option");
      blank.value = "";
      blank.textContent = "-- Select --";
      select.appendChild(blank);
    }

    options.forEach((value) => {
      const option = document.createElement("option");
      option.value = String(value);
      option.textContent = String(value);
      select.appendChild(option);
    });
    return select;
  }

  if (type === "textarea") {
    const textarea = document.createElement("textarea");
    textarea.id = inputId;
    textarea.className = "question-textarea";
    textarea.rows = 2;
    textarea.placeholder = "Enter details";
    return textarea;
  }

  const input = document.createElement("input");
  input.id = inputId;
  input.className = "question-input";
  if (type === "integer" || type === "number") {
    input.type = "number";
    input.step = type === "integer" ? "1" : "any";
  } else {
    input.type = "text";
  }
  input.placeholder = "Enter value";
  return input;
}

function renderQuestions() {
  const questions = Array.isArray(state.questions) ? state.questions : [];
  if (!questions.length) {
    questionsHostEl.innerHTML = '<div class="empty-state">No open questions. Report may be complete.</div>';
    return;
  }

  const grouped = new Map();
  questions.forEach((question, index) => {
    const group = question?.group || "General";
    if (!grouped.has(group)) grouped.set(group, []);
    grouped.get(group).push({ question, index });
  });

  questionsHostEl.innerHTML = "";
  grouped.forEach((items, groupName) => {
    const groupWrap = document.createElement("div");
    groupWrap.className = "question-group";

    const header = document.createElement("div");
    header.className = "question-group-header";
    header.textContent = groupName;
    groupWrap.appendChild(header);

    items.forEach(({ question, index }) => {
      const row = document.createElement("div");
      row.className = "question-row";

      const label = document.createElement("label");
      label.className = "question-label";
      label.setAttribute("for", `question-input-${index}`);
      label.innerHTML = `${escapeHtml(question.label || "Question")} ${
        question.required ? '<span class="req">*</span>' : ""
      }<span class="question-meta">${escapeHtml(question.pointer || "")}</span>`;
      row.appendChild(label);

      row.appendChild(createQuestionInput(question, index));

      if (question.help) {
        const help = document.createElement("div");
        help.className = "question-help";
        help.textContent = question.help;
        row.appendChild(help);
      }

      groupWrap.appendChild(row);
    });

    questionsHostEl.appendChild(groupWrap);
  });
}

function renderBundleAndPatchJson() {
  bundleJsonEl.textContent = state.bundle
    ? JSON.stringify(state.bundle, null, 2)
    : "No bundle yet.";
  patchJsonEl.textContent = JSON.stringify(state.lastPatch || [], null, 2);
}

function renderAll() {
  renderSummary();
  renderMarkdown();
  renderCompletenessPrompts();
  renderQuestions();
  renderValidation();
  renderBundleAndPatchJson();
  renderSpeechSourceBadge();
  updateControls();
}

function decodePointerToken(token) {
  return token.replace(/~1/g, "/").replace(/~0/g, "~");
}

function pointerExists(documentValue, pointer) {
  if (!pointer || pointer === "/") return true;
  const tokens = pointer.split("/").slice(1).map(decodePointerToken);
  let current = documentValue;

  for (const token of tokens) {
    if (Array.isArray(current)) {
      if (!/^\d+$/.test(token)) return false;
      const index = Number(token);
      if (!Number.isInteger(index) || index < 0 || index >= current.length) return false;
      current = current[index];
      continue;
    }
    if (current && typeof current === "object") {
      if (!(token in current)) return false;
      current = current[token];
      continue;
    }
    return false;
  }
  return true;
}

function parseQuestionValue(question, index) {
  const input = document.getElementById(`question-input-${index}`);
  if (!input) return { hasValue: false, value: null };

  const type = question?.input_type || "string";
  if (type === "multiselect") {
    const selected = Array.from(input.selectedOptions || [])
      .map((opt) => String(opt.value || "").trim())
      .filter((value) => value !== "");
    if (!selected.length) return { hasValue: false, value: null };
    return { hasValue: true, value: selected };
  }

  if (type === "boolean") {
    const raw = String(input.value || "").trim().toLowerCase();
    if (!raw) return { hasValue: false, value: null };
    if (raw === "true") return { hasValue: true, value: true };
    if (raw === "false") return { hasValue: true, value: false };
    throw new Error(`Invalid boolean value for "${question.label}".`);
  }

  const raw = String(input.value || "").trim();
  if (!raw) return { hasValue: false, value: null };

  if (type === "integer") {
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed)) {
      throw new Error(`"${question.label}" must be an integer.`);
    }
    return { hasValue: true, value: parsed };
  }

  if (type === "number") {
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) {
      throw new Error(`"${question.label}" must be a number.`);
    }
    return { hasValue: true, value: parsed };
  }

  return { hasValue: true, value: raw };
}

function buildPatchOpsFromAnswers() {
  const bundle = state.bundle;
  if (!bundle) return [];
  const ops = [];

  state.questions.forEach((question, index) => {
    const parsed = parseQuestionValue(question, index);
    if (!parsed.hasValue) return;
    const pointer = String(question.pointer || "").trim();
    if (!pointer) return;
    let value = parsed.value;
    if (question?.input_type === "multiselect" && Array.isArray(value)) {
      // Some bundle fields are stored as strings even if the UI offers a multiselect.
      // Normalize those to avoid schema-validation failures during render.
      if (pointer.toLowerCase().endsWith("/echo_features")) {
        value = value.join(", ");
      }
    }
    if (pointer.toLowerCase().endsWith("/tests")) {
      if (typeof value === "string") {
        const parts = value
          .split(/[,;\n]+/)
          .map((item) => String(item || "").trim())
          .filter((item) => item !== "");
        if (!parts.length) return;
        value = parts;
      } else if (Array.isArray(value)) {
        const parts = value
          .map((item) => String(item || "").trim())
          .filter((item) => item !== "");
        if (!parts.length) return;
        value = parts;
      }
    }
    ops.push({
      op: pointerExists(bundle, pointer) ? "replace" : "add",
      path: pointer,
      value,
    });
  });
  return ops;
}

async function withBusy(taskName, fn) {
  state.busy = true;
  setStatus(taskName);
  updateControls();
  try {
    await fn();
  } finally {
    state.busy = false;
    setStatus("Ready");
    updateControls();
  }
}

async function seedBundleFromText() {
  const text = String(seedTextEl.value || "").trim();
  if (!text) {
    showBanner("warning", "Enter procedure text before seeding.");
    return;
  }

  if (phiRunBtn && !phiScrubbedConfirmed) {
    showBanner("warning", "Run PHI detection and apply redactions before seeding.");
    return;
  }

  if (phiConfirmAckEl && !phiConfirmAckEl.checked) {
    showBanner("warning", "Confirm PHI removal before seeding.");
    return;
  }

  await withBusy("Seeding bundle...", async () => {
    const strict = Boolean(strictToggleEl.checked);
    const seed = await postJSON("/report/seed_from_text", {
      text,
      already_scrubbed: true,
      metadata: {},
      strict,
    });

    state.seed = seed;
    completenessPrompts = Array.isArray(seed?.missing_field_prompts) ? seed.missing_field_prompts : [];
    completenessValuesByPath = new Map();
    state.render = {
      bundle: seed.bundle,
      markdown: seed.markdown,
      issues: seed.issues || [],
      warnings: seed.warnings || [],
      inference_notes: seed.inference_notes || [],
      suggestions: seed.suggestions || [],
      quality_flags: seed.quality_flags || [],
      needs_manual_review: Boolean(seed.needs_manual_review),
    };
    state.verify = {
      bundle: seed.bundle,
      issues: seed.issues || [],
      warnings: seed.warnings || [],
      inference_notes: seed.inference_notes || [],
      suggestions: seed.suggestions || [],
      questions: seed.questions || [],
      quality_flags: seed.quality_flags || [],
      needs_manual_review: Boolean(seed.needs_manual_review),
    };
    state.bundle = seed.bundle;
    state.questions = seed.questions || [];
    state.lastPatch = [];
    renderAll();

    const blockerCount = currentBlockerFlags().length;
    showBanner(
      blockerCount ? "warning" : "success",
      blockerCount
        ? `Bundle seeded, but ${blockerCount} blocker flag${blockerCount === 1 ? "" : "s"} require manual review before dashboard transfer.`
        : `Bundle seeded. ${state.questions.length} follow-up question${state.questions.length === 1 ? "" : "s"} generated.`,
    );
  });
}

async function refreshQuestions() {
  if (!state.bundle) {
    showBanner("warning", "Seed a bundle before refreshing questions.");
    return;
  }

  await withBusy("Refreshing questions...", async () => {
    const strict = Boolean(strictToggleEl.checked);
    const verify = await postJSON("/report/questions", {
      bundle: state.bundle,
      strict,
    });

    state.verify = verify;
    state.bundle = verify.bundle;
    state.questions = verify.questions || [];
    renderAll();

    const blockerCount = currentBlockerFlags().length;
    showBanner(
      blockerCount ? "warning" : "success",
      blockerCount
        ? `Questions refreshed. ${blockerCount} blocker flag${blockerCount === 1 ? "" : "s"} still require manual review.`
        : `Questions refreshed. ${state.questions.length} question${state.questions.length === 1 ? "" : "s"} remaining.`,
    );
  });
}

async function applyPatchAndRender() {
  if (!state.bundle) {
    showBanner("warning", "Seed a bundle before applying answers.");
    return;
  }

  let patchOps = [];
  try {
    patchOps = buildPatchOpsFromAnswers();
  } catch (error) {
    showBanner("error", error?.message || "Failed parsing question answers.");
    return;
  }

  if (!patchOps.length) {
    showBanner("warning", "No answers entered. Fill at least one question value.");
    return;
  }

  await withBusy("Applying JSON Patch...", async () => {
    const strict = Boolean(strictToggleEl.checked);
    const render = await postJSON("/report/render", {
      bundle: state.bundle,
      patch: patchOps,
      embed_metadata: false,
      strict,
    });

    const verify = await postJSON("/report/questions", {
      bundle: render.bundle,
      strict,
    });

    state.render = render;
    state.verify = verify;
    state.bundle = verify.bundle;
    state.questions = verify.questions || [];
    state.lastPatch = patchOps;
    renderAll();

    const blockerCount = currentBlockerFlags().length;
    showBanner(
      blockerCount ? "warning" : "success",
      blockerCount
        ? `Patch applied, but ${blockerCount} blocker flag${blockerCount === 1 ? "" : "s"} still require manual review.`
        : `Patch applied (${patchOps.length} op${patchOps.length === 1 ? "" : "s"}). ${state.questions.length} question${
            state.questions.length === 1 ? "" : "s"
          } remaining.`,
    );
  });
}

function clearState() {
  state.bundle = null;
  state.seed = null;
  state.verify = null;
  state.render = null;
  state.questions = [];
  state.lastPatch = [];
  completenessPrompts = [];
  completenessValuesByPath = new Map();
  resetSpeechState({ clearText: false });
  resetSpeechStartConfirmation();
  seedTextEl.value = "";
  phiDetections = [];
  phiExcludedDetections = new Set();
  phiHasRunDetection = false;
  phiScrubbedConfirmed = false;
  phiOriginalText = "";
  phiSelection = null;
  setPhiConfirmAck(false);
  setPhiSummary("");
  setPhiProgress("");
  if (phiWorkerReady) setPhiStatus("Ready (local model loaded)");
  setSpeechStatus(resolveSpeechIdleStatus());
  setSpeechWarningText("");
  renderPhiDetections();
  hideBanner();
  renderAll();
}

function insertCompletenessAddendumIntoNote() {
  const block = buildCompletenessAddendumBlock();
  if (!block) {
    showBanner("warning", "No completeness values entered yet.");
    return;
  }
  seedTextEl.value = upsertCompletenessAddendum(seedTextEl.value, block);
  showBanner("success", "Completeness addendum inserted into the note.");
  updateControls();
}

seedBtn.addEventListener("click", () => {
  seedBundleFromText().catch((error) => {
    showBanner("error", error?.message || "Seed request failed.");
  });
});

refreshBtn.addEventListener("click", () => {
  refreshQuestions().catch((error) => {
    showBanner("error", error?.message || "Refresh request failed.");
  });
});

applyPatchBtn.addEventListener("click", () => {
  applyPatchAndRender().catch((error) => {
    showBanner("error", error?.message || "Patch request failed.");
  });
});

clearBtn.addEventListener("click", clearState);
if (transferToDashboardBtn) transferToDashboardBtn.addEventListener("click", transferToDashboard);
if (completenessInsertBtn) completenessInsertBtn.addEventListener("click", insertCompletenessAddendumIntoNote);
if (completenessCopyBtn) {
  completenessCopyBtn.addEventListener("click", () => {
    const block = buildCompletenessAddendumBlock();
    if (!block) {
      showBanner("warning", "No completeness values entered yet.");
      return;
    }
    copyToClipboard(block).then((ok) => {
      showBanner(ok ? "success" : "warning", ok ? "Completeness addendum copied." : "Copy failed.");
    });
  });
}
if (speechStartBtn) {
  speechStartBtn.addEventListener("click", () => {
    startSpeechDictation().catch((error) => {
      showBanner("error", error?.message || "Unable to start reporter dictation.");
    });
  });
}
if (speechStartCancelBtn) {
  speechStartCancelBtn.addEventListener("click", () => {
    closeSpeechStartConfirmModal();
  });
}
if (speechStartConfirmBtn) {
  speechStartConfirmBtn.addEventListener("click", () => {
    speechStartConfirmedForCurrentNote = true;
    closeSpeechStartConfirmModal();
    startSpeechDictation().catch((error) => {
      showBanner("error", error?.message || "Unable to start reporter dictation.");
    });
  });
}
if (speechStopBtn) {
  speechStopBtn.addEventListener("click", stopSpeechDictation);
}
if (speechDiscardBtn) {
  speechDiscardBtn.addEventListener("click", discardSpeechTranscript);
}
if (speechCloudFallbackBtn) {
  speechCloudFallbackBtn.addEventListener("click", () => {
    runCloudSpeechFallback().catch((error) => {
      showBanner("error", error?.message || "Cloud fallback transcription failed.");
      setSpeechStatus("Cloud fallback failed.");
      setSpeechWarningText(error?.message || "Cloud fallback transcription failed");
      renderSpeechState();
    });
  });
}
if (speechModelSelectEl) {
  speechModelSelectEl.addEventListener("change", () => {
    applySpeechModelSelection(speechModelSelectEl.value, { restartWorker: true });
  });
}
seedTextEl.addEventListener("input", () => {
  if (!phiRunBtn) {
    updateControls();
    return;
  }

  const preservePhiState = preservePhiStateOnNextSeedInput;
  preservePhiStateOnNextSeedInput = false;

  // Any text edit invalidates prior detection + scrub confirmation.
  if (!preservePhiState && (phiHasRunDetection || phiScrubbedConfirmed || phiDetections.length)) {
    phiHasRunDetection = false;
    phiScrubbedConfirmed = false;
    phiDetections = [];
    phiExcludedDetections = new Set();
    phiOriginalText = "";
    setPhiConfirmAck(false);
    setPhiSummary("");
    setPhiProgress("");
    setPhiStatus("Text changed. Run detection to confirm redactions.");
    renderPhiDetections();
  }

  updatePhiSelectionFromTextarea();
});

if (seedTextEl) {
  ["select", "mouseup", "keyup"].forEach((eventName) => {
    seedTextEl.addEventListener(eventName, updatePhiSelectionFromTextarea);
  });
}

if (phiAddRedactionBtn) {
  phiAddRedactionBtn.addEventListener("click", () => {
    if (!phiSelection) return;

    const text = String(seedTextEl?.value || "");
    const start = clamp(Number(phiSelection.start) || 0, 0, text.length);
    const end = clamp(Number(phiSelection.end) || 0, 0, text.length);
    if (end <= start) return;

    const label = String(phiEntityTypeSelectEl?.value || "OTHER");
    const newDetection = {
      id: `manual_${Date.now()}_${Math.random().toString(36).slice(2)}`,
      label,
      start,
      end,
      score: 1.0,
      source: "manual",
    };

    phiDetections = [...phiDetections, newDetection];
    phiExcludedDetections.delete(newDetection.id);
    phiScrubbedConfirmed = false;
    setPhiConfirmAck(false);

    setPhiSummary(`${phiDetections.length} PHI span${phiDetections.length === 1 ? "" : "s"} detected`);
    setPhiStatus(`Added manual redaction: ${label}`);
    renderPhiDetections();

    if (seedTextEl) {
      seedTextEl.focus();
      seedTextEl.setSelectionRange(0, 0);
    }
    phiSelection = null;
    updateControls();
  });
}

if (phiConfirmAckEl) {
  phiConfirmAckEl.addEventListener("change", () => {
    updateControls();
  });
}

if (speechStartBtn) {
  applySpeechModelSelection(speechModelKey, { persist: false });
  setSpeechStatus(resolveSpeechIdleStatus());
  startSpeechWorker();
}

if (phiRunBtn) {
  const forceLegacy = shouldForceLegacyPhiWorker();
  setPhiStatus(forceLegacy ? "Loading legacy PHI worker…" : "Loading local PHI model…");
  startPhiWorker({ forceLegacy });

  phiRunBtn.addEventListener("click", () => {
    if (!phiWorker || phiRunning || !phiWorkerReady) return;
    phiRunning = true;
    phiHasRunDetection = false;
    phiScrubbedConfirmed = false;
    phiDetections = [];
    phiExcludedDetections = new Set();
    phiOriginalText = String(seedTextEl.value || "");
    setPhiConfirmAck(false);
    setPhiStatus("Detecting… (client-side)");
    setPhiProgress("");
    setPhiSummary("");
    renderPhiDetections();
    updateControls();
    try {
      phiWorker.postMessage({
        type: "start",
        text: phiOriginalText,
        config: buildPhiWorkerConfigForRun(),
      });
    } catch (e) {
      phiRunning = false;
      setPhiStatus(`Detection failed to start: ${e?.message || e}`);
      updateControls();
    }
  });
}

if (phiApplyBtn) {
  phiApplyBtn.addEventListener("click", () => {
    if (!phiHasRunDetection) return;

    const spans = getIncludedPhiDetections()
      .filter((d) => Number.isFinite(d.start) && Number.isFinite(d.end) && d.end > d.start)
      .sort((a, b) => b.start - a.start);

    let text = String(seedTextEl.value || "");
    for (const d of spans) {
      const start = Math.max(0, Math.min(text.length, Number(d.start)));
      const end = Math.max(0, Math.min(text.length, Number(d.end)));
      if (end <= start) continue;
      text = `${text.slice(0, start)}[REDACTED]${text.slice(end)}`;
    }
    seedTextEl.value = text;
    phiScrubbedConfirmed = true;
    setPhiConfirmAck(false);
    setPhiStatus("Redactions applied (scrubbed text ready to seed)");
    setPhiProgress("");
    setPhiSummary(`${spans.length} span${spans.length === 1 ? "" : "s"} redacted`);
    phiSelection = null;
    renderPhiDetections();
    updateControls();
    void autoCleanSpeechTranscriptAfterRedaction();
  });
}

if (phiRevertBtn) {
  phiRevertBtn.addEventListener("click", () => {
    if (!phiOriginalText) return;
    seedTextEl.value = phiOriginalText;
    phiDetections = [];
    phiExcludedDetections = new Set();
    phiHasRunDetection = false;
    phiScrubbedConfirmed = false;
    phiOriginalText = "";
    phiSelection = null;
    setPhiConfirmAck(false);
    setPhiSummary("");
    setPhiProgress("");
    setPhiStatus("Reverted. Run detection to confirm redactions.");
    renderPhiDetections();
    updateControls();
  });
}

const dashboardTransfer = consumeDashboardTransferPayload();
if (dashboardTransfer?.note) {
  resetSpeechState({ clearText: false });
  resetSpeechStartConfirmation();
  seedTextEl.value = dashboardTransfer.note;
  if (phiRunBtn) {
    phiDetections = [];
    phiExcludedDetections = new Set();
    phiHasRunDetection = false;
    phiScrubbedConfirmed = false;
    phiOriginalText = "";
    phiSelection = null;
    setPhiConfirmAck(false);
    setPhiSummary("");
    setPhiProgress("");
    setPhiStatus("Note loaded. Run detection and apply redactions before seeding.");
    renderPhiDetections();
  }
  setSpeechStatus(resolveSpeechIdleStatus());
  setSpeechWarningText("");
  showBanner("success", "Loaded note from dashboard. Run PHI detection, apply redactions, then seed.");
}

updatePhiSelectionFromTextarea();
renderPhiDetections();
renderAll();
