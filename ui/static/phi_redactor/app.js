/* global monaco */
import { buildPdfDocumentModel, extractPdfAdaptive } from "./pdf_local/pdf/pipeline.js";
import { canUseCameraScan, captureBestFrame, captureFrame, startCamera, stopCamera } from "./camera_local/cameraCapture.js";
import { createCameraCaptureQueue } from "./camera_local/cameraUi.js";
import {
  buildCameraOcrDocumentText,
  cancelCameraOcrJob,
  makeCameraWorkerUrl,
  runCameraOcrJob,
} from "./camera_local/imagePipeline.js";
import {
  buildCaptureWarnings,
  computeCaptureQualityMetrics,
  computeGrayFromImageData,
} from "./camera_local/imagePreprocess.js";
import { initPrivacyShield } from "./camera_local/privacyShield.js";

const statusTextEl = document.getElementById("statusText");
const progressTextEl = document.getElementById("progressText");
const detectionsListEl = document.getElementById("detectionsList");
const detectionsCountEl = document.getElementById("detectionsCount");
let serverResponseEl = document.getElementById("serverResponse");

const runBtn = document.getElementById("runBtn");
const cancelBtn = document.getElementById("cancelBtn");
const applyBtn = document.getElementById("applyBtn");
const revertBtn = document.getElementById("revertBtn");
const submitBtn = document.getElementById("submitBtn");
const exportBtn = document.getElementById("exportBtn");
const exportTablesBtn = document.getElementById("exportTablesBtn");
const exportEditedBtn = document.getElementById("exportEditedBtn");
const exportPatchBtn = document.getElementById("exportPatchBtn");
const newNoteBtn = document.getElementById("newNoteBtn");
const addRedactionBtn = document.getElementById("addRedactionBtn");
const entityTypeSelect = document.getElementById("entityTypeSelect");
const redactProvidersToggle = document.getElementById("redactProvidersToggle");
const editedResponseEl = document.getElementById("editedResponse");
const flattenedTablesHost = document.getElementById("flattenedTablesHost");
const feedbackPanelEl = document.getElementById("feedbackPanel");
const runIdDisplayEl = document.getElementById("runIdDisplay");
const submitterNameEl = document.getElementById("submitterName");
const feedbackRatingEl = document.getElementById("feedbackRating");
const feedbackCommentEl = document.getElementById("feedbackComment");
const submitFeedbackBtn = document.getElementById("submitFeedbackBtn");
const saveCorrectionsBtn = document.getElementById("saveCorrectionsBtn");
const feedbackStatusEl = document.getElementById("feedbackStatus");
const phiConfirmModalEl = document.getElementById("phiConfirmModal");
const bundlePanelEl = document.getElementById("bundlePanel");
const indexDateInputEl = document.getElementById("indexDateInput");
const docDateInputEl = document.getElementById("docDateInput");
const timepointRoleSelectEl = document.getElementById("timepointRoleSelect");
const docSeqInputEl = document.getElementById("docSeqInput");
const translateDatesToggleEl = document.getElementById("translateDatesToggle");
const chronoPreviewBtn = document.getElementById("chronoPreviewBtn");
const addToBundleBtn = document.getElementById("addToBundleBtn");
const submitBundleBtn = document.getElementById("submitBundleBtn");
const clearBundleBtn = document.getElementById("clearBundleBtn");
const clearCurrentNoteBtn = document.getElementById("clearCurrentNoteBtn");
const zkPatientIdInputEl = document.getElementById("zkPatientIdInput");
const episodeIdInputEl = document.getElementById("episodeIdInput");
const genBundleIdsBtn = document.getElementById("genBundleIdsBtn");
const bundleDocsHostEl = document.getElementById("bundleDocsHost");
const bundleSummaryHostEl = document.getElementById("bundleSummaryHost");
const chronoPreviewModalEl = document.getElementById("chronoPreviewModal");
const chronoPreviewBodyEl = document.getElementById("chronoPreviewBody");
const registryGridRootEl = document.getElementById("registryGridRoot");
const registryLegacyRootEl = document.getElementById("registryLegacyRoot");
const registryLegacyRightRootEl = document.getElementById("registryLegacyRightRoot");
const completenessPromptsCardEl = document.getElementById("completenessPromptsCard");
const completenessPromptsBodyEl = document.getElementById("completenessPromptsBody");
const completenessCopyBtn = document.getElementById("completenessCopyBtn");
const completenessOpenReporterBtn = document.getElementById("completenessOpenReporterBtn");
const focusClinicalBtn = document.getElementById("focusClinicalBtn");
const focusBillingBtn = document.getElementById("focusBillingBtn");
const splitReviewBtn = document.getElementById("splitReviewBtn");
const toggleDetectionsPaneBtn = document.getElementById("toggleDetectionsPaneBtn");
const pdfUploadInputEl = document.getElementById("pdfUploadInput");
const pdfOcrQualitySelectEl = document.getElementById("pdfOcrQualitySelect");
const pdfOcrMaskSelectEl = document.getElementById("pdfOcrMaskSelect");
const pdfExtractBtn = document.getElementById("pdfExtractBtn");
const pdfExtractSummaryEl = document.getElementById("pdfExtractSummary");
const cameraScanBtn = document.getElementById("cameraScanBtn");
const cameraScanSummaryEl = document.getElementById("cameraScanSummary");
const cameraScanModalEl = document.getElementById("cameraScanModal");
const cameraPreviewEl = document.getElementById("cameraPreview");
const cameraGuideOverlayEl = document.getElementById("cameraGuideOverlay");
const cameraGuideHintEl = document.getElementById("cameraGuideHint");
const cameraStartBtn = document.getElementById("cameraStartBtn");
const cameraCaptureBtn = document.getElementById("cameraCaptureBtn");
const cameraRetakeBtn = document.getElementById("cameraRetakeBtn");
const cameraClearBtn = document.getElementById("cameraClearBtn");
const cameraRunOcrBtn = document.getElementById("cameraRunOcrBtn");
const cameraCloseBtn = document.getElementById("cameraCloseBtn");
const cameraOcrQualitySelectEl = document.getElementById("cameraOcrQualitySelect");
const cameraEnhanceSelectEl = document.getElementById("cameraEnhanceSelect");
const cameraSceneHintSelectEl = document.getElementById("cameraSceneHintSelect");
const cameraStatusTextEl = document.getElementById("cameraStatusText");
const cameraProgressTextEl = document.getElementById("cameraProgressText");
const cameraThumbStripEl = document.getElementById("cameraThumbStrip");
const cameraWarningListEl = document.getElementById("cameraWarningList");
const cameraCropPanelEl = document.getElementById("cameraCropPanel");
const cameraCropPageSelectEl = document.getElementById("cameraCropPageSelect");
const cameraCropTopRangeEl = document.getElementById("cameraCropTopRange");
const cameraCropRightRangeEl = document.getElementById("cameraCropRightRange");
const cameraCropBottomRangeEl = document.getElementById("cameraCropBottomRange");
const cameraCropLeftRangeEl = document.getElementById("cameraCropLeftRange");
const cameraCropTopValueEl = document.getElementById("cameraCropTopValue");
const cameraCropRightValueEl = document.getElementById("cameraCropRightValue");
const cameraCropBottomValueEl = document.getElementById("cameraCropBottomValue");
const cameraCropLeftValueEl = document.getElementById("cameraCropLeftValue");
const cameraCropPreviewStageEl = document.getElementById("cameraCropPreviewStage");
const cameraCropPreviewImgEl = document.getElementById("cameraCropPreviewImg");
const cameraCropPreviewBoxEl = document.getElementById("cameraCropPreviewBox");
const cameraCropZoomWrapEl = document.getElementById("cameraCropZoomWrap");
const cameraCropZoomImgEl = document.getElementById("cameraCropZoomImg");
const cameraCropApplyBtn = document.getElementById("cameraCropApplyBtn");
const cameraCropApplyAllBtn = document.getElementById("cameraCropApplyAllBtn");
const cameraCropResetBtn = document.getElementById("cameraCropResetBtn");
const cameraCropResetAllBtn = document.getElementById("cameraCropResetAllBtn");
const privacyShieldEl = document.getElementById("privacyShield");

function detectCameraWarningProfile(env = globalThis) {
  try {
    const nav = env?.navigator;
    const ua = String(nav?.userAgent || "");
    const platform = String(nav?.platform || "");
    const maxTouchPoints = Number(nav?.maxTouchPoints || 0);
    const isIOSDevice = /iP(?:hone|ad|od)\b/i.test(ua) || (platform === "MacIntel" && maxTouchPoints > 1);
    if (!isIOSDevice) return "default";
    const isSafari = /\bSafari\//.test(ua) && !/\b(?:CriOS|FxiOS|EdgiOS|OPiOS|YaBrowser)\//.test(ua);
    return isSafari ? "ios_safari" : "default";
  } catch {
    return "default";
  }
}

const cameraWarningProfile = detectCameraWarningProfile();
let cameraGuideFrameRafId = 0;

let lastServerResponse = null;
let flatTablesBase = null;
let flatTablesState = null;
let editedPayload = null;
let editedDirty = false;
let registryGridEdits = null; // latest export from React RegistryGrid (JSON Patch + fields)
let currentRunId = null;
let feedbackSubmitted = false;
let fieldFeedbackStore = new Map(); // key: path, value: {path, error_type, correction, comment, ...}
let fieldFeedbackModalEl = null;
let activeFieldFeedbackContext = null;

let registryGridMonacoGetter = () => null;
let registryGridMounted = false;
let registryGridLoadPromise = null;
let lastCompletenessPrompts = [];
let completenessEdits = null; // {edited_patch, edited_fields} generated from completeness inputs
let completenessRawValueByPath = new Map(); // key: effective dotted path (with indices), value: raw string
let completenessSelectedIndexByPromptPath = new Map(); // key: prompt.path (with [*]), value: selected index (number)

const TESTER_MODE = new URLSearchParams(location.search).get("tester") === "1";
if (TESTER_MODE && feedbackPanelEl) feedbackPanelEl.open = true;

/**
 * Get merge mode from query param or localStorage.
 * - ?merge=union (default, safer - keeps all candidates until after veto)
 * - ?merge=best_of (legacy - may lose ML spans if regex span is vetoed)
 * - localStorage.phi_merge_mode (persistent dev override)
 */
function getConfiguredMergeMode() {
  const params = new URLSearchParams(location.search);
  const qp = params.get("merge");
  if (qp === "union" || qp === "best_of") return qp;

  const ls = localStorage.getItem("phi_merge_mode");
  if (ls === "union" || ls === "best_of") return ls;

  return "union"; // default: safer mode
}

function getConfiguredRedactProviders() {
  const params = new URLSearchParams(location.search);
  const qp = params.get("redact_providers");
  if (qp === "1") return true;
  if (qp === "0") return false;

  const ls = localStorage.getItem("phi_redact_providers");
  if (ls === "1") return true;
  if (ls === "0") return false;

  return true; // default: safer - treat clinician names as PHI
}

/**
 * Feature flag (opt-out): embedded React-based Registry grid.
 * - Default: ON
 * - Disable: ?reactGrid=0 or localStorage ui.reactGrid=0
 * - Enable: ?reactGrid=1 or localStorage ui.reactGrid=1
 */
function isReactRegistryGridEnabled() {
  const params = new URLSearchParams(location.search);
  const qp = params.get("reactGrid");
  if (qp === "1" || qp === "0") {
    try {
      localStorage.setItem("ui.reactGrid", qp);
    } catch {
      // ignore storage failures (private mode)
    }
    return qp === "1";
  }

  try {
    const ls = localStorage.getItem("ui.reactGrid");
    if (ls === "1") return true;
    if (ls === "0") return false;
  } catch {
    // ignore storage failures (private mode)
  }

  // Default ON (opt-out via ?reactGrid=0 or localStorage ui.reactGrid=0).
  return true;
}

/**
 * Review layout helpers (ergonomics).
 *
 * Goal: make it easier to review evidence by keeping the Monaco note and the
 * clinical editor visible together.
 *
 * Controls:
 * - Focus modes: "clinical" (hide billing col), "billing" (hide clinical col), "all"
 *   - Query param: ?focus=clinical|billing|all
 *   - localStorage: ui.reviewFocus
 * - Split review: side-by-side note + review
 *   - Query param: ?splitReview=1|0
 *   - localStorage: ui.reviewSplit
 * - Collapse detections sidebar:
 *   - Query param: ?detectionsCollapsed=1|0
 *   - localStorage: ui.detectionsCollapsed
 */
const UI_REVIEW_FOCUS_LS_KEY = "ui.reviewFocus";
const UI_REVIEW_SPLIT_LS_KEY = "ui.reviewSplit";
const UI_DETECTIONS_COLLAPSED_LS_KEY = "ui.detectionsCollapsed";
const REPORTER_DASHBOARD_TRANSFER_KEY = "ps.reporter_to_dashboard_note_v1";
const DASHBOARD_REPORTER_TRANSFER_KEY = "ps.dashboard_to_reporter_note_v1";

function safeGetLocalStorageItem(key) {
  try {
    return localStorage.getItem(key);
  } catch {
    return null;
  }
}

function safeSetLocalStorageItem(key, value) {
  try {
    localStorage.setItem(key, value);
  } catch {
    // ignore storage failures (private mode)
  }
}

function safeGetSessionStorageItem(key) {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null;
  }
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

function safeRemoveStorageItem(storage, key) {
  if (!storage || typeof storage.removeItem !== "function") return;
  try {
    storage.removeItem(key);
  } catch {
    // ignore storage failures (private mode)
  }
}

function parseReporterTransferPayload(raw) {
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

function consumeReporterTransferPayload() {
  const parsed =
    parseReporterTransferPayload(safeGetSessionStorageItem(REPORTER_DASHBOARD_TRANSFER_KEY)) ||
    parseReporterTransferPayload(safeGetLocalStorageItem(REPORTER_DASHBOARD_TRANSFER_KEY));
  safeRemoveStorageItem(globalThis.sessionStorage, REPORTER_DASHBOARD_TRANSFER_KEY);
  safeRemoveStorageItem(globalThis.localStorage, REPORTER_DASHBOARD_TRANSFER_KEY);
  return parsed;
}

function readBoolSetting(queryKey, storageKey, defaultValue) {
  const params = new URLSearchParams(location.search);
  const qp = params.get(queryKey);
  if (qp === "1") return true;
  if (qp === "0") return false;
  const ls = safeGetLocalStorageItem(storageKey);
  if (ls === "1") return true;
  if (ls === "0") return false;
  return Boolean(defaultValue);
}

function readEnumSetting(queryKey, storageKey, allowed, defaultValue) {
  const params = new URLSearchParams(location.search);
  const qp = params.get(queryKey);
  if (qp && allowed.includes(qp)) return qp;
  const ls = safeGetLocalStorageItem(storageKey);
  if (ls && allowed.includes(ls)) return ls;
  return defaultValue;
}

function setButtonActive(btn, active) {
  if (!btn) return;
  btn.classList.toggle("active", Boolean(active));
  btn.setAttribute("aria-pressed", active ? "true" : "false");
}

function getFocusModeFromBody() {
  const body = document.body;
  if (!body) return "all";
  if (body.classList.contains("ps-focus-clinical")) return "clinical";
  if (body.classList.contains("ps-focus-billing")) return "billing";
  return "all";
}

function applyFocusMode(mode, opts = {}) {
  const { persist = true } = opts;
  const body = document.body;
  if (!body) return;

  const next = mode === "clinical" || mode === "billing" ? mode : "all";
  body.classList.remove("ps-focus-clinical", "ps-focus-billing");
  if (next === "clinical") body.classList.add("ps-focus-clinical");
  if (next === "billing") body.classList.add("ps-focus-billing");

  if (persist) safeSetLocalStorageItem(UI_REVIEW_FOCUS_LS_KEY, next);
  syncLayoutControls();
}

function applySplitReview(enabled, opts = {}) {
  const { persist = true } = opts;
  const body = document.body;
  if (!body) return;
  body.classList.toggle("ps-review-split", Boolean(enabled));
  if (persist) safeSetLocalStorageItem(UI_REVIEW_SPLIT_LS_KEY, enabled ? "1" : "0");

  // If split is enabled and focus hasn't been explicitly set, default to Clinical
  // so the grid has enough room to be usable alongside the note.
  if (enabled) {
    const params = new URLSearchParams(location.search);
    const qp = params.get("focus");
    const stored = safeGetLocalStorageItem(UI_REVIEW_FOCUS_LS_KEY);
    if (!qp && !stored) applyFocusMode("clinical", { persist: true });
  }

  syncLayoutControls();
}

function applyDetectionsCollapsed(enabled, opts = {}) {
  const { persist = true } = opts;
  const body = document.body;
  if (!body) return;
  body.classList.toggle("ps-detections-collapsed", Boolean(enabled));
  if (persist) safeSetLocalStorageItem(UI_DETECTIONS_COLLAPSED_LS_KEY, enabled ? "1" : "0");
  syncLayoutControls();
}

function syncLayoutControls() {
  const body = document.body;
  if (!body) return;

  const focus = getFocusModeFromBody();
  setButtonActive(focusClinicalBtn, focus === "clinical");
  setButtonActive(focusBillingBtn, focus === "billing");
  setButtonActive(splitReviewBtn, body.classList.contains("ps-review-split"));

  if (toggleDetectionsPaneBtn) {
    const collapsed = body.classList.contains("ps-detections-collapsed");
    const label = collapsed ? "Expand detections panel" : "Collapse detections panel";
    toggleDetectionsPaneBtn.title = label;
    toggleDetectionsPaneBtn.setAttribute("aria-label", label);
  }
}

function applyInitialLayoutPrefs() {
  const body = document.body;
  if (!body) return;

  const split = readBoolSetting("splitReview", UI_REVIEW_SPLIT_LS_KEY, false);
  const detectionsCollapsed = readBoolSetting("detectionsCollapsed", UI_DETECTIONS_COLLAPSED_LS_KEY, false);

  let focus = "all";
  const params = new URLSearchParams(location.search);
  const focusQp = params.get("focus");
  const focusStored = safeGetLocalStorageItem(UI_REVIEW_FOCUS_LS_KEY);
  if (focusQp) focus = focusQp;
  else if (focusStored) focus = focusStored;
  else if (split) focus = "clinical";

  focus = readEnumSetting("focus", UI_REVIEW_FOCUS_LS_KEY, ["all", "clinical", "billing"], focus);
  applyFocusMode(focus, { persist: false });
  applySplitReview(split, { persist: false });
  applyDetectionsCollapsed(detectionsCollapsed, { persist: false });
  syncLayoutControls();
}

function initLayoutControls() {
  if (focusClinicalBtn) {
    focusClinicalBtn.disabled = false;
    focusClinicalBtn.addEventListener("click", () => {
      const current = getFocusModeFromBody();
      applyFocusMode(current === "clinical" ? "all" : "clinical");
    });
  }
  if (focusBillingBtn) {
    focusBillingBtn.disabled = false;
    focusBillingBtn.addEventListener("click", () => {
      const current = getFocusModeFromBody();
      applyFocusMode(current === "billing" ? "all" : "billing");
    });
  }
  if (splitReviewBtn) {
    splitReviewBtn.disabled = false;
    splitReviewBtn.addEventListener("click", () => {
      const enabled = document.body?.classList.contains("ps-review-split");
      applySplitReview(!enabled);
    });
  }
  if (toggleDetectionsPaneBtn) {
    toggleDetectionsPaneBtn.disabled = false;
    toggleDetectionsPaneBtn.addEventListener("click", () => {
      const enabled = document.body?.classList.contains("ps-detections-collapsed");
      applyDetectionsCollapsed(!enabled);
    });
  }

  syncLayoutControls();
}

applyInitialLayoutPrefs();

function showRegistryGridUi() {
  if (registryLegacyRightRootEl) registryLegacyRightRootEl.classList.add("hidden");
  if (registryGridRootEl) registryGridRootEl.classList.remove("hidden");
}

function showRegistryLegacyUi() {
  if (registryLegacyRightRootEl) registryLegacyRightRootEl.classList.remove("hidden");
  if (registryGridRootEl) registryGridRootEl.classList.add("hidden");
}

function setRegistryGridMonacoGetter(getterFn) {
  registryGridMonacoGetter = typeof getterFn === "function" ? getterFn : () => null;
}

function getRegistryGridMonacoEditorSafe() {
  try {
    return registryGridMonacoGetter?.() || null;
  } catch {
    return null;
  }
}

function setRegistryGridLoadingPlaceholder(message) {
  if (!registryGridRootEl) return;
  const msg = message || "Loading registry grid…";
  registryGridRootEl.innerHTML = `<div class="dash-empty" style="padding: 12px;">${msg}</div>`;
}

function loadRegistryGridBundle() {
  if (registryGridLoadPromise) return registryGridLoadPromise;

  registryGridLoadPromise = new Promise((resolve, reject) => {
    try {
      if (window.RegistryGrid && typeof window.RegistryGrid.mount === "function") {
        resolve(window.RegistryGrid);
        return;
      }

      // CSS is best-effort (JS load is the gating factor).
      const cssId = "registryGridCss";
      if (!document.getElementById(cssId)) {
        const link = document.createElement("link");
        link.id = cssId;
        link.rel = "stylesheet";
        link.href = "/ui/registry_grid/registry_grid.css";
        document.head.appendChild(link);
      }

      const scriptId = "registryGridScript";
      const existing = document.getElementById(scriptId);
      if (existing) {
        // If the script tag exists but the global isn't ready yet, wait briefly.
        const maxWaitMs = 10_000;
        const start = Date.now();
        const tick = () => {
          if (window.RegistryGrid && typeof window.RegistryGrid.mount === "function") {
            resolve(window.RegistryGrid);
            return;
          }
          if (Date.now() - start > maxWaitMs) {
            reject(new Error("RegistryGrid bundle tag present but global did not initialize"));
            return;
          }
          setTimeout(tick, 50);
        };
        tick();
        return;
      }

      const script = document.createElement("script");
      script.id = scriptId;
      script.src = "/ui/registry_grid/registry_grid.iife.js";
      script.async = true;
      script.onload = () => {
        if (window.RegistryGrid && typeof window.RegistryGrid.mount === "function") {
          resolve(window.RegistryGrid);
        } else {
          reject(new Error("RegistryGrid bundle loaded but window.RegistryGrid is missing"));
        }
      };
      script.onerror = () => reject(new Error("Failed to load /ui/registry_grid/registry_grid.iife.js"));
      document.head.appendChild(script);
    } catch (e) {
      reject(e);
    }
  }).catch((e) => {
    // Allow retry on the next render (e.g., after a fresh build or transient network failure).
    registryGridLoadPromise = null;
    throw e;
  });

  return registryGridLoadPromise;
}

async function maybeRenderRegistryGrid(data) {
  if (!registryGridRootEl || !registryLegacyRightRootEl) return false;
  if (!isReactRegistryGridEnabled()) {
    unmountRegistryGrid();
    showRegistryLegacyUi();
    return false;
  }

  showRegistryGridUi();
  setRegistryGridLoadingPlaceholder("Loading registry grid…");

  try {
    const api = await loadRegistryGridBundle();
    if (!api || typeof api.mount !== "function") {
      throw new Error("RegistryGrid API missing mount()");
    }

    const mountArgs = {
      rootEl: registryGridRootEl,
      getMonacoEditor: getRegistryGridMonacoEditorSafe,
      processResponse: data,
      onExportEditedJson: setRegistryGridEdits,
    };

    if (!registryGridMounted) {
      api.mount(mountArgs);
      registryGridMounted = true;
      return true;
    }

    if (typeof api.update === "function") {
      api.update({ processResponse: data });
      return true;
    }

    // Back-compat: if update() isn't present yet, re-mount.
    api.mount(mountArgs);
    return true;
  } catch (e) {
    console.error("RegistryGrid failed; falling back to legacy renderer.", e);
    registryGridMounted = false;
    try {
      window.RegistryGrid?.unmount?.();
    } catch {
      // ignore
    }
    showRegistryLegacyUi();
    return false;
  }
}

function unmountRegistryGrid() {
  if (!registryGridMounted) return;
  registryGridMounted = false;
  try {
    window.RegistryGrid?.unmount?.();
  } catch (e) {
    console.warn("RegistryGrid unmount failed (ignored).", e);
  }
}

const WORKER_CONFIG = {
  aiThreshold: 0.5,
  debug: true,
  // Quantized INT8 ONNX can silently collapse to all-"O" under WASM.
  // Keep this ON until quantized inference is validated end-to-end.
  forceUnquantized: true,
  // Merge mode: "union" (default, safer) or "best_of" (legacy)
  mergeMode: getConfiguredMergeMode(),
  // If false, clinician/provider/staff names are treated as PHI and can be redacted.
  protectProviders: false,
};

function buildWorkerConfigForRun() {
  const cfg = { ...WORKER_CONFIG };
  const redactProviders = redactProvidersToggle ? Boolean(redactProvidersToggle.checked) : true;
  cfg.protectProviders = !redactProviders;
  return cfg;
}

function ensureDetectionIds(list) {
  const seen = new Map();
  const input = Array.isArray(list) ? list : [];
  return input.map((d) => {
    if (d && typeof d.id === "string" && d.id) return d;
    const label = String(d?.label ?? "OTHER");
    const source = String(d?.source ?? "unknown");
    const start = Number.isFinite(d?.start) ? d.start : -1;
    const end = Number.isFinite(d?.end) ? d.end : -1;
    const base = `${label}:${source}:${start}:${end}`;
    const n = (seen.get(base) || 0) + 1;
    seen.set(base, n);
    const id = n === 1 ? base : `${base}:${n}`;
    return { ...d, id };
  });
}

if (redactProvidersToggle) {
  const initial = getConfiguredRedactProviders();
  redactProvidersToggle.checked = initial;
  redactProvidersToggle.addEventListener("change", () => {
    try {
      localStorage.setItem("phi_redact_providers", redactProvidersToggle.checked ? "1" : "0");
    } catch {
      // ignore storage failures (private mode)
    }
  });
}

const YES_NO_OPTIONS = [
  { value: "", label: "—" },
  { value: "Yes", label: "Yes" },
  { value: "No", label: "No" },
];
const SEDATION_TYPE_OPTIONS = [
  { value: "", label: "—" },
  { value: "Moderate", label: "Moderate" },
  { value: "Deep", label: "Deep" },
  { value: "General", label: "General" },
  { value: "MAC", label: "MAC" },
  { value: "Local Only", label: "Local Only" },
  { value: "Topical Only", label: "Topical Only" },
  { value: "None", label: "None" },
];
const AIRWAY_TYPE_OPTIONS = [
  { value: "", label: "—" },
  { value: "Native", label: "Native" },
  { value: "ETT", label: "ETT" },
  { value: "LMA", label: "LMA" },
  { value: "iGel", label: "iGel" },
  { value: "Tracheostomy", label: "Tracheostomy" },
];
const ROLE_OPTIONS = [
  { value: "primary", label: "Primary" },
  { value: "add_on", label: "Add On" },
];
const STATUS_OPTIONS = [
  { value: "Dropped", label: "Dropped" },
  { value: "Suppressed", label: "Suppressed" },
];
const RULE_OUTCOME_OPTIONS = [
  { value: "dropped", label: "Dropped" },
  { value: "suppressed", label: "Suppressed" },
  { value: "informational", label: "Informational" },
  { value: "allowed", label: "Allowed" },
];
const EBUS_ACTION_OPTIONS = [
  { value: "", label: "—" },
  { value: "inspected_only", label: "Inspected only" },
  { value: "needle_aspiration", label: "Needle aspiration" },
  { value: "core_biopsy", label: "Core biopsy" },
  { value: "forceps_biopsy", label: "Forceps biopsy" },
  { value: "other", label: "Other" },
];
const DISPOSITION_OPTIONS = [
  { value: "", label: "—" },
  { value: "Outpatient discharge", label: "Outpatient discharge" },
  { value: "Observation unit", label: "Observation unit" },
  { value: "Floor admission", label: "Floor admission" },
  { value: "ICU admission", label: "ICU admission" },
  { value: "Already inpatient - return to floor", label: "Already inpatient - return to floor" },
  { value: "Already inpatient - transfer to ICU", label: "Already inpatient - transfer to ICU" },
  { value: "Transfer to another facility", label: "Transfer to another facility" },
  { value: "OR", label: "OR" },
  { value: "Death", label: "Death" },
];

const HEMITHORAX_OPTIONS = [
  { value: "", label: "—" },
  { value: "Right", label: "Right" },
  { value: "Left", label: "Left" },
  { value: "Bilateral", label: "Bilateral" },
];
const CHEST_US_EFFUSION_VOLUME_OPTIONS = [
  { value: "", label: "—" },
  { value: "None", label: "None" },
  { value: "Minimal", label: "Minimal" },
  { value: "Small", label: "Small" },
  { value: "Moderate", label: "Moderate" },
  { value: "Large", label: "Large" },
];
const CHEST_US_ECHOGENICITY_OPTIONS = [
  { value: "", label: "—" },
  { value: "Anechoic", label: "Anechoic" },
  { value: "Hypoechoic", label: "Hypoechoic" },
  { value: "Isoechoic", label: "Isoechoic" },
  { value: "Hyperechoic", label: "Hyperechoic" },
];
const CHEST_US_LOCULATIONS_OPTIONS = [
  { value: "", label: "—" },
  { value: "None", label: "None" },
  { value: "Thin", label: "Thin" },
  { value: "Thick", label: "Thick" },
];
const CHEST_US_DIAPHRAGM_MOTION_OPTIONS = [
  { value: "", label: "—" },
  { value: "Normal", label: "Normal" },
  { value: "Diminished", label: "Diminished" },
  { value: "Absent", label: "Absent" },
];
const CHEST_US_LUNG_SLIDING_OPTIONS = [
  { value: "", label: "—" },
  { value: "Present", label: "Present" },
  { value: "Absent", label: "Absent" },
];
const CHEST_US_PLEURA_OPTIONS = [
  { value: "", label: "—" },
  { value: "Normal", label: "Normal" },
  { value: "Thick", label: "Thick" },
  { value: "Nodular", label: "Nodular" },
];
const ASA_CLASS_OPTIONS = [
  { value: "", label: "—" },
  { value: "1", label: "1" },
  { value: "2", label: "2" },
  { value: "3", label: "3" },
  { value: "4", label: "4" },
  { value: "5", label: "5" },
  { value: "6", label: "6" },
];
const BRONCHUS_SIGN_OPTIONS = [
  { value: "", label: "—" },
  { value: "Positive", label: "Positive" },
  { value: "Negative", label: "Negative" },
  { value: "Not assessed", label: "Not assessed" },
];
const RADIAL_EBUS_PROBE_POSITION_OPTIONS = [
  { value: "", label: "—" },
  { value: "Concentric", label: "Concentric" },
  { value: "Eccentric", label: "Eccentric" },
  { value: "Adjacent", label: "Adjacent" },
  { value: "Not visualized", label: "Not visualized" },
];
const NAV_CONFIRMATION_METHOD_OPTIONS = [
  { value: "", label: "—" },
  { value: "Radial EBUS", label: "Radial EBUS" },
  { value: "CBCT", label: "CBCT" },
  { value: "Fluoroscopy", label: "Fluoroscopy" },
  { value: "Augmented Fluoroscopy", label: "Augmented Fluoroscopy" },
  { value: "None", label: "None" },
];
const NAV_TARGET_CONFIRMATION_METHOD_OPTIONS = [
  { value: "", label: "—" },
  { value: "Radial EBUS", label: "Radial EBUS" },
  { value: "CBCT", label: "CBCT" },
  { value: "Fluoroscopy", label: "Fluoroscopy" },
  { value: "Augmented fluoroscopy", label: "Augmented fluoroscopy" },
  { value: "None", label: "None" },
];
const FIBRINOLYTIC_INDICATION_OPTIONS = [
  { value: "", label: "—" },
  { value: "Complex parapneumonic", label: "Complex parapneumonic" },
  { value: "Empyema", label: "Empyema" },
  { value: "Hemothorax", label: "Hemothorax" },
  { value: "Malignant effusion", label: "Malignant effusion" },
];

runBtn.disabled = true;
cancelBtn.disabled = true;
applyBtn.disabled = true;
revertBtn.disabled = true;
submitBtn.disabled = true;
if (exportBtn) exportBtn.disabled = true;
if (exportTablesBtn) exportTablesBtn.disabled = true;
if (exportEditedBtn) exportEditedBtn.disabled = true;
if (exportPatchBtn) exportPatchBtn.disabled = true;
if (newNoteBtn) newNoteBtn.disabled = true;
if (statusTextEl) statusTextEl.textContent = "Booting UI…";
resetRunState();
if (submitterNameEl) submitterNameEl.addEventListener("input", updateFeedbackButtons);

function setStatus(text) {
  if (!statusTextEl) return;
  statusTextEl.textContent = text;
}

function setProgress(text) {
  if (!progressTextEl) return;
  progressTextEl.textContent = text || "";
}

function setFeedbackStatus(text) {
  if (!feedbackStatusEl) return;
  feedbackStatusEl.textContent = text || "";
}

function getSubmitterName() {
  return String(submitterNameEl?.value || "").trim();
}

function updateFeedbackButtons() {
  const hasName = getSubmitterName().length > 0;
  const flagCount = fieldFeedbackStore ? fieldFeedbackStore.size : 0;

  if (runIdDisplayEl) {
    runIdDisplayEl.textContent = currentRunId ? `Run ID: ${currentRunId}` : "Run ID: (not persisted)";
  }

  if (submitFeedbackBtn) {
    submitFeedbackBtn.disabled = !currentRunId || feedbackSubmitted || !hasName;
  }
  if (saveCorrectionsBtn) {
    saveCorrectionsBtn.disabled = !currentRunId || !editedPayload;
    saveCorrectionsBtn.textContent =
      flagCount > 0 ? `Save corrections (${flagCount} flag${flagCount === 1 ? "" : "s"})` : "Save corrections";
  }
}

  function resetRunState() {
    currentRunId = null;
    feedbackSubmitted = false;
    setFeedbackStatus("");
    updateFeedbackButtons();
  }

  function resetFeedbackDraft() {
    if (feedbackRatingEl) feedbackRatingEl.value = feedbackRatingEl.defaultValue || "8";
    if (feedbackCommentEl) feedbackCommentEl.value = feedbackCommentEl.defaultValue || "";
  }

  function setRunId(runId) {
    currentRunId = runId || null;
    feedbackSubmitted = false;
    setFeedbackStatus("");
    updateFeedbackButtons();
  }

async function confirmPhiRemoval() {
  if (!phiConfirmModalEl || typeof phiConfirmModalEl.showModal !== "function") {
    return window.confirm(
      "Confirm PHI removal before persistence. The server must store scrubbed-only text."
    );
  }

  return new Promise((resolve) => {
    const onClose = () => {
      phiConfirmModalEl.removeEventListener("close", onClose);
      resolve(phiConfirmModalEl.returnValue === "confirm");
    };
    phiConfirmModalEl.addEventListener("close", onClose);
    phiConfirmModalEl.showModal();
  });
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function buildLineStartOffsets(text) {
  const starts = [0];
  for (let i = 0; i < text.length; i++) {
    if (text.charCodeAt(i) === 10) starts.push(i + 1);
  }
  return starts;
}

function offsetToPosition(offset, lineStarts, textLength) {
  const safeOffset = clamp(offset, 0, textLength);

  let lo = 0;
  let hi = lineStarts.length - 1;
  let best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (lineStarts[mid] <= safeOffset) {
      best = mid;
      lo = mid + 1;
    } else {
      hi = mid - 1;
    }
  }
  const lineStart = lineStarts[best] ?? 0;
  return { lineNumber: best + 1, column: safeOffset - lineStart + 1 };
}

function formatScore(score) {
  if (typeof score !== "number") return "—";
  return score.toFixed(2);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "className") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v != null) {
      node.setAttribute(k, String(v));
    }
  }
  for (const child of children) node.appendChild(child);
  return node;
}

function safeSnippet(text, start, end) {
  const s = clamp(start, 0, text.length);
  const e = clamp(end, 0, text.length);
  const raw = text.slice(s, e);
  const oneLine = raw.replace(/\s+/g, " ").trim();
  if (oneLine.length <= 120) return oneLine || "(empty)";
  return `${oneLine.slice(0, 117)}…`;
}

function safeHtml(str) {
  return String(str || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

// =============================================================================
// Zero-knowledge temporal tokens (client-side only)
// =============================================================================

const MS_PER_DAY = 24 * 60 * 60 * 1000;

const ZK_BRACKET_TOKEN_RE = /\[[A-Z_ ]{2,32}:[^\]]*\]/g;
const ZK_ISO_DATE_RE = /\b(?:19|20)\d{2}[-/](?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])\b/g;
const ZK_US_NUMERIC_DATE_RE = /\b(?:0?[1-9]|1[0-2])[-/](?:0?[1-9]|[12]\d|3[01])[-/](?:\d{2}|\d{4})\b/g;
const ZK_MONTH_NAME_DATE_RE = /\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:\s*,?\s*(?:19|20)\d{2})?\b/gi;

function parseIsoDateInput(value) {
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(String(value || "").trim());
  if (!m) return null;
  return { year: Number(m[1]), month: Number(m[2]), day: Number(m[3]) };
}

function isValidYmd(ymd) {
  if (!ymd) return false;
  const year = Number(ymd.year);
  const month = Number(ymd.month);
  const day = Number(ymd.day);
  if (!Number.isInteger(year) || year < 1800 || year > 2200) return false;
  if (!Number.isInteger(month) || month < 1 || month > 12) return false;
  if (!Number.isInteger(day) || day < 1 || day > 31) return false;

  const d = new Date(Date.UTC(year, month - 1, day, 12, 0, 0, 0));
  return d.getUTCFullYear() === year && d.getUTCMonth() === month - 1 && d.getUTCDate() === day;
}

function utcNoonMs(ymd) {
  if (!isValidYmd(ymd)) return null;
  return Date.UTC(ymd.year, ymd.month - 1, ymd.day, 12, 0, 0, 0);
}

function diffDaysUtcNoon(indexYmd, targetYmd) {
  const a = utcNoonMs(indexYmd);
  const b = utcNoonMs(targetYmd);
  if (a == null || b == null) return null;
  return Math.round((b - a) / MS_PER_DAY);
}

function formatTOffset(days) {
  const n = Number(days);
  if (!Number.isFinite(n)) return "T+0";
  if (n >= 0) return `T+${Math.trunc(n)}`;
  return `T-${Math.abs(Math.trunc(n))}`;
}

function buildDateToken(days) {
  return `[DATE: ${formatTOffset(days)} DAYS]`;
}

function buildSystemHeaderToken({ role, seq, docOffsetDays }) {
  const safeRole = String(role || "").toUpperCase() || "UNKNOWN";
  const safeSeq = Number.isFinite(Number(seq)) ? Math.trunc(Number(seq)) : 0;
  return `[SYSTEM: ROLE=${safeRole} SEQ=${safeSeq} DOC_OFFSET=${formatTOffset(docOffsetDays)} DAYS]`;
}

function stripBracketTokensForLeakScan(text) {
  return String(text || "").replace(ZK_BRACKET_TOKEN_RE, " ");
}

function countDateLikeStringsForLeakScan(text) {
  const clean = String(text || "");
  if (!clean.trim()) return 0;
  const regexes = [ZK_ISO_DATE_RE, ZK_US_NUMERIC_DATE_RE, ZK_MONTH_NAME_DATE_RE];
  let count = 0;
  for (const re of regexes) {
    // Ensure we don't rely on global regex state
    const clone = new RegExp(re.source, re.flags);
    count += Array.from(clean.matchAll(clone)).length;
  }
  return count;
}

function sanitizeDateCandidate(raw) {
  let s = String(raw || "").trim();
  s = s.replace(/^_+|_+$/g, ""); // template underscores
  s = s.replace(/^[\[(]+|[\])]+$/g, ""); // brackets/parens
  s = s.replace(/[.,;:]+$/g, ""); // trailing punctuation
  return s.trim();
}

function monthNameToNumber(name) {
  const key = String(name || "").toLowerCase().slice(0, 3);
  const map = {
    jan: 1,
    feb: 2,
    mar: 3,
    apr: 4,
    may: 5,
    jun: 6,
    jul: 7,
    aug: 8,
    sep: 9,
    oct: 10,
    nov: 11,
    dec: 12,
  };
  return map[key] ?? null;
}

function normalizeYear(rawYear) {
  const y = Number(rawYear);
  if (!Number.isFinite(y)) return null;
  if (y >= 1000) return Math.trunc(y);
  // Pivot: 00-30 => 2000-2030, else 1900-1999
  if (y >= 0 && y <= 30) return 2000 + Math.trunc(y);
  if (y >= 31 && y <= 99) return 1900 + Math.trunc(y);
  return null;
}

function parseAbsoluteDateCandidate(raw) {
  const candidate = sanitizeDateCandidate(raw);
  if (!candidate) return { ymd: null, normalized: "", pattern: "", warning: "empty" };

  // ISO: YYYY-MM-DD or YYYY/MM/DD
  let m = /^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$/.exec(candidate);
  if (m) {
    const ymd = { year: Number(m[1]), month: Number(m[2]), day: Number(m[3]) };
    if (!isValidYmd(ymd)) return { ymd: null, normalized: "", pattern: "iso", warning: "invalid date" };
    return { ymd, normalized: `${m[1]}-${String(m[2]).padStart(2, "0")}-${String(m[3]).padStart(2, "0")}`, pattern: "iso", warning: "" };
  }

  // Numeric: M/D/YYYY or D/M/YYYY (best-effort disambiguation)
  m = /^(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})$/.exec(candidate);
  if (m) {
    const a = Number(m[1]);
    const b = Number(m[2]);
    const year = normalizeYear(m[3]);
    if (!Number.isFinite(a) || !Number.isFinite(b) || year == null) {
      return { ymd: null, normalized: "", pattern: "numeric", warning: "unparseable numeric date" };
    }
    let month = a;
    let day = b;
    let warning = "";
    if (a > 12 && b <= 12) {
      day = a;
      month = b;
      warning = "interpreted as D/M";
    } else if (a <= 12 && b <= 12) {
      warning = "ambiguous; interpreted as M/D";
    }
    const ymd = { year, month, day };
    if (!isValidYmd(ymd)) return { ymd: null, normalized: "", pattern: "numeric", warning: "invalid date" };
    return { ymd, normalized: `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`, pattern: "numeric", warning };
  }

  // DD-MMM-YYYY, DD MMM YYYY, DDMMMYYYY (month names)
  m = /^(\d{1,2})[-\s]?([A-Za-z]{3,9})[-\s]?(\d{2,4})$/.exec(candidate);
  if (m) {
    const day = Number(m[1]);
    const month = monthNameToNumber(m[2]);
    const year = normalizeYear(m[3]);
    const ymd = { year, month, day };
    if (!isValidYmd(ymd)) return { ymd: null, normalized: "", pattern: "dd_mmm", warning: "invalid date" };
    return { ymd, normalized: `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`, pattern: "dd_mmm", warning: "" };
  }

  // MMM DD, YYYY (year optional)
  m = /^([A-Za-z]{3,9})\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s*,?\s*(\d{2,4}))?$/.exec(candidate);
  if (m) {
    const month = monthNameToNumber(m[1]);
    const day = Number(m[2]);
    const year = m[3] ? normalizeYear(m[3]) : null;
    if (!year) {
      return { ymd: null, normalized: "", pattern: "mmm_dd", warning: "missing year" };
    }
    const ymd = { year, month, day };
    if (!isValidYmd(ymd)) return { ymd: null, normalized: "", pattern: "mmm_dd", warning: "invalid date" };
    return { ymd, normalized: `${year}-${String(month).padStart(2, "0")}-${String(day).padStart(2, "0")}`, pattern: "mmm_dd", warning: "" };
  }

  return { ymd: null, normalized: "", pattern: "", warning: "unrecognized date format" };
}

function buildDateRedactionReplacement(raw, { translateDates, indexYmd }) {
  if (!translateDates) return "[REDACTED]";
  if (!indexYmd) return "[DATE: REDACTED]";

  const parsed = parseAbsoluteDateCandidate(raw);
  if (!parsed.ymd) return "[DATE: REDACTED]";
  const offsetDays = diffDaysUtcNoon(indexYmd, parsed.ymd);
  if (offsetDays == null) return "[DATE: REDACTED]";
  return buildDateToken(offsetDays);
}

function highlightSpanInEditor(start, end) {
  try {
    // Fallback: if Monaco is still loading/unavailable, highlight in the basic textarea.
    if (!window.editor) {
      const ta = document.getElementById("fallbackTextarea");
      if (!ta) return;
      const textLength = ta.value.length;
      const s = clamp(Number(start) || 0, 0, textLength);
      const e = clamp(Number(end) || 0, 0, textLength);
      ta.focus();
      ta.setSelectionRange(s, e);
      return;
    }

    const model = window.editor.getModel();
    if (!model) return;

    const s = model.getPositionAt(Math.max(0, start));
    const e = model.getPositionAt(Math.max(0, end));
    const range = new monaco.Range(s.lineNumber, s.column, e.lineNumber, e.column);

    window.editor.setSelection(range);
    window.editor.revealRangeInCenter(range);
    window.editor.focus();
  } catch (err) {
    console.warn("Failed to highlight evidence span", err);
  }
}

window.__highlightEvidence = (start, end) => highlightSpanInEditor(start, end);

function normalizeSpans(spans) {
  // Accept shapes:
  // {start,end,text} or {span:[start,end],text} or {start,end,snippet}
  if (!Array.isArray(spans)) return [];
  return spans
    .map((sp) => ({
      text: sp.text ?? sp.snippet ?? "",
      start: sp.start ?? sp.span?.[0] ?? sp.span?.start ?? 0,
      end: sp.end ?? sp.span?.[1] ?? sp.span?.end ?? 0,
    }))
    .filter(
      (sp) =>
        Number.isFinite(sp.start) && Number.isFinite(sp.end) && sp.end > sp.start
    );
}

function renderEvidenceChips(spans) {
  const normalized = normalizeSpans(spans);
  if (normalized.length === 0) return "—";

  return normalized
    .map(
      (sp) => `
    <button class="ev-chip" title="Click to highlight ${sp.start}-${sp.end}"
      onclick="window.__highlightEvidence(${sp.start}, ${sp.end})">
      ${safeHtml(sp.text || "(evidence)")}
      <span class="ev-range">(${sp.start}-${sp.end})</span>
    </button>
  `
    )
    .join(" ");
}

function getEvidenceMap(data) {
  // Prefer registry.evidence; fall back to top-level evidence
  return data?.registry?.evidence || data?.evidence || {};
}

function formatNumber(value, digits = 2) {
  if (!Number.isFinite(value)) return "—";
  return value.toFixed(digits);
}

function formatCurrency(value) {
  if (!Number.isFinite(value)) return "—";
  return `$${value.toFixed(2)}`;
}

function titleCaseKey(key) {
  const raw = String(key || "");
  if (!raw) return "—";

  const special = {
    bal: "BAL",
    linear_ebus: "Linear EBUS",
    radial_ebus: "Radial EBUS",
    navigational_bronchoscopy: "Navigational Bronchoscopy",
    diagnostic_bronchoscopy: "Diagnostic Bronchoscopy",
    therapeutic_aspiration: "Therapeutic Aspiration",
    tbna_conventional: "Conventional TBNA",
    peripheral_tbna: "Peripheral TBNA",
    ipc: "IPC",
    chest_tube: "Chest Tube",
  };
  if (special[raw]) return special[raw];

  return raw
    .replaceAll("_", " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function normalizeCptCode(code) {
  return String(code || "").trim().replace(/^\+/, "");
}

function clearEl(node) {
  if (!node) return;
  while (node.firstChild) node.removeChild(node.firstChild);
}

function fmtBool(val) {
  if (val === null || val === undefined) return "—";
  return val ? "Yes" : "No";
}

function fmtMaybe(val) {
  if (val === null || val === undefined) return "—";
  const s = String(val).trim();
  return s ? s : "—";
}

function isLikelyHeaderDump(text) {
  const s = String(text || "").trim();
  if (!s) return false;
  const lower = s.toLowerCase();
  if (lower.startsWith("pt:") || lower.startsWith("patient:")) return true;
  if (lower.includes("||")) return true;
  if (/\bmrn\b\s*:/i.test(s)) return true;
  if (/\bdob\b\s*:/i.test(s)) return true;
  if (/\battending\b\s*:/i.test(s)) return true;
  if (/\bfellow\b\s*:/i.test(s)) return true;
  return false;
}

function cleanLocationForDisplay(value) {
  const s = String(value || "").trim();
  if (!s) return "";
  if (isLikelyHeaderDump(s)) return "";
  return s.replace(/\s+/g, " ").trim();
}

function cleanLocationsListForDisplay(value) {
  const arr = Array.isArray(value) ? value : [];
  const cleaned = arr
    .map((v) => cleanLocationForDisplay(v))
    .filter((v) => v && String(v).trim() !== "");
  return cleaned;
}

function cleanIndicationForDisplay(value) {
  const raw = String(value || "").trim();
  if (!raw) return "";
  let s = raw;
  s = s.replace(/\\bPROCEDURE\\b[\\s\\S]*$/i, "").trim();
  s = s.replace(/\\bTARGET(?:S)?\\b\\s*:[\\s\\S]*$/i, "").trim();
  s = s.split(/\n\\s*\n/)[0] || s;
  s = s.replace(/\s+/g, " ").trim();
  if (s.length > 220) s = `${s.slice(0, 220)}…`;
  return s;
}

function fmtSpan(span) {
  const start = span?.[0];
  const end = span?.[1];
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return "—";
  return `${start}–${end}`;
}

function splitWarning(w) {
  const text = String(w || "").trim();
  if (!text) return null;
  const idx = text.indexOf(":");
  if (idx > 0 && idx < 48) {
    return { category: text.slice(0, idx).trim() || "Other", message: text.slice(idx + 1).trim() || "—" };
  }
  return { category: "Other", message: text };
}

function groupWarnings(warnings) {
  const grouped = new Map();
  (Array.isArray(warnings) ? warnings : []).forEach((w) => {
    const parsed = splitWarning(w);
    if (!parsed) return;
    if (!grouped.has(parsed.category)) grouped.set(parsed.category, []);
    grouped.get(parsed.category).push(parsed.message);
  });
  return grouped;
}

function getRegistry(data) {
  return data?.registry || {};
}

function hasDisplayValue(value) {
  return value !== null && value !== undefined && (typeof value !== "string" || value.trim() !== "");
}

function pickRegistryPathValue(registry, paths) {
  const candidates = Array.isArray(paths) ? paths : [];
  for (const path of candidates) {
    const value = getByPath(registry, path);
    if (hasDisplayValue(value)) return { path, value };
  }
  return { path: candidates[0] || "", value: null };
}

function normalizeSexDisplay(value) {
  const raw = String(value ?? "").trim();
  if (!raw) return null;
  const upper = raw.toUpperCase();
  if (upper === "M") return "Male";
  if (upper === "F") return "Female";
  if (upper === "O") return "Other";
  return raw;
}

function getCodingSupport(data) {
  const cs = getRegistry(data)?.coding_support;
  return cs && typeof cs === "object" ? cs : null;
}

function getCodingLines(data) {
  const cs = getCodingSupport(data);
  const lines = cs?.coding_summary?.lines;
  if (Array.isArray(lines) && lines.length > 0) return lines;

  // Fallback: synthesize from suggestions/cpt_codes (selected-only)
  const suggestions = Array.isArray(data?.suggestions) ? data.suggestions : [];
  if (suggestions.length > 0) {
    return suggestions.map((s, idx) => ({
      sequence: idx + 1,
      code: normalizeCptCode(s.code),
      description: s.description || null,
      units: 1,
      role: "primary",
      selection_status: "selected",
      selection_reason: s.rationale || null,
      note_spans: null,
    }));
  }

  const codes = Array.isArray(data?.cpt_codes) ? data.cpt_codes : [];
  if (codes.length > 0) {
    return codes.map((c, idx) => ({
      sequence: idx + 1,
      code: normalizeCptCode(c),
      description: null,
      units: 1,
      role: "primary",
      selection_status: "selected",
      selection_reason: null,
      note_spans: null,
    }));
  }

  return [];
}

function getCodingRationale(data) {
  const cs = getCodingSupport(data);
  const cr = cs?.coding_rationale;
  return cr && typeof cr === "object" ? cr : {};
}

function getEvidence(data) {
  return data?.evidence || getRegistry(data)?.evidence || {};
}

function getPerCodeBilling(data) {
  const lines = Array.isArray(data?.per_code_billing) ? data.per_code_billing : [];
  return lines;
}

function getBillingByCode(data) {
  const map = new Map();
  getPerCodeBilling(data).forEach((b) => {
    const code = normalizeCptCode(b?.cpt_code);
    if (code) map.set(code, b);
  });
  return map;
}

function deepClone(value) {
  if (typeof structuredClone === "function") return structuredClone(value);
  return JSON.parse(JSON.stringify(value));
}

function decodeJsonPointerSegment(seg) {
  // JSON Pointer (RFC 6901): "~1" => "/", "~0" => "~" (order matters).
  return String(seg || "").replace(/~1/g, "/").replace(/~0/g, "~");
}

function getJsonPointerParent(root, pointer, createMissing = false) {
  const p = String(pointer || "");
  if (!p || p === "/") return null;
  if (!p.startsWith("/")) return null;
  const parts = p.split("/").slice(1).map(decodeJsonPointerSegment);
  if (parts.length === 0) return null;

  let curr = root;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = parts[i];
    const nextKey = parts[i + 1];
    if (Array.isArray(curr)) {
      const idx = key === "-" ? curr.length : Number(key);
      if (!Number.isInteger(idx) || idx < 0) return null;
      if (curr[idx] === undefined) {
        if (!createMissing) return null;
        curr[idx] = nextKey === "-" || Number.isInteger(Number(nextKey)) ? [] : {};
      }
      curr = curr[idx];
      continue;
    }
    if (!curr || typeof curr !== "object") return null;
    if (curr[key] === undefined) {
      if (!createMissing) return null;
      curr[key] = nextKey === "-" || Number.isInteger(Number(nextKey)) ? [] : {};
    }
    curr = curr[key];
  }

  return { parent: curr, key: parts[parts.length - 1] };
}

function applyJsonPatchOps(root, ops) {
  const list = Array.isArray(ops) ? ops : [];
  list.forEach((op) => {
    if (!op || typeof op !== "object") return;
    const kind = String(op.op || "").toLowerCase();
    const path = String(op.path || "");
    if (!path) return;

    const loc = getJsonPointerParent(root, path, kind === "add");
    if (!loc) return;

    const { parent, key } = loc;
    if (Array.isArray(parent)) {
      const idx = key === "-" ? parent.length : Number(key);
      if (!Number.isInteger(idx) || idx < 0) return;
      if (kind === "remove") parent.splice(idx, 1);
      else if (kind === "add") parent.splice(idx, 0, op.value);
      else if (kind === "replace") parent[idx] = op.value;
      return;
    }

    if (!parent || typeof parent !== "object") return;
    if (kind === "remove") delete parent[key];
    else if (kind === "add" || kind === "replace") parent[key] = op.value;
  });
}

function setRegistryGridEdits(next) {
  const obj = next && typeof next === "object" ? next : null;
  const patch = Array.isArray(obj?.edited_patch) ? obj.edited_patch : [];
  const fields = Array.isArray(obj?.edited_fields) ? obj.edited_fields : [];
  registryGridEdits = patch.length || fields.length ? { edited_patch: patch, edited_fields: fields } : null;
  updateEditedPayload();
}

function parseList(value) {
  if (Array.isArray(value)) return value.filter((v) => String(v || "").trim() !== "");
  const text = String(value || "").trim();
  if (!text) return [];
  return text
    .split(",")
    .map((v) => v.trim())
    .filter(Boolean);
}

function parseNumber(value) {
  if (value === null || value === undefined || value === "") return null;
  const n = Number(value);
  return Number.isFinite(n) ? n : null;
}

function toYesNo(value) {
  if (value === true) return "Yes";
  if (value === false) return "No";
  const s = String(value || "").toLowerCase();
  if (s === "yes" || s === "true") return "Yes";
  if (s === "no" || s === "false") return "No";
  return "";
}

function parseYesNo(value) {
  const s = String(value || "").toLowerCase();
  if (s === "yes" || s === "true") return true;
  if (s === "no" || s === "false") return false;
  return null;
}

function getByPath(obj, path) {
  const root = obj && typeof obj === "object" ? obj : null;
  const rawPath = String(path || "").trim();
  if (!root || !rawPath) return undefined;

  const parts = [];
  rawPath.split(".").forEach((segment) => {
    const token = String(segment || "").trim();
    if (!token) return;
    const matches = token.matchAll(/([^[\]]+)|\[(\d+)\]/g);
    for (const match of matches) {
      if (match[1]) parts.push(match[1]);
      else if (match[2] !== undefined) parts.push(Number(match[2]));
    }
  });

  let curr = root;
  for (const part of parts) {
    if (curr === null || curr === undefined) return undefined;
    if (typeof part === "number") {
      if (!Array.isArray(curr)) return undefined;
      curr = curr[part];
    } else {
      curr = curr[part];
    }
  }
  return curr;
}

function ensurePath(obj, path) {
  const parts = path.split(".");
  let curr = obj;
  for (let i = 0; i < parts.length; i += 1) {
    const key = parts[i];
    if (!curr[key] || typeof curr[key] !== "object") curr[key] = {};
    curr = curr[key];
  }
  return curr;
}

function setByPath(obj, path, value) {
  const parts = path.split(".");
  let curr = obj;
  for (let i = 0; i < parts.length - 1; i += 1) {
    const key = parts[i];
    if (!curr[key] || typeof curr[key] !== "object") curr[key] = {};
    curr = curr[key];
  }
  curr[parts[parts.length - 1]] = value;
}

function resetEditedState() {
  flatTablesBase = null;
  flatTablesState = null;
  editedPayload = null;
  editedDirty = false;
  registryGridEdits = null;
  completenessEdits = null;
  completenessRawValueByPath = new Map();
  completenessSelectedIndexByPromptPath = new Map();
  fieldFeedbackStore = new Map();
  activeFieldFeedbackContext = null;
  if (editedResponseEl) editedResponseEl.textContent = "(no edits yet)";
  if (exportEditedBtn) exportEditedBtn.disabled = true;
  if (exportPatchBtn) exportPatchBtn.disabled = true;
  updateFeedbackButtons();
}

function collectRegistryCodeEvidence(data, code) {
  const registry = getRegistry(data);
  const items = registry?.billing?.cpt_codes;
  if (!Array.isArray(items)) return [];
  const match = items.find((c) => normalizeCptCode(c?.code) === code);
  const ev = match?.evidence;
  return Array.isArray(ev) ? ev : [];
}

function makeEvidenceChip(span) {
  const start = span?.start ?? span?.span?.[0];
  const end = span?.end ?? span?.span?.[1];
  const text = span?.text ?? span?.snippet ?? span?.quote ?? "";
  const btn = document.createElement("button");
  btn.className = "ev-chip";
  btn.type = "button";
  btn.title = Number.isFinite(start) && Number.isFinite(end) ? `Click to highlight ${start}-${end}` : "Evidence";
  btn.appendChild(document.createTextNode(String(text || "(evidence)")));
  if (Number.isFinite(start) && Number.isFinite(end) && end > start) {
    const range = document.createElement("span");
    range.className = "ev-range";
    range.textContent = `(${start}-${end})`;
    btn.appendChild(range);
    btn.addEventListener("click", () => highlightSpanInEditor(start, end));
  } else {
    btn.disabled = true;
  }
  return btn;
}

function makeEvidenceDetails(spans, summaryText = "Evidence") {
  const normalized = normalizeSpans(spans);
  if (normalized.length === 0) return null;

  const details = document.createElement("details");
  details.className = "inline-details";
  const summary = document.createElement("summary");
  summary.textContent = summaryText;
  details.appendChild(summary);

  const wrap = document.createElement("div");
  wrap.style.marginTop = "8px";
  normalized.slice(0, 6).forEach((sp) => wrap.appendChild(makeEvidenceChip(sp)));
  details.appendChild(wrap);
  return details;
}

/**
 * Main Orchestrator: Renders the clean clinical dashboard
 */
function renderDashboard(data) {
  renderStatusBannerHost(data);
  renderStatCards(data);

  renderBillingSelected(data);
  renderBillingSuppressed(data);
  renderCodingRationaleTable(data);
  renderRulesAppliedTable(data);
  renderFinancialSummary(data);
  renderAuditFlags(data);
  renderPipelineMetadata(data);
  renderCompletenessPrompts(data);

  // Right column (clinical) - procedures first, then context
  renderProceduresSummaryTable(data);
  renderClinicalContextTable(data);

  // Clear detail panels host, then render sub-panels
  const detailPanelsHost = document.getElementById("procedureDetailPanels");
  if (detailPanelsHost) clearEl(detailPanelsHost);
  renderDiagnosticFindings(data);
  renderBalDetails(data);
  renderLinearEbusSummary(data);
  renderEbusNodeEvents(data);

  renderEvidenceTraceability(data);

  renderDebugLogs(data);
}

function severityRank(severity) {
  const s = String(severity || "").toLowerCase();
  if (s === "required") return 0;
  return 1;
}

function buildCompletenessChecklistText(prompts) {
  const list = Array.isArray(prompts) ? prompts : [];
  if (list.length === 0) return "";

  const grouped = new Map();
  list.forEach((p) => {
    const group = String(p?.group || "Other").trim() || "Other";
    if (!grouped.has(group)) grouped.set(group, []);
    grouped.get(group).push(p);
  });

  const order = [];
  if (grouped.has("Global")) order.push(["Global", grouped.get("Global")]);
  grouped.forEach((items, group) => {
    if (group === "Global") return;
    order.push([group, items]);
  });

  const lines = [];
  order.forEach(([group, items]) => {
    lines.push(`${group}:`);
    const sorted = items
      .map((p, idx) => ({ p, idx }))
      .sort((a, b) => {
        const rank = severityRank(a.p?.severity) - severityRank(b.p?.severity);
        return rank !== 0 ? rank : a.idx - b.idx;
      })
      .map(({ p }) => p);
    sorted.forEach((p) => {
      const sev = String(p?.severity || "recommended").toLowerCase() === "required" ? "Required" : "Recommended";
      const label = String(p?.label || "").trim() || String(p?.path || "").trim() || "Missing field";
      const msg = String(p?.message || "").trim();
      const suffix = msg ? `: ${msg}` : "";
      lines.push(`- [${sev}] ${label}${suffix}`);
    });
    lines.push("");
  });

  return lines.join("\n").trim();
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

function encodeJsonPointerSegment(seg) {
  return String(seg || "").replace(/~/g, "~0").replace(/\//g, "~1");
}

function registryDottedPathToPointer(path) {
  const dotted = String(path || "").trim();
  if (!dotted) return "";
  const parts = dotted.split(".").filter(Boolean);
  const segments = [];

  parts.forEach((part) => {
    const token = String(part || "");
    // Match "foo[0]" -> ["foo", "0"]
    const match = token.match(/^([^\[]+?)(?:\[(\d+)\])?$/);
    if (!match) {
      segments.push(token);
      return;
    }
    const key = match[1];
    const idx = match[2];
    if (key) segments.push(key);
    if (idx !== undefined) segments.push(idx);
  });

  return `/registry/${segments.map(encodeJsonPointerSegment).join("/")}`;
}

function getWildcardItemsForPrompt(registry, promptPath) {
  const path = String(promptPath || "");
  if (path.startsWith("granular_data.navigation_targets[*].")) {
    const list = registry?.granular_data?.navigation_targets;
    return Array.isArray(list) ? list : [];
  }
  if (path.startsWith("granular_data.linear_ebus_stations_detail[*].")) {
    const list = registry?.granular_data?.linear_ebus_stations_detail;
    return Array.isArray(list) ? list : [];
  }
  return [];
}

function describeWildcardItemForPrompt(promptPath, item, idx) {
  const path = String(promptPath || "");
  if (path.startsWith("granular_data.navigation_targets[*].")) {
    const number = Number.isFinite(item?.target_number) ? item.target_number : idx + 1;
    const loc = String(item?.target_location_text || "").trim();
    return loc ? `Target ${number} — ${loc}` : `Target ${number}`;
  }
  if (path.startsWith("granular_data.linear_ebus_stations_detail[*].")) {
    const station = String(item?.station || "").trim();
    return station ? `Station ${station}` : `Station #${idx + 1}`;
  }
  return `Item #${idx + 1}`;
}

function resolvePromptPath(registry, promptPath) {
  const base = String(promptPath || "").trim();
  if (!base.includes("[*]")) return { effectivePath: base, hasWildcard: false, wildcardCount: 0 };

  const items = getWildcardItemsForPrompt(registry, base);
  const count = items.length;
  if (count <= 1) {
    completenessSelectedIndexByPromptPath.set(base, 0);
    return { effectivePath: base.replaceAll("[*]", "[0]"), hasWildcard: true, wildcardCount: count };
  }

  const existing = completenessSelectedIndexByPromptPath.get(base);
  const idx = Number.isInteger(existing) && existing >= 0 && existing < count ? existing : 0;
  completenessSelectedIndexByPromptPath.set(base, idx);
  return { effectivePath: base.replaceAll("[*]", `[${idx}]`), hasWildcard: true, wildcardCount: count };
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
    "complications.bleeding.bleeding_grade_nashville": { type: "integer", placeholder: "0–4" },
    "complications.pneumothorax.intervention": {
      type: "multiselect",
      options: ["Observation", "Aspiration", "Pigtail catheter", "Chest tube", "Heimlich valve", "Surgery"],
    },
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
    "granular_data.linear_ebus_stations_detail[*].needle_gauge": {
      type: "enum",
      options: [19, 21, 22, 25],
    },
    "granular_data.linear_ebus_stations_detail[*].number_of_passes": { type: "integer", placeholder: "passes" },
    "granular_data.linear_ebus_stations_detail[*].short_axis_mm": { type: "number", placeholder: "mm" },
    "granular_data.linear_ebus_stations_detail[*].lymphocytes_present": { type: "boolean" },

    // Pleural / chest ultrasound
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

    // Pleural fibrinolytic therapy
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

function coerceCompletenessValue(spec, rawValue) {
  if (!spec || !spec.type) return null;

  if (spec.type === "multiselect") {
    if (Array.isArray(rawValue)) {
      const clean = rawValue.map((v) => String(v || "").trim()).filter((v) => v !== "");
      return clean.length ? clean : null;
    }
    return null;
  }

  const raw = String(rawValue ?? "").trim();
  if (!raw) return null;

  if (spec.type === "boolean") {
    const lower = raw.toLowerCase();
    if (lower === "true" || lower === "yes") return true;
    if (lower === "false" || lower === "no") return false;
    return null;
  }

  if (spec.type === "integer") {
    const n = Number.parseInt(raw, 10);
    return Number.isFinite(n) ? n : null;
  }

  if (spec.type === "number") {
    const n = Number.parseFloat(raw);
    return Number.isFinite(n) ? n : null;
  }

  // enum/string/ecog are handled elsewhere or can return raw string
  return raw;
}

function escapeRegExp(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getStoredWildcardEffectivePathsForPrompt(promptPath) {
  const base = String(promptPath || "").trim();
  if (!base.includes("[*]")) return [];
  const parts = base.split("[*]");
  const pattern = `^${parts.map(escapeRegExp).join("\\[(\\d+)\\]")}$`;
  let re = null;
  try {
    re = new RegExp(pattern);
  } catch (e) {
    console.warn("Invalid completeness wildcard path pattern (ignored):", base, e);
    return [];
  }
  const out = [];
  for (const key of completenessRawValueByPath.keys()) {
    if (re.test(key)) out.push(key);
  }
  out.sort();
  return out;
}

function buildCompletenessEdits(registry, prompts) {
  const list = Array.isArray(prompts) ? prompts : [];
  const ops = [];
  const fields = [];

  list.forEach((prompt) => {
    const promptPath = String(prompt?.target_path || prompt?.path || "").trim();
    if (!promptPath) return;

    const resolved = resolvePromptPath(registry, promptPath);
    if (resolved.hasWildcard && resolved.wildcardCount === 0) return;
    const spec = getCompletenessInputSpec(promptPath);

    if (spec.type === "ecog") {
      const effectivePath = resolved.effectivePath;
      const rawValue = completenessRawValueByPath.get(effectivePath);
      const raw = String(rawValue || "").trim();
      if (!raw) return;
      const match = raw.match(/^\s*([0-4])\s*$/);
      const isSingle = Boolean(match);
      const targetPath = isSingle ? "clinical_context.ecog_score" : "clinical_context.ecog_text";
      const value = isSingle ? Number.parseInt(match[1], 10) : raw;
      const ptr = registryDottedPathToPointer(targetPath);
      if (!ptr) return;
      ops.push({ op: "add", path: ptr, value });
      fields.push(targetPath);
      return;
    }

    const effectivePaths = resolved.hasWildcard
      ? getStoredWildcardEffectivePathsForPrompt(promptPath)
      : [resolved.effectivePath];

    effectivePaths.forEach((effectivePath) => {
      const rawValue = completenessRawValueByPath.get(effectivePath);
      if (
        rawValue === undefined ||
        rawValue === null ||
        (typeof rawValue === "string" && rawValue.trim() === "") ||
        (Array.isArray(rawValue) && rawValue.length === 0)
      ) {
        return;
      }

      const coerced = coerceCompletenessValue(spec, rawValue);
      if (coerced === null) return;

      const ptr = registryDottedPathToPointer(effectivePath);
      if (!ptr) return;
      ops.push({ op: "add", path: ptr, value: coerced });
      fields.push(effectivePath);
    });
  });

  return { edited_patch: ops, edited_fields: fields };
}

function recomputeCompletenessEdits(registry, prompts) {
  const next = buildCompletenessEdits(registry, prompts);
  completenessEdits = next.edited_patch.length || next.edited_fields.length ? next : null;
  updateEditedPayload();
}

function getFlatTableStateById(tableId) {
  const tables = Array.isArray(flatTablesState) ? flatTablesState : [];
  return tables.find((t) => t?.id === tableId) || null;
}

function findFlatFieldValueRowByRegistryPath(tables, registryPath) {
  const list = Array.isArray(tables) ? tables : [];
  const full = String(registryPath || "").trim();
  if (!full) return null;
  for (const table of list) {
    const rows = Array.isArray(table?.rows) ? table.rows : [];
    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      const meta = row?.__meta || {};
      if (meta.path === full) return { table, row, rowIndex: i };
    }
  }
  return null;
}

function formatCompletenessValueForFlatTable(meta, value) {
  const valueType = String(meta?.valueType || "text").toLowerCase();
  if (value === null || value === undefined) return "";
  if (valueType === "boolean") return toYesNo(value);
  if (valueType === "number") {
    if (typeof value === "number") return Number.isFinite(value) ? String(value) : "";
    const raw = String(value || "").trim();
    return raw;
  }
  if (valueType === "list") {
    if (Array.isArray(value)) {
      return value
        .map((v) => String(v || "").trim())
        .filter((v) => v !== "")
        .join(", ");
    }
    return String(value || "").trim();
  }
  return String(value ?? "");
}

function applyCompletenessValueToFlatTables(targetEffectivePath, coercedValue) {
  const path = String(targetEffectivePath || "").trim();
  if (!path) return false;
  if (!Array.isArray(flatTablesState) || flatTablesState.length === 0) return false;

  const restoreBase = coercedValue === null || coercedValue === undefined;

  // Granular arrays (direct cell addressing by index).
  const navMatch = path.match(/^granular_data\.navigation_targets\[(\d+)\]\.([^.]+)$/);
  if (navMatch) {
    const rowIndex = Number.parseInt(navMatch[1], 10);
    const key = navMatch[2];
    if (!Number.isFinite(rowIndex) || rowIndex < 0) return false;
    const table = getFlatTableStateById("navigation_targets");
    if (!table || !Array.isArray(table.rows) || rowIndex >= table.rows.length) return false;
    const row = table.rows[rowIndex];

    const baseTable = getFlatTableBaseById("navigation_targets");
    const baseRow = Array.isArray(baseTable?.rows) ? baseTable.rows[rowIndex] : null;

    if (restoreBase) {
      row[key] = baseRow ? baseRow[key] ?? "" : "";
      return true;
    }

    if (typeof coercedValue === "boolean") row[key] = toYesNo(coercedValue);
    else if (typeof coercedValue === "number") row[key] = Number.isFinite(coercedValue) ? String(coercedValue) : "";
    else if (Array.isArray(coercedValue))
      row[key] = coercedValue.map((v) => String(v || "").trim()).filter((v) => v !== "").join(", ");
    else row[key] = String(coercedValue ?? "");
    return true;
  }

  const ebusMatch = path.match(/^granular_data\.linear_ebus_stations_detail\[(\d+)\]\.([^.]+)$/);
  if (ebusMatch) {
    const rowIndex = Number.parseInt(ebusMatch[1], 10);
    const key = ebusMatch[2];
    if (!Number.isFinite(rowIndex) || rowIndex < 0) return false;
    const table = getFlatTableStateById("linear_ebus_stations_detail");
    if (!table || !Array.isArray(table.rows) || rowIndex >= table.rows.length) return false;
    const row = table.rows[rowIndex];

    const baseTable = getFlatTableBaseById("linear_ebus_stations_detail");
    const baseRow = Array.isArray(baseTable?.rows) ? baseTable.rows[rowIndex] : null;

    if (restoreBase) {
      row[key] = baseRow ? baseRow[key] ?? "" : "";
      return true;
    }

    if (typeof coercedValue === "boolean") row[key] = toYesNo(coercedValue);
    else if (typeof coercedValue === "number") row[key] = Number.isFinite(coercedValue) ? String(coercedValue) : "";
    else if (Array.isArray(coercedValue))
      row[key] = coercedValue.map((v) => String(v || "").trim()).filter((v) => v !== "").join(", ");
    else row[key] = String(coercedValue ?? "");
    return true;
  }

  // Field/value rows (meta.path points at registry.*).
  const registryPath = `registry.${path}`;
  const stateLoc = findFlatFieldValueRowByRegistryPath(flatTablesState, registryPath);
  if (!stateLoc) return false;
  const meta = stateLoc.row?.__meta || {};

  if (restoreBase) {
    const baseLoc = findFlatFieldValueRowByRegistryPath(flatTablesBase, registryPath);
    stateLoc.row.value = baseLoc?.row?.value ?? "";
    return true;
  }

  stateLoc.row.value = formatCompletenessValueForFlatTable(meta, coercedValue);
  return true;
}

function renderCompletenessPrompts(data) {
  if (!completenessPromptsCardEl || !completenessPromptsBodyEl) return;
  clearEl(completenessPromptsBodyEl);

  const prompts = Array.isArray(data?.missing_field_prompts) ? data.missing_field_prompts : [];
  lastCompletenessPrompts = prompts;
  const registry = getRegistry(data) || {};

  if (!prompts.length || data?.error) {
    completenessPromptsCardEl.classList.add("hidden");
    if (completenessCopyBtn) completenessCopyBtn.disabled = true;
    return;
  }

  completenessPromptsCardEl.classList.remove("hidden");
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
  hint.textContent =
    "Edits here update the Flattened Tables below and are recorded in “Edited JSON (Training)”. For evidence-backed extraction, add to the note and re-run.";
  summary.appendChild(hint);

  completenessPromptsBodyEl.appendChild(summary);

  const commitCompletenessValue = (promptPath, effectivePath) => {
    const basePromptPath = String(promptPath || "").trim();
    const baseEffectivePath = String(effectivePath || "").trim();
    if (!basePromptPath || !baseEffectivePath) return;

    const spec = getCompletenessInputSpec(basePromptPath);
    const rawValue = completenessRawValueByPath.get(baseEffectivePath);

    if (spec.type === "ecog") {
      const raw = String(rawValue ?? "").trim();
      if (!raw) {
        applyCompletenessValueToFlatTables("clinical_context.ecog_score", null);
        applyCompletenessValueToFlatTables("clinical_context.ecog_text", null);
      } else {
        const match = raw.match(/^\s*([0-4])\s*$/);
        if (match) {
          applyCompletenessValueToFlatTables("clinical_context.ecog_score", Number.parseInt(match[1], 10));
          applyCompletenessValueToFlatTables("clinical_context.ecog_text", null);
        } else {
          applyCompletenessValueToFlatTables("clinical_context.ecog_text", raw);
          applyCompletenessValueToFlatTables("clinical_context.ecog_score", null);
        }
      }
    } else {
      const coerced = coerceCompletenessValue(spec, rawValue);
      const ok = applyCompletenessValueToFlatTables(baseEffectivePath, coerced);
      if (!ok) console.warn("Completeness prompt update target not found in flattened tables:", baseEffectivePath);
    }

    renderFlatTablesFromState();
    recomputeCompletenessEdits(registry, prompts);
  };

  const grouped = new Map();
  prompts.forEach((p) => {
    const group = String(p?.group || "Other").trim() || "Other";
    if (!grouped.has(group)) grouped.set(group, []);
    grouped.get(group).push(p);
  });

  const groupEntries = [];
  if (grouped.has("Global")) groupEntries.push(["Global", grouped.get("Global")]);
  grouped.forEach((items, group) => {
    if (group === "Global") return;
    groupEntries.push([group, items]);
  });

  groupEntries.forEach(([group, items]) => {
    const groupWrap = document.createElement("div");
    groupWrap.className = "completeness-group";

    const title = document.createElement("div");
    title.className = "completeness-group-title";
    title.textContent = group;
    groupWrap.appendChild(title);

    const list = document.createElement("ul");
    list.className = "completeness-list";

    const sorted = items
      .map((p, idx) => ({ p, idx }))
      .sort((a, b) => {
        const rank = severityRank(a.p?.severity) - severityRank(b.p?.severity);
        return rank !== 0 ? rank : a.idx - b.idx;
      })
      .map(({ p }) => p);

    sorted.forEach((p) => {
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
      const targetPath = String(p?.target_path || "").trim();
      const displayPath =
        path && targetPath && path !== targetPath ? `${path} → ${targetPath}` : (targetPath || path);
      if (displayPath) {
        const meta = document.createElement("div");
        meta.className = "completeness-item-path";
        meta.textContent = displayPath;
        main.appendChild(meta);
      }

      const controls = document.createElement("div");
      controls.className = "completeness-item-controls";

      const promptPath = String(p?.target_path || p?.path || "").trim();
      const resolved = resolvePromptPath(registry, promptPath);

      if (resolved.hasWildcard) {
        const itemsForSelect = getWildcardItemsForPrompt(registry, promptPath);
        if (itemsForSelect.length > 1) {
          const select = document.createElement("select");
          select.className = "flat-select";
          const current = completenessSelectedIndexByPromptPath.get(promptPath) || 0;
          select.value = String(current);
          itemsForSelect.forEach((item, idx) => {
            const opt = document.createElement("option");
            opt.value = String(idx);
            opt.textContent = describeWildcardItemForPrompt(promptPath, item, idx);
            select.appendChild(opt);
          });
          select.addEventListener("change", () => {
            const nextIdx = Number.parseInt(String(select.value || "0"), 10);
            const safeIdx = Number.isFinite(nextIdx) && nextIdx >= 0 ? nextIdx : 0;
            completenessSelectedIndexByPromptPath.set(promptPath, safeIdx);
            renderCompletenessPrompts(lastServerResponse || data);
            recomputeCompletenessEdits(registry, prompts);
          });
          controls.appendChild(select);
        } else if (itemsForSelect.length === 0) {
          const msg = document.createElement("div");
          msg.className = "subtle";
          msg.textContent = "No list items detected to attach this value. Add detail to note and re-run extraction.";
          controls.appendChild(msg);
        }
      }

      const spec = getCompletenessInputSpec(promptPath);
      const effectivePath = resolved.effectivePath;
      const stored = completenessRawValueByPath.get(effectivePath);

      let input = null;
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
          completenessRawValueByPath.set(effectivePath, String(select.value || ""));
          commitCompletenessValue(promptPath, effectivePath);
        });
        input = select;
      } else if (spec.type === "boolean") {
        const select = document.createElement("select");
        select.className = "flat-select";
        [
          ["", "—"],
          ["true", "Yes"],
          ["false", "No"],
        ].forEach(([value, label]) => {
          const option = document.createElement("option");
          option.value = value;
          option.textContent = label;
          select.appendChild(option);
        });
        select.value = stored === undefined || stored === null ? "" : String(stored);
        select.addEventListener("change", () => {
          completenessRawValueByPath.set(effectivePath, String(select.value || ""));
          commitCompletenessValue(promptPath, effectivePath);
        });
        input = select;
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
          completenessRawValueByPath.set(effectivePath, vals);
          commitCompletenessValue(promptPath, effectivePath);
        });
        input = select;
      } else {
        const el = document.createElement("input");
        el.className = "flat-input";
        el.type = spec.type === "integer" || spec.type === "number" ? "number" : "text";
        if (spec.type === "integer") el.step = "1";
        if (spec.type === "number") el.step = "any";
        el.placeholder = spec.placeholder || "Enter value";
        el.value = stored === undefined || stored === null ? "" : String(stored);
        el.addEventListener("input", () => {
          completenessRawValueByPath.set(effectivePath, String(el.value || ""));
        });
        el.addEventListener("blur", () => commitCompletenessValue(promptPath, effectivePath));
        input = el;
      }

      if (input) {
        const isWildcardWithoutItems = resolved.hasWildcard && resolved.wildcardCount === 0;
        if (isWildcardWithoutItems) input.disabled = true;
        controls.appendChild(input);
      }

      li.appendChild(main);
      li.appendChild(controls);
      list.appendChild(li);
    });

    groupWrap.appendChild(list);
    completenessPromptsBodyEl.appendChild(groupWrap);
  });

  recomputeCompletenessEdits(registry, prompts);
}

/**
 * 1. Renders the Executive Summary (Stat Cards)
 */
function renderStatCards(data) {
  const container = document.getElementById("statCards");
  if (!container) return;

  // Determine review status
  let statusText = "Ready";
  let statusClass = "";
  if (data.needs_manual_review || (data.audit_warnings && data.audit_warnings.length > 0)) {
    statusText = "⚠️ Review Required";
    statusClass = "warning";
  }

  // Format currency and RVU
  const payment = data.estimated_payment
    ? new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(data.estimated_payment)
    : "$0.00";
  const rvu = data.total_work_rvu ? data.total_work_rvu.toFixed(2) : "0.00";

  container.innerHTML = `
    <div class="stat-card">
      <span class="stat-label">Review Status</span>
      <div class="stat-value ${statusClass}">${statusText}</div>
    </div>
    <div class="stat-card">
      <span class="stat-label">Total wRVU</span>
      <div class="stat-value">${rvu}</div>
    </div>
    <div class="stat-card">
      <span class="stat-label">Est. Payment</span>
      <div class="stat-value currency">${payment}</div>
    </div>
    <div class="stat-card">
      <span class="stat-label">CPT Count</span>
      <div class="stat-value">${(data.per_code_billing || []).length}</div>
    </div>
  `;
}

function renderStatusBannerHost(data) {
  const host = document.getElementById("statusBannerHost");
  if (!host) return;

  clearEl(host);

  const banner = document.createElement("div");

  const warnings = Array.isArray(data?.audit_warnings) ? data.audit_warnings : [];
  const hasError = Boolean(data?.error);
  const needsReview = data?.review_status === "pending_phi_review" || data?.needs_manual_review;

  if (hasError) {
    banner.className = "status-banner error";
    banner.textContent = `Error: ${String(data.error)}`;
  } else if (needsReview) {
    banner.className = "status-banner error";
    banner.textContent = "⚠️ Manual review required";
  } else if (warnings.length > 0) {
    banner.className = "status-banner warning";
    banner.textContent = `⚠️ ${warnings.length} warning(s) – review recommended`;
  } else {
    banner.className = "status-banner success";
    banner.textContent = "✓ Extraction complete";
  }

  host.appendChild(banner);
}

function renderBillingSelected(data) {
  const tbody = document.getElementById("billingSelectedBody");
  if (!tbody) return;
  clearEl(tbody);

  const billingByCode = getBillingByCode(data);
  const codingLines = getCodingLines(data);
  const selected = codingLines.filter(
    (ln) => String(ln?.selection_status || "selected").toLowerCase() === "selected"
  );

  if (selected.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "dash-empty";
    td.textContent = "No selected codes.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  selected.forEach((ln) => {
    const code = normalizeCptCode(ln?.code);
    const billing = billingByCode.get(code);
    const desc = ln?.description || billing?.description || "—";
    const units = Number.isFinite(ln?.units) ? ln.units : Number.isFinite(billing?.units) ? billing.units : 1;
    const roleRaw = String(ln?.role || (ln?.is_add_on ? "add_on" : "primary") || "primary");
    const role = roleRaw.toLowerCase() === "add_on" ? "add_on" : "primary";
    const reason = ln?.selection_reason || ln?.rationale || "";

    const tr = document.createElement("tr");

    const tdCode = document.createElement("td");
    const codeSpan = document.createElement("span");
    codeSpan.className = "code-cell";
    codeSpan.textContent = code || "—";
    tdCode.appendChild(codeSpan);

    const tdDesc = document.createElement("td");
    const descDiv = document.createElement("div");
    descDiv.style.fontWeight = "600";
    descDiv.textContent = String(desc || "—");
    tdDesc.appendChild(descDiv);

    if (reason) {
      const reasonDiv = document.createElement("div");
      reasonDiv.className = "qa-line";
      reasonDiv.textContent = `Rationale: ${String(reason)}`;
      tdDesc.appendChild(reasonDiv);
    }

    const combinedEvidence = [];
    if (Array.isArray(ln?.note_spans)) combinedEvidence.push(...ln.note_spans);
    combinedEvidence.push(...collectRegistryCodeEvidence(data, code));
    const evDetails = makeEvidenceDetails(combinedEvidence, "Evidence");
    if (evDetails) tdDesc.appendChild(evDetails);

    const tdUnits = document.createElement("td");
    tdUnits.textContent = String(units ?? "—");

    const tdRole = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `status-badge ${role === "add_on" ? "role-addon" : "role-primary"}`;
    badge.textContent = role === "add_on" ? "Add On" : "Primary";
    tdRole.appendChild(badge);

    tr.appendChild(tdCode);
    tr.appendChild(tdDesc);
    tr.appendChild(tdUnits);
    tr.appendChild(tdRole);
    tbody.appendChild(tr);
  });
}

function renderBillingSuppressed(data) {
  const tbody = document.getElementById("billingSuppressedBody");
  if (!tbody) return;
  clearEl(tbody);

  const codingLines = getCodingLines(data);
  const rules = Array.isArray(getCodingRationale(data)?.rules_applied)
    ? getCodingRationale(data).rules_applied
    : [];
  const warnings = Array.isArray(data?.audit_warnings) ? data.audit_warnings : [];

  const entries = new Map(); // code -> {status, reason}

  // 1) Prefer explicit coding_support dropped lines (stable order)
  codingLines.forEach((ln) => {
    const status = String(ln?.selection_status || "").toLowerCase();
    if (status === "selected") return;
    const code = normalizeCptCode(ln?.code);
    if (!code) return;
    const reason = String(ln?.selection_reason || "").trim() || "Dropped by rule";
    const inferred = /^suppressed\b/i.test(reason) ? "Suppressed" : "Dropped";
    entries.set(code, { status: inferred, reason });
  });

  // 2) Add any rule-driven dropped codes not already present
  rules.forEach((r) => {
    const affected = Array.isArray(r?.codes_affected) ? r.codes_affected : [];
    const outcome = String(r?.outcome || "").toLowerCase();
    if (outcome !== "dropped" && outcome !== "suppressed") return;
    affected.forEach((c) => {
      const code = normalizeCptCode(c);
      if (!code || entries.has(code)) return;
      entries.set(code, { status: "Dropped", reason: String(r?.details || "Dropped by rule") });
    });
  });

  // 3) Add suppressed codes hinted by warnings (e.g., "Suppressed 31645: ...")
  warnings.forEach((w) => {
    const text = String(w || "");
    const match = text.match(/\bSuppressed\s+(\d{5})\b/i);
    if (!match) return;
    const code = match[1];
    if (entries.has(code)) return;
    entries.set(code, { status: "Suppressed", reason: text.replace(/\s+/g, " ").trim() });
  });

  if (entries.size === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 3;
    td.className = "dash-empty";
    td.textContent = "No codes dropped/suppressed.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  for (const [code, info] of entries.entries()) {
    const tr = document.createElement("tr");
    const tdCode = document.createElement("td");
    const codeSpan = document.createElement("span");
    codeSpan.className = "code-cell";
    codeSpan.textContent = code;
    tdCode.appendChild(codeSpan);

    const tdStatus = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = `status-badge ${info.status === "Suppressed" ? "status-suppressed" : "status-dropped"}`;
    badge.textContent = info.status;
    tdStatus.appendChild(badge);

    const tdReason = document.createElement("td");
    tdReason.textContent = String(info.reason || "—");

    tr.appendChild(tdCode);
    tr.appendChild(tdStatus);
    tr.appendChild(tdReason);
    tbody.appendChild(tr);
  }
}

function renderCodingRationaleTable(data) {
  const tbody = document.getElementById("codingRationaleBody");
  if (!tbody) return;
  clearEl(tbody);

  const codingLines = getCodingLines(data);
  const perCode = Array.isArray(getCodingRationale(data)?.per_code) ? getCodingRationale(data).per_code : [];
  const perCodeByCode = new Map();
  perCode.forEach((pc) => {
    const code = normalizeCptCode(pc?.code);
    if (code) perCodeByCode.set(code, pc);
  });

  const codesInOrder = [];
  codingLines.forEach((ln) => {
    const code = normalizeCptCode(ln?.code);
    if (code && !codesInOrder.includes(code)) codesInOrder.push(code);
  });
  // Include any per_code entries not present in coding_lines
  Array.from(perCodeByCode.keys())
    .sort((a, b) => a.localeCompare(b))
    .forEach((code) => {
      if (!codesInOrder.includes(code)) codesInOrder.push(code);
    });

  if (codesInOrder.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 3;
    td.className = "dash-empty";
    td.textContent = "No coding rationale returned.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  codesInOrder.forEach((code) => {
    const pc = perCodeByCode.get(code);
    const summary = pc?.summary || codingLines.find((ln) => normalizeCptCode(ln?.code) === code)?.selection_reason || "—";

    const tr = document.createElement("tr");

    const tdCode = document.createElement("td");
    const codeSpan = document.createElement("span");
    codeSpan.className = "code-cell";
    codeSpan.textContent = code;
    tdCode.appendChild(codeSpan);

    const tdLogic = document.createElement("td");
    tdLogic.textContent = String(summary || "—");

    const tdEvidence = document.createElement("td");

    const docEv = Array.isArray(pc?.documentation_evidence) ? pc.documentation_evidence : [];
    const spans = docEv
      .map((e) => ({
        text: e?.snippet || e?.text || "",
        start: e?.span?.start,
        end: e?.span?.end,
      }))
      .filter((e) => Number.isFinite(e.start) && Number.isFinite(e.end) && e.end > e.start);

    const evDetails = makeEvidenceDetails(spans, "Evidence");
    if (evDetails) tdEvidence.appendChild(evDetails);
    else tdEvidence.appendChild(document.createTextNode("—"));

    const qaFlags = Array.isArray(pc?.qa_flags) ? pc.qa_flags : [];
    if (qaFlags.length > 0) {
      const qaWrap = document.createElement("div");
      qaWrap.style.marginTop = "8px";
      qaFlags.forEach((q) => {
        const line = document.createElement("div");
        line.className = "qa-line";
        const sev = String(q?.severity || "info").toUpperCase();
        const msg = String(q?.message || "");
        line.textContent = `${sev}: ${msg}`;
        qaWrap.appendChild(line);
      });
      tdEvidence.appendChild(qaWrap);
    }

    tr.appendChild(tdCode);
    tr.appendChild(tdLogic);
    tr.appendChild(tdEvidence);
    tbody.appendChild(tr);
  });
}

function renderRulesAppliedTable(data) {
  const tbody = document.getElementById("rulesAppliedBody");
  if (!tbody) return;
  clearEl(tbody);

  const rules = Array.isArray(getCodingRationale(data)?.rules_applied)
    ? getCodingRationale(data).rules_applied
    : [];

  if (rules.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 4;
    td.className = "dash-empty";
    td.textContent = "No rules applied returned.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  rules.forEach((r) => {
    const tr = document.createElement("tr");
    const tdType = document.createElement("td");
    const type = String(r?.rule_type || "—");
    const id = r?.rule_id ? String(r.rule_id) : "";
    tdType.textContent = id ? `${type} (${id})` : type;

    const tdCodes = document.createElement("td");
    const codes = Array.isArray(r?.codes_affected) ? r.codes_affected.map(normalizeCptCode).filter(Boolean) : [];
    tdCodes.textContent = codes.length ? codes.join(", ") : "—";

    const tdOutcome = document.createElement("td");
    tdOutcome.textContent = fmtMaybe(r?.outcome);

    const tdDetails = document.createElement("td");
    tdDetails.textContent = fmtMaybe(r?.details);

    tr.appendChild(tdType);
    tr.appendChild(tdCodes);
    tr.appendChild(tdOutcome);
    tr.appendChild(tdDetails);
    tbody.appendChild(tr);
  });
}

function renderFinancialSummary(data) {
  const totalRvuEl = document.getElementById("financialTotalRVU");
  const totalPayEl = document.getElementById("financialTotalPayment");
  const tbody = document.getElementById("financialSummaryBody");
  if (!tbody) return;

  if (totalRvuEl) totalRvuEl.textContent = Number.isFinite(data?.total_work_rvu) ? data.total_work_rvu.toFixed(2) : "—";
  if (totalPayEl) totalPayEl.textContent = Number.isFinite(data?.estimated_payment) ? formatCurrency(data.estimated_payment) : "—";

  clearEl(tbody);

  const billing = getPerCodeBilling(data);
  const selectedUnits = new Map();
  getCodingLines(data)
    .filter((ln) => String(ln?.selection_status || "selected").toLowerCase() === "selected")
    .forEach((ln) => {
      const code = normalizeCptCode(ln?.code);
      if (!code) return;
      const units = Number.isFinite(ln?.units) ? ln.units : 1;
      selectedUnits.set(code, units);
    });

  if (!Array.isArray(billing) || billing.length === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 6;
    td.className = "dash-empty";
    td.textContent = "No financial breakdown available.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  billing.forEach((b) => {
    const code = normalizeCptCode(b?.cpt_code);
    const units = Number.isFinite(b?.units) ? b.units : 1;
    const selUnits = selectedUnits.get(code);
    const mismatch =
      Number.isFinite(selUnits) && Number.isFinite(units) && selUnits !== units;

    const tr = document.createElement("tr");

    const tdCode = document.createElement("td");
    const codeSpan = document.createElement("span");
    codeSpan.className = "code-cell";
    codeSpan.textContent = code || "—";
    tdCode.appendChild(codeSpan);

    const tdUnits = document.createElement("td");
    tdUnits.textContent = String(units);

    const tdWork = document.createElement("td");
    tdWork.textContent = formatNumber(b?.work_rvu);

    const tdFac = document.createElement("td");
    tdFac.textContent = formatNumber(b?.total_facility_rvu);

    const tdPay = document.createElement("td");
    tdPay.textContent = formatCurrency(b?.facility_payment);

    const tdNotes = document.createElement("td");
    if (mismatch) {
      tdNotes.textContent = `⚠ Units mismatch (selected ${selUnits}, billed ${units})`;
    } else {
      tdNotes.textContent = "—";
    }

    tr.appendChild(tdCode);
    tr.appendChild(tdUnits);
    tr.appendChild(tdWork);
    tr.appendChild(tdFac);
    tr.appendChild(tdPay);
    tr.appendChild(tdNotes);
    tbody.appendChild(tr);
  });
}

function renderAuditFlags(data) {
  const tbody = document.getElementById("auditFlagsBody");
  if (!tbody) return;
  clearEl(tbody);

  const grouped = groupWarnings(data?.audit_warnings || []);
  const validationErrors = Array.isArray(data?.validation_errors) ? data.validation_errors : [];
  if (validationErrors.length > 0) grouped.set("Validation errors", validationErrors.map((e) => String(e || "—")));

  if (grouped.size === 0) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 2;
    td.className = "dash-empty";
    td.textContent = "No audit or validation flags.";
    tr.appendChild(td);
    tbody.appendChild(tr);
    return;
  }

  for (const [category, messages] of grouped.entries()) {
    const tr = document.createElement("tr");
    const tdCat = document.createElement("td");
    tdCat.textContent = category;

    const tdMsg = document.createElement("td");
    const list = document.createElement("div");
    (Array.isArray(messages) ? messages : []).forEach((m) => {
      const line = document.createElement("div");
      line.className = "qa-line";
      line.textContent = `• ${String(m || "—")}`;
      list.appendChild(line);
    });
    tdMsg.appendChild(list);

    tr.appendChild(tdCat);
    tr.appendChild(tdMsg);
    tbody.appendChild(tr);
  }
}

function renderPipelineMetadata(data) {
  const tbody = document.getElementById("pipelineMetadataBody");
  if (!tbody) return;
  clearEl(tbody);

  const rows = [
    ["Needs manual review", data?.needs_manual_review ? "Yes" : "No"],
    ["Review status", fmtMaybe(data?.review_status)],
    ["Coder difficulty", fmtMaybe(data?.coder_difficulty)],
    ["Pipeline mode", fmtMaybe(data?.pipeline_mode)],
    ["KB version", fmtMaybe(data?.kb_version)],
    ["Policy version", fmtMaybe(data?.policy_version)],
    [
      "Processing time",
      Number.isFinite(data?.processing_time_ms)
        ? `${Math.round(data.processing_time_ms).toLocaleString()} ms`
        : "—",
    ],
  ];

  rows.forEach(([k, v]) => {
    const tr = document.createElement("tr");
    const tdK = document.createElement("td");
    tdK.textContent = k;
    const tdV = document.createElement("td");
    tdV.textContent = fmtMaybe(v);
    tr.appendChild(tdK);
    tr.appendChild(tdV);
    tbody.appendChild(tr);
  });
}

/**
 * Build a collapsible detail panel (<details>) for procedure sub-sections.
 */
function createDetailPanel(title, badgeClass, performed, contentBuilder) {
  const panel = document.createElement("details");
  panel.className = "proc-detail-panel";
  if (!performed) panel.classList.add("panel-notperformed");
  if (performed) panel.setAttribute("open", "");

  const summary = document.createElement("summary");
  summary.className = "proc-detail-panel-header";

  if (badgeClass) {
    const badge = document.createElement("span");
    badge.className = `panel-type-badge ${badgeClass}`;
    badge.textContent = badgeClass.replace("badge-", "").toUpperCase();
    summary.appendChild(badge);
  }

  const titleSpan = document.createElement("span");
  titleSpan.textContent = title;
  summary.appendChild(titleSpan);

  panel.appendChild(summary);

  const body = document.createElement("div");
  body.className = "proc-detail-panel-body";
  contentBuilder(body);
  panel.appendChild(body);

  return panel;
}

/**
 * Return structured key-detail pairs for a procedure instead of a flat string.
 * Falls back to parsing the existing summarizeProcedure() output.
 */
function summarizeProcedureStructured(procKey, procObj) {
  const performed = isPerformedProcedure(procObj);
  const p = procObj && typeof procObj === "object" ? procObj : {};

  if (procKey === "diagnostic_bronchoscopy") {
    const abn = Array.isArray(p.airway_abnormalities) ? p.airway_abnormalities.filter(Boolean) : [];
    const findings = String(p.inspection_findings || "").trim();
    const parts = [];
    if (abn.length > 0) parts.push({ label: "Abnormalities", value: abn.join(", ") });
    if (findings) parts.push({ label: "Findings", value: findings });
    return parts;
  }

  if (procKey === "bal") {
    const parts = [];
    const loc = cleanLocationForDisplay(p.location);
    if (loc) parts.push({ label: "Location", value: loc });
    if (Number.isFinite(p.volume_instilled_ml)) parts.push({ label: "Instilled", value: `${p.volume_instilled_ml} mL` });
    if (Number.isFinite(p.volume_recovered_ml)) parts.push({ label: "Recovered", value: `${p.volume_recovered_ml} mL` });
    return parts;
  }

  if (procKey === "chest_ultrasound") {
    const parts = [];
    if (p.hemithorax) parts.push({ label: "Side", value: String(p.hemithorax).trim() });
    if (p.effusion_volume) parts.push({ label: "Effusion", value: String(p.effusion_volume).trim() });
    if (p.effusion_echogenicity) parts.push({ label: "Echo", value: String(p.effusion_echogenicity).trim() });
    if (p.effusion_loculations) parts.push({ label: "Loculations", value: String(p.effusion_loculations).trim() });
    return parts;
  }

  if (procKey === "rigid_bronchoscopy") {
    const parts = [];
    if (Number.isFinite(p.rigid_scope_size)) parts.push({ label: "Size", value: `${p.rigid_scope_size} mm` });
    return parts;
  }

  if (procKey === "chest_tube") {
    const parts = [];
    if (p.action) parts.push({ label: "Action", value: String(p.action).trim() });
    if (p.tube_type) parts.push({ label: "Type", value: String(p.tube_type).trim() });
    if (p.tube_size_fr) parts.push({ label: "Size", value: `${p.tube_size_fr} Fr` });
    if (p.guidance) parts.push({ label: "Guidance", value: String(p.guidance).trim() });
    return parts;
  }

  if (procKey === "thoracentesis") {
    const parts = [];
    if (p.side) parts.push({ label: "Side", value: String(p.side).trim() });
    if (p.guidance) parts.push({ label: "Guidance", value: String(p.guidance).trim() });
    const vol = Number.isFinite(p.volume_removed_ml) ? p.volume_removed_ml : Number.isFinite(p.volume_drained_ml) ? p.volume_drained_ml : null;
    if (Number.isFinite(vol)) parts.push({ label: "Removed", value: `${vol} mL` });
    if (p.fluid_appearance) parts.push({ label: "Fluid", value: String(p.fluid_appearance).trim() });
    if (p.manometry_performed !== null && p.manometry_performed !== undefined) {
      parts.push({ label: "Manometry", value: p.manometry_performed ? "Yes" : "No" });
    }
    return parts;
  }

  if (procKey === "ipc") {
    const parts = [];
    if (p.action) parts.push({ label: "Action", value: String(p.action).trim() });
    if (p.side) parts.push({ label: "Side", value: String(p.side).trim() });
    if (p.catheter_brand) parts.push({ label: "Brand", value: String(p.catheter_brand).trim() });
    if (p.tunneled !== null && p.tunneled !== undefined) parts.push({ label: "Tunneled", value: p.tunneled ? "Yes" : "No" });
    return parts;
  }

  if (procKey === "pleural_biopsy") {
    const parts = [];
    if (p.side) parts.push({ label: "Side", value: String(p.side).trim() });
    if (p.guidance) parts.push({ label: "Guidance", value: String(p.guidance).trim() });
    if (p.needle_type) parts.push({ label: "Needle", value: String(p.needle_type).trim() });
    if (Number.isFinite(p.number_of_samples)) parts.push({ label: "Samples", value: String(p.number_of_samples) });
    return parts;
  }

  if (procKey === "pleurodesis") {
    const parts = [];
    if (p.method) parts.push({ label: "Method", value: String(p.method).trim() });
    if (p.agent) parts.push({ label: "Agent", value: String(p.agent).trim() });
    if (Number.isFinite(p.talc_dose_grams)) parts.push({ label: "Talc", value: `${p.talc_dose_grams} g` });
    if (p.indication) parts.push({ label: "Indication", value: String(p.indication).trim() });
    return parts;
  }

  if (procKey === "fibrinolytic_therapy") {
    const parts = [];
    if (Array.isArray(p.agents) && p.agents.length > 0) parts.push({ label: "Agents", value: p.agents.filter(Boolean).join(", ") });
    if (Number.isFinite(p.tpa_dose_mg)) parts.push({ label: "tPA", value: `${p.tpa_dose_mg} mg` });
    if (Number.isFinite(p.dnase_dose_mg)) parts.push({ label: "DNase", value: `${p.dnase_dose_mg} mg` });
    if (Number.isFinite(p.number_of_doses)) parts.push({ label: "Doses", value: String(p.number_of_doses) });
    if (p.indication) parts.push({ label: "Indication", value: String(p.indication).trim() });
    return parts;
  }

  if (procKey === "medical_thoracoscopy") {
    const parts = [];
    if (p.side) parts.push({ label: "Side", value: String(p.side).trim() });
    if (p.scope_type) parts.push({ label: "Scope", value: String(p.scope_type).trim() });
    if (p.anesthesia_type) parts.push({ label: "Anesthesia", value: String(p.anesthesia_type).trim() });
    if (p.biopsies_taken !== null && p.biopsies_taken !== undefined) parts.push({ label: "Biopsies", value: p.biopsies_taken ? "Yes" : "No" });
    if (Number.isFinite(p.number_of_biopsies)) parts.push({ label: "Count", value: String(p.number_of_biopsies) });
    if (p.adhesiolysis_performed !== null && p.adhesiolysis_performed !== undefined) parts.push({ label: "Adhesiolysis", value: p.adhesiolysis_performed ? "Yes" : "No" });
    if (p.findings) parts.push({ label: "Findings", value: String(p.findings).trim() });
    return parts;
  }

  if (procKey === "linear_ebus") {
    if (!performed) return [];
    const parts = [];
    const stations = Array.isArray(p.stations_sampled) ? p.stations_sampled.filter(Boolean) : [];
    if (stations.length > 0) parts.push({ label: "Stations", value: stations.join(", ") });
    if (p.needle_gauge) parts.push({ label: "Needle", value: p.needle_gauge });
    if (p.elastography_used !== null && p.elastography_used !== undefined)
      parts.push({ label: "Elastography", value: p.elastography_used ? "Yes" : "No" });
    const pattern = deriveLinearEbusElastographyPattern(p);
    if (pattern) parts.push({ label: "Pattern", value: pattern });
    return parts;
  }

  if (procKey === "therapeutic_aspiration") {
    const parts = [];
    if (p.material) parts.push({ label: "Material", value: p.material });
    const loc = cleanLocationForDisplay(p.location);
    if (loc) parts.push({ label: "Location", value: loc });
    return parts;
  }

  if (procKey === "radial_ebus") {
    const parts = [];
    if (p.probe_position) parts.push({ label: "Probe", value: p.probe_position });
    if (p.guide_sheath_used !== null && p.guide_sheath_used !== undefined)
      parts.push({ label: "Guide sheath", value: p.guide_sheath_used ? "Yes" : "No" });
    return parts;
  }

  if (procKey === "navigational_bronchoscopy") {
    const parts = [];
    if (p.target_reached !== null && p.target_reached !== undefined)
      parts.push({ label: "Target reached", value: p.target_reached ? "Yes" : "No" });
    if (Number.isFinite(p.divergence_mm)) parts.push({ label: "Divergence", value: `${p.divergence_mm} mm` });
    if (p.confirmation_method) parts.push({ label: "Confirmed by", value: p.confirmation_method });
    return parts;
  }

  // Fallback: parse existing flat string
  const flat = summarizeProcedure(procKey, procObj);
  if (flat === "\u2014" || !flat) return [];
  return flat.split(" \u00b7 ").map((seg) => {
    const colon = seg.indexOf(": ");
    if (colon > -1) return { label: seg.slice(0, colon), value: seg.slice(colon + 2) };
    return { label: "", value: seg };
  });
}

function renderClinicalContextTable(data) {
  const host = document.getElementById("clinicalContextHost");
  if (!host) return;
  clearEl(host);

  const registry = getRegistry(data);
  const age = pickRegistryPathValue(registry, ["patient.age", "patient_demographics.age_years"]).value;
  const sexRaw = pickRegistryPathValue(registry, ["patient.sex", "patient_demographics.gender"]).value;
  const indication = pickRegistryPathValue(registry, ["procedure.indication", "clinical_context.primary_indication"]).value;
  const asa = pickRegistryPathValue(registry, ["risk_assessment.asa_class", "clinical_context.asa_class"]).value;
  const sex = normalizeSexDisplay(sexRaw);

  const ctx = registry?.clinical_context || {};
  const sed = registry?.sedation || {};
  const setting = registry?.procedure_setting || {};

  const rows = [
    { label: "Primary indication", value: cleanIndicationForDisplay(indication), fullWidth: true },
    { label: "Patient age", value: age },
    { label: "Patient sex", value: sex },
    { label: "ASA class", value: asa },
    { label: "Indication category", value: ctx?.indication_category },
    { label: "Bronchus sign", value: ctx?.bronchus_sign },
    { label: "Sedation type", value: sed?.type },
    { label: "Anesthesia provider", value: sed?.anesthesia_provider },
    { label: "Airway type", value: setting?.airway_type },
    { label: "Procedure location", value: setting?.location },
    { label: "Patient position", value: setting?.patient_position },
  ].filter((r) => r.value !== null && r.value !== undefined && String(r.value).trim() !== "");

  if (rows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "dash-empty";
    empty.style.padding = "10px 12px";
    empty.textContent = "No clinical context available.";
    host.appendChild(empty);
    return;
  }

  const grid = document.createElement("div");
  grid.className = "clinical-kv-grid";

  rows.forEach((r) => {
    const cell = document.createElement("div");
    cell.className = "kv-cell";
    if (r.fullWidth) cell.classList.add("kv-full-width");

    const label = document.createElement("div");
    label.className = "kv-cell-label";
    label.textContent = r.label;

    const value = document.createElement("div");
    value.className = "kv-cell-value";
    value.textContent = fmtMaybe(r.value);

    cell.appendChild(label);
    cell.appendChild(value);
    grid.appendChild(cell);
  });

  host.appendChild(grid);
}

function isPerformedProcedure(procObj) {
  if (procObj === true) return true;
  if (procObj === false || procObj === null || procObj === undefined) return false;
  if (typeof procObj === "object" && typeof procObj.performed === "boolean") return procObj.performed;
  return false;
}

function hasMeaningfulValue(value) {
  if (value === null || value === undefined) return false;
  if (typeof value === "string") return value.trim() !== "";
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return true; // number/boolean/etc.
}

function hasProcedureDetails(procObj) {
  if (!procObj || typeof procObj !== "object" || Array.isArray(procObj)) return false;
  for (const [k, v] of Object.entries(procObj)) {
    if (k === "performed" || k === "summary") continue;
    if (hasMeaningfulValue(v)) return true;
  }
  return false;
}

function deriveLinearEbusElastographyPattern(procObj) {
  const direct = String(procObj?.elastography_pattern || "").trim();
  if (direct) return direct;

  const events = Array.isArray(procObj?.node_events) ? procObj.node_events : [];
  const patterns = events
    .map((ev) => String(ev?.elastography_pattern || "").trim())
    .filter(Boolean);
  const unique = Array.from(new Set(patterns));
  if (unique.length === 0) return "";
  if (unique.length === 1) return unique[0];
  return unique.join(", ");
}

function summarizeProcedure(procKey, procObj) {
  const performed = isPerformedProcedure(procObj);
  const p = procObj && typeof procObj === "object" ? procObj : {};

  if (procKey === "diagnostic_bronchoscopy") {
    const abn = Array.isArray(p.airway_abnormalities) ? p.airway_abnormalities.filter(Boolean) : [];
    const findings = String(p.inspection_findings || "").trim();
    const parts = [];
    if (abn.length > 0) parts.push(`Abnormalities: ${abn.join(", ")}`);
    if (findings) parts.push(`Findings: ${findings}`);
    return parts.join(" · ") || "—";
  }

  if (procKey === "bal") {
    const parts = [];
    const loc = cleanLocationForDisplay(p.location);
    if (loc) parts.push(`Location: ${loc}`);
    if (Number.isFinite(p.volume_instilled_ml)) parts.push(`Instilled: ${p.volume_instilled_ml} mL`);
    if (Number.isFinite(p.volume_recovered_ml)) parts.push(`Recovered: ${p.volume_recovered_ml} mL`);
    return parts.join(" · ") || "—";
  }

  if (procKey === "chest_ultrasound") {
    const parts = [];
    if (p.hemithorax) parts.push(String(p.hemithorax).trim());
    if (p.effusion_volume) parts.push(`Effusion: ${String(p.effusion_volume).trim()}`);
    if (p.effusion_echogenicity) parts.push(String(p.effusion_echogenicity).trim());
    if (p.effusion_loculations) parts.push(`Loculations: ${String(p.effusion_loculations).trim()}`);
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "rigid_bronchoscopy") {
    const parts = [];
    if (Number.isFinite(p.rigid_scope_size)) parts.push(`${p.rigid_scope_size} mm`);
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "chest_tube") {
    const parts = [];
    if (p.action) parts.push(String(p.action).trim());
    if (p.tube_type) parts.push(String(p.tube_type).trim());
    if (p.tube_size_fr) parts.push(`${p.tube_size_fr} Fr`);
    if (p.guidance) parts.push(`${String(p.guidance).trim()} guided`);
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "thoracentesis") {
    const parts = [];
    if (p.side) parts.push(String(p.side).trim());
    if (p.guidance) parts.push(`${String(p.guidance).trim()} guided`);
    const vol =
      Number.isFinite(p.volume_removed_ml) ? p.volume_removed_ml : Number.isFinite(p.volume_drained_ml) ? p.volume_drained_ml : null;
    if (Number.isFinite(vol)) parts.push(`${vol} mL removed`);
    if (p.fluid_appearance) parts.push(String(p.fluid_appearance).trim());
    if (p.manometry_performed !== null && p.manometry_performed !== undefined) {
      parts.push(`Manometry: ${p.manometry_performed ? "Yes" : "No"}`);
    }
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "ipc") {
    const parts = [];
    if (p.action) parts.push(String(p.action).trim());
    if (p.side) parts.push(String(p.side).trim());
    if (p.catheter_brand) parts.push(String(p.catheter_brand).trim());
    if (p.tunneled !== null && p.tunneled !== undefined) parts.push(p.tunneled ? "Tunneled" : "Not tunneled");
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "pleural_biopsy") {
    const parts = [];
    if (p.side) parts.push(String(p.side).trim());
    if (p.guidance) parts.push(`${String(p.guidance).trim()} guided`);
    if (p.needle_type) parts.push(String(p.needle_type).trim());
    if (Number.isFinite(p.number_of_samples)) parts.push(`${p.number_of_samples} samples`);
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "pleurodesis") {
    const parts = [];
    if (p.method) parts.push(String(p.method).trim());
    if (p.agent) parts.push(String(p.agent).trim());
    if (Number.isFinite(p.talc_dose_grams)) parts.push(`${p.talc_dose_grams} g`);
    if (p.indication) parts.push(String(p.indication).trim());
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "fibrinolytic_therapy") {
    const parts = [];
    if (Array.isArray(p.agents) && p.agents.length > 0) parts.push(p.agents.filter(Boolean).join(", "));
    if (Number.isFinite(p.tpa_dose_mg)) parts.push(`tPA ${p.tpa_dose_mg} mg`);
    if (Number.isFinite(p.dnase_dose_mg)) parts.push(`DNase ${p.dnase_dose_mg} mg`);
    if (Number.isFinite(p.number_of_doses)) parts.push(`${p.number_of_doses} dose(s)`);
    if (p.indication) parts.push(String(p.indication).trim());
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "medical_thoracoscopy") {
    const parts = [];
    if (p.side) parts.push(String(p.side).trim());
    if (p.scope_type) parts.push(String(p.scope_type).trim());
    if (p.anesthesia_type) parts.push(String(p.anesthesia_type).trim());
    if (p.biopsies_taken !== null && p.biopsies_taken !== undefined) {
      parts.push(`Biopsies: ${p.biopsies_taken ? "Yes" : "No"}`);
    }
    if (Number.isFinite(p.number_of_biopsies)) parts.push(`${p.number_of_biopsies} biopsy(ies)`);
    if (p.adhesiolysis_performed !== null && p.adhesiolysis_performed !== undefined) {
      parts.push(`Adhesiolysis: ${p.adhesiolysis_performed ? "Yes" : "No"}`);
    }
    if (p.findings) parts.push(`Findings: ${String(p.findings).trim()}`);
    return parts.join(" · ") || (performed ? "Performed" : "—");
  }

  if (procKey === "linear_ebus") {
    if (!performed) return "—";
    const stations = Array.isArray(p.stations_sampled) ? p.stations_sampled.filter(Boolean) : [];
    const parts = [];
    if (stations.length > 0) parts.push(`Stations: ${stations.join(", ")}`);
    if (p.needle_gauge) parts.push(`Needle: ${p.needle_gauge}`);
    if (p.elastography_used !== null && p.elastography_used !== undefined) parts.push(`Elastography: ${p.elastography_used ? "Yes" : "No"}`);
    const pattern = deriveLinearEbusElastographyPattern(p);
    if (pattern) parts.push(`Pattern: ${pattern}`);
    return parts.join(" · ") || "—";
  }

  if (procKey === "therapeutic_aspiration") {
    const parts = [];
    if (p.material) parts.push(`Material: ${p.material}`);
    const loc = cleanLocationForDisplay(p.location);
    if (loc) parts.push(`Location: ${loc}`);
    return parts.join(" · ") || "—";
  }

  if (procKey === "radial_ebus") {
    const parts = [];
    if (p.probe_position) parts.push(`Probe: ${p.probe_position}`);
    if (p.guide_sheath_used !== null && p.guide_sheath_used !== undefined) parts.push(`Guide sheath: ${p.guide_sheath_used ? "Yes" : "No"}`);
    return parts.join(" · ") || "—";
  }

  if (procKey === "navigational_bronchoscopy") {
    const parts = [];
    if (p.target_reached !== null && p.target_reached !== undefined) parts.push(`Target reached: ${p.target_reached ? "Yes" : "No"}`);
    if (Number.isFinite(p.divergence_mm)) parts.push(`Divergence: ${p.divergence_mm} mm`);
    if (p.confirmation_method) parts.push(`Confirmed by: ${p.confirmation_method}`);
    return parts.join(" · ") || "—";
  }

  // Fallback: surface a few common fields without being noisy
  const parts = [];
  const loc = cleanLocationForDisplay(p.location);
  if (loc) parts.push(`Location: ${loc}`);
  const locations = cleanLocationsListForDisplay(p.locations);
  if (locations.length > 0) parts.push(`Locations: ${locations.join(", ")}`);
  if (Number.isFinite(p.number_of_samples)) parts.push(`${p.number_of_samples} samples`);
  return parts.join(" · ") || "—";
}

function renderProceduresSummaryTable(data) {
  const host = document.getElementById("proceduresSummaryHost");
  if (!host) return;
  clearEl(host);

  const registry = getRegistry(data);
  const procs = registry?.procedures_performed;
  const pleural = registry?.pleural_procedures;
  const hasProcs = procs && typeof procs === "object";
  const hasPleural = pleural && typeof pleural === "object";

  if (!hasProcs && !hasPleural) {
    const empty = document.createElement("div");
    empty.className = "dash-empty";
    empty.style.padding = "10px 12px";
    empty.textContent = "No procedures available.";
    host.appendChild(empty);
    return;
  }

  const items = [];
  if (hasProcs) {
    Object.keys(procs)
      .sort((a, b) => titleCaseKey(a).localeCompare(titleCaseKey(b)))
      .forEach((k) => items.push({ section: "procedures_performed", key: k, obj: procs[k] }));
  }
  if (hasPleural) {
    Object.keys(pleural)
      .sort((a, b) => titleCaseKey(a).localeCompare(titleCaseKey(b)))
      .forEach((k) => items.push({ section: "pleural_procedures", key: k, obj: pleural[k] }));
  }

  const withPerformed = items.map((it) => ({ ...it, performed: isPerformedProcedure(it.obj) }));
  withPerformed.sort((a, b) => {
    if (a.performed !== b.performed) return a.performed ? -1 : 1;
    return titleCaseKey(a.key).localeCompare(titleCaseKey(b.key));
  });

  if (withPerformed.length === 0) {
    const empty = document.createElement("div");
    empty.className = "dash-empty";
    empty.style.padding = "10px 12px";
    empty.textContent = "No procedures found.";
    host.appendChild(empty);
    return;
  }

  const table = document.createElement("table");
  table.className = "dash-table";
  const thead = document.createElement("thead");
  const headTr = document.createElement("tr");
  [{ text: "Procedure", width: "35%" }, { text: "Performed", width: "15%" }, { text: "Key Details", width: "50%" }].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h.text;
    th.style.width = h.width;
    headTr.appendChild(th);
  });
  thead.appendChild(headTr);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  let insertedSeparator = false;

  withPerformed.forEach(({ key, obj, performed }) => {
    if (!performed && !insertedSeparator) {
      insertedSeparator = true;
      const sepTr = document.createElement("tr");
      sepTr.className = "proc-group-separator";
      const sepTd = document.createElement("td");
      sepTd.colSpan = 3;
      sepTr.appendChild(sepTd);
      tbody.appendChild(sepTr);
    }

    const tr = document.createElement("tr");
    tr.className = performed ? "proc-row-performed" : "proc-row-notperformed";

    const tdName = document.createElement("td");
    tdName.className = "proc-name-cell";
    const actionSuffix =
      (key === "chest_tube" || key === "ipc") && obj && typeof obj === "object" && typeof obj.action === "string"
        ? String(obj.action).trim()
        : "";
    tdName.textContent = actionSuffix ? `${titleCaseKey(key)} ${actionSuffix}` : titleCaseKey(key);

    const tdPerf = document.createElement("td");
    const badge = document.createElement("span");
    badge.className = performed ? "proc-badge-yes" : "proc-badge-no";
    badge.textContent = performed ? "Yes" : "No";
    tdPerf.appendChild(badge);

    const tdDetails = document.createElement("td");
    const detailItems = summarizeProcedureStructured(key, obj);
    if (detailItems.length === 0) {
      tdDetails.textContent = "\u2014";
    } else {
      const ul = document.createElement("ul");
      ul.className = "proc-detail-list";
      detailItems.forEach(({ label, value }) => {
        const li = document.createElement("li");
        li.className = "proc-detail-item";
        if (label) {
          const lbl = document.createElement("span");
          lbl.className = "proc-detail-label";
          lbl.textContent = label;
          li.appendChild(lbl);
        }
        const val = document.createElement("span");
        val.className = "proc-detail-value";
        val.textContent = value;
        li.appendChild(val);
        ul.appendChild(li);
      });
      tdDetails.appendChild(ul);
    }

    tr.appendChild(tdName);
    tr.appendChild(tdPerf);
    tr.appendChild(tdDetails);
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  host.appendChild(table);
}

function toggleCard(cardId, visible) {
  const card = document.getElementById(cardId);
  if (!card) return;
  card.classList.toggle("hidden", !visible);
}

function renderDiagnosticFindings(data) {
  const host = document.getElementById("procedureDetailPanels");
  if (!host) return;

  const proc = getRegistry(data)?.procedures_performed?.diagnostic_bronchoscopy;
  const performed = isPerformedProcedure(proc);
  const hasData = performed || hasProcedureDetails(proc);
  if (!hasData) return;

  const panel = createDetailPanel("Diagnostic Bronchoscopy Findings", "badge-diagnostic", performed, (body) => {
    const table = document.createElement("table");
    table.className = "dash-table kv-table";
    const tbody = document.createElement("tbody");

    const abn = Array.isArray(proc?.airway_abnormalities) ? proc.airway_abnormalities.filter(Boolean) : [];
    const rows = [
      ["Airway abnormalities", abn.length ? abn.join(", ") : "\u2014"],
      ["Findings (free text)", proc?.inspection_findings || "\u2014"],
    ];
    rows.forEach(([k, v]) => {
      const tr = document.createElement("tr");
      const tdK = document.createElement("td");
      tdK.textContent = k;
      const tdV = document.createElement("td");
      tdV.textContent = fmtMaybe(v);
      tr.appendChild(tdK);
      tr.appendChild(tdV);
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    body.appendChild(table);
  });

  host.appendChild(panel);
}

function renderBalDetails(data) {
  const host = document.getElementById("procedureDetailPanels");
  if (!host) return;

  const proc = getRegistry(data)?.procedures_performed?.bal;
  const performed = isPerformedProcedure(proc);
  const hasData = performed || hasProcedureDetails(proc);
  if (!hasData) return;

  const panel = createDetailPanel("BAL Details", "badge-sampling", performed, (body) => {
    const table = document.createElement("table");
    table.className = "dash-table kv-table";
    const tbody = document.createElement("tbody");

    const rows = [
      ["Location", proc?.location || "\u2014"],
      ["Instilled (mL)", Number.isFinite(proc?.volume_instilled_ml) ? String(proc.volume_instilled_ml) : "\u2014"],
      ["Recovered (mL)", Number.isFinite(proc?.volume_recovered_ml) ? String(proc.volume_recovered_ml) : "\u2014"],
      ["Appearance", proc?.appearance || "\u2014"],
    ];
    rows.forEach(([k, v]) => {
      const tr = document.createElement("tr");
      const tdK = document.createElement("td");
      tdK.textContent = k;
      const tdV = document.createElement("td");
      tdV.textContent = fmtMaybe(v);
      tr.appendChild(tdK);
      tr.appendChild(tdV);
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    body.appendChild(table);
  });

  host.appendChild(panel);
}

function renderLinearEbusSummary(data) {
  const host = document.getElementById("procedureDetailPanels");
  if (!host) return;

  const proc = getRegistry(data)?.procedures_performed?.linear_ebus;
  const performed = isPerformedProcedure(proc);
  const events = Array.isArray(proc?.node_events) ? proc.node_events : [];
  const hasData = performed || hasProcedureDetails(proc) || events.length > 0;
  if (!hasData) return;

  const panel = createDetailPanel("Linear EBUS Technical Summary", "badge-ebus", performed, (body) => {
    const table = document.createElement("table");
    table.className = "dash-table kv-table";
    const tbody = document.createElement("tbody");

    const derivedPattern = deriveLinearEbusElastographyPattern(proc);
    const stations =
      Array.isArray(proc?.stations_sampled) && proc.stations_sampled.length > 0
        ? proc.stations_sampled.filter(Boolean)
        : Array.isArray(events)
          ? events
              .filter((e) => e?.action && e.action !== "inspected_only" && e.station)
              .map((e) => e.station)
          : [];
    const uniqueStations = Array.from(new Set(stations));

    const rows = [
      ["Stations sampled", uniqueStations.length ? uniqueStations.join(", ") : "\u2014"],
      ["Needle gauge", proc?.needle_gauge || "\u2014"],
      ["Elastography used", proc?.elastography_used === null || proc?.elastography_used === undefined ? "\u2014" : (proc.elastography_used ? "Yes" : "No")],
      ["Elastography pattern", derivedPattern || "\u2014"],
    ];

    rows.forEach(([k, v]) => {
      const tr = document.createElement("tr");
      const tdK = document.createElement("td");
      tdK.textContent = k;
      const tdV = document.createElement("td");
      tdV.textContent = fmtMaybe(v);
      tr.appendChild(tdK);
      tr.appendChild(tdV);
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    body.appendChild(table);
  });

  host.appendChild(panel);
}

function renderEbusNodeEvents(data) {
  const host = document.getElementById("procedureDetailPanels");
  if (!host) return;

  const proc = getRegistry(data)?.procedures_performed?.linear_ebus;
  const performed = isPerformedProcedure(proc);
  const events = Array.isArray(proc?.node_events) ? proc.node_events : [];
  if (events.length === 0) return;

  const actionLabel = (action) => {
    const a = String(action || "");
    if (a === "inspected_only") return "Inspected only";
    if (a === "needle_aspiration") return "Needle aspiration";
    if (a === "core_biopsy") return "Core biopsy";
    if (a === "forceps_biopsy") return "Forceps biopsy";
    return a || "\u2014";
  };

  const panel = createDetailPanel("Linear EBUS Node Events (Granular)", "badge-ebus", performed, (body) => {
    const table = document.createElement("table");
    table.className = "dash-table striped";
    const thead = document.createElement("thead");
    const headTr = document.createElement("tr");
    [
      { text: "Station", width: "12%" },
      { text: "Action", width: "36%" },
      { text: "Passes", width: "10%" },
      { text: "Elastography", width: "20%" },
      { text: "Evidence", width: "22%" },
    ].forEach((h) => {
      const th = document.createElement("th");
      th.textContent = h.text;
      th.style.width = h.width;
      headTr.appendChild(th);
    });
    thead.appendChild(headTr);
    table.appendChild(thead);

    const tbody = document.createElement("tbody");

    events.forEach((ev) => {
      const tr = document.createElement("tr");

      const tdStation = document.createElement("td");
      tdStation.textContent = fmtMaybe(ev?.station);

      const tdAction = document.createElement("td");
      tdAction.textContent = actionLabel(ev?.action);

      const tdPasses = document.createElement("td");
      tdPasses.textContent = Number.isFinite(ev?.passes) ? String(ev.passes) : "\u2014";

      const tdElast = document.createElement("td");
      tdElast.textContent = fmtMaybe(ev?.elastography_pattern);

      const tdEvidence = document.createElement("td");
      const quote = String(ev?.evidence_quote || "").trim();
      if (!quote) {
        tdEvidence.textContent = "\u2014";
      } else {
        const details = document.createElement("details");
        details.className = "inline-details";
        const summary = document.createElement("summary");
        summary.textContent = safeSnippet(quote, 0, quote.length);
        details.appendChild(summary);
        const detBody = document.createElement("div");
        detBody.style.marginTop = "8px";
        const pre = document.createElement("pre");
        pre.style.whiteSpace = "pre-wrap";
        pre.style.margin = "0";
        pre.textContent = quote;
        detBody.appendChild(pre);
        details.appendChild(detBody);
        tdEvidence.appendChild(details);
      }

      tr.appendChild(tdStation);
      tr.appendChild(tdAction);
      tr.appendChild(tdPasses);
      tr.appendChild(tdElast);
      tr.appendChild(tdEvidence);
      tbody.appendChild(tr);
    });

    table.appendChild(tbody);
    body.appendChild(table);
  });

  host.appendChild(panel);
}

function renderEvidenceTraceability(data) {
  const host = document.getElementById("evidenceTraceabilityHost");
  if (!host) return;
  clearEl(host);

  const evidence = getEvidence(data);
  if (!evidence || typeof evidence !== "object") {
    const empty = document.createElement("div");
    empty.className = "dash-empty";
    empty.textContent = "No evidence available.";
    host.appendChild(empty);
    return;
  }

  const fields = Object.keys(evidence).sort((a, b) => a.localeCompare(b));
  const rows = [];
  fields.forEach((field) => {
    const items = evidence[field];
    if (!Array.isArray(items)) return;
    items.forEach((item) => rows.push({ field, item }));
  });

  if (rows.length === 0) {
    const empty = document.createElement("div");
    empty.className = "dash-empty";
    empty.textContent = "No evidence spans.";
    host.appendChild(empty);
    return;
  }

  const wrap = document.createElement("div");
  wrap.className = "dash-table-wrap";

  const table = document.createElement("table");
  table.className = "dash-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Field", "Evidence", "Span", "Confidence", "Source"].forEach((h) => {
    const th = document.createElement("th");
    th.textContent = h;
    headRow.appendChild(th);
  });
  thead.appendChild(headRow);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");

  rows.slice(0, 250).forEach(({ field, item }) => {
    const tr = document.createElement("tr");

    const tdField = document.createElement("td");
    tdField.textContent = field;

    const tdEv = document.createElement("td");
    const text = String(item?.text || item?.quote || "").trim();
    const span = Array.isArray(item?.span) ? item.span : null;
    const chip = makeEvidenceChip({ text: safeSnippet(text || "(evidence)", 0, Math.min(text.length || 0, 240)), span });
    tdEv.appendChild(chip);

    const tdSpan = document.createElement("td");
    tdSpan.textContent = fmtSpan(span);

    const tdConf = document.createElement("td");
    tdConf.textContent = typeof item?.confidence === "number" ? item.confidence.toFixed(2) : "—";

    const tdSource = document.createElement("td");
    tdSource.textContent = fmtMaybe(item?.source);

    tr.appendChild(tdField);
    tr.appendChild(tdEv);
    tr.appendChild(tdSpan);
    tr.appendChild(tdConf);
    tr.appendChild(tdSource);
    tbody.appendChild(tr);
  });

  table.appendChild(tbody);
  wrap.appendChild(table);
  host.appendChild(wrap);

  if (rows.length > 250) {
    const note = document.createElement("div");
    note.className = "qa-line";
    note.style.marginTop = "8px";
    note.textContent = `Showing first 250 evidence items (${rows.length} total).`;
    host.appendChild(note);
  }
}

function buildFlattenedTables(data) {
  const registry = getRegistry(data);
  const tables = [];

  const codingLines = getCodingLines(data);
  const selectedLines = codingLines.filter(
    (ln) => String(ln?.selection_status || "selected").toLowerCase() === "selected"
  );
  const suppressedLines = codingLines.filter(
    (ln) => String(ln?.selection_status || "").toLowerCase() !== "selected"
  );

  tables.push({
    id: "coding_selected",
    title: "CPT Codes – Selected",
    columns: [
      { key: "code", label: "CPT Code", type: "text" },
      { key: "description", label: "Description", type: "text" },
      { key: "units", label: "Units", type: "number" },
      { key: "role", label: "Role", type: "select", options: ROLE_OPTIONS },
      { key: "rationale", label: "Rationale", type: "text" },
    ],
    rows: selectedLines.map((ln) => ({
      code: normalizeCptCode(ln?.code),
      description: ln?.description || "",
      units: Number.isFinite(ln?.units) ? String(ln.units) : "",
      role: String(ln?.role || (ln?.is_add_on ? "add_on" : "primary") || "primary").toLowerCase(),
      rationale: ln?.selection_reason || ln?.rationale || "",
    })),
    allowAdd: true,
    allowDelete: true,
    emptyMessage: "No selected CPT codes.",
  });

  tables.push({
    id: "coding_suppressed",
    title: "CPT Codes – Dropped / Suppressed",
    columns: [
      { key: "code", label: "CPT Code", type: "text" },
      { key: "status", label: "Status", type: "select", options: STATUS_OPTIONS },
      { key: "reason", label: "Reason", type: "text" },
    ],
    rows: suppressedLines.map((ln) => {
      const statusRaw = String(ln?.selection_status || "").toLowerCase();
      const status =
        statusRaw === "suppressed" || /suppress/i.test(String(ln?.selection_reason || ""))
          ? "Suppressed"
          : "Dropped";
      return {
        code: normalizeCptCode(ln?.code),
        status,
        reason: ln?.selection_reason || "",
      };
    }),
    allowAdd: true,
    allowDelete: true,
    emptyMessage: "No dropped/suppressed CPT codes.",
  });

  const perCode = Array.isArray(getCodingRationale(data)?.per_code)
    ? getCodingRationale(data).per_code
    : [];
  tables.push({
    id: "coding_rationale",
    title: "Coding Logic & Rationale",
    columns: [
      { key: "code", label: "CPT Code", type: "text" },
      { key: "summary", label: "Summary", type: "text" },
    ],
    rows: perCode.map((pc) => ({
      code: normalizeCptCode(pc?.code),
      summary: pc?.summary || "",
    })),
    allowAdd: true,
    allowDelete: true,
    emptyMessage: "No coding rationale entries.",
  });

  const rules = Array.isArray(getCodingRationale(data)?.rules_applied)
    ? getCodingRationale(data).rules_applied
    : [];
  tables.push({
    id: "rules_applied",
    title: "Rules Applied (Bundling & Policy)",
    columns: [
      { key: "rule_type", label: "Rule Type", type: "text" },
      { key: "codes_affected", label: "Codes Affected", type: "text" },
      { key: "outcome", label: "Outcome", type: "select", options: RULE_OUTCOME_OPTIONS },
      { key: "details", label: "Details", type: "text" },
    ],
    rows: rules.map((r) => ({
      rule_type: r?.rule_type || "",
      codes_affected: Array.isArray(r?.codes_affected) ? r.codes_affected.join(", ") : "",
      outcome: String(r?.outcome || "").toLowerCase() || "informational",
      details: r?.details || "",
    })),
    allowAdd: true,
    allowDelete: true,
    emptyMessage: "No rules applied.",
  });

  const billing = getPerCodeBilling(data);
  tables.push({
    id: "financial_summary",
    title: "Financial Summary",
    columns: [
      { key: "cpt_code", label: "CPT Code", type: "text" },
      { key: "units", label: "Units", type: "number" },
      { key: "work_rvu", label: "Work RVU", type: "number" },
      { key: "total_facility_rvu", label: "Facility RVU", type: "number" },
      { key: "facility_payment", label: "Payment", type: "number" },
      { key: "notes", label: "Notes", type: "text" },
    ],
    rows: billing.map((b) => ({
      cpt_code: normalizeCptCode(b?.cpt_code),
      units: Number.isFinite(b?.units) ? String(b.units) : "",
      work_rvu: Number.isFinite(b?.work_rvu) ? String(b.work_rvu) : "",
      total_facility_rvu: Number.isFinite(b?.total_facility_rvu) ? String(b.total_facility_rvu) : "",
      facility_payment: Number.isFinite(b?.facility_payment) ? String(b.facility_payment) : "",
      notes: b?.notes || "",
    })),
    allowAdd: true,
    allowDelete: true,
    emptyMessage: "No financial summary lines.",
  });

  const groupedWarnings = groupWarnings(data?.audit_warnings || []);
  const auditRows = [];
  for (const [category, messages] of groupedWarnings.entries()) {
    (Array.isArray(messages) ? messages : []).forEach((msg) => {
      auditRows.push({ category, notes: msg || "" });
    });
  }
  const validationErrors = Array.isArray(data?.validation_errors) ? data.validation_errors : [];
  validationErrors.forEach((msg) => auditRows.push({ category: "Validation", notes: String(msg || "") }));

  tables.push({
    id: "audit_flags",
    title: "Audit & Quality Flags",
    columns: [
      { key: "category", label: "Category", type: "text" },
      { key: "notes", label: "Notes", type: "text" },
    ],
    rows: auditRows,
    allowAdd: true,
    allowDelete: true,
    emptyMessage: "No audit flags.",
  });

  const clinicalRows = [];
  const ageInfo = pickRegistryPathValue(registry, ["patient.age", "patient_demographics.age_years"]);
  const sexInfo = pickRegistryPathValue(registry, ["patient.sex", "patient_demographics.gender"]);
  const indicationInfo = pickRegistryPathValue(registry, ["procedure.indication", "clinical_context.primary_indication"]);
  const asaInfo = pickRegistryPathValue(registry, ["risk_assessment.asa_class", "clinical_context.asa_class"]);
  clinicalRows.push({
    field: "Patient age",
    value: hasDisplayValue(ageInfo.value) ? String(ageInfo.value) : "",
    __meta: { path: `registry.${ageInfo.path || "patient.age"}`, valueType: "number" },
  });
  clinicalRows.push({
    field: "Patient sex",
    value: normalizeSexDisplay(sexInfo.value) || "",
    __meta: { path: `registry.${sexInfo.path || "patient.sex"}`, valueType: "text" },
  });
  clinicalRows.push({
    field: "Primary indication",
    value: cleanIndicationForDisplay(indicationInfo.value) || "",
    __meta: { path: `registry.${indicationInfo.path || "procedure.indication"}`, valueType: "text" },
  });
  clinicalRows.push({
    field: "ASA class",
    value: hasDisplayValue(asaInfo.value) ? String(asaInfo.value) : "",
    __meta: {
      path: `registry.${asaInfo.path || "risk_assessment.asa_class"}`,
      valueType: "number",
      inputType: "select",
      options: ASA_CLASS_OPTIONS,
    },
  });
  const ecogScore = registry?.clinical_context?.ecog_score;
  const ecogText = registry?.clinical_context?.ecog_text;
  clinicalRows.push({
    field: "ECOG score",
    value: Number.isFinite(ecogScore) ? String(ecogScore) : "",
    __meta: { path: "registry.clinical_context.ecog_score", valueType: "number" },
  });
  clinicalRows.push({
    field: "ECOG (raw text/range)",
    value: ecogText || "",
    __meta: { path: "registry.clinical_context.ecog_text", valueType: "text" },
  });
  clinicalRows.push({
    field: "CT bronchus sign",
    value: registry?.clinical_context?.bronchus_sign || "",
    __meta: {
      path: "registry.clinical_context.bronchus_sign",
      valueType: "text",
      inputType: "select",
      options: BRONCHUS_SIGN_OPTIONS,
    },
  });

  tables.push({
    id: "clinical_context",
    title: "Patient & Clinical Context",
    columns: [
      { key: "field", label: "Field", readOnly: true },
      { key: "value", label: "Value", type: "text" },
    ],
    rows: clinicalRows,
    allowAdd: false,
    allowDelete: false,
  });

  const procedures = registry?.procedures_performed || {};
  const pleuralProcs = registry?.pleural_procedures || {};
  const procRows = [];

  Object.keys(procedures)
    .sort((a, b) => titleCaseKey(a).localeCompare(titleCaseKey(b)))
    .forEach((key) => {
      procRows.push({
        procedure: titleCaseKey(key),
        performed: toYesNo(isPerformedProcedure(procedures[key])),
        details: summarizeProcedure(key, procedures[key]),
        __meta: { section: "procedures_performed", procKey: key },
      });
    });

  Object.keys(pleuralProcs)
    .sort((a, b) => titleCaseKey(a).localeCompare(titleCaseKey(b)))
    .forEach((key) => {
      procRows.push({
        procedure: titleCaseKey(key),
        performed: toYesNo(isPerformedProcedure(pleuralProcs[key])),
        details: summarizeProcedure(key, pleuralProcs[key]),
        __meta: { section: "pleural_procedures", procKey: key },
      });
    });

  tables.push({
    id: "procedures_summary",
    title: "Procedures Performed (Summary)",
    columns: [
      { key: "procedure", label: "Procedure", readOnly: true },
      { key: "performed", label: "Performed", type: "select", options: YES_NO_OPTIONS },
      { key: "details", label: "Key Details", type: "text" },
    ],
    rows: procRows,
    allowAdd: false,
    allowDelete: false,
  });

  const navAgg = registry?.procedures_performed?.navigational_bronchoscopy || {};
  const navAggPerformed = isPerformedProcedure(navAgg);
  const navTargetsPresent =
    Array.isArray(registry?.granular_data?.navigation_targets) && registry.granular_data.navigation_targets.length > 0;
  if (navAggPerformed || hasProcedureDetails(navAgg) || navTargetsPresent) {
    tables.push({
      id: "navigation_bronchoscopy_details",
      title: "Navigational Bronchoscopy (Aggregate)",
      columns: [
        { key: "field", label: "Field", readOnly: true },
        { key: "value", label: "Value", type: "text" },
      ],
      rows: [
        {
          field: "Tool-in-lesion confirmed",
          value: toYesNo(navAgg?.tool_in_lesion_confirmed),
          __meta: {
            path: "registry.procedures_performed.navigational_bronchoscopy.tool_in_lesion_confirmed",
            valueType: "boolean",
          },
        },
        {
          field: "Tool-in-lesion method",
          value: navAgg?.confirmation_method || "",
          __meta: {
            path: "registry.procedures_performed.navigational_bronchoscopy.confirmation_method",
            valueType: "text",
            inputType: "select",
            options: NAV_CONFIRMATION_METHOD_OPTIONS,
          },
        },
        {
          field: "CT-to-body divergence (mm)",
          value: Number.isFinite(navAgg?.divergence_mm) ? String(navAgg.divergence_mm) : "",
          __meta: {
            path: "registry.procedures_performed.navigational_bronchoscopy.divergence_mm",
            valueType: "number",
          },
        },
      ],
      allowAdd: false,
      allowDelete: false,
    });
  }

  const radial = registry?.procedures_performed?.radial_ebus || {};
  const radialPerformed = isPerformedProcedure(radial);
  const rebusUsed =
    Array.isArray(registry?.granular_data?.navigation_targets) &&
    registry.granular_data.navigation_targets.some(
      (t) => t?.rebus_used === true || String(t?.rebus_view || "").trim() !== ""
    );
  if (radialPerformed || hasProcedureDetails(radial) || rebusUsed) {
    tables.push({
      id: "radial_ebus_details",
      title: "Radial EBUS (Aggregate)",
      columns: [
        { key: "field", label: "Field", readOnly: true },
        { key: "value", label: "Value", type: "text" },
      ],
      rows: [
        {
          field: "Probe position",
          value: radial?.probe_position || "",
          __meta: {
            path: "registry.procedures_performed.radial_ebus.probe_position",
            valueType: "text",
            inputType: "select",
            options: RADIAL_EBUS_PROBE_POSITION_OPTIONS,
          },
        },
      ],
      allowAdd: false,
      allowDelete: false,
    });
  }

  const fibrinolytic = registry?.pleural_procedures?.fibrinolytic_therapy || {};
  const fibrinolyticPerformed = isPerformedProcedure(fibrinolytic);
  if (fibrinolyticPerformed || hasProcedureDetails(fibrinolytic)) {
    tables.push({
      id: "pleural_fibrinolytic_therapy",
      title: "Pleural Fibrinolytic Therapy",
      columns: [
        { key: "field", label: "Field", readOnly: true },
        { key: "value", label: "Value", type: "text" },
      ],
      rows: [
        {
          field: "Agents",
          value: Array.isArray(fibrinolytic?.agents) ? fibrinolytic.agents.join(", ") : "",
          __meta: { path: "registry.pleural_procedures.fibrinolytic_therapy.agents", valueType: "list" },
        },
        {
          field: "tPA dose (mg)",
          value: Number.isFinite(fibrinolytic?.tpa_dose_mg) ? String(fibrinolytic.tpa_dose_mg) : "",
          __meta: { path: "registry.pleural_procedures.fibrinolytic_therapy.tpa_dose_mg", valueType: "number" },
        },
        {
          field: "DNase dose (mg)",
          value: Number.isFinite(fibrinolytic?.dnase_dose_mg) ? String(fibrinolytic.dnase_dose_mg) : "",
          __meta: { path: "registry.pleural_procedures.fibrinolytic_therapy.dnase_dose_mg", valueType: "number" },
        },
        {
          field: "Number of doses",
          value: Number.isFinite(fibrinolytic?.number_of_doses) ? String(fibrinolytic.number_of_doses) : "",
          __meta: {
            path: "registry.pleural_procedures.fibrinolytic_therapy.number_of_doses",
            valueType: "number",
          },
        },
        {
          field: "Indication",
          value: fibrinolytic?.indication || "",
          __meta: {
            path: "registry.pleural_procedures.fibrinolytic_therapy.indication",
            valueType: "text",
            inputType: "select",
            options: FIBRINOLYTIC_INDICATION_OPTIONS,
          },
        },
      ],
      allowAdd: false,
      allowDelete: false,
    });
  }

  const complications = registry?.complications || {};
  const ptx = complications?.pneumothorax || {};
  const bleed = complications?.bleeding || {};
  const showComplications =
    ptx?.occurred === true ||
    bleed?.occurred === true ||
    hasMeaningfulValue(ptx?.intervention) ||
    hasMeaningfulValue(bleed?.bleeding_grade_nashville);
  if (showComplications) {
    tables.push({
      id: "complications_details",
      title: "Complications (Key Fields)",
      columns: [
        { key: "field", label: "Field", readOnly: true },
        { key: "value", label: "Value", type: "text" },
      ],
      rows: [
        {
          field: "Pneumothorax intervention",
          value: Array.isArray(ptx?.intervention) ? ptx.intervention.join(", ") : "",
          __meta: { path: "registry.complications.pneumothorax.intervention", valueType: "list" },
        },
        {
          field: "Bleeding grade (Nashville 0–4)",
          value: Number.isFinite(bleed?.bleeding_grade_nashville) ? String(bleed.bleeding_grade_nashville) : "",
          __meta: { path: "registry.complications.bleeding.bleeding_grade_nashville", valueType: "number" },
        },
      ],
      allowAdd: false,
      allowDelete: false,
    });
  }

  const granular = registry?.granular_data || {};
  const navigationTargets = Array.isArray(granular?.navigation_targets) ? granular.navigation_targets : [];
  tables.push({
    id: "navigation_targets",
    title: "Navigation Targets (Lesion Characteristics)",
    columns: [
      { key: "target_number", label: "Target #", readOnly: true },
      { key: "target_location_text", label: "Location", type: "text" },
      { key: "target_lobe", label: "Lobe", type: "text" },
      { key: "target_segment", label: "Segment", type: "text" },
      { key: "lesion_size_mm", label: "Size (mm)", type: "number" },
      { key: "distance_from_pleura_mm", label: "Dist from pleura (mm)", type: "number" },
      { key: "ct_characteristics", label: "CT Characteristics", type: "text" },
      { key: "pet_suv_max", label: "PET SUV Max", type: "number" },
      { key: "bronchus_sign", label: "Bronchus Sign", type: "text" },
      { key: "registration_error_mm", label: "Registration Error (mm)", type: "number" },
      { key: "rebus_view", label: "rEBUS View", type: "text" },
      { key: "tool_in_lesion_confirmed", label: "TIL Confirmed", type: "select", options: YES_NO_OPTIONS },
      { key: "confirmation_method", label: "TIL Method", type: "select", options: NAV_TARGET_CONFIRMATION_METHOD_OPTIONS },
      { key: "sampling_tools_used", label: "Sampling Tools", type: "text" },
      { key: "number_of_needle_passes", label: "Needle Passes", type: "number" },
      { key: "number_of_forceps_biopsies", label: "Forceps Specimens", type: "number" },
      { key: "number_of_cryo_biopsies", label: "Cryobiopsy Specimens", type: "number" },
      { key: "rose_result", label: "ROSE Result", type: "text" },
      { key: "notes", label: "Notes", type: "text" },
    ],
    rows: navigationTargets.map((t, idx) => ({
      target_number: String(t?.target_number ?? idx + 1),
      target_location_text: cleanLocationForDisplay(t?.target_location_text) || "",
      target_lobe: t?.target_lobe || "",
      target_segment: t?.target_segment || "",
      lesion_size_mm: Number.isFinite(t?.lesion_size_mm) ? String(t.lesion_size_mm) : "",
      distance_from_pleura_mm: Number.isFinite(t?.distance_from_pleura_mm) ? String(t.distance_from_pleura_mm) : "",
      ct_characteristics: t?.ct_characteristics || "",
      pet_suv_max: Number.isFinite(t?.pet_suv_max) ? String(t.pet_suv_max) : "",
      bronchus_sign: t?.bronchus_sign || "",
      registration_error_mm: Number.isFinite(t?.registration_error_mm) ? String(t.registration_error_mm) : "",
      rebus_view: t?.rebus_view || "",
      tool_in_lesion_confirmed: toYesNo(t?.tool_in_lesion_confirmed),
      confirmation_method: t?.confirmation_method || "",
      sampling_tools_used: Array.isArray(t?.sampling_tools_used) ? t.sampling_tools_used.join(", ") : "",
      number_of_needle_passes: Number.isFinite(t?.number_of_needle_passes) ? String(t.number_of_needle_passes) : "",
      number_of_forceps_biopsies: Number.isFinite(t?.number_of_forceps_biopsies) ? String(t.number_of_forceps_biopsies) : "",
      number_of_cryo_biopsies: Number.isFinite(t?.number_of_cryo_biopsies) ? String(t.number_of_cryo_biopsies) : "",
      rose_result: t?.rose_result || "",
      notes: t?.notes || "",
    })),
    allowAdd: false,
    allowDelete: false,
    emptyMessage: "No navigation targets.",
  });

  const ebusStations = Array.isArray(granular?.linear_ebus_stations_detail) ? granular.linear_ebus_stations_detail : [];
  tables.push({
    id: "linear_ebus_stations_detail",
    title: "Linear EBUS Stations (Morphology)",
    columns: [
      { key: "station", label: "Station", readOnly: true },
      { key: "sampled", label: "Sampled", type: "select", options: YES_NO_OPTIONS },
      { key: "short_axis_mm", label: "Short Axis (mm)", type: "number" },
      { key: "long_axis_mm", label: "Long Axis (mm)", type: "number" },
      { key: "shape", label: "Shape", type: "text" },
      { key: "margin", label: "Margin", type: "text" },
      { key: "echogenicity", label: "Echogenicity", type: "text" },
      { key: "chs_present", label: "CHS Present", type: "select", options: YES_NO_OPTIONS },
      { key: "necrosis_present", label: "Necrosis", type: "select", options: YES_NO_OPTIONS },
      { key: "calcification_present", label: "Calcification", type: "select", options: YES_NO_OPTIONS },
      { key: "needle_gauge", label: "Needle Gauge", type: "text" },
      { key: "number_of_passes", label: "Passes", type: "number" },
      { key: "rose_result", label: "ROSE Result", type: "text" },
      { key: "lymphocytes_present", label: "Lymphocytes", type: "select", options: YES_NO_OPTIONS },
      { key: "morphologic_impression", label: "Morphologic Impression", type: "text" },
    ],
    rows: ebusStations.map((s) => ({
      station: s?.station || "",
      sampled: toYesNo(s?.sampled),
      short_axis_mm: Number.isFinite(s?.short_axis_mm) ? String(s.short_axis_mm) : "",
      long_axis_mm: Number.isFinite(s?.long_axis_mm) ? String(s.long_axis_mm) : "",
      shape: s?.shape || "",
      margin: s?.margin || "",
      echogenicity: s?.echogenicity || "",
      chs_present: toYesNo(s?.chs_present),
      necrosis_present: toYesNo(s?.necrosis_present),
      calcification_present: toYesNo(s?.calcification_present),
      needle_gauge: s?.needle_gauge ? String(s.needle_gauge) : "",
      number_of_passes: Number.isFinite(s?.number_of_passes) ? String(s.number_of_passes) : "",
      rose_result: s?.rose_result || "",
      lymphocytes_present: toYesNo(s?.lymphocytes_present),
      morphologic_impression: s?.morphologic_impression || "",
    })),
    allowAdd: false,
    allowDelete: false,
    emptyMessage: "No station detail entries.",
  });

  const caoSites = Array.isArray(granular?.cao_interventions_detail) ? granular.cao_interventions_detail : [];
  tables.push({
    id: "cao_interventions_detail",
    title: "Central Airway Obstruction (Sites)",
    columns: [
      { key: "location", label: "Location", readOnly: true },
      { key: "obstruction_type", label: "Obstruction Type", type: "text" },
      { key: "etiology", label: "Etiology", type: "text" },
      { key: "lesion_morphology", label: "Morphology", type: "text" },
      { key: "lesion_count_text", label: "Lesion Count", type: "text" },
      { key: "length_mm", label: "Length (mm)", type: "number" },
      { key: "pre_obstruction_pct", label: "Pre Obstruction (%)", type: "number" },
      { key: "post_obstruction_pct", label: "Post Obstruction (%)", type: "number" },
      { key: "pre_diameter_mm", label: "Pre Diameter (mm)", type: "number" },
      { key: "post_diameter_mm", label: "Post Diameter (mm)", type: "number" },
      { key: "modalities_applied", label: "Modalities (Summary)", readOnly: true },
      { key: "stent_placed_at_site", label: "Stent Placed", type: "select", options: YES_NO_OPTIONS },
      { key: "hemostasis_required", label: "Hemostasis Required", type: "select", options: YES_NO_OPTIONS },
      { key: "notes", label: "Notes", type: "text" },
    ],
    rows: caoSites.map((site) => {
      const modalities = Array.isArray(site?.modalities_applied) ? site.modalities_applied : [];
      const modalitySummary = modalities
        .map((m) => {
          const name = String(m?.modality || "").trim();
          if (!name) return null;
          const parts = [];
          if (Number.isFinite(m?.power_setting_watts)) parts.push(`${m.power_setting_watts}W`);
          if (Number.isFinite(m?.balloon_diameter_mm)) parts.push(`${m.balloon_diameter_mm}mm`);
          if (Number.isFinite(m?.freeze_time_seconds)) parts.push(`${m.freeze_time_seconds}s`);
          if (Number.isFinite(m?.number_of_applications)) parts.push(`x${m.number_of_applications}`);
          return parts.length ? `${name} (${parts.join(", ")})` : name;
        })
        .filter(Boolean)
        .join(" · ");

      return {
        location: cleanLocationForDisplay(site?.location) || "",
        obstruction_type: site?.obstruction_type || "",
        etiology: site?.etiology || "",
        lesion_morphology: site?.lesion_morphology || "",
        lesion_count_text: site?.lesion_count_text || "",
        length_mm: Number.isFinite(site?.length_mm) ? String(site.length_mm) : "",
        pre_obstruction_pct: Number.isFinite(site?.pre_obstruction_pct) ? String(site.pre_obstruction_pct) : "",
        post_obstruction_pct: Number.isFinite(site?.post_obstruction_pct) ? String(site.post_obstruction_pct) : "",
        pre_diameter_mm: Number.isFinite(site?.pre_diameter_mm) ? String(site.pre_diameter_mm) : "",
        post_diameter_mm: Number.isFinite(site?.post_diameter_mm) ? String(site.post_diameter_mm) : "",
        modalities_applied: modalitySummary,
        stent_placed_at_site: toYesNo(site?.stent_placed_at_site),
        hemostasis_required: toYesNo(site?.hemostasis_required),
        notes: site?.notes || "",
      };
    }),
    allowAdd: false,
    allowDelete: false,
    emptyMessage: "No CAO sites.",
  });

  const diag = registry?.procedures_performed?.diagnostic_bronchoscopy || {};
  tables.push({
    id: "diagnostic_findings",
    title: "Diagnostic Bronchoscopy Findings",
    columns: [
      { key: "field", label: "Field", readOnly: true },
      { key: "value", label: "Value", type: "text" },
    ],
    rows: [
      {
        field: "Airway abnormalities",
        value: Array.isArray(diag?.airway_abnormalities) ? diag.airway_abnormalities.join(", ") : "",
        __meta: {
          path: "registry.procedures_performed.diagnostic_bronchoscopy.airway_abnormalities",
          valueType: "list",
        },
      },
      {
        field: "Findings (free text)",
        value: diag?.inspection_findings || "",
        __meta: {
          path: "registry.procedures_performed.diagnostic_bronchoscopy.inspection_findings",
          valueType: "text",
        },
      },
    ],
    allowAdd: false,
    allowDelete: false,
  });

  const bal = registry?.procedures_performed?.bal || {};
  tables.push({
    id: "bal_details",
    title: "BAL Details",
    columns: [
      { key: "field", label: "Field", readOnly: true },
      { key: "value", label: "Value", type: "text" },
    ],
    rows: [
      {
        field: "Location",
        value: cleanLocationForDisplay(bal?.location) || "",
        __meta: { path: "registry.procedures_performed.bal.location", valueType: "text" },
      },
      {
        field: "Instilled (mL)",
        value: Number.isFinite(bal?.volume_instilled_ml) ? String(bal.volume_instilled_ml) : "",
        __meta: { path: "registry.procedures_performed.bal.volume_instilled_ml", valueType: "number" },
      },
      {
        field: "Recovered (mL)",
        value: Number.isFinite(bal?.volume_recovered_ml) ? String(bal.volume_recovered_ml) : "",
        __meta: { path: "registry.procedures_performed.bal.volume_recovered_ml", valueType: "number" },
      },
      {
        field: "Appearance",
        value: bal?.appearance || "",
        __meta: { path: "registry.procedures_performed.bal.appearance", valueType: "text" },
      },
    ],
    allowAdd: false,
    allowDelete: false,
  });

  const chestUs = registry?.procedures_performed?.chest_ultrasound || {};
  const chestUsPerformed = isPerformedProcedure(chestUs);
  if (chestUsPerformed || hasProcedureDetails(chestUs)) {
    tables.push({
      id: "chest_ultrasound_details",
      title: "Chest Ultrasound Findings",
      columns: [
        { key: "field", label: "Field", readOnly: true },
        { key: "value", label: "Value", type: "text" },
      ],
      rows: [
        {
          field: "Performed",
          value: toYesNo(chestUsPerformed),
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.performed",
            valueType: "boolean",
            inputType: "select",
            options: YES_NO_OPTIONS,
          },
        },
        {
          field: "Image documentation",
          value: toYesNo(chestUs?.image_documentation),
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.image_documentation",
            valueType: "boolean",
            inputType: "select",
            options: YES_NO_OPTIONS,
          },
        },
        {
          field: "Hemithorax",
          value: chestUs?.hemithorax || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.hemithorax",
            valueType: "text",
            inputType: "select",
            options: HEMITHORAX_OPTIONS,
          },
        },
        {
          field: "Effusion volume",
          value: chestUs?.effusion_volume || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.effusion_volume",
            valueType: "text",
            inputType: "select",
            options: CHEST_US_EFFUSION_VOLUME_OPTIONS,
          },
        },
        {
          field: "Effusion echogenicity",
          value: chestUs?.effusion_echogenicity || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.effusion_echogenicity",
            valueType: "text",
            inputType: "select",
            options: CHEST_US_ECHOGENICITY_OPTIONS,
          },
        },
        {
          field: "Effusion loculations",
          value: chestUs?.effusion_loculations || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.effusion_loculations",
            valueType: "text",
            inputType: "select",
            options: CHEST_US_LOCULATIONS_OPTIONS,
          },
        },
        {
          field: "Diaphragmatic motion",
          value: chestUs?.diaphragmatic_motion || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.diaphragmatic_motion",
            valueType: "text",
            inputType: "select",
            options: CHEST_US_DIAPHRAGM_MOTION_OPTIONS,
          },
        },
        {
          field: "Lung sliding (pre)",
          value: chestUs?.lung_sliding_pre || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.lung_sliding_pre",
            valueType: "text",
            inputType: "select",
            options: CHEST_US_LUNG_SLIDING_OPTIONS,
          },
        },
        {
          field: "Lung sliding (post)",
          value: chestUs?.lung_sliding_post || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.lung_sliding_post",
            valueType: "text",
            inputType: "select",
            options: CHEST_US_LUNG_SLIDING_OPTIONS,
          },
        },
        {
          field: "Consolidation/atelectasis present",
          value: toYesNo(chestUs?.lung_consolidation_present),
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.lung_consolidation_present",
            valueType: "boolean",
            inputType: "select",
            options: YES_NO_OPTIONS,
          },
        },
        {
          field: "Pleura",
          value: chestUs?.pleura_characteristics || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.pleura_characteristics",
            valueType: "text",
            inputType: "select",
            options: CHEST_US_PLEURA_OPTIONS,
          },
        },
        {
          field: "Impression (free text)",
          value: chestUs?.impression_text || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.impression_text",
            valueType: "text",
          },
        },
        {
          field: "Plan (free text)",
          value: chestUs?.plan_text || "",
          __meta: {
            path: "registry.procedures_performed.chest_ultrasound.plan_text",
            valueType: "text",
          },
        },
      ],
      allowAdd: false,
      allowDelete: false,
    });
  }

  const outcomes = registry?.outcomes || {};
  const outcomesHasAny = outcomes && typeof outcomes === "object" && Object.keys(outcomes).length > 0;
  tables.push({
    id: "outcomes",
    title: "Outcomes & Disposition",
    columns: [
      { key: "field", label: "Field", readOnly: true },
      { key: "value", label: "Value", type: "text" },
    ],
    rows: [
      {
        field: "Procedure completed",
        value: toYesNo(outcomes?.procedure_completed),
        __meta: {
          path: "registry.outcomes.procedure_completed",
          valueType: "boolean",
          inputType: "select",
          options: YES_NO_OPTIONS,
        },
      },
      {
        field: "Aborted reason",
        value: outcomes?.procedure_aborted_reason || "",
        __meta: {
          path: "registry.outcomes.procedure_aborted_reason",
          valueType: "text",
        },
      },
      {
        field: "Disposition",
        value: outcomes?.disposition || "",
        __meta: {
          path: "registry.outcomes.disposition",
          valueType: "text",
          inputType: "select",
          options: DISPOSITION_OPTIONS,
        },
      },
      {
        field: "Preliminary diagnosis",
        value: outcomes?.preliminary_diagnosis || "",
        __meta: {
          path: "registry.outcomes.preliminary_diagnosis",
          valueType: "text",
        },
      },
      {
        field: "Preliminary staging",
        value: outcomes?.preliminary_staging || "",
        __meta: {
          path: "registry.outcomes.preliminary_staging",
          valueType: "text",
        },
      },
      {
        field: "Follow-up imaging ordered",
        value: toYesNo(outcomes?.follow_up_imaging_ordered),
        __meta: {
          path: "registry.outcomes.follow_up_imaging_ordered",
          valueType: "boolean",
          inputType: "select",
          options: YES_NO_OPTIONS,
        },
      },
      {
        field: "Follow-up imaging type",
        value: outcomes?.follow_up_imaging_type || "",
        __meta: {
          path: "registry.outcomes.follow_up_imaging_type",
          valueType: "text",
        },
      },
      {
        field: "Follow-up plan (free text)",
        value: outcomes?.follow_up_plan_text || "",
        __meta: {
          path: "registry.outcomes.follow_up_plan_text",
          valueType: "text",
        },
      },
    ],
    allowAdd: false,
    allowDelete: false,
    emptyMessage: outcomesHasAny ? "" : "No outcomes documented.",
  });

  const ebus = registry?.procedures_performed?.linear_ebus || {};
  const ebusPerformed = isPerformedProcedure(ebus);
  tables.push({
    id: "linear_ebus_summary",
    title: "Linear EBUS Technical Summary",
    columns: [
      { key: "field", label: "Field", readOnly: true },
      { key: "value", label: "Value", type: "text" },
    ],
    rows: [
      {
        field: "Stations sampled",
        value: ebusPerformed && Array.isArray(ebus?.stations_sampled) ? ebus.stations_sampled.join(", ") : "",
        __meta: {
          path: "registry.procedures_performed.linear_ebus.stations_sampled",
          valueType: "list",
        },
      },
      {
        field: "Needle gauge",
        value: ebusPerformed ? ebus?.needle_gauge || "" : "",
        __meta: { path: "registry.procedures_performed.linear_ebus.needle_gauge", valueType: "text" },
      },
      {
        field: "Elastography used",
        value: ebusPerformed ? toYesNo(ebus?.elastography_used) : "",
        __meta: {
          path: "registry.procedures_performed.linear_ebus.elastography_used",
          valueType: "boolean",
          inputType: "select",
          options: YES_NO_OPTIONS,
        },
      },
      {
        field: "Elastography pattern",
        value: ebusPerformed ? deriveLinearEbusElastographyPattern(ebus) : "",
        __meta: {
          path: "registry.procedures_performed.linear_ebus.elastography_pattern",
          valueType: "text",
        },
      },
    ],
    allowAdd: false,
    allowDelete: false,
  });

  const nodeEvents = Array.isArray(ebus?.node_events) ? ebus.node_events : [];
  tables.push({
    id: "ebus_node_events",
    title: "Linear EBUS Node Events",
    columns: [
      { key: "station", label: "Station", type: "text" },
      { key: "action", label: "Action", type: "select", options: EBUS_ACTION_OPTIONS },
      { key: "passes", label: "Passes", type: "number" },
      { key: "elastography_pattern", label: "Elastography", type: "text" },
      { key: "evidence_quote", label: "Evidence", type: "text" },
    ],
    rows: nodeEvents.map((ev) => ({
      station: ev?.station || "",
      action: ev?.action || "",
      passes: Number.isFinite(ev?.passes) ? String(ev.passes) : "",
      elastography_pattern: ev?.elastography_pattern || "",
      evidence_quote: ev?.evidence_quote || "",
    })),
    allowAdd: true,
    allowDelete: true,
    emptyMessage: "No node events.",
  });

  const evidence = getEvidence(data);
  const evRows = [];
  if (evidence && typeof evidence === "object") {
    Object.keys(evidence)
      .sort((a, b) => a.localeCompare(b))
      .forEach((field) => {
        const items = evidence[field];
        if (!Array.isArray(items)) return;
        items.forEach((item) => {
          const text = String(item?.text || item?.quote || "").trim();
          const span = Array.isArray(item?.span) ? item.span : null;
          evRows.push({
            field,
            evidence: text || "(evidence)",
            span: fmtSpan(span),
            confidence: typeof item?.confidence === "number" ? item.confidence.toFixed(2) : "—",
            source: item?.source || "",
          });
        });
      });
  }

  tables.push({
    id: "evidence_traceability",
    title: "Evidence Traceability (Read-only)",
    columns: [
      { key: "field", label: "Field", readOnly: true },
      { key: "evidence", label: "Evidence", readOnly: true },
      { key: "span", label: "Span", readOnly: true },
      { key: "confidence", label: "Confidence", readOnly: true },
      { key: "source", label: "Source", readOnly: true },
    ],
    rows: evRows.slice(0, 250),
    allowAdd: false,
    allowDelete: false,
    readOnly: true,
    note:
      evRows.length > 250 ? `Showing first 250 evidence items (${evRows.length} total).` : "",
  });

  return tables;
}

function renderFlattenedTables(data) {
  if (!flattenedTablesHost) return;
  if (!data) {
    flattenedTablesHost.innerHTML =
      '<div class="dash-empty" style="padding: 12px;">No results to show.</div>';
    return;
  }

  const tables = buildFlattenedTables(data);
  flatTablesBase = deepClone(tables);
  flatTablesState = deepClone(tables);
  editedDirty = false;
  fieldFeedbackStore = new Map();
  activeFieldFeedbackContext = null;
  if (editedResponseEl) editedResponseEl.textContent = "(no edits yet)";
  renderFlatTablesFromState();
}

function getFlatTableBaseById(tableId) {
  const base = Array.isArray(flatTablesBase) ? flatTablesBase : [];
  return base.find((t) => t?.id === tableId) || null;
}

function getFlatTableRowKey(table, row, rowIndex) {
  if (table?.allowAdd || table?.allowDelete) return `idx:${rowIndex}`;
  const meta = row?.__meta || {};
  if (meta.path) return `path:${meta.path}`;
  if (meta.procKey) {
    const section = meta.section === "pleural_procedures" ? "pleural_procedures" : "procedures_performed";
    return `proc:${section}:${meta.procKey}`;
  }
  return `idx:${rowIndex}`;
}

function buildBaseRowMap(table) {
  const baseTable = getFlatTableBaseById(table?.id);
  const map = new Map();
  const rows = Array.isArray(baseTable?.rows) ? baseTable.rows : [];
  rows.forEach((row, idx) => map.set(getFlatTableRowKey(table, row, idx), row));
  return map;
}

function isFlatCellModified(table, row, rowIndex, colKey, baseRowMap) {
  const baseRow = baseRowMap?.get(getFlatTableRowKey(table, row, rowIndex));
  const baseVal = baseRow ? baseRow[colKey] : "";
  const currVal = row ? row[colKey] : "";
  return String(currVal ?? "") !== String(baseVal ?? "");
}

function computeFieldFeedbackPath(table, row, rowIndex, col) {
  if (!table || !row || !col) return null;
  if (table.readOnly || col.readOnly) return null;

  const meta = row.__meta || {};

  // Common "field/value" tables.
  if (meta.path && col.key === "value") return meta.path;

  // Procedures performed summary: performed is the editable canonical signal.
  if (table.id === "procedures_summary" && col.key === "performed" && meta.procKey) {
    const section = meta.section === "pleural_procedures" ? "pleural_procedures" : "procedures_performed";
    return `registry.${section}.${meta.procKey}.performed`;
  }

  // Granular arrays.
  if (table.id === "navigation_targets" && col.key !== "target_number") {
    return `registry.granular_data.navigation_targets[${rowIndex}].${col.key}`;
  }
  if (table.id === "linear_ebus_stations_detail") {
    return `registry.granular_data.linear_ebus_stations_detail[${rowIndex}].${col.key}`;
  }
  if (table.id === "cao_interventions_detail") {
    return `registry.granular_data.cao_interventions_detail[${rowIndex}].${col.key}`;
  }
  if (table.id === "ebus_node_events") {
    return `registry.procedures_performed.linear_ebus.node_events[${rowIndex}].${col.key}`;
  }

  // Fallback (still useful for QA, even if not a registry field).
  return `ui_tables.${table.id}[${rowIndex}].${col.key}`;
}

let fieldFeedbackPathEl = null;
let fieldFeedbackValueEl = null;
let fieldFeedbackTypeEl = null;
let fieldFeedbackCorrectionEl = null;
let fieldFeedbackCommentEl = null;

function ensureFieldFeedbackModal() {
  if (fieldFeedbackModalEl) return;

  fieldFeedbackModalEl = document.createElement("dialog");
  fieldFeedbackModalEl.className = "modal";

  const form = document.createElement("form");
  form.className = "modal-content";
  form.method = "dialog";

  const title = document.createElement("h3");
  title.textContent = "Flag field for review";

  const metaBox = document.createElement("div");
  metaBox.className = "subtle";
  metaBox.style.marginBottom = "10px";

  fieldFeedbackPathEl = document.createElement("div");
  fieldFeedbackPathEl.className = "mono";
  fieldFeedbackPathEl.style.marginBottom = "6px";

  fieldFeedbackValueEl = document.createElement("div");
  fieldFeedbackValueEl.className = "subtle";

  metaBox.appendChild(fieldFeedbackPathEl);
  metaBox.appendChild(fieldFeedbackValueEl);

  const grid = document.createElement("div");
  grid.className = "field-feedback-grid";

  const typeField = document.createElement("div");
  typeField.className = "feedback-field";
  const typeLabel = document.createElement("label");
  typeLabel.textContent = "Error type";
  fieldFeedbackTypeEl = document.createElement("select");
  fieldFeedbackTypeEl.className = "flat-select";
  [
    { value: "extraction_miss", label: "Extraction Miss" },
    { value: "hallucination", label: "Hallucination" },
    { value: "wrong_value", label: "Wrong Value" },
  ].forEach((opt) => {
    const o = document.createElement("option");
    o.value = opt.value;
    o.textContent = opt.label;
    fieldFeedbackTypeEl.appendChild(o);
  });
  typeField.appendChild(typeLabel);
  typeField.appendChild(fieldFeedbackTypeEl);

  const correctionField = document.createElement("div");
  correctionField.className = "feedback-field";
  const correctionLabel = document.createElement("label");
  correctionLabel.textContent = "Correction (optional)";
  fieldFeedbackCorrectionEl = document.createElement("input");
  fieldFeedbackCorrectionEl.className = "flat-input";
  fieldFeedbackCorrectionEl.type = "text";
  correctionField.appendChild(correctionLabel);
  correctionField.appendChild(fieldFeedbackCorrectionEl);

  grid.appendChild(typeField);
  grid.appendChild(correctionField);

  const commentField = document.createElement("div");
  commentField.className = "feedback-field";
  const commentLabel = document.createElement("label");
  commentLabel.textContent = "Comment (optional)";
  fieldFeedbackCommentEl = document.createElement("textarea");
  fieldFeedbackCommentEl.className = "flat-input";
  fieldFeedbackCommentEl.rows = 4;
  commentField.appendChild(commentLabel);
  commentField.appendChild(fieldFeedbackCommentEl);

  const actions = document.createElement("div");
  actions.className = "modal-actions";

  const cancelBtnLocal = document.createElement("button");
  cancelBtnLocal.value = "cancel";
  cancelBtnLocal.className = "secondary";
  cancelBtnLocal.textContent = "Cancel";

  const removeBtn = document.createElement("button");
  removeBtn.value = "remove";
  removeBtn.className = "secondary";
  removeBtn.textContent = "Remove flag";

  const saveBtn = document.createElement("button");
  saveBtn.value = "save";
  saveBtn.className = "primary";
  saveBtn.textContent = "Save flag";

  actions.appendChild(cancelBtnLocal);
  actions.appendChild(removeBtn);
  actions.appendChild(saveBtn);

  form.appendChild(title);
  form.appendChild(metaBox);
  form.appendChild(grid);
  form.appendChild(commentField);
  form.appendChild(actions);
  fieldFeedbackModalEl.appendChild(form);

  fieldFeedbackModalEl.addEventListener("close", () => {
    const ctx = activeFieldFeedbackContext;
    const action = fieldFeedbackModalEl.returnValue;
    activeFieldFeedbackContext = null;
    if (!ctx) return;

    if (action === "save") {
      const now = new Date().toISOString();
      const prev = fieldFeedbackStore.get(ctx.path);
      const correction = String(fieldFeedbackCorrectionEl?.value || "").trim() || null;
      const comment = String(fieldFeedbackCommentEl?.value || "").trim() || null;
      const entry = {
        path: ctx.path,
        error_type: String(fieldFeedbackTypeEl?.value || "wrong_value"),
        correction,
        comment,
        table_id: ctx.tableId || null,
        column_key: ctx.columnKey || null,
        label: ctx.label || null,
        ui_current_value: String(ctx.currentValue ?? ""),
        created_at: prev?.created_at || now,
        updated_at: now,
      };
      fieldFeedbackStore.set(ctx.path, entry);
      syncFeedbackButtonsForPath(ctx.path);
      updateEditedPayload();
      setFeedbackStatus("Field flag saved.");
      return;
    }

    if (action === "remove") {
      fieldFeedbackStore.delete(ctx.path);
      syncFeedbackButtonsForPath(ctx.path);
      updateEditedPayload();
      setFeedbackStatus("Field flag removed.");
    }
  });

  document.body.appendChild(fieldFeedbackModalEl);
}

function showFieldFeedbackModal(ctx) {
  if (!ctx?.path) return;
  ensureFieldFeedbackModal();
  activeFieldFeedbackContext = ctx;

  const existing = fieldFeedbackStore.get(ctx.path);
  if (fieldFeedbackPathEl) fieldFeedbackPathEl.textContent = ctx.path;
  if (fieldFeedbackValueEl) fieldFeedbackValueEl.textContent = `Current value: ${String(ctx.currentValue ?? "")}`;
  if (fieldFeedbackTypeEl) fieldFeedbackTypeEl.value = existing?.error_type || "wrong_value";
  if (fieldFeedbackCorrectionEl) fieldFeedbackCorrectionEl.value = existing?.correction || "";
  if (fieldFeedbackCommentEl) fieldFeedbackCommentEl.value = existing?.comment || "";

  if (typeof fieldFeedbackModalEl.showModal === "function") fieldFeedbackModalEl.showModal();
  else window.alert("Your browser does not support dialogs. Please update.");
}

function syncFeedbackButtonsForPath(path) {
  const has = fieldFeedbackStore.has(path);
  const raw = String(path || "");
  const escaped =
    typeof CSS !== "undefined" && typeof CSS.escape === "function"
      ? CSS.escape(raw)
      : raw.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
  document.querySelectorAll(`button.feedback-btn[data-feedback-path="${escaped}"]`).forEach((btn) => {
    btn.classList.toggle("flagged", has);
  });
}

function getFlatTableGroupId(tableId) {
  const id = String(tableId || "");
  if (id.startsWith("coding_") || id === "rules_applied" || id === "financial_summary") return "coding";
  if (id === "audit_flags") return "quality";
  if (id === "complications_details") return "quality";
  if (id === "clinical_context") return "patient";
  if (id === "diagnostic_findings") return "diagnostics";
  if (id === "evidence_traceability") return "evidence";
  return "procedures";
}

const FLAT_TABLE_GROUP_META = [
  { id: "patient", title: "Patient & Context", defaultOpen: true },
  { id: "procedures", title: "Procedures & Technical Details", defaultOpen: true },
  { id: "diagnostics", title: "Diagnostics & Pathology", defaultOpen: true },
  { id: "coding", title: "Coding & Billing", defaultOpen: true },
  { id: "quality", title: "Audit & Quality Flags", defaultOpen: false },
  { id: "evidence", title: "Evidence (Read-only)", defaultOpen: false },
  { id: "other", title: "Other", defaultOpen: false },
];

function renderFlatTablesFromState() {
  if (!flattenedTablesHost) return;
  clearEl(flattenedTablesHost);

  const tables = Array.isArray(flatTablesState) ? flatTablesState : [];
  if (tables.length === 0) {
    flattenedTablesHost.innerHTML =
      '<div class="dash-empty" style="padding: 12px;">No tables available.</div>';
    return;
  }

  const grouped = new Map();
  tables.forEach((table) => {
    const groupId = getFlatTableGroupId(table?.id);
    if (!grouped.has(groupId)) grouped.set(groupId, []);
    grouped.get(groupId).push(table);
  });

  FLAT_TABLE_GROUP_META.forEach((group) => {
    const groupTables = grouped.get(group.id) || [];
    if (!groupTables.length) return;

    const groupEl = document.createElement("details");
    groupEl.className = "registry-group";
    groupEl.open = Boolean(group.defaultOpen);

    const summary = document.createElement("summary");
    summary.className = "registry-group-header";
    summary.textContent = group.title;
    groupEl.appendChild(summary);

    const groupBody = document.createElement("div");
    groupBody.className = "registry-group-body";

    groupTables.forEach((table) => {
      const section = document.createElement("div");
      section.className = "flat-table-section";

      const header = document.createElement("div");
      header.className = "flat-table-header";

      const title = document.createElement("div");
      title.className = "flat-table-title";
      title.textContent = table.title || table.id;

      const actions = document.createElement("div");
      actions.className = "flat-table-actions";

      if (table.allowAdd) {
        const addBtn = document.createElement("button");
        addBtn.type = "button";
        addBtn.className = "secondary row-action-btn";
        addBtn.textContent = "Add row";
        addBtn.addEventListener("click", () => {
          const newRow = {};
          table.columns.forEach((col) => {
            if (col.readOnly) return;
            if (col.type === "select" && Array.isArray(col.options) && col.options.length > 0) {
              newRow[col.key] = col.options[0].value ?? "";
            } else {
              newRow[col.key] = "";
            }
          });
          table.rows.push(newRow);
          editedDirty = true;
          renderFlatTablesFromState();
          updateEditedPayload();
        });
        actions.appendChild(addBtn);
      }

      header.appendChild(title);
      header.appendChild(actions);
      section.appendChild(header);

      const tableEl = document.createElement("table");
      tableEl.className = "flat-table";

      const thead = document.createElement("thead");
      const headRow = document.createElement("tr");
      table.columns.forEach((col) => {
        const th = document.createElement("th");
        th.textContent = col.label || col.key;
        headRow.appendChild(th);
      });
      if (table.allowDelete) {
        const th = document.createElement("th");
        th.textContent = "Remove";
        headRow.appendChild(th);
      }
      thead.appendChild(headRow);
      tableEl.appendChild(thead);

      const tbody = document.createElement("tbody");
      if (!Array.isArray(table.rows) || table.rows.length === 0) {
        const tr = document.createElement("tr");
        const td = document.createElement("td");
        td.colSpan = table.columns.length + (table.allowDelete ? 1 : 0);
        td.className = "dash-empty";
        td.textContent = table.emptyMessage || "No rows.";
        tr.appendChild(td);
        tbody.appendChild(tr);
      } else {
        const baseRowMap = buildBaseRowMap(table);

        table.rows.forEach((row, rowIndex) => {
          const tr = document.createElement("tr");

          table.columns.forEach((col) => {
            const td = document.createElement("td");
            const rawValue = row[col.key] ?? "";
            const meta = row.__meta || {};
            const inputType = meta.inputType || col.type;
            const options = meta.options || col.options;

            if (col.readOnly || table.readOnly) {
              const span = document.createElement("span");
              span.className = "flat-readonly";
              span.textContent = String(rawValue ?? "");
              td.appendChild(span);
              tr.appendChild(td);
              return;
            }

            const cellWrap = document.createElement("div");
            cellWrap.className = "cell-input-wrap";

            const isYesNo =
              Array.isArray(options) &&
              options.length === YES_NO_OPTIONS.length &&
              options.every((o, idx) => o.value === YES_NO_OPTIONS[idx].value);

            const needsToggle = meta.valueType === "boolean" || (inputType === "select" && isYesNo);

            if (needsToggle) {
              const toggle = document.createElement("div");
              toggle.className = "bool-toggle";

              const buttons = [
                { value: "", label: "—", title: "Unset" },
                { value: "Yes", label: "✅", title: "Yes" },
                { value: "No", label: "❌", title: "No" },
              ];

              const setActive = (val) => {
                Array.from(toggle.querySelectorAll("button")).forEach((btn) => {
                  btn.classList.toggle("active", btn.dataset.value === String(val ?? ""));
                });
              };

              buttons.forEach((b) => {
                const btn = document.createElement("button");
                btn.type = "button";
                btn.className = "bool-toggle-btn";
                btn.textContent = b.label;
                btn.title = b.title;
                btn.dataset.value = b.value;
                btn.addEventListener("click", () => {
                  row[col.key] = b.value;
                  editedDirty = true;
                  setActive(b.value);
                  toggle.classList.toggle(
                    "cell-modified",
                    isFlatCellModified(table, row, rowIndex, col.key, baseRowMap)
                  );
                  updateEditedPayload();
                });
                toggle.appendChild(btn);
              });

              setActive(rawValue ?? "");
              toggle.classList.toggle("cell-modified", isFlatCellModified(table, row, rowIndex, col.key, baseRowMap));
              cellWrap.appendChild(toggle);
            } else if (inputType === "select") {
              const select = document.createElement("select");
              select.className = "flat-select";
              (Array.isArray(options) ? options : []).forEach((opt) => {
                const option = document.createElement("option");
                option.value = opt.value;
                option.textContent = opt.label ?? opt.value;
                select.appendChild(option);
              });
              select.value = rawValue ?? "";
              select.classList.toggle("cell-modified", isFlatCellModified(table, row, rowIndex, col.key, baseRowMap));
              select.addEventListener("change", () => {
                row[col.key] = select.value;
                editedDirty = true;
                select.classList.toggle(
                  "cell-modified",
                  isFlatCellModified(table, row, rowIndex, col.key, baseRowMap)
                );
                updateEditedPayload();
              });
              cellWrap.appendChild(select);
            } else {
              const input = document.createElement("input");
              input.className = "flat-input";
              input.type = inputType === "number" ? "number" : "text";
              input.value = rawValue ?? "";
              input.classList.toggle("cell-modified", isFlatCellModified(table, row, rowIndex, col.key, baseRowMap));
              input.addEventListener("input", () => {
                row[col.key] = input.value;
                editedDirty = true;
                input.classList.toggle(
                  "cell-modified",
                  isFlatCellModified(table, row, rowIndex, col.key, baseRowMap)
                );
                updateEditedPayload();
              });
              cellWrap.appendChild(input);
            }

            const feedbackPath = computeFieldFeedbackPath(table, row, rowIndex, col);
            if (feedbackPath) {
              const btn = document.createElement("button");
              btn.type = "button";
              btn.className = "feedback-btn";
              btn.textContent = "🚩";
              btn.dataset.feedbackPath = feedbackPath;
              btn.title = "Flag this field for review";
              if (fieldFeedbackStore.has(feedbackPath)) btn.classList.add("flagged");
              btn.addEventListener("click", () => {
                showFieldFeedbackModal({
                  path: feedbackPath,
                  tableId: table.id,
                  columnKey: col.key,
                  label: row?.field || row?.procedure || col.label || col.key,
                  currentValue: row?.[col.key] ?? "",
                });
              });
              cellWrap.appendChild(btn);
            }

            td.appendChild(cellWrap);
            tr.appendChild(td);
          });

          if (table.allowDelete) {
            const td = document.createElement("td");
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = "secondary row-action-btn";
            btn.textContent = "Remove";
            btn.addEventListener("click", () => {
              table.rows.splice(rowIndex, 1);
              editedDirty = true;
              renderFlatTablesFromState();
              updateEditedPayload();
            });
            td.appendChild(btn);
            tr.appendChild(td);
          }
          tbody.appendChild(tr);
        });
      }

      tableEl.appendChild(tbody);
      section.appendChild(tableEl);

      if (table.note) {
        const note = document.createElement("div");
        note.className = "flat-table-note";
        note.textContent = table.note;
        section.appendChild(note);
      }

      groupBody.appendChild(section);
    });

    groupEl.appendChild(groupBody);
    flattenedTablesHost.appendChild(groupEl);
  });
}

function exportableRows(table) {
  return table.rows.map((row) => {
    const obj = {};
    table.columns.forEach((col) => {
      obj[col.key] = row[col.key] ?? "";
    });
    return obj;
  });
}

function getTablesForExport() {
  const tables = Array.isArray(flatTablesState) ? flatTablesState : [];
  return tables.map((table) => ({
    id: table.id,
    title: table.title,
    columns: table.columns,
    rows: exportableRows(table),
  }));
}

function updateEditedPayload() {
  if (!editedResponseEl) return;
  const hasFieldFeedback = fieldFeedbackStore && fieldFeedbackStore.size > 0;
  const hasRegistryGridEdits = Boolean(
    registryGridEdits?.edited_patch?.length || registryGridEdits?.edited_fields?.length
  );
  const hasCompletenessEdits = Boolean(
    completenessEdits?.edited_patch?.length || completenessEdits?.edited_fields?.length
  );
  if ((!editedDirty && !hasFieldFeedback && !hasRegistryGridEdits && !hasCompletenessEdits) || !lastServerResponse) {
    editedResponseEl.textContent = "(no edits yet)";
    editedPayload = null;
    if (exportEditedBtn) exportEditedBtn.disabled = true;
    if (exportPatchBtn) exportPatchBtn.disabled = true;
    updateFeedbackButtons();
    return;
  }

  const payload = deepClone(lastServerResponse);
  if (editedDirty && flatTablesState) applyEditsToPayload(payload, flatTablesState);
  if (hasCompletenessEdits) {
    payload.edited_completeness_patch = completenessEdits.edited_patch;
    payload.edited_completeness_fields = completenessEdits.edited_fields;
    try {
      applyJsonPatchOps(payload, completenessEdits.edited_patch);
    } catch (e) {
      console.warn("Failed to apply completeness JSON Patch (ignored).", e);
    }
  }
  if (hasRegistryGridEdits) {
    payload.edited_patch = registryGridEdits.edited_patch;
    payload.edited_fields = registryGridEdits.edited_fields;
    try {
      applyJsonPatchOps(payload, registryGridEdits.edited_patch);
    } catch (e) {
      console.warn("Failed to apply registry grid JSON Patch (ignored).", e);
    }
  }

  payload.edited_for_training = true;
  payload.edited_at = new Date().toISOString();
  const sources = [];
  if (editedDirty || hasFieldFeedback) sources.push("ui_flattened_tables");
  if (hasCompletenessEdits) sources.push("ui_completeness_prompts");
  if (hasRegistryGridEdits) sources.push("ui_registry_grid");
  payload.edited_sources = sources;
  payload.edited_source = sources.includes("ui_flattened_tables") ? "ui_flattened_tables" : sources[0] || "ui";

  if (flatTablesState && (editedDirty || hasFieldFeedback)) {
    payload.edited_tables = getTablesForExport().map((table) => ({
      id: table.id,
      title: table.title,
      rows: table.rows,
    }));
  }
  if (hasFieldFeedback) {
    payload.edited_field_feedback = Array.from(fieldFeedbackStore.values()).sort((a, b) =>
      String(a?.path || "").localeCompare(String(b?.path || ""))
    );
  }

  editedPayload = payload;
  editedResponseEl.textContent = JSON.stringify(payload, null, 2);
  if (exportEditedBtn) exportEditedBtn.disabled = !editedPayload;
  if (exportPatchBtn) exportPatchBtn.disabled = !(registryGridEdits?.edited_patch?.length > 0);
  updateFeedbackButtons();
}

function applyEditsToPayload(payload, tables) {
  const tableMap = new Map();
  (Array.isArray(tables) ? tables : []).forEach((t) => tableMap.set(t.id, t));

  const selectedRows = tableMap.get("coding_selected")?.rows || [];
  const suppressedRows = tableMap.get("coding_suppressed")?.rows || [];
  const combinedLines = [];
  let sequence = 1;

  selectedRows.forEach((row) => {
    const code = normalizeCptCode(row?.code);
    if (!code) return;
    combinedLines.push({
      sequence: sequence++,
      code,
      description: row?.description || null,
      units: parseNumber(row?.units) ?? 1,
      role: row?.role || "primary",
      selection_status: "selected",
      selection_reason: row?.rationale || null,
    });
  });

  suppressedRows.forEach((row) => {
    const code = normalizeCptCode(row?.code);
    if (!code) return;
    const status = String(row?.status || "").toLowerCase() === "suppressed" ? "suppressed" : "dropped";
    combinedLines.push({
      sequence: sequence++,
      code,
      description: null,
      units: 1,
      role: "primary",
      selection_status: status,
      selection_reason: row?.reason || null,
    });
  });

  ensurePath(payload, "registry");
  ensurePath(payload, "registry.coding_support");
  ensurePath(payload, "registry.coding_support.coding_summary");
  payload.registry.coding_support.coding_summary.lines = combinedLines;

  const rationaleRows = tableMap.get("coding_rationale")?.rows || [];
  ensurePath(payload, "registry.coding_support.coding_rationale");
  const existingRationale = Array.isArray(payload.registry.coding_support.coding_rationale.per_code)
    ? payload.registry.coding_support.coding_rationale.per_code
    : [];
  const rationaleByCode = new Map();
  existingRationale.forEach((pc) => {
    const code = normalizeCptCode(pc?.code);
    if (code) rationaleByCode.set(code, pc);
  });
  rationaleRows.forEach((row) => {
    const code = normalizeCptCode(row?.code);
    if (!code) return;
    const base = rationaleByCode.get(code) || { code };
    base.summary = row?.summary || null;
    rationaleByCode.set(code, base);
  });
  payload.registry.coding_support.coding_rationale.per_code = Array.from(rationaleByCode.values());

  const rulesRows = tableMap.get("rules_applied")?.rows || [];
  payload.registry.coding_support.coding_rationale.rules_applied = rulesRows
    .map((row) => ({
      rule_type: row?.rule_type || null,
      codes_affected: parseList(row?.codes_affected),
      outcome: String(row?.outcome || "").toLowerCase() || null,
      details: row?.details || null,
    }))
    .filter((row) => row.rule_type || row.codes_affected.length || row.details);

  const billingRows = tableMap.get("financial_summary")?.rows || [];
  const existingBilling = Array.isArray(payload.per_code_billing) ? payload.per_code_billing : [];
  const billingByCode = new Map();
  existingBilling.forEach((b) => {
    const code = normalizeCptCode(b?.cpt_code);
    if (code) billingByCode.set(code, b);
  });

  payload.per_code_billing = billingRows
    .map((row) => {
      const code = normalizeCptCode(row?.cpt_code);
      if (!code) return null;
      const base = billingByCode.get(code) || {};
      return {
        ...base,
        cpt_code: code,
        units: parseNumber(row?.units),
        work_rvu: parseNumber(row?.work_rvu),
        total_facility_rvu: parseNumber(row?.total_facility_rvu),
        facility_payment: parseNumber(row?.facility_payment),
        notes: row?.notes || null,
      };
    })
    .filter(Boolean);

  const auditRows = tableMap.get("audit_flags")?.rows || [];
  payload.audit_warnings = auditRows
    .map((row) => {
      const cat = String(row?.category || "").trim();
      const note = String(row?.notes || "").trim();
      if (!cat && !note) return null;
      return cat ? `${cat}: ${note || "—"}` : note;
    })
    .filter(Boolean);

  const clinicalRows = tableMap.get("clinical_context")?.rows || [];
  clinicalRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "boolean") value = parseYesNo(value);
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const procRows = tableMap.get("procedures_summary")?.rows || [];
  procRows.forEach((row) => {
    const meta = row.__meta || {};
    const procKey = meta.procKey;
    const section = meta.section === "pleural_procedures" ? "pleural_procedures" : "procedures_performed";
    if (!procKey) return;
    ensurePath(payload, `registry.${section}`);
    if (!payload.registry[section][procKey]) payload.registry[section][procKey] = {};
    const performed = parseYesNo(row?.performed);
    if (performed !== null) payload.registry[section][procKey].performed = performed;
  });

  const diagRows = tableMap.get("diagnostic_findings")?.rows || [];
  diagRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const balRows = tableMap.get("bal_details")?.rows || [];
  balRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const chestUsRows = tableMap.get("chest_ultrasound_details")?.rows || [];
  chestUsRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "boolean") value = parseYesNo(value);
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const outcomesRows = tableMap.get("outcomes")?.rows || [];
  outcomesRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "boolean") value = parseYesNo(value);
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const navAggRows = tableMap.get("navigation_bronchoscopy_details")?.rows || [];
  navAggRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "boolean") value = parseYesNo(value);
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const radialEbusRows = tableMap.get("radial_ebus_details")?.rows || [];
  radialEbusRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "boolean") value = parseYesNo(value);
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const fibrinolyticRows = tableMap.get("pleural_fibrinolytic_therapy")?.rows || [];
  fibrinolyticRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "boolean") value = parseYesNo(value);
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const complicationsRows = tableMap.get("complications_details")?.rows || [];
  complicationsRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "boolean") value = parseYesNo(value);
    if (meta.valueType === "number") value = parseNumber(value);
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const ebusRows = tableMap.get("linear_ebus_summary")?.rows || [];
  ebusRows.forEach((row) => {
    const meta = row.__meta || {};
    if (!meta.path) return;
    let value = row.value;
    if (meta.valueType === "boolean") value = parseYesNo(value);
    if (meta.valueType === "list") value = parseList(value);
    if (meta.valueType === "text") {
      const trimmed = String(value || "").trim();
      value = trimmed ? trimmed : null;
    }
    setByPath(payload, meta.path, value);
  });

  const nodeRows = tableMap.get("ebus_node_events")?.rows || [];
  if (nodeRows.length > 0) {
    ensurePath(payload, "registry.procedures_performed.linear_ebus");
    payload.registry.procedures_performed.linear_ebus.node_events = nodeRows
      .map((row) => ({
        station: row?.station || null,
        action: row?.action || null,
        passes: parseNumber(row?.passes),
        elastography_pattern: row?.elastography_pattern || null,
        evidence_quote: row?.evidence_quote || null,
      }))
      .filter((row) => row.station || row.action || row.passes || row.elastography_pattern || row.evidence_quote);
  }

  const navRows = tableMap.get("navigation_targets")?.rows || [];
  const existingNav = Array.isArray(payload?.registry?.granular_data?.navigation_targets)
    ? payload.registry.granular_data.navigation_targets
    : [];
  if (navRows.length > 0) {
    ensurePath(payload, "registry.granular_data");
    payload.registry.granular_data.navigation_targets = navRows.map((row, idx) => {
      const base = existingNav[idx] && typeof existingNav[idx] === "object" ? existingNav[idx] : {};
      const out = { ...base };

      out.target_number = base?.target_number ?? idx + 1;

      const loc = String(row?.target_location_text || "").trim();
      if (loc) out.target_location_text = loc;

      const lobe = String(row?.target_lobe || "").trim();
      out.target_lobe = lobe ? lobe : null;

      const segment = String(row?.target_segment || "").trim();
      out.target_segment = segment ? segment : null;

      const size = parseNumber(row?.lesion_size_mm);
      out.lesion_size_mm = size;

      const dist = parseNumber(row?.distance_from_pleura_mm);
      out.distance_from_pleura_mm = dist;

      const ct = String(row?.ct_characteristics || "").trim();
      out.ct_characteristics = ct ? ct : null;

      const suv = parseNumber(row?.pet_suv_max);
      out.pet_suv_max = suv;

      const bs = String(row?.bronchus_sign || "").trim();
      out.bronchus_sign = bs ? bs : null;

      const regErr = parseNumber(row?.registration_error_mm);
      out.registration_error_mm = regErr;

      const view = String(row?.rebus_view || "").trim();
      out.rebus_view = view ? view : null;

      const til = parseYesNo(row?.tool_in_lesion_confirmed);
      out.tool_in_lesion_confirmed = til;

      const method = String(row?.confirmation_method || "").trim();
      out.confirmation_method = method ? method : null;

      const tools = parseList(row?.sampling_tools_used);
      out.sampling_tools_used = tools.length ? tools : null;

      const needlePasses = parseNumber(row?.number_of_needle_passes);
      out.number_of_needle_passes = needlePasses === null ? null : Math.trunc(needlePasses);

      const forceps = parseNumber(row?.number_of_forceps_biopsies);
      out.number_of_forceps_biopsies = forceps === null ? null : Math.trunc(forceps);

      const cryo = parseNumber(row?.number_of_cryo_biopsies);
      out.number_of_cryo_biopsies = cryo === null ? null : Math.trunc(cryo);

      const roseResult = String(row?.rose_result || "").trim();
      out.rose_result = roseResult ? roseResult : null;
      out.rose_performed = roseResult ? true : null;

      const notes = String(row?.notes || "").trim();
      out.notes = notes ? notes : null;

      return out;
    });
  }

  const stationRows = tableMap.get("linear_ebus_stations_detail")?.rows || [];
  const existingStations = Array.isArray(payload?.registry?.granular_data?.linear_ebus_stations_detail)
    ? payload.registry.granular_data.linear_ebus_stations_detail
    : [];
  if (stationRows.length > 0) {
    ensurePath(payload, "registry.granular_data");
    payload.registry.granular_data.linear_ebus_stations_detail = stationRows.map((row, idx) => {
      const base = existingStations[idx] && typeof existingStations[idx] === "object" ? existingStations[idx] : {};
      const out = { ...base };

      const station = String(row?.station || "").trim();
      if (station) out.station = station;

      const sampled = parseYesNo(row?.sampled);
      out.sampled = sampled;

      const shortAxis = parseNumber(row?.short_axis_mm);
      out.short_axis_mm = shortAxis;

      const longAxis = parseNumber(row?.long_axis_mm);
      out.long_axis_mm = longAxis;

      const shape = String(row?.shape || "").trim();
      out.shape = shape ? shape : null;

      const margin = String(row?.margin || "").trim();
      out.margin = margin ? margin : null;

      const echo = String(row?.echogenicity || "").trim();
      out.echogenicity = echo ? echo : null;

      const chs = parseYesNo(row?.chs_present);
      out.chs_present = chs;

      const nec = parseYesNo(row?.necrosis_present);
      out.necrosis_present = nec;

      const calc = parseYesNo(row?.calcification_present);
      out.calcification_present = calc;

      const gauge = String(row?.needle_gauge || "").trim();
      out.needle_gauge = gauge ? gauge : null;

      const passes = parseNumber(row?.number_of_passes);
      out.number_of_passes = passes === null ? null : Math.trunc(passes);

      const rose = String(row?.rose_result || "").trim();
      out.rose_result = rose ? rose : null;

      const lymph = parseYesNo(row?.lymphocytes_present);
      out.lymphocytes_present = lymph;

      const imp = String(row?.morphologic_impression || "").trim();
      out.morphologic_impression = imp ? imp : null;

      return out;
    });
  }

  const caoRows = tableMap.get("cao_interventions_detail")?.rows || [];
  const existingCao = Array.isArray(payload?.registry?.granular_data?.cao_interventions_detail)
    ? payload.registry.granular_data.cao_interventions_detail
    : [];
  if (caoRows.length > 0) {
    ensurePath(payload, "registry.granular_data");
    payload.registry.granular_data.cao_interventions_detail = caoRows.map((row, idx) => {
      const base = existingCao[idx] && typeof existingCao[idx] === "object" ? existingCao[idx] : {};
      const out = { ...base };

      const loc = String(row?.location || "").trim();
      if (loc) out.location = loc;

      const obstruction = String(row?.obstruction_type || "").trim();
      out.obstruction_type = obstruction ? obstruction : null;

      const etiology = String(row?.etiology || "").trim();
      out.etiology = etiology ? etiology : null;

      const morph = String(row?.lesion_morphology || "").trim();
      out.lesion_morphology = morph ? morph : null;

      const countText = String(row?.lesion_count_text || "").trim();
      out.lesion_count_text = countText ? countText : null;

      const length = parseNumber(row?.length_mm);
      out.length_mm = length;

      const prePct = parseNumber(row?.pre_obstruction_pct);
      out.pre_obstruction_pct = prePct === null ? null : Math.trunc(prePct);

      const postPct = parseNumber(row?.post_obstruction_pct);
      out.post_obstruction_pct = postPct === null ? null : Math.trunc(postPct);

      const preDiam = parseNumber(row?.pre_diameter_mm);
      out.pre_diameter_mm = preDiam;

      const postDiam = parseNumber(row?.post_diameter_mm);
      out.post_diameter_mm = postDiam;

      const stent = parseYesNo(row?.stent_placed_at_site);
      out.stent_placed_at_site = stent;

      const hemo = parseYesNo(row?.hemostasis_required);
      out.hemostasis_required = hemo;

      const notes = String(row?.notes || "").trim();
      out.notes = notes ? notes : null;

      return out;
    });
  }
}

function getOptionLabel(options, value) {
  if (!Array.isArray(options)) return value ?? "";
  const match = options.find((opt) => String(opt.value) === String(value));
  return match ? match.label ?? match.value : value ?? "";
}

function buildExcelHtml(tables) {
  const blocks = tables.map((table) => {
    const headers = table.columns.map((c) => `<th>${safeHtml(c.label || c.key)}</th>`).join("");
    const rows = table.rows
      .map((row) => {
        const cells = table.columns
          .map((c) => {
            const raw = row[c.key] ?? "";
            const display =
              c.type === "select" ? getOptionLabel(c.options, raw) : raw;
            return `<td>${safeHtml(display ?? "")}</td>`;
          })
          .join("");
        return `<tr>${cells}</tr>`;
      })
      .join("");
    return `
      <h3>${safeHtml(table.title || table.id)}</h3>
      <table border="1">
        <thead><tr>${headers}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
      <br />
    `;
  });

  return `
    <html>
      <head><meta charset="UTF-8"></head>
      <body>${blocks.join("")}</body>
    </html>
  `;
}

function exportTablesToExcel() {
  const tables = getTablesForExport();
  if (!tables || tables.length === 0) {
    setStatus("No tables to export");
    return;
  }
  const html = buildExcelHtml(tables);
  const blob = new Blob([html], { type: "application/vnd.ms-excel" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `procedure_suite_tables_${Date.now()}.xls`;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
  setStatus("Exported tables");
}

/**
 * Transforms API data into a unified "Golden Record"
 * FIX: Now prioritizes backend rationale over generic placeholders
 */
function transformToUnifiedTable(rawData) {
  const unifiedMap = new Map();

  // Helper: Get explanation from specific coding_support backend map
  const getBackendRationale = (code) => {
    if (rawData.registry?.coding_support?.code_rationales?.[code]) {
      return rawData.registry.coding_support.code_rationales[code];
    }
    // Fallback to "evidence" array if available
    const billingEntry = rawData.registry?.billing?.cpt_codes?.find((c) => c.code === code);
    if (billingEntry?.evidence?.length > 0) {
      return billingEntry.evidence.map((e) => e.text).join("; ");
    }
    return null;
  };

  // 1. Process Header Codes (Raw)
  (rawData.header_codes || []).forEach((item) => {
    unifiedMap.set(item.code, {
      code: item.code,
      desc: item.description || "Unknown Procedure",
      inHeader: true,
      inBody: false,
      status: "pending",
      rationale: "Found in header scan",
      rvu: 0.0,
      payment: 0.0,
    });
  });

  // 2. Process Derived Codes (Body)
  (rawData.derived_codes || []).forEach((item) => {
    const existing = unifiedMap.get(item.code) || {
      code: item.code,
      inHeader: false,
      rvu: 0.0,
      payment: 0.0,
    };

    existing.desc = item.description || existing.desc;
    existing.inBody = true;

    // FIX: Grab specific backend rationale if available
    const backendReason = getBackendRationale(item.code);
    if (backendReason) {
      existing.rationale = backendReason;
    } else {
      existing.rationale = "Derived from procedure actions";
    }

    unifiedMap.set(item.code, existing);
  });

  // 3. Process Final Selection (The "Truth")
  (rawData.per_code_billing || []).forEach((item) => {
    const existing = unifiedMap.get(item.cpt_code) || {
      code: item.cpt_code,
      inHeader: false,
      inBody: true,
      rationale: "Selected",
    };

    existing.code = item.cpt_code; // Ensure code is set
    existing.desc = item.description || existing.desc;
    existing.status = item.status || "selected";
    existing.rvu = item.work_rvu;
    existing.payment = item.facility_payment;

    // FIX: Ensure suppression/bundling logic is visible
    if (item.work_rvu === 0) {
      existing.status = "Bundled/Suppressed";
      // If we have a specific bundling warning, append it
      const warning = (rawData.audit_warnings || []).find((w) => w.includes(item.cpt_code));
      if (warning) existing.rationale = warning;
    } else {
      // Refresh rationale from backend to ensure it's not "Derived..."
      const backendReason = getBackendRationale(item.cpt_code);
      if (backendReason) existing.rationale = backendReason;
    }

    unifiedMap.set(item.cpt_code, existing);
  });

  // Sort: High Value -> Suppressed -> Header Only
  return Array.from(unifiedMap.values()).sort((a, b) => {
    if (a.rvu > 0 && b.rvu === 0) return -1;
    if (b.rvu > 0 && a.rvu === 0) return 1;
    return a.code.localeCompare(b.code);
  });
}

/**
 * 2. Renders the Unified Billing Reconciliation Table
 * Merges Header, Derived, and Final codes into one view.
 */
function renderUnifiedTable(data) {
  const tbody = document.getElementById("unifiedTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const sortedRows = transformToUnifiedTable(data);

  // Render Rows
  sortedRows.forEach((row) => {
    const tr = document.createElement("tr");

    // Logic Badges
    let sourceBadge = "";
    if (row.inHeader && row.inBody) sourceBadge = `<span class="badge badge-both">Match</span>`;
    else if (row.inHeader) sourceBadge = `<span class="badge badge-header">Header Only</span>`;
    else sourceBadge = `<span class="badge badge-body">Derived</span>`;

    // Status Badge
    let statusBadge = `<span class="badge badge-primary">Primary</span>`;
    if (row.rvu === 0 || row.status === "Bundled/Suppressed") {
      statusBadge = `<span class="badge badge-bundled">Bundled</span>`;
      tr.classList.add("row-suppressed");
    }

    // Rationale cleaning
    const rationale = row.rationale || (row.inBody ? "Derived from procedure actions" : "Found in header scan");
    const rvuDisplay = Number.isFinite(row.rvu) ? row.rvu.toFixed(2) : "0.00";
    const paymentDisplay = Number.isFinite(row.payment) ? row.payment.toFixed(2) : "0.00";

    tr.innerHTML = `
      <td><span class="code-cell">${row.code}</span></td>
      <td>
        <span class="desc-text">${row.desc || "Unknown Procedure"}</span>
        ${sourceBadge}
      </td>
      <td>${statusBadge}</td>
      <td><span class="rationale-text">${rationale}</span></td>
      <td><strong>${rvuDisplay}</strong></td>
      <td>$${paymentDisplay}</td>
    `;
    tbody.appendChild(tr);
  });
}

/**
 * Renders the Clinical/Registry Data Table (Restored)
 * Flattens nested registry objects into a clean key-value view.
 */
function renderRegistrySummary(data) {
  const tbody = document.getElementById("registryTableBody");
  if (!tbody) return;
  tbody.innerHTML = "";

  const registry = data.registry || {};

  // 1. Clinical Context (Top Priority)
  if (registry.clinical_context) {
    addRegistryRow(tbody, "Indication", registry.clinical_context.primary_indication);
    if (registry.clinical_context.indication_category) {
      addRegistryRow(tbody, "Category", registry.clinical_context.indication_category);
    }
  }

  // 2. Anesthesia/Sedation
  if (registry.sedation) {
    const sedationStr = `${registry.sedation.type || "Not specified"} (${registry.sedation.anesthesia_provider || "Provider unknown"})`;
    addRegistryRow(tbody, "Sedation", sedationStr);
  }

  // 3. EBUS Details (Granular)
  if (registry.procedures_performed?.linear_ebus?.performed) {
    const ebus = registry.procedures_performed.linear_ebus;
    const stations = Array.isArray(ebus.stations_sampled) ? ebus.stations_sampled.join(", ") : "None";
    const needle = ebus.needle_gauge || "Not specified";
    addRegistryRow(
      tbody,
      "Linear EBUS",
      `<strong>Stations:</strong> ${stations} <br> <span style="font-size:11px; color:#64748b;">Gauge: ${needle} | Elastography: ${ebus.elastography_used ? "Yes" : "No"}</span>`
    );
  }

  // 4. Other Procedures (Iterate generic performed flags)
  const procs = registry.procedures_performed || {};
  Object.keys(procs).forEach((key) => {
    if (key === "linear_ebus") return; // Handled above
    const p = procs[key];
    if (p === true || (p && p.performed)) {
      // Convert snake_case to Title Case (e.g., radial_ebus -> Radial Ebus)
      const label = key.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase());

      // Extract useful details if they exist (e.g., "lobes", "sites")
      let details = "Performed";
      if (p?.sites) details = `Sites: ${Array.isArray(p.sites) ? p.sites.join(", ") : p.sites}`;
      else if (p?.target_lobes) details = `Lobes: ${p.target_lobes.join(", ")}`;
      else if (p?.action) details = p.action;

      addRegistryRow(tbody, label, details);
    }
  });
}

// Helper to append rows
function addRegistryRow(tbody, label, content) {
  if (!content) return;
  const tr = document.createElement("tr");
  tr.innerHTML = `
    <td style="font-weight:600; color:#475569;">${label}</td>
    <td>${content}</td>
  `;
  tbody.appendChild(tr);
}

/**
 * 3. Renders Technical Logs (Collapsed by default)
 */
function renderDebugLogs(data) {
  const logBox = document.getElementById("systemLogs");
  if (!logBox) return;

  let logs = [];

  // Collect all warnings and logs
  if (data.audit_warnings) logs.push(...data.audit_warnings.map((w) => `[AUDIT] ${w}`));
  if (data.warnings) logs.push(...data.warnings.map((w) => `[WARN] ${w}`));
  if (data.self_correction) {
    data.self_correction.forEach((sc) => {
      logs.push(`[SELF-CORRECT] Applied patch for ${sc.trigger.target_cpt}: ${sc.trigger.reason}`);
    });
  }

  if (logs.length === 0) {
    logBox.textContent = "No system warnings or overrides.";
  } else {
    logBox.textContent = logs.join("\n");
  }
}

/**
 * Render the formatted results from the server response.
 * Shows status banner, CPT codes table, and registry form.
 */
async function renderResults(data, options = {}) {
  const container = document.getElementById("resultsContainer");
  if (!container) return;

  const rawData = options?.rawData ?? data;
  lastServerResponse = data;
  if (exportBtn) exportBtn.disabled = !data;
  if (exportTablesBtn) exportTablesBtn.disabled = !data;
  if (newNoteBtn) newNoteBtn.disabled = !data;

  container.classList.remove("hidden");
  const preferGrid = isReactRegistryGridEnabled() && Boolean(registryGridRootEl && registryLegacyRightRootEl);

  // Always render the legacy dashboard (CPT + RVU tables stay on the left).
  renderDashboard(data);
  renderFlattenedTables(data);

  if (preferGrid) {
    await maybeRenderRegistryGrid(data);
  } else {
    unmountRegistryGrid();
    showRegistryLegacyUi();
  }

  if (serverResponseEl) {
    serverResponseEl.textContent = JSON.stringify(rawData, null, 2);
  }
}

function clearResultsUi() {
  const container = document.getElementById("resultsContainer");
  if (container) container.classList.add("hidden");
  unmountRegistryGrid();
  showRegistryLegacyUi();
  if (registryGridRootEl) registryGridRootEl.innerHTML = "";
  if (bundleSummaryHostEl) {
    bundleSummaryHostEl.classList.add("hidden");
    bundleSummaryHostEl.innerHTML = "";
  }
  lastServerResponse = null;
  lastCompletenessPrompts = [];
  if (completenessPromptsCardEl) completenessPromptsCardEl.classList.add("hidden");
  if (completenessPromptsBodyEl) clearEl(completenessPromptsBodyEl);
  if (completenessCopyBtn) completenessCopyBtn.disabled = true;
  resetRunState();
  if (serverResponseEl) serverResponseEl.textContent = "(none)";
  if (flattenedTablesHost) {
    flattenedTablesHost.innerHTML =
      '<div class="dash-empty" style="padding: 12px;">No results yet.</div>';
  }
  if (exportBtn) exportBtn.disabled = true;
  if (exportTablesBtn) exportTablesBtn.disabled = true;
  if (exportEditedBtn) exportEditedBtn.disabled = true;
  if (exportPatchBtn) exportPatchBtn.disabled = true;
  if (newNoteBtn) newNoteBtn.disabled = true;
  resetEditedState();
}

function renderStatusBanner(data, container) {
  const statusBanner = document.createElement("div");

  if (data.needs_manual_review) {
    statusBanner.className = "status-banner error";
    statusBanner.textContent = "⚠️ Manual review required";
  } else if (data.audit_warnings?.length > 0) {
    statusBanner.className = "status-banner warning";
    statusBanner.textContent = `⚠️ ${data.audit_warnings.length} warning(s) - review recommended`;
  } else {
    statusBanner.className = "status-banner success";
    const difficulty = data.coder_difficulty || "HIGH_CONF";
    statusBanner.textContent = `✓ High confidence extraction (${difficulty})`;
  }

  container.appendChild(statusBanner);
}

// --- Helper: Create Section Wrapper ---
function createSection(title, icon) {
  const div = document.createElement('div');
  div.className = 'report-section';
  div.innerHTML = `<div class="report-header">${icon} ${title}</div><div class="report-body"></div>`;
  return div;
}

function buildQaByCode(codingSupport) {
  const qaByCode = {};
  const perCode = codingSupport?.coding_rationale?.per_code || [];
  perCode.forEach((pc) => {
    if (!pc?.code) return;
    qaByCode[pc.code] = pc.qa_flags || [];
  });
  return qaByCode;
}

function renderCPTRawHeader(registry, codingSupport, data) {
  const section = createSection("CPT Codes – Raw (Header)", "🧾");

  const ev = getEvidenceMap(data);
  const header = ev.code_evidence || []; // [{text,start,end}] in schema
  if (!Array.isArray(header) || header.length === 0) {
    section.querySelector(
      ".report-body"
    ).innerHTML = `<div class="subtle" style="text-align:center;">No header CPT codes found</div>`;
    return section;
  }

  const derivedSet = new Set((registry?.billing?.cpt_codes || []).map((c) => c.code));
  const decisionByCode = new Map(
    (codingSupport?.coding_summary?.lines || []).map((ln) => [
      ln.code,
      (ln.selection_status || "selected").toLowerCase(),
    ])
  );

  const rows = header
    .map((ce) => {
      const code = typeof ce === "string" ? ce : ce?.text;
      if (!code) return "";

      const status = (
        decisionByCode.get(code) || (derivedSet.has(code) ? "derived_only" : "header_only")
      ).toLowerCase();
      const bodyEv = derivedSet.has(code) ? "Yes" : "No";
      const evidenceHtml = typeof ce === "string" ? "—" : renderEvidenceChips([ce]);

      return `
      <tr>
        <td><strong>${safeHtml(code)}</strong></td>
        <td><span class="status-badge status-${safeHtml(status)}">${safeHtml(status)}</span></td>
        <td>${bodyEv}</td>
        <td>${evidenceHtml}</td>
      </tr>
    `;
    })
    .filter(Boolean)
    .join("");

  section.querySelector(".report-body").innerHTML = `
    <table class="data-table">
      <thead><tr><th>Code</th><th>Status</th><th>Body evidence?</th><th>Evidence</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  return section;
}

function renderCPTDerivedEvidence(registry, data) {
  const section = createSection("CPT Codes – Derived (Body Evidence)", "🔎");
  const derived = registry?.billing?.cpt_codes || [];

  if (!Array.isArray(derived) || derived.length === 0) {
    section.querySelector(
      ".report-body"
    ).innerHTML = `<div class="subtle" style="text-align:center;">No derived CPT codes</div>`;
    return section;
  }

  const rows = derived
    .map((c) => {
      const derivedFrom = Array.isArray(c.derived_from) ? c.derived_from.join(", ") : "";
      return `
    <tr>
      <td><strong>${safeHtml(c.code)}</strong></td>
      <td>${safeHtml(c.description || "-")}</td>
      <td>${safeHtml(derivedFrom || "-")}</td>
      <td>${renderEvidenceChips(c.evidence || [])}</td>
    </tr>
  `;
    })
    .join("");

  section.querySelector(".report-body").innerHTML = `
    <table class="data-table">
      <thead><tr><th>Code</th><th>Description</th><th>Derived From</th><th>Evidence</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
  return section;
}

// --- 2. CPT Coding Summary ---
function renderCPTSummary(lines, qaByCode = {}) {
  const section = createSection('CPT Coding Summary (Final Selection)', '💳');
  if (!Array.isArray(lines) || lines.length === 0) {
    section.querySelector('.report-body').innerHTML = `
      <div class="subtle" style="text-align:center;">No CPT summary lines returned</div>
    `;
    return section;
  }
  const tbody = lines
    .map((line) => {
      const code = line.code || "-";
      const qa =
        (qaByCode[code] || [])
          .map(
            (q) =>
              `<div class="qa-line">${safeHtml(
                (q.severity || "info").toUpperCase()
              )}: ${safeHtml(q.message || "")}</div>`
          )
          .join("") || "—";

      const evidence = renderEvidenceChips(line.note_spans || []);

      return `
        <tr class="${line.selection_status === 'dropped' ? 'opacity-50' : ''}">
            <td>${safeHtml(line.sequence ?? '-')}</td>
            <td><strong>${safeHtml(code)}</strong></td>
            <td>${safeHtml(line.description || '-')}</td>
            <td>${safeHtml(line.units ?? '-')}</td>
            <td><span class="status-badge ${
              line.role === "primary" ? "role-primary" : "role-addon"
            }">${safeHtml(line.role || "-")}</span></td>
            <td><span class="status-badge status-${safeHtml(
              (line.selection_status || "selected").toLowerCase()
            )}">${safeHtml(line.selection_status || "selected")}</span></td>
            <td>${safeHtml(line.selection_reason || '-')}</td>
            <td>${evidence}</td>
            <td>${qa}</td>
        </tr>
    `;
    })
    .join('');

  section.querySelector('.report-body').innerHTML = `
        <table class="data-table">
            <thead>
                <tr>
                    <th>Seq</th><th>CPT Code</th><th>Description</th><th>Units</th><th>Role</th><th>Status</th><th>Selection Rationale</th><th>Evidence</th><th>QA</th>
                </tr>
            </thead>
            <tbody>${tbody}</tbody>
        </table>
    `;
  return section;
}

// --- 3. Bundling & Suppression Decisions ---
function renderBundlingDecisions(rules) {
  const section = createSection('Bundling & Suppression Decisions', '🧾');

  // Filter for rules that actually affected codes (dropped/informational)
  const validRules = rules.filter(r => Array.isArray(r.codes_affected) && r.codes_affected.length > 0);

  if (validRules.length === 0) return document.createDocumentFragment();

  const tbody = validRules.map(rule => `
        <tr>
            <td>${rule.codes_affected.join(', ')}</td>
            <td><span class="status-badge status-${rule.outcome === 'dropped' ? 'dropped' : 'suppressed'}">${rule.outcome === 'dropped' ? 'Dropped' : 'Suppressed'}</span></td>
            <td>${rule.details || '-'}</td>
        </tr>
    `).join('');

  section.querySelector('.report-body').innerHTML = `
        <table class="data-table">
            <thead><tr><th>CPT Code</th><th>Action</th><th>Reason</th></tr></thead>
            <tbody>${tbody}</tbody>
        </table>
    `;
  return section;
}

// --- 4. RVU & Payment Summary ---
function renderRVUSummary(billingLines, totalRVU, totalPay) {
  const section = createSection('RVU & Payment Summary', '💰');

  const rows = (Array.isArray(billingLines) ? billingLines : []).map(line => `
        <tr>
            <td><strong>${line.cpt_code || '-'}</strong></td>
            <td>${line.units ?? '-'}</td>
            <td>${formatNumber(line.work_rvu)}</td>
            <td>${formatNumber(line.total_facility_rvu)}</td>
            <td>${formatCurrency(line.facility_payment)}</td>
        </tr>
    `).join('');

  const totals = `
        <tr class="totals-row">
            <td colspan="2" style="text-align:right">Totals:</td>
            <td>${formatNumber(totalRVU)}</td>
            <td>-</td>
            <td>${formatCurrency(totalPay)}</td>
        </tr>
    `;

  section.querySelector('.report-body').innerHTML = `
        <table class="data-table">
            <thead>
                <tr><th>CPT Code</th><th>Units</th><th>Work RVU</th><th>Facility RVU</th><th>Est. Payment ($)</th></tr>
            </thead>
            <tbody>${rows}${totals}</tbody>
        </table>
    `;
  return section;
}

// --- 5. Clinical Context ---
function renderClinicalContext(registry, data) {
  const section = createSection('Clinical Context', '🩺');
  const ev = getEvidenceMap(data);

  const rows = [];

  if (registry.clinical_context?.primary_indication) {
    rows.push([
      "Primary Indication",
      cleanIndicationForDisplay(registry.clinical_context.primary_indication),
      renderEvidenceChips(ev["clinical_context.primary_indication"] || []),
    ]);
  }
  if (registry.clinical_context?.bronchus_sign) {
    rows.push(["Bronchus Sign", registry.clinical_context.bronchus_sign, "—"]);
  }
  if (registry.sedation?.type) {
    rows.push([
      "Sedation Type",
      registry.sedation.type,
      renderEvidenceChips(ev["sedation.type"] || []),
    ]);
  }
  if (registry.procedure_setting?.airway_type) {
    rows.push([
      "Airway Type",
      registry.procedure_setting.airway_type,
      renderEvidenceChips(ev["procedure_setting.airway_type"] || []),
    ]);
  }

  if (rows.length === 0) {
    section.querySelector(
      ".report-body"
    ).innerHTML = `<div class="subtle" style="text-align:center;">No clinical context found</div>`;
    return section;
  }

  section.querySelector(".report-body").innerHTML = `
    <table class="data-table">
      <thead><tr><th style="width:25%">Field</th><th>Value</th><th>Evidence</th></tr></thead>
      <tbody>
        ${rows
          .map(
            ([k, v, e]) =>
              `<tr><td><strong>${safeHtml(k)}</strong></td><td>${safeHtml(
                v
              )}</td><td>${e}</td></tr>`
          )
          .join("")}
      </tbody>
    </table>
  `;
  return section;
}

// --- 6. Procedures Performed ---
function renderProceduresSection(procedures, data) {
  const container = document.createElement('div');
  const ev = getEvidenceMap(data);

  // A. Main Procedures List
  const summarySection = createSection('Procedures Performed', '🔍');
  const summaryRows = Object.entries(procedures).map(([key, proc]) => {
    if (!proc || proc.performed !== true) return '';
    const name = key.replace(/_/g, ' ').toUpperCase();
    const evKey = `procedures_performed.${key}.performed`;
    const evidenceHtml = renderEvidenceChips(ev[evKey] || []);

    // Extract key details based on procedure type
    let details = [];
    if (typeof proc.inspection_findings === "string") {
      const snippet = proc.inspection_findings.length > 50
        ? `${proc.inspection_findings.substring(0, 50)}...`
        : proc.inspection_findings;
      details.push(`Findings: ${snippet}`);
    }
    if (proc.material) details.push(`Material: ${proc.material}`);
    if (Array.isArray(proc.stations_sampled))
      details.push(`Stations: ${proc.stations_sampled.join(', ')}`);

    return `
            <tr>
                <td><strong>${safeHtml(name)}</strong></td>
                <td><span class="status-badge status-selected">Yes</span></td>
                <td>${safeHtml(details.join('; ') || '-')}</td>
                <td>${evidenceHtml}</td>
            </tr>
        `;
  }).filter(Boolean).join('');

  summarySection.querySelector('.report-body').innerHTML = `
        <table class="data-table">
            <thead><tr><th>Procedure</th><th>Performed</th><th>Key Details</th><th>Evidence</th></tr></thead>
            <tbody>${summaryRows}</tbody>
        </table>
    `;
  container.appendChild(summarySection);

  // B. Special Linear EBUS Detail Table
  if (procedures.linear_ebus && procedures.linear_ebus.performed) {
    const ebus = procedures.linear_ebus;
    const ebusSection = createSection('Linear EBUS Details', '📊');

    // General Attributes
    const stationsSampled = Array.isArray(ebus.stations_sampled) ? ebus.stations_sampled.join(", ") : "-";
    let attrRows = `
            <tr><td><strong>Stations Sampled</strong></td><td>${stationsSampled}</td></tr>
            <tr><td><strong>Needle Gauge</strong></td><td>${ebus.needle_gauge || '-'}</td></tr>
            <tr><td><strong>Elastography Used</strong></td><td>${ebus.elastography_used ? 'Yes' : 'No'}</td></tr>
        `;

    let nodeTable = '';
    if (Array.isArray(ebus.node_events) && ebus.node_events.length > 0) {
      const nodeRows = ebus.node_events.map(ev => `
                <tr>
                    <td>${ev.station || '-'}</td>
                    <td>${ev.action || '-'}</td>
                    <td style="font-style:italic; color:#64748b">"${ev.evidence_quote || ''}"</td>
                </tr>
            `).join('');

      nodeTable = `
                <h4 style="margin:1rem 0 0.5rem; font-size:0.9rem; color:#475569">Node Events</h4>
                <table class="data-table">
                    <thead><tr><th>Station</th><th>Action</th><th>Documentation Evidence</th></tr></thead>
                    <tbody>${nodeRows}</tbody>
                </table>
            `;
    }

    ebusSection.querySelector('.report-body').innerHTML = `
            <table class="data-table" style="margin-bottom:1rem">
                <thead><tr><th style="width:30%">Attribute</th><th>Value</th></tr></thead>
                <tbody>${attrRows}</tbody>
            </table>
            ${nodeTable}
        `;
    container.appendChild(ebusSection);
  }

  return container;
}

// --- 7. Audit & QA Notes ---
function renderAuditNotes(warnings) {
  const section = createSection('Audit & QA Notes (Condensed)', '⚠️');

  // Simple heuristic to categorize strings like "CATEGORY: Message"
  const parsed = warnings.map(w => {
    const match = w.match(/^([A-Z_]+):\s*(.+)$/);
    return match
      ? { cat: match[1], msg: match[2] }
      : { cat: 'General', msg: w };
  });

  const rows = parsed.map(item => `
        <tr>
            <td style="width:25%"><strong>${item.cat}</strong></td>
            <td>${item.msg}</td>
        </tr>
    `).join('');

  section.querySelector('.report-body').innerHTML = `
        <table class="data-table">
            <thead><tr><th>Category</th><th>Message</th></tr></thead>
            <tbody>${rows}</tbody>
        </table>
    `;
  return section;
}

/**
 * Render the CPT codes table with descriptions, confidence, RVU, and payment.
 */
function renderLegacyCPTTable(data) {
  const section = createSection("CPT Codes", "💳");
  const suggestions = data.suggestions || [];
  const billing = data.per_code_billing || [];

  // Create billing lookup
  const billingMap = {};
  billing.forEach(b => billingMap[b.cpt_code] = b);

  let rows = "";

  // If we have suggestions, use those (more detailed)
  if (suggestions.length > 0) {
    suggestions.forEach((s) => {
      const b = billingMap[s.code] || {};
      const confidence = s.confidence ? `${(s.confidence * 100).toFixed(0)}%` : "—";
      const rvu = b.work_rvu?.toFixed(2) || "—";
      const payment = b.facility_payment ? `$${b.facility_payment.toFixed(2)}` : "—";

      rows += `
        <tr>
          <td><code>${s.code}</code></td>
          <td>${s.description || "—"}</td>
          <td>${confidence}</td>
          <td>${rvu}</td>
          <td>${payment}</td>
        </tr>
      `;
    });
  } else if (data.cpt_codes?.length > 0) {
    // Fallback to simple cpt_codes list
    data.cpt_codes.forEach((code) => {
      const b = billingMap[code] || {};
      const rvu = b.work_rvu?.toFixed(2) || "—";
      const payment = b.facility_payment ? `$${b.facility_payment.toFixed(2)}` : "—";

      rows += `
        <tr>
          <td><code>${code}</code></td>
          <td>${b.description || "—"}</td>
          <td>—</td>
          <td>${rvu}</td>
          <td>${payment}</td>
        </tr>
      `;
    });
  } else {
    rows = '<tr><td colspan="5" class="subtle" style="text-align: center;">No CPT codes returned</td></tr>';
  }

  // Totals row
  if (data.total_work_rvu || data.estimated_payment) {
    const totalRvu = data.total_work_rvu?.toFixed(2) || "—";
    const totalPayment = data.estimated_payment ? `$${data.estimated_payment.toFixed(2)}` : "—";
    rows += `
      <tr class="totals-row">
        <td colspan="3"><strong>TOTALS</strong></td>
        <td><strong>${totalRvu}</strong></td>
        <td><strong>${totalPayment}</strong></td>
      </tr>
    `;
  }

  section.querySelector(".report-body").innerHTML = `
    <table class="data-table">
      <thead>
        <tr>
          <th>Code</th>
          <th>Description</th>
          <th>Confidence</th>
          <th>RVU</th>
          <th>Payment</th>
        </tr>
      </thead>
      <tbody>${rows}</tbody>
    </table>
  `;

  return section;
}

/**
 * Format a value for display in the registry table.
 * Handles primitives, arrays, and objects.
 */
function formatValueForDisplay(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "number") return String(value);
  if (typeof value === "string") return value;

  if (Array.isArray(value)) {
    if (value.length === 0) return "—";
    // Check if array contains objects
    if (typeof value[0] === "object" && value[0] !== null) {
      // For arrays of objects, extract meaningful info
      return value.map(item => {
        if (item.code) return item.code; // CPT code object
        if (item.text) return `"${item.text.slice(0, 50)}${item.text.length > 50 ? '...' : ''}"`; // Evidence span
        if (item.name) return item.name;
        // Fallback: show first few properties
        const keys = Object.keys(item).slice(0, 2);
        return keys.map(k => `${k}: ${item[k]}`).join(", ");
      }).join("; ");
    }
    return value.join(", ");
  }

  if (typeof value === "object") {
    // For single objects, extract meaningful info
    if (value.code) return value.code;
    if (value.text) return `"${value.text.slice(0, 50)}${value.text.length > 50 ? '...' : ''}"`;
    // Fallback: JSON but truncated
    const json = JSON.stringify(value);
    return json.length > 100 ? json.slice(0, 97) + "..." : json;
  }

  return String(value);
}

/**
 * Recursively render all non-null registry fields as a form.
 * Flattens nested objects with arrow notation paths.
 */
function renderRegistryForm(registry) {
  const container = document.getElementById("registryForm");
  const rows = [];

  // Keys to skip (complex nested structures shown separately or not useful)
  const skipKeys = new Set(["evidence", "billing", "ner_spans"]);

  // Recursively extract non-null fields
  function extractFields(obj, prefix = "") {
    if (obj === null || obj === undefined) return;

    for (const [key, value] of Object.entries(obj)) {
      const path = prefix ? `${prefix}.${key}` : key;
      const lowKey = key.toLowerCase();

      if (value === null || value === undefined) continue;
      if (value === false) continue; // Skip false booleans (procedures not performed)
      if (Array.isArray(value) && value.length === 0) continue; // Skip empty arrays

      // Skip complex evidence/billing structures at top level
      if (!prefix && skipKeys.has(lowKey)) continue;

      if (typeof value === "object" && !Array.isArray(value)) {
        // Recurse into nested objects
        extractFields(value, path);
      } else {
        // Format the value for display
        const displayValue = formatValueForDisplay(value);
        if (displayValue === "—") continue; // Skip empty values

        // Format the key for display (snake_case → Title Case with arrow separators)
        const displayKey = path
          .split(".")
          .map((part) => part.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()))
          .join(" > ");

        rows.push({ path, label: displayKey, value: displayValue, rawValue: value });
      }
    }
  }

  extractFields(registry);

  // Build form HTML
  let html = "";
  if (rows.length === 0) {
    html =
      '<div class="subtle" style="text-align: center; padding: 20px;">No registry data extracted</div>';
  } else {
    // Group by top-level category
    const groups = {};
    rows.forEach((row) => {
      const category = row.path
        .split(".")[0]
        .replace(/_/g, " ")
        .replace(/\b\w/g, (c) => c.toUpperCase());
      if (!groups[category]) groups[category] = [];
      groups[category].push(row);
    });

    for (const [category, items] of Object.entries(groups)) {
      html += `<div class="collapsible-section open">`;
      html += `<button type="button" class="collapsible-header" onclick="this.parentElement.classList.toggle('open')">`;
      html += `<span>${category}</span>`;
      html += `<span class="collapsible-icon">▼</span>`;
      html += `</button>`;
      html += `<div class="collapsible-content">`;

      items.forEach(({ path, label, value }) => {
        // Escape HTML in values to prevent XSS
        const safeValue = String(value).replace(/</g, "&lt;").replace(/>/g, "&gt;");
        const shortLabel = label.includes(" > ") ? label.split(" > ").slice(1).join(" > ") : label;

        html += `<div class="form-group" data-path="${path}">`;
        html += `<label class="form-label">${shortLabel}</label>`;
        html += `<input type="text" class="form-control" value="${safeValue}" readonly>`;
        html += `</div>`;
      });

      html += `</div></div>`;
    }
  }

  container.innerHTML = html;
}

async function main() {
  const editorHost = document.getElementById("editor");
  const fallbackTextarea = document.getElementById("fallbackTextarea");
  const isMobileSafariBrowser = (() => {
    const ua = navigator.userAgent || "";
    const isIOSDevice =
      /iPad|iPhone|iPod/i.test(ua) || (ua.includes("Mac") && "ontouchend" in document);
    const isSafari =
      /Safari/i.test(ua) && !/Chrome|CriOS|FxiOS|EdgiOS|OPR/i.test(ua);
    return isIOSDevice && isSafari;
  })();

  // Let users paste/typing immediately; Monaco boot can lag on first load.
  if (fallbackTextarea) {
    fallbackTextarea.classList.remove("hidden");
    fallbackTextarea.disabled = false;
  }

  setStatus("Ready to type (model loading in background)...");
  initLayoutControls();

  let usingPlainEditor = true;
  let editor = null;
  let model = null;

  // Try to boot Monaco quickly, but never hang the app if it stalls.
  if (!isMobileSafariBrowser && window.__monacoReady) {
    try {
      await Promise.race([
        window.__monacoReady,
        new Promise((_, reject) => setTimeout(() => reject(new Error("Monaco load timeout")), 2500)),
      ]);
      usingPlainEditor = false;
    } catch {
      usingPlainEditor = true;
    }
  }

  const crossOriginIsolated = globalThis.crossOriginIsolated === true;
  if (!crossOriginIsolated) {
    setStatus(
      "Cross-origin isolation is OFF (SharedArrayBuffer unavailable). Running in single-threaded mode."
    );
  } else if (isMobileSafariBrowser) {
    setStatus("Mobile Safari detected: using basic editor mode for reliable select/copy behavior.");
  } else if (usingPlainEditor) {
    setStatus("Loading… (basic editor mode; Monaco still initializing)");
  }

  if (!usingPlainEditor && editorHost) {
    const initialValue = fallbackTextarea ? fallbackTextarea.value : "";
    if (fallbackTextarea) fallbackTextarea.remove();

    editor = monaco.editor.create(editorHost, {
      value: initialValue,
      language: "plaintext",
      theme: "vs-dark",
      minimap: { enabled: false },
      wordWrap: "on",
      fontSize: 13,
      automaticLayout: true,
    });
    // Expose for evidence click-to-highlight
    window.editor = editor;
    model = editor.getModel();
	  } else {
	    // Monaco unavailable/slow: use the built-in textarea as the editor surface.
	    window.editor = null;
	    model = {
	      getValue: () => (fallbackTextarea ? fallbackTextarea.value : ""),
      setValue: (value) => {
        if (!fallbackTextarea) return;
        fallbackTextarea.value = String(value ?? "");
      },
	    };
	  }

	  // RegistryGrid embed expects a getter for the live Monaco editor (or null in textarea mode).
	  setRegistryGridMonacoGetter(() => window.editor);

  const reporterTransfer = consumeReporterTransferPayload();
  if (reporterTransfer?.note) {
    model.setValue(reporterTransfer.note);
    setStatus("Reporter note loaded. Run detection, apply redactions, then submit.");
    setProgress("");
  }

	  let originalText = model.getValue();
	  let hasRunDetection = false;
	  let scrubbedConfirmed = false;
	  let suppressDirtyFlag = false;
    let bundleDocs = [];
    let lastBundleResponse = null;
    let bundleBusy = false;
    let running = false;
    let extractingPdf = false;
    let extractingCamera = false;
    let cameraModalOpen = false;
    let cameraStream = null;
    let cameraWorker = null;
    let cameraOcrJobId = "";
    let privacyShieldTeardown = () => {};
    const cameraQueue = createCameraCaptureQueue();
    let cameraSelectedCropPageIndex = 0;
    let cameraCropDragState = null;
    let cameraCropShowZoomPreview = false;

  function formatFileSize(bytes) {
    if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function setEditorText(nextText) {
    suppressDirtyFlag = true;
    try {
      if (!usingPlainEditor && editor) editor.setValue(nextText);
      else model.setValue(nextText);
    } finally {
      suppressDirtyFlag = false;
    }
  }

  const PDF_UPLOAD_HELP_TEXT =
    "Input options: paste note text directly, or upload a PDF for layout-aware local extraction with safety gating before PHI detection.";
  const CAMERA_SCAN_HELP_TEXT =
    "Camera scan captures pages in-memory only. Run OCR locally, then use Run Detection and Apply Redactions.";

  function setPdfExtractSummary(message, tone = "neutral") {
    if (!pdfExtractSummaryEl) return;
    pdfExtractSummaryEl.textContent = String(message || "");
    pdfExtractSummaryEl.classList.remove("is-warning", "is-error", "is-success");
    if (tone === "warning") pdfExtractSummaryEl.classList.add("is-warning");
    if (tone === "error") pdfExtractSummaryEl.classList.add("is-error");
    if (tone === "success") pdfExtractSummaryEl.classList.add("is-success");
  }

  function setCameraScanSummary(message, tone = "neutral") {
    if (!cameraScanSummaryEl) return;
    cameraScanSummaryEl.classList.remove("hidden");
    cameraScanSummaryEl.textContent = String(message || "");
    cameraScanSummaryEl.classList.remove("is-warning", "is-error", "is-success");
    if (tone === "warning") cameraScanSummaryEl.classList.add("is-warning");
    if (tone === "error") cameraScanSummaryEl.classList.add("is-error");
    if (tone === "success") cameraScanSummaryEl.classList.add("is-success");
  }

  function formatPdfStage(stage) {
    switch (stage) {
      case "layout_analysis":
        return "layout analysis";
      case "contamination_detection":
        return "contamination checks";
      case "adaptive_assembly":
        return "adaptive text assembly";
      case "native_extraction_done":
        return "native extraction complete";
      case "ocr_prepare":
        return "OCR preparation";
      case "ocr_loading_assets":
        return "OCR asset load";
      case "ocr_rendering":
        return "OCR page rendering";
      case "ocr_recognizing":
        return "OCR recognition";
      case "ocr_failed":
        return "OCR failure";
      default:
        return "processing";
    }
  }

  function buildPageReasonSummary(docModel, maxPages = 3) {
    if (!docModel || !Array.isArray(docModel.pages)) return "";
    const reasons = docModel.pages
      .filter((page) => page?.classification?.reason)
      .slice(0, maxPages)
      .map((page) => `p${page.pageIndex + 1}: ${page.classification.reason}`);

    if (docModel.pages.length > maxPages) {
      reasons.push(`+${docModel.pages.length - maxPages} more page(s)`);
    }
    return reasons.join(" | ");
  }

  function buildOcrDebugSummary(docModel, maxPages = 2) {
    if (!docModel || !Array.isArray(docModel.pages)) return "";
    const ocrPages = docModel.pages.filter((page) => page?.sourceDecision !== "native" && page?.ocrMeta);
    if (!ocrPages.length) return "";

    const maskModes = new Set();
    let imageRegionCandidates = 0;
    let maskedPages = 0;
    let coverageTotal = 0;
    let coverageCount = 0;
    const cropBoxes = [];
    const figureFilterModes = new Set();

    for (const page of ocrPages) {
      const masking = page.ocrMeta?.masking;
      if (masking?.mode) maskModes.add(String(masking.mode));
      if (masking?.applied) maskedPages += 1;
      if (Number.isFinite(masking?.candidateCount)) {
        imageRegionCandidates += Number(masking.candidateCount);
      }
      if (Number.isFinite(masking?.coverageRatio)) {
        coverageTotal += Number(masking.coverageRatio);
        coverageCount += 1;
      }

      const crop = page.ocrMeta?.crop;
      if (crop?.applied && Array.isArray(crop.box) && crop.box.length === 4 && cropBoxes.length < maxPages) {
        const [x0, y0, x1, y1] = crop.box.map((value) => Math.round(Number(value) || 0));
        cropBoxes.push(`p${page.pageIndex + 1}[${x0},${y0},${x1},${y1}]`);
      }
      if (page.ocrMeta?.filterMode?.reason) {
        figureFilterModes.add(String(page.ocrMeta.filterMode.reason));
      }
    }

    const maskModeText = maskModes.size ? [...maskModes].join("/") : "n/a";
    const avgCoveragePct = coverageCount ? Math.round((coverageTotal / coverageCount) * 100) : null;
    const coverageText = avgCoveragePct !== null ? `, maskedArea~${avgCoveragePct}%` : "";
    const cropText = cropBoxes.length ? `, crop=${cropBoxes.join(" | ")}` : ", crop=none";
    const figureFilterText = figureFilterModes.size
      ? `, figFilter=${[...figureFilterModes].join("/")}`
      : "";

    return ` OCR debug: mask=${maskModeText}, imageRegions=${imageRegionCandidates}, maskedPages=${maskedPages}/${ocrPages.length}${coverageText}${cropText}${figureFilterText}.`;
  }

  function buildPageMetricsSummary(docModel, maxPages = 3) {
    const lines = Array.isArray(docModel?.qualitySummary?.pageMetrics)
      ? docModel.qualitySummary.pageMetrics
      : [];
    if (!lines.length) return "";
    const preview = lines.slice(0, maxPages);
    if (lines.length > maxPages) {
      preview.push(`+${lines.length - maxPages} more page(s)`);
    }
    return ` Metrics: ${preview.join(" | ")}`;
  }

  function buildPdfExtractionMetricsReport(docModel, context = {}) {
    if (!docModel || !Array.isArray(docModel.pages)) return null;
    const pages = docModel.pages.map((page) => {
      const metrics = page?.extractionMetrics && typeof page.extractionMetrics === "object"
        ? page.extractionMetrics
        : {};
      const quality = page?.qualityMetrics && typeof page.qualityMetrics === "object"
        ? page.qualityMetrics
        : {};
      const before = Number(metrics.junkScoreBeforeMerge);
      const after = Number(metrics.junkScoreAfterMerge);
      return {
        pageIndex: Number(page?.pageIndex) + 1,
        nativeTextDensity: Number.isFinite(Number(metrics.nativeTextDensity)) ? Number(metrics.nativeTextDensity) : 0,
        backfillVotes: Number(metrics.backfillVotes) || 0,
        backfillStrongVotes: Number(metrics.backfillStrongVotes) || 0,
        backfillScore: Number(metrics.backfillScore) || 0,
        backfillSignals: Array.isArray(metrics.backfillSignals) ? metrics.backfillSignals : [],
        needsOcrBackfill: Boolean(metrics.needsOcrBackfill),
        mode: String(metrics.mode || "native_only"),
        ocrRoiCount: Number(metrics.ocrRoiCount) || 0,
        ocrRoiAreaPx: Number(metrics.ocrRoiAreaPx) || 0,
        ocrRoiAreaRatio: Number.isFinite(Number(metrics.ocrRoiAreaRatio)) ? Number(metrics.ocrRoiAreaRatio) : 0,
        junkScoreBeforeMerge: Number.isFinite(before)
          ? before
          : (Number.isFinite(quality.junkScoreBeforeMerge) ? Number(quality.junkScoreBeforeMerge) : 0),
        junkScoreAfterMerge: Number.isFinite(after)
          ? after
          : (Number.isFinite(quality.junkScoreAfterMerge) ? Number(quality.junkScoreAfterMerge) : Number(quality.junkScore) || 0),
      };
    });

    const byMode = pages.reduce((acc, page) => {
      const key = String(page.mode || "unknown");
      acc[key] = (acc[key] || 0) + 1;
      return acc;
    }, {});
    const backfillPages = pages.filter((page) => page.needsOcrBackfill).map((page) => page.pageIndex);
    const fullOcrPages = pages
      .filter((page) => page.mode === "full_ocr")
      .map((page) => page.pageIndex);

    return {
      generatedAt: new Date().toISOString(),
      status: context.status || (docModel.blocked ? "blocked" : "success"),
      blocked: Boolean(docModel.blocked),
      blockReason: docModel.blockReason || "",
      fileName: String(docModel.fileName || context.fileName || ""),
      pageCount: pages.length,
      context: {
        ocrQualityMode: context.ocrQualityMode || "",
        ocrMaskMode: context.ocrMaskMode || "",
      },
      byMode,
      backfillPages,
      fullOcrPages,
      pages,
    };
  }

  function publishPdfExtractionMetricsReport(report) {
    if (!report) return;
    try {
      window.__lastPdfExtractionMetrics = report;
      const history = Array.isArray(window.__pdfExtractionMetricsHistory)
        ? window.__pdfExtractionMetricsHistory
        : [];
      history.push(report);
      window.__pdfExtractionMetricsHistory = history.slice(-20);
      console.info("[pdf_extract_metrics]", report);
    } catch {
      // Ignore diagnostics write failures.
    }
  }

  function resetPdfUploadUi() {
    if (pdfUploadInputEl) pdfUploadInputEl.value = "";
    setPdfExtractSummary(PDF_UPLOAD_HELP_TEXT, "neutral");
  }

  function setCameraStatus(message) {
    if (!cameraStatusTextEl) return;
    cameraStatusTextEl.textContent = String(message || "");
  }

  function setCameraProgress(message) {
    if (!cameraProgressTextEl) return;
    cameraProgressTextEl.textContent = String(message || "");
  }

  function resolveCameraSceneHint() {
    const sceneHintValue = String(cameraSceneHintSelectEl?.value || "auto");
    if (sceneHintValue === "monitor") return "monitor";
    if (sceneHintValue === "document") return "document";
    return "auto";
  }

  function updateCameraGuideHint() {
    if (!cameraGuideHintEl) return;
    const sceneHint = resolveCameraSceneHint();
    if (sceneHint === "monitor") {
      cameraGuideHintEl.textContent = "Screen capture: tilt phone ~15 degrees, hold steady, reduce glare.";
      return;
    }
    if (sceneHint === "document") {
      cameraGuideHintEl.textContent = "Align paper inside frame. Hold steady and avoid shadowing.";
      return;
    }
    cameraGuideHintEl.textContent = "Align document inside frame. Hold steady and reduce glare.";
  }

  function clearCameraGuideFrameStyle() {
    if (!cameraGuideOverlayEl) return;
    cameraGuideOverlayEl.style.removeProperty("--camera-guide-left");
    cameraGuideOverlayEl.style.removeProperty("--camera-guide-top");
    cameraGuideOverlayEl.style.removeProperty("--camera-guide-width");
    cameraGuideOverlayEl.style.removeProperty("--camera-guide-height");
  }

  function updateCameraGuideFrame() {
    if (!cameraGuideOverlayEl || !cameraPreviewEl) return;
    const overlayRect = cameraGuideOverlayEl.getBoundingClientRect();
    const previewRect = cameraPreviewEl.getBoundingClientRect();
    if (!overlayRect?.width || !overlayRect?.height || !previewRect?.width || !previewRect?.height) return;

    let left = previewRect.left - overlayRect.left;
    let top = previewRect.top - overlayRect.top;
    let width = previewRect.width;
    let height = previewRect.height;
    const videoWidth = Math.max(0, Number(cameraPreviewEl.videoWidth) || 0);
    const videoHeight = Math.max(0, Number(cameraPreviewEl.videoHeight) || 0);

    if (videoWidth > 0 && videoHeight > 0 && width > 0 && height > 0) {
      const boxRatio = width / height;
      const videoRatio = videoWidth / videoHeight;
      const ratioEpsilon = 0.0001;
      if (videoRatio < boxRatio - ratioEpsilon) {
        const contentWidth = Math.max(1, height * videoRatio);
        const padX = Math.max(0, (width - contentWidth) / 2);
        left += padX;
        width = contentWidth;
      } else if (videoRatio > boxRatio + ratioEpsilon) {
        const contentHeight = Math.max(1, width / videoRatio);
        const padY = Math.max(0, (height - contentHeight) / 2);
        top += padY;
        height = contentHeight;
      }
    }

    const insetX = Math.max(6, width * 0.03);
    const insetY = Math.max(8, height * 0.035);
    const frameLeft = left + insetX;
    const frameTop = top + insetY;
    const frameWidth = Math.max(24, width - insetX * 2);
    const frameHeight = Math.max(24, height - insetY * 2);

    cameraGuideOverlayEl.style.setProperty("--camera-guide-left", `${frameLeft.toFixed(2)}px`);
    cameraGuideOverlayEl.style.setProperty("--camera-guide-top", `${frameTop.toFixed(2)}px`);
    cameraGuideOverlayEl.style.setProperty("--camera-guide-width", `${frameWidth.toFixed(2)}px`);
    cameraGuideOverlayEl.style.setProperty("--camera-guide-height", `${frameHeight.toFixed(2)}px`);
  }

  function scheduleCameraGuideFrameUpdate() {
    if (cameraGuideFrameRafId) {
      cancelAnimationFrame(cameraGuideFrameRafId);
      cameraGuideFrameRafId = 0;
    }
    cameraGuideFrameRafId = requestAnimationFrame(() => {
      cameraGuideFrameRafId = 0;
      updateCameraGuideFrame();
    });
  }

  function createCameraProbeCanvas(width, height) {
    const safeWidth = Math.max(1, Math.floor(Number(width) || 1));
    const safeHeight = Math.max(1, Math.floor(Number(height) || 1));
    if (typeof OffscreenCanvas === "function") {
      return new OffscreenCanvas(safeWidth, safeHeight);
    }
    const canvas = document.createElement("canvas");
    canvas.width = safeWidth;
    canvas.height = safeHeight;
    return canvas;
  }

  function evaluateCapturedFrameQuality(frame) {
    try {
      const srcWidth = Math.max(1, Number(frame?.width) || 1);
      const srcHeight = Math.max(1, Number(frame?.height) || 1);
      const probeMaxDim = 1200;
      const scale = Math.min(1, probeMaxDim / Math.max(srcWidth, srcHeight));
      const probeWidth = Math.max(1, Math.round(srcWidth * scale));
      const probeHeight = Math.max(1, Math.round(srcHeight * scale));
      const canvas = createCameraProbeCanvas(probeWidth, probeHeight);
      const ctx = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
      if (!ctx) return { metrics: null, warnings: [] };

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, probeWidth, probeHeight);
      ctx.drawImage(frame.bitmap, 0, 0, srcWidth, srcHeight, 0, 0, probeWidth, probeHeight);

      const imageData = ctx.getImageData(0, 0, probeWidth, probeHeight);
      const gray = computeGrayFromImageData(imageData.data);
      const metrics = computeCaptureQualityMetrics(gray, probeWidth, probeHeight);
      const warnings = buildCaptureWarnings(metrics, { warningProfile: cameraWarningProfile });
      return { metrics, warnings };
    } catch {
      return { metrics: null, warnings: [] };
    }
  }

  function normalizeCameraCropMargins(input = {}) {
    return {
      top: clamp(Number(input.top) || 0, 0, 45),
      right: clamp(Number(input.right) || 0, 0, 45),
      bottom: clamp(Number(input.bottom) || 0, 0, 45),
      left: clamp(Number(input.left) || 0, 0, 45),
    };
  }

  function cameraCropRectToMargins(crop) {
    if (!crop || typeof crop !== "object") {
      return { top: 0, right: 0, bottom: 0, left: 0 };
    }
    const x0 = clamp(Number(crop.x0) || 0, 0, 1);
    const y0 = clamp(Number(crop.y0) || 0, 0, 1);
    const x1 = clamp(Number(crop.x1) || 1, 0, 1);
    const y1 = clamp(Number(crop.y1) || 1, 0, 1);
    return normalizeCameraCropMargins({
      top: Math.round(Math.min(y0, y1) * 100),
      right: Math.round((1 - Math.max(x0, x1)) * 100),
      bottom: Math.round((1 - Math.max(y0, y1)) * 100),
      left: Math.round(Math.min(x0, x1) * 100),
    });
  }

  function cameraCropMarginsToRect(marginsInput) {
    const margins = normalizeCameraCropMargins(marginsInput);
    const x0 = clamp(margins.left / 100, 0, 1);
    const y0 = clamp(margins.top / 100, 0, 1);
    const x1 = clamp(1 - margins.right / 100, 0, 1);
    const y1 = clamp(1 - margins.bottom / 100, 0, 1);
    const fullPage = margins.top < 0.5 && margins.right < 0.5 && margins.bottom < 0.5 && margins.left < 0.5;
    if (fullPage) return null;
    if (x1 - x0 < 0.05 || y1 - y0 < 0.05) return null;
    return { x0, y0, x1, y1 };
  }

  function readCameraCropMarginsFromInputs() {
    return normalizeCameraCropMargins({
      top: Number(cameraCropTopRangeEl?.value || 0),
      right: Number(cameraCropRightRangeEl?.value || 0),
      bottom: Number(cameraCropBottomRangeEl?.value || 0),
      left: Number(cameraCropLeftRangeEl?.value || 0),
    });
  }

  function writeCameraCropMarginsToInputs(marginsInput) {
    const margins = normalizeCameraCropMargins(marginsInput);
    if (cameraCropTopRangeEl) cameraCropTopRangeEl.value = String(margins.top);
    if (cameraCropRightRangeEl) cameraCropRightRangeEl.value = String(margins.right);
    if (cameraCropBottomRangeEl) cameraCropBottomRangeEl.value = String(margins.bottom);
    if (cameraCropLeftRangeEl) cameraCropLeftRangeEl.value = String(margins.left);
    if (cameraCropTopValueEl) cameraCropTopValueEl.textContent = `${Math.round(margins.top)}%`;
    if (cameraCropRightValueEl) cameraCropRightValueEl.textContent = `${Math.round(margins.right)}%`;
    if (cameraCropBottomValueEl) cameraCropBottomValueEl.textContent = `${Math.round(margins.bottom)}%`;
    if (cameraCropLeftValueEl) cameraCropLeftValueEl.textContent = `${Math.round(margins.left)}%`;
  }

  function normalizeCameraCropRect(input) {
    if (!input || typeof input !== "object") return { x0: 0, y0: 0, x1: 1, y1: 1 };
    const x0 = clamp(Number(input.x0) || 0, 0, 1);
    const y0 = clamp(Number(input.y0) || 0, 0, 1);
    const x1 = clamp(Number(input.x1) || 1, 0, 1);
    const y1 = clamp(Number(input.y1) || 1, 0, 1);
    return {
      x0: Math.min(x0, x1),
      y0: Math.min(y0, y1),
      x1: Math.max(x0, x1),
      y1: Math.max(y0, y1),
    };
  }

  function getCameraCropRectFromInputsOrFull() {
    const raw = cameraCropMarginsToRect(readCameraCropMarginsFromInputs());
    return normalizeCameraCropRect(raw || { x0: 0, y0: 0, x1: 1, y1: 1 });
  }

  function setCameraCropRectToInputs(rect) {
    writeCameraCropMarginsToInputs(cameraCropRectToMargins(normalizeCameraCropRect(rect)));
  }

  function getCameraCropFrameRect() {
    if (!cameraCropPreviewStageEl || !cameraCropPreviewImgEl) return null;
    if (cameraCropPreviewImgEl.classList.contains("hidden")) return null;

    const stageRect = cameraCropPreviewStageEl.getBoundingClientRect();
    const imgRect = cameraCropPreviewImgEl.getBoundingClientRect();
    if (!stageRect?.width || !stageRect?.height || !imgRect?.width || !imgRect?.height) return null;

    let left = imgRect.left - stageRect.left;
    let top = imgRect.top - stageRect.top;
    let width = imgRect.width;
    let height = imgRect.height;

    // The preview image uses object-fit: contain, which can add internal bars.
    // Map crop coordinates to the actual rendered image content, not the full element box.
    const naturalWidth = Math.max(0, Number(cameraCropPreviewImgEl.naturalWidth) || 0);
    const naturalHeight = Math.max(0, Number(cameraCropPreviewImgEl.naturalHeight) || 0);
    if (naturalWidth > 0 && naturalHeight > 0 && width > 0 && height > 0) {
      const boxRatio = width / height;
      const contentRatio = naturalWidth / naturalHeight;
      const ratioEpsilon = 0.0001;
      if (contentRatio < boxRatio - ratioEpsilon) {
        const contentWidth = Math.max(1, height * contentRatio);
        const padX = Math.max(0, (width - contentWidth) / 2);
        left += padX;
        width = contentWidth;
      } else if (contentRatio > boxRatio + ratioEpsilon) {
        const contentHeight = Math.max(1, width / contentRatio);
        const padY = Math.max(0, (height - contentHeight) / 2);
        top += padY;
        height = contentHeight;
      }
    }

    return {
      left,
      top,
      width,
      height,
    };
  }

  function renderCameraCropZoomPreview() {
    if (!cameraCropZoomWrapEl || !cameraCropZoomImgEl) return;
    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    const page = pages[cameraSelectedCropPageIndex] || null;
    const activeRect = getCameraCropRectFromInputsOrFull();
    const hasCrop = activeRect.x1 - activeRect.x0 < 0.999 || activeRect.y1 - activeRect.y0 < 0.999;

    if (!cameraCropShowZoomPreview || !page || !hasCrop) {
      cameraCropZoomWrapEl.classList.add("hidden");
      cameraCropZoomImgEl.removeAttribute("src");
      return;
    }

    const img = cameraCropPreviewImgEl;
    const naturalWidth = Math.max(1, Number(img?.naturalWidth) || 0);
    const naturalHeight = Math.max(1, Number(img?.naturalHeight) || 0);
    if (!naturalWidth || !naturalHeight || !img?.complete) {
      cameraCropZoomWrapEl.classList.add("hidden");
      return;
    }

    const sx = Math.max(0, Math.floor(activeRect.x0 * naturalWidth));
    const sy = Math.max(0, Math.floor(activeRect.y0 * naturalHeight));
    const ex = Math.min(naturalWidth, Math.ceil(activeRect.x1 * naturalWidth));
    const ey = Math.min(naturalHeight, Math.ceil(activeRect.y1 * naturalHeight));
    const sw = Math.max(1, ex - sx);
    const sh = Math.max(1, ey - sy);
    const maxDim = 900;
    const scale = Math.min(1, maxDim / Math.max(sw, sh));
    const outW = Math.max(1, Math.round(sw * scale));
    const outH = Math.max(1, Math.round(sh * scale));

    try {
      const canvas = document.createElement("canvas");
      canvas.width = outW;
      canvas.height = outH;
      const ctx = canvas.getContext("2d");
      if (!ctx) throw new Error("missing 2d context");
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, outW, outH);
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, outW, outH);
      cameraCropZoomImgEl.src = canvas.toDataURL("image/jpeg", 0.95);
      cameraCropZoomWrapEl.classList.remove("hidden");
    } catch {
      cameraCropZoomWrapEl.classList.add("hidden");
      cameraCropZoomImgEl.removeAttribute("src");
    }
  }

  function beginCameraCropDrag(pointerEvent) {
    if (!cameraCropPreviewStageEl || !cameraCropPreviewBoxEl) return;
    if (running || bundleBusy || extractingPdf || extractingCamera) return;
    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    if (!pages.length) return;

    const handleEl = pointerEvent.target?.closest?.("[data-crop-handle]");
    const handleName = String(handleEl?.dataset?.cropHandle || "").toLowerCase();
    const clickedOnBox = pointerEvent.target === cameraCropPreviewBoxEl || pointerEvent.target?.closest?.("#cameraCropPreviewBox");
    if (!clickedOnBox) return;

    const mode = ["nw", "ne", "se", "sw"].includes(handleName) ? handleName : "move";
    const frame = getCameraCropFrameRect();
    if (!frame) return;
    const startRect = getCameraCropRectFromInputsOrFull();
    if (!startRect) return;

    cameraCropDragState = {
      pointerId: Number(pointerEvent.pointerId),
      mode,
      startClientX: Number(pointerEvent.clientX) || 0,
      startClientY: Number(pointerEvent.clientY) || 0,
      startRect,
      frame,
    };
    cameraCropShowZoomPreview = false;

    try {
      cameraCropPreviewStageEl.setPointerCapture(pointerEvent.pointerId);
    } catch {
      // ignore
    }
    pointerEvent.preventDefault();
  }

  function updateCameraCropDrag(pointerEvent) {
    if (!cameraCropDragState) return;
    if (Number(pointerEvent.pointerId) !== cameraCropDragState.pointerId) return;

    const minSpan = 0.05;
    const dx = (Number(pointerEvent.clientX) - cameraCropDragState.startClientX) / Math.max(1, cameraCropDragState.frame.width);
    const dy = (Number(pointerEvent.clientY) - cameraCropDragState.startClientY) / Math.max(1, cameraCropDragState.frame.height);
    const start = cameraCropDragState.startRect;
    let next = { ...start };

    if (cameraCropDragState.mode === "move") {
      const spanX = Math.max(minSpan, start.x1 - start.x0);
      const spanY = Math.max(minSpan, start.y1 - start.y0);
      let x0 = start.x0 + dx;
      let y0 = start.y0 + dy;
      x0 = clamp(x0, 0, 1 - spanX);
      y0 = clamp(y0, 0, 1 - spanY);
      next = { x0, y0, x1: x0 + spanX, y1: y0 + spanY };
    } else if (cameraCropDragState.mode === "nw") {
      next.x0 = clamp(start.x0 + dx, 0, start.x1 - minSpan);
      next.y0 = clamp(start.y0 + dy, 0, start.y1 - minSpan);
    } else if (cameraCropDragState.mode === "ne") {
      next.x1 = clamp(start.x1 + dx, start.x0 + minSpan, 1);
      next.y0 = clamp(start.y0 + dy, 0, start.y1 - minSpan);
    } else if (cameraCropDragState.mode === "se") {
      next.x1 = clamp(start.x1 + dx, start.x0 + minSpan, 1);
      next.y1 = clamp(start.y1 + dy, start.y0 + minSpan, 1);
    } else if (cameraCropDragState.mode === "sw") {
      next.x0 = clamp(start.x0 + dx, 0, start.x1 - minSpan);
      next.y1 = clamp(start.y1 + dy, start.y0 + minSpan, 1);
    }

    setCameraCropRectToInputs(next);
    renderCameraCropPreview();
    pointerEvent.preventDefault();
  }

  function endCameraCropDrag(pointerEvent) {
    if (!cameraCropDragState) return;
    if (pointerEvent && Number(pointerEvent.pointerId) !== cameraCropDragState.pointerId) return;
    cameraCropDragState = null;
  }

  function updateCameraCropPreviewFromInputs() {
    const margins = readCameraCropMarginsFromInputs();
    writeCameraCropMarginsToInputs(margins);
    cameraCropShowZoomPreview = false;
    renderCameraCropPreview();
  }

  function loadCameraCropControlsFromPage(pageIndex) {
    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    const idx = clamp(Number(pageIndex) || 0, 0, Math.max(0, pages.length - 1));
    cameraSelectedCropPageIndex = idx;
    cameraCropDragState = null;
    const page = pages[idx] || null;
    cameraCropShowZoomPreview = Boolean(page?.crop);
    writeCameraCropMarginsToInputs(cameraCropRectToMargins(page?.crop || null));
  }

  function renderCameraCropPreview() {
    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    const page = pages[cameraSelectedCropPageIndex] || null;
    if (!cameraCropPreviewImgEl || !cameraCropPreviewBoxEl || !cameraCropPreviewStageEl) return;

    if (page?.previewUrl) {
      if (cameraCropPreviewImgEl.getAttribute("src") !== page.previewUrl) {
        cameraCropPreviewImgEl.src = page.previewUrl;
      }
      cameraCropPreviewImgEl.classList.remove("hidden");
    } else {
      cameraCropPreviewImgEl.removeAttribute("src");
      cameraCropPreviewImgEl.classList.add("hidden");
      cameraCropPreviewBoxEl.classList.add("hidden");
      renderCameraCropZoomPreview();
      return;
    }

    const frame = getCameraCropFrameRect();
    if (!frame) {
      cameraCropPreviewBoxEl.classList.add("hidden");
      renderCameraCropZoomPreview();
      return;
    }

    const cropRect = getCameraCropRectFromInputsOrFull();
    const left = frame.left + cropRect.x0 * frame.width;
    const top = frame.top + cropRect.y0 * frame.height;
    const width = Math.max(1, (cropRect.x1 - cropRect.x0) * frame.width);
    const height = Math.max(1, (cropRect.y1 - cropRect.y0) * frame.height);

    cameraCropPreviewBoxEl.style.left = `${left}px`;
    cameraCropPreviewBoxEl.style.top = `${top}px`;
    cameraCropPreviewBoxEl.style.width = `${width}px`;
    cameraCropPreviewBoxEl.style.height = `${height}px`;
    cameraCropPreviewBoxEl.classList.remove("hidden");
    renderCameraCropZoomPreview();
  }

  function renderCameraCropPanel() {
    if (!cameraCropPanelEl) return;
    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    const hasPages = pages.length > 0;
    cameraCropPanelEl.classList.toggle("hidden", !hasPages);
    if (!hasPages) {
      if (cameraCropPreviewImgEl) {
        cameraCropPreviewImgEl.removeAttribute("src");
        cameraCropPreviewImgEl.classList.add("hidden");
      }
      if (cameraCropPreviewBoxEl) cameraCropPreviewBoxEl.classList.add("hidden");
      if (cameraCropZoomWrapEl) cameraCropZoomWrapEl.classList.add("hidden");
      return;
    }

    cameraSelectedCropPageIndex = clamp(cameraSelectedCropPageIndex, 0, pages.length - 1);
    if (cameraCropPageSelectEl) {
      cameraCropPageSelectEl.innerHTML = "";
      for (let i = 0; i < pages.length; i += 1) {
        const option = document.createElement("option");
        const cropTag = pages[i]?.crop ? " (cropped)" : "";
        option.value = String(i);
        option.textContent = `Page ${i + 1}${cropTag}`;
        cameraCropPageSelectEl.appendChild(option);
      }
      cameraCropPageSelectEl.value = String(cameraSelectedCropPageIndex);
    }
    renderCameraCropPreview();
  }

  function renderCameraWarnings(warnings = []) {
    if (!cameraWarningListEl) return;
    cameraWarningListEl.innerHTML = "";
    for (const warning of Array.isArray(warnings) ? warnings : []) {
      const text = String(warning || "").trim();
      if (!text) continue;
      const item = document.createElement("div");
      item.className = "camera-warning-item";
      item.textContent = text;
      cameraWarningListEl.appendChild(item);
    }
  }

  function renderCameraThumbnails() {
    if (!cameraThumbStripEl) return;
    cameraThumbStripEl.innerHTML = "";

    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    if (!pages.length) {
      const empty = document.createElement("div");
      empty.className = "subtle";
      empty.textContent = "No captured pages yet.";
      cameraThumbStripEl.appendChild(empty);
      renderCameraCropPanel();
      return;
    }

    cameraSelectedCropPageIndex = clamp(cameraSelectedCropPageIndex, 0, pages.length - 1);

    for (let i = 0; i < pages.length; i += 1) {
      const page = pages[i];
      const item = document.createElement("div");
      item.className = "camera-thumb-item";
      if (page?.crop) item.classList.add("cropped");
      if (i === cameraSelectedCropPageIndex) item.classList.add("selected");

      const button = document.createElement("button");
      button.type = "button";
      button.className = "camera-thumb-button";
      button.addEventListener("click", () => {
        cameraSelectedCropPageIndex = i;
        loadCameraCropControlsFromPage(i);
        renderCameraThumbnails();
      });

      const img = document.createElement("img");
      if (page.previewUrl) {
        img.src = page.previewUrl;
        img.alt = `Captured page ${i + 1}`;
      } else {
        img.alt = `Captured page ${i + 1} preview unavailable`;
      }

      const label = document.createElement("div");
      label.className = "camera-thumb-label";
      label.textContent = `Page ${i + 1}${page?.crop ? " • Crop" : ""}`;

      button.appendChild(img);
      button.appendChild(label);
      item.appendChild(button);
      cameraThumbStripEl.appendChild(item);
    }

    renderCameraCropPanel();
  }

  function stopCameraPreviewStream() {
    if (cameraGuideOverlayEl) cameraGuideOverlayEl.classList.remove("active");
    if (cameraGuideFrameRafId) {
      cancelAnimationFrame(cameraGuideFrameRafId);
      cameraGuideFrameRafId = 0;
    }
    clearCameraGuideFrameStyle();
    if (cameraPreviewEl) {
      stopCamera(cameraPreviewEl);
    } else if (cameraStream) {
      stopCamera(cameraStream);
    }
    cameraStream = null;
  }

  function clearCapturedCameraPages() {
    const cleared = cameraQueue.clearAll();
    cameraSelectedCropPageIndex = 0;
    cameraCropDragState = null;
    cameraCropShowZoomPreview = false;
    writeCameraCropMarginsToInputs({ top: 0, right: 0, bottom: 0, left: 0 });
    renderCameraWarnings([]);
    renderCameraThumbnails();
    return cleared;
  }

  function updateCameraControls() {
    const hasPages = Array.isArray(cameraQueue.pages) && cameraQueue.pages.length > 0;
    const hasStream = Boolean(cameraStream);
    const busy = running || bundleBusy || extractingPdf || extractingCamera;
    const cropDisabled = busy || !hasPages;

    if (cameraStartBtn) cameraStartBtn.disabled = busy || hasStream;
    if (cameraCaptureBtn) cameraCaptureBtn.disabled = busy || !hasStream;
    if (cameraRetakeBtn) cameraRetakeBtn.disabled = busy || !hasPages;
    if (cameraClearBtn) cameraClearBtn.disabled = busy || !hasPages;
    if (cameraRunOcrBtn) cameraRunOcrBtn.disabled = busy || !hasPages;
    if (cameraCropPageSelectEl) cameraCropPageSelectEl.disabled = cropDisabled;
    if (cameraCropTopRangeEl) cameraCropTopRangeEl.disabled = cropDisabled;
    if (cameraCropRightRangeEl) cameraCropRightRangeEl.disabled = cropDisabled;
    if (cameraCropBottomRangeEl) cameraCropBottomRangeEl.disabled = cropDisabled;
    if (cameraCropLeftRangeEl) cameraCropLeftRangeEl.disabled = cropDisabled;
    if (cameraCropApplyBtn) cameraCropApplyBtn.disabled = cropDisabled;
    if (cameraCropApplyAllBtn) cameraCropApplyAllBtn.disabled = cropDisabled;
    if (cameraCropResetBtn) cameraCropResetBtn.disabled = cropDisabled;
    if (cameraCropResetAllBtn) cameraCropResetAllBtn.disabled = cropDisabled;
  }

  function resetCameraModalUi() {
    setCameraStatus("Start camera to capture pages.");
    setCameraProgress("");
    updateCameraGuideHint();
    cameraCropDragState = null;
    cameraCropShowZoomPreview = false;
    writeCameraCropMarginsToInputs({ top: 0, right: 0, bottom: 0, left: 0 });
    renderCameraWarnings([]);
    renderCameraThumbnails();
    updateCameraControls();
  }

  async function ensureCameraWorker() {
    if (cameraWorker) return cameraWorker;
    cameraWorker = new Worker(makeCameraWorkerUrl(), { type: "module" });
    return cameraWorker;
  }

  function cancelCameraOcrIfRunning() {
    if (!extractingCamera) return;
    if (!cameraWorker || !cameraOcrJobId) return;
    cancelCameraOcrJob(cameraWorker, cameraOcrJobId);
  }

  function setCameraCapabilityState() {
    const support = canUseCameraScan();
    if (cameraScanBtn) {
      cameraScanBtn.disabled = !support.ok;
    }

    if (!support.ok) {
      setCameraScanSummary(
        "Camera scan requires HTTPS and a browser with live camera + worker OCR support.",
        "warning",
      );
      return;
    }

    setCameraScanSummary(CAMERA_SCAN_HELP_TEXT, "neutral");
  }

  function collectCameraWarnings(pages) {
    const warnings = [];
    for (const page of Array.isArray(pages) ? pages : []) {
      for (const warning of Array.isArray(page?.warnings) ? page.warnings : []) {
        warnings.push(`Page ${Number(page.pageIndex) + 1}: ${String(warning)}`);
      }
    }
    return warnings;
  }

  function applyCurrentCropToSelectedPage() {
    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    if (!pages.length) return;
    cameraSelectedCropPageIndex = clamp(cameraSelectedCropPageIndex, 0, pages.length - 1);
    const crop = cameraCropMarginsToRect(readCameraCropMarginsFromInputs());
    cameraQueue.setPageCrop(cameraSelectedCropPageIndex, crop);
    cameraCropShowZoomPreview = Boolean(crop);
    setCameraStatus(
      crop
        ? `Applied crop to page ${cameraSelectedCropPageIndex + 1}.`
        : `Cleared crop on page ${cameraSelectedCropPageIndex + 1}.`,
    );
    renderCameraThumbnails();
    updateCameraControls();
  }

  function applyCurrentCropToAllPages() {
    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    if (!pages.length) return;
    const crop = cameraCropMarginsToRect(readCameraCropMarginsFromInputs());
    for (let i = 0; i < pages.length; i += 1) {
      cameraQueue.setPageCrop(i, crop);
    }
    cameraCropShowZoomPreview = Boolean(crop);
    setCameraStatus(
      crop
        ? `Applied crop to all ${pages.length} page(s).`
        : `Cleared crop on all ${pages.length} page(s).`,
    );
    renderCameraThumbnails();
    updateCameraControls();
  }

  function resetSelectedPageCrop() {
    const pages = Array.isArray(cameraQueue.pages) ? cameraQueue.pages : [];
    if (!pages.length) return;
    cameraSelectedCropPageIndex = clamp(cameraSelectedCropPageIndex, 0, pages.length - 1);
    cameraQueue.setPageCrop(cameraSelectedCropPageIndex, null);
    cameraCropShowZoomPreview = false;
    loadCameraCropControlsFromPage(cameraSelectedCropPageIndex);
    setCameraStatus(`Reset crop on page ${cameraSelectedCropPageIndex + 1}.`);
    renderCameraThumbnails();
    updateCameraControls();
  }

  function resetAllPageCrops() {
    const cleared = cameraQueue.clearAllCrops();
    cameraCropShowZoomPreview = false;
    writeCameraCropMarginsToInputs({ top: 0, right: 0, bottom: 0, left: 0 });
    if (cleared > 0) {
      setCameraStatus(`Reset crop on ${cleared} page(s).`);
    } else {
      setCameraStatus("No cropped pages to reset.");
    }
    renderCameraThumbnails();
    updateCameraControls();
  }

  function closeCameraModal() {
    cameraModalOpen = false;
    cancelCameraOcrIfRunning();
    stopCameraPreviewStream();
    clearCapturedCameraPages();
    if (cameraScanModalEl?.open) {
      try {
        cameraScanModalEl.close();
      } catch {
        // ignore
      }
    }
    setCameraStatus("Start camera to capture pages.");
    setCameraProgress("");
    updateCameraControls();
    updateZkControls();
  }

  function openCameraModal() {
    const support = canUseCameraScan();
    if (!support.ok) {
      setCameraScanSummary(
        "Camera scan requires HTTPS and a browser with live camera + worker OCR support.",
        "warning",
      );
      setStatus("Camera scan unavailable in this browser.");
      return;
    }

    cameraModalOpen = true;
    resetCameraModalUi();
    if (!cameraScanModalEl) return;
    if (!cameraScanModalEl.open) {
      try {
        cameraScanModalEl.showModal();
      } catch {
        cameraScanModalEl.setAttribute("open", "");
      }
    }
    updateCameraGuideHint();
    scheduleCameraGuideFrameUpdate();
    setCameraStatus("Tap Start Camera to begin.");
  }

  async function startCameraPreview() {
    if (running || bundleBusy || extractingPdf || extractingCamera) return;
    if (!cameraPreviewEl) return;

    try {
      stopCameraPreviewStream();
      cameraStream = await startCamera(cameraPreviewEl, {
        facingMode: "environment",
        preferredWidth: 1280,
      });
      if (cameraGuideOverlayEl) cameraGuideOverlayEl.classList.add("active");
      scheduleCameraGuideFrameUpdate();
      setCameraStatus("Camera ready. Capture one or more pages.");
      setCameraProgress("");
      updateCameraControls();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setCameraStatus(`Camera start failed: ${msg}`);
      setCameraScanSummary(`Camera start failed: ${msg}`, "error");
      stopCameraPreviewStream();
      updateCameraControls();
    }
  }

  async function captureCameraPage() {
    if (!cameraPreviewEl || !cameraStream) return;
    try {
      const captureMaxDim = cameraOcrQualitySelectEl?.value === "high_accuracy" ? 3400 : 2500;
      const sceneHint = resolveCameraSceneHint();
      let frame = null;
      if (sceneHint === "monitor") {
        const burstSamples = cameraOcrQualitySelectEl?.value === "high_accuracy" ? 6 : 5;
        setCameraStatus(`Holding steady... sampling ${burstSamples} frames for best monitor capture.`);
        frame = await captureBestFrame(cameraPreviewEl, {
          maxDim: captureMaxDim,
          framesToSample: burstSamples,
          delayMs: 110,
          onProgress: (progress) => {
            const sampleIndex = Number(progress?.sampleIndex) || 0;
            const sampleCount = Number(progress?.sampleCount) || burstSamples;
            setCameraProgress(`stabilize (${sampleIndex}/${sampleCount})`);
          },
        });
      } else {
        frame = await captureFrame(cameraPreviewEl, { maxDim: captureMaxDim });
      }
      const quality = evaluateCapturedFrameQuality(frame);
      cameraQueue.addPage({
        bitmap: frame.bitmap,
        blob: frame.blob,
        width: frame.width,
        height: frame.height,
        warnings: quality.warnings,
      });
      renderCameraThumbnails();
      renderCameraWarnings(collectCameraWarnings(cameraQueue.pages));
      if (quality.warnings.length > 0) {
        setCameraStatus(
          `Captured page ${cameraQueue.pages.length} with quality warning: ${quality.warnings[0]}`,
        );
      } else {
        setCameraStatus(`Captured page ${cameraQueue.pages.length}.`);
      }
      setCameraProgress("");
      updateCameraControls();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setCameraStatus(`Capture failed: ${msg}`);
      setCameraProgress("");
    }
  }

  async function runCameraOcrAndLoadEditor() {
    if (running || bundleBusy || extractingPdf || extractingCamera) return;
    const pagesForOcr = cameraQueue.exportForOcr();
    if (!pagesForOcr.length) {
      setCameraStatus("Capture at least one page before running OCR.");
      return;
    }
    const croppedPageCount = pagesForOcr.filter((page) => page?.crop).length;

    const ocrMode = cameraOcrQualitySelectEl?.value === "high_accuracy" ? "high_accuracy" : "fast";
    const enhanceValue = String(cameraEnhanceSelectEl?.value || "auto");
    const preprocessMode = enhanceValue === "bw_high_contrast"
      ? "bw_high_contrast"
      : enhanceValue === "grayscale"
        ? "grayscale"
        : enhanceValue === "off"
          ? "off"
          : "auto";
    const sceneHint = resolveCameraSceneHint();

    extractingCamera = true;
    setCameraStatus("Running local OCR...");
    setCameraProgress("Preparing OCR worker...");
    updateCameraControls();
    updateZkControls();

    const jobId = `camera_job_${Date.now()}`;
    cameraOcrJobId = jobId;

    try {
      const worker = await ensureCameraWorker();
      const result = await runCameraOcrJob(
        worker,
        pagesForOcr,
        {
          jobId,
          lang: "eng",
          mode: ocrMode,
          preprocess: { mode: preprocessMode },
          sceneHint,
          warningProfile: cameraWarningProfile,
        },
        {
          onProgress: (event) => {
            const stage = String(event?.stage || "ocr");
            const pageText = Number.isFinite(event?.pageIndex) ? ` page ${Number(event.pageIndex) + 1}` : "";
            const pct = Number.isFinite(event?.pct) ? ` (${Math.round(Number(event.pct) * 100)}%)` : "";
            setCameraProgress(`${stage}${pageText}${pct}`);
          },
        },
      );

      const ocrPages = Array.isArray(result.pages) ? result.pages : [];
      const mergedText = buildCameraOcrDocumentText(ocrPages);

      if (!mergedText.trim()) {
        setCameraStatus("OCR returned no text. Retake images and try again.");
        return;
      }

      const warnings = collectCameraWarnings(ocrPages);
      renderCameraWarnings(warnings);

      setEditorText(mergedText);
      originalText = mergedText;
      hasRunDetection = false;
      setScrubbedConfirmed(false);
      clearDetections();
      clearResultsUi();
      resetFeedbackDraft();
      resetPdfUploadUi();

      const summary =
        `Loaded ${ocrPages.length} camera page(s). OCR mode: ${ocrMode === "high_accuracy" ? "high accuracy" : "fast"}. ` +
        `Enhance: ${preprocessMode}. Capture: ${sceneHint}.` +
        (croppedPageCount > 0 ? ` Cropped pages: ${croppedPageCount}/${pagesForOcr.length}.` : "");
      setCameraScanSummary(summary, "success");
      setStatus("Camera OCR text loaded into editor. Run Detection to use existing PHI redaction workflow.");
      setProgress("");

      extractingCamera = false;
      closeCameraModal();
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      if (error?.name === "AbortError" || /cancel/i.test(msg)) {
        setCameraStatus("Camera OCR cancelled.");
      } else {
        setCameraStatus(`Camera OCR failed: ${msg}`);
        setCameraScanSummary(`Camera OCR failed: ${msg}`, "error");
      }
    } finally {
      extractingCamera = false;
      cameraOcrJobId = "";
      setCameraProgress("");
      updateCameraControls();
      updateZkControls();
    }
  }

  async function extractSelectedPdfIntoEditor(file) {
    if (!file) return;
    if (running || bundleBusy || extractingPdf || extractingCamera) return;

    const looksLikePdf = /\.pdf$/i.test(file.name) || file.type === "application/pdf";
    if (!looksLikePdf) {
      setStatus("Please choose a PDF file.");
      return;
    }

    extractingPdf = true;
    publishPdfExtractionMetricsReport({
      generatedAt: new Date().toISOString(),
      status: "running",
      fileName: file?.name || "",
    });
    if (runBtn) runBtn.disabled = true;
    if (cancelBtn) cancelBtn.disabled = true;
    if (applyBtn) applyBtn.disabled = true;
    if (submitBtn) submitBtn.disabled = true;
    updateZkControls();

    setStatus(`Extracting text from ${file.name} with layout-aware local parser...`);
    setProgress("Starting PDF worker...");
    setPdfExtractSummary("Analyzing PDF layout and contamination risk locally...", "neutral");

    try {
      const rawPages = [];
      let completedPages = 0;
      let totalPages = 0;
      let docModel = null;
      let lastOcrError = "";
      const ocrQualityMode = pdfOcrQualitySelectEl?.value === "high_accuracy"
        ? "high_accuracy"
        : "fast";
      const ocrMaskMode = pdfOcrMaskSelectEl?.value === "on"
        ? "on"
        : pdfOcrMaskSelectEl?.value === "off"
          ? "off"
          : "auto";

      for await (const event of extractPdfAdaptive(file, {
        ocr: {
          available: true,
          enabled: true,
          lang: "eng",
          qualityMode: ocrQualityMode,
          maskImages: ocrMaskMode,
        },
        gate: {
          minCompletenessConfidence: 0.72,
          maxContaminationScore: 0.24,
          hardBlockWhenUnsafeWithoutOcr: true,
        },
      })) {
        if (event.kind === "progress") {
          completedPages = event.completedPages;
          totalPages = event.totalPages;
          setProgress(`PDF extraction: ${completedPages}/${totalPages} pages`);
        } else if (event.kind === "ocr_progress") {
          setProgress(`OCR pass: ${event.completedPages}/${event.totalPages} pages`);
        } else if (event.kind === "ocr_status") {
          const pct = Number.isFinite(event.progress) ? Math.round(event.progress * 100) : 0;
          setProgress(`OCR ${event.status || "processing"} (${pct}%)`);
        } else if (event.kind === "ocr_error") {
          lastOcrError = String(event.error || "");
          setStatus(`OCR failed (${event.error}); using native fallback and gate checks.`);
        } else if (event.kind === "stage") {
          const stageLabel = formatPdfStage(event.stage);
          if (Number.isFinite(event.totalPages) && event.totalPages > 0) {
            setProgress(`PDF ${stageLabel}: page ${event.pageIndex + 1}/${event.totalPages}`);
          } else {
            setProgress(`PDF ${stageLabel}...`);
          }
        } else if (event.kind === "page") {
          rawPages[event.page.pageIndex] = event.page;
        } else if (event.kind === "done") {
          docModel = event.document || null;
        }
      }

      const pages = rawPages.filter(Boolean).sort((a, b) => a.pageIndex - b.pageIndex);
      if (!pages.length) {
        setStatus("No text could be extracted from that PDF.");
        setPdfExtractSummary("No extractable text was found in this PDF.", "warning");
        setProgress("");
        return;
      }

      if (!docModel) {
        docModel = buildPdfDocumentModel(file, pages, {
          ocr: {
            available: true,
            enabled: true,
            lang: "eng",
            qualityMode: ocrQualityMode,
          },
          gate: {
            minCompletenessConfidence: 0.72,
            maxContaminationScore: 0.24,
            hardBlockWhenUnsafeWithoutOcr: true,
          },
        });
      }

      if (docModel.blocked) {
        const baseSummary =
          `Blocked loading extracted text from ${file.name}. ` +
          `Safety gate triggered because native extraction appears incomplete and OCR is unavailable or failed.`;
        const qualityDetail =
          `Low-confidence pages: ${docModel.qualitySummary?.lowConfidencePages || 0}; ` +
          `contaminated pages: ${docModel.qualitySummary?.contaminatedPages || 0}.`;
        const reason = docModel.blockReason || "Unsafe native extraction.";
        const ocrFailureDetail = lastOcrError ? ` OCR error: ${lastOcrError}.` : "";
        setPdfExtractSummary(
          `${baseSummary} ${qualityDetail} Reason: ${reason}.${ocrFailureDetail}`,
          "error",
        );
        setStatus("PDF extraction blocked for safety. OCR path is required for this document.");
        setProgress(
          totalPages > 0
            ? `PDF extraction blocked after ${completedPages}/${totalPages} pages`
            : "PDF extraction blocked",
        );
        publishPdfExtractionMetricsReport(buildPdfExtractionMetricsReport(docModel, {
          fileName: file.name,
          ocrQualityMode,
          ocrMaskMode,
          status: "blocked",
        }));
        return;
      }

      const normalizedText = docModel.fullText.startsWith("\n")
        ? docModel.fullText.slice(1)
        : docModel.fullText;

      setEditorText(normalizedText);
      originalText = normalizedText;
      hasRunDetection = false;
      setScrubbedConfirmed(false);
      clearDetections();
      clearResultsUi();
      resetFeedbackDraft();
      if (runBtn) runBtn.disabled = extractingPdf || extractingCamera || !workerReady;

      const ocrNeededPages = docModel.pages.filter((page) => page.sourceDecision !== "native").length;
      const summaryText =
        `Loaded ${docModel.pages.length} page(s) from ${file.name} (${formatFileSize(file.size)}). ` +
        (ocrNeededPages
          ? `${ocrNeededPages} page(s) used OCR/hybrid recovery after layout contamination checks.`
          : "All pages passed native layout safety checks.");
      const ocrModeText = ` OCR mode: ${ocrQualityMode === "high_accuracy" ? "high accuracy" : "fast"}.`;
      const ocrMaskText = ` OCR mask: ${ocrMaskMode}.`;
      const qualityTail =
        ` Low-confidence pages: ${docModel.qualitySummary?.lowConfidencePages || 0}; ` +
        `contaminated pages: ${docModel.qualitySummary?.contaminatedPages || 0}.`;
      const debugTail = buildOcrDebugSummary(docModel, 2);
      const metricTail = buildPageMetricsSummary(docModel, 2);
      const reasonTail = buildPageReasonSummary(docModel, 2);
      setPdfExtractSummary(`${summaryText}${ocrModeText}${ocrMaskText}${qualityTail}${debugTail}${metricTail}${reasonTail ? ` ${reasonTail}` : ""}`, "success");
      publishPdfExtractionMetricsReport(buildPdfExtractionMetricsReport(docModel, {
        fileName: file.name,
        ocrQualityMode,
        ocrMaskMode,
        status: "success",
      }));

      setStatus("PDF text loaded into editor. Run Detection to use the existing PHI redaction workflow.");
      if (totalPages > 0) {
        setProgress(`PDF extraction complete: ${completedPages}/${totalPages} pages`);
      } else {
        setProgress("");
      }
    } catch (error) {
      const msg = error instanceof Error ? error.message : String(error);
      setStatus(`PDF extraction failed: ${msg}`);
      setPdfExtractSummary(`PDF extraction failed: ${msg}`, "error");
      setProgress("");
      publishPdfExtractionMetricsReport({
        generatedAt: new Date().toISOString(),
        status: "error",
        fileName: file?.name || "",
        error: msg,
      });
    } finally {
      extractingPdf = false;
      if (runBtn) runBtn.disabled = extractingPdf || extractingCamera || !workerReady;
      if (applyBtn) applyBtn.disabled = running || !hasRunDetection;
      if (revertBtn) revertBtn.disabled = running || originalText === model.getValue();
      if (submitBtn) submitBtn.disabled = !scrubbedConfirmed || running;
      updateZkControls();
    }
  }

  if (completenessCopyBtn) {
    completenessCopyBtn.disabled = true;
    completenessCopyBtn.addEventListener("click", () => {
      const text = buildCompletenessChecklistText(lastCompletenessPrompts);
      copyToClipboard(text).then((ok) => {
        setStatus(ok ? "Completeness checklist copied to clipboard." : "Copy failed (clipboard unavailable).");
      });
    });
  }

  if (completenessOpenReporterBtn) {
    completenessOpenReporterBtn.disabled = true;
    completenessOpenReporterBtn.addEventListener("click", () => {
      if (!scrubbedConfirmed) {
        setStatus("Apply redactions before sending note to Reporter Builder.");
        return;
      }
      const note = String(model.getValue() || "").trim();
      if (!note) {
        setStatus("No scrubbed note text available to send.");
        return;
      }

      const payload = JSON.stringify({
        note,
        source: "dashboard",
        note_type: "scrubbed_note",
        transferred_at: new Date().toISOString(),
      });

      const wroteSession = safeSetStorageItem(globalThis.sessionStorage, DASHBOARD_REPORTER_TRANSFER_KEY, payload);
      const wroteLocal = safeSetStorageItem(globalThis.localStorage, DASHBOARD_REPORTER_TRANSFER_KEY, payload);
      if (!wroteSession && !wroteLocal) {
        setStatus("Browser storage unavailable. Transfer to Reporter Builder failed.");
        return;
      }
      window.location.href = "./reporter_builder.html";
    });
  }

  if (pdfUploadInputEl) {
    pdfUploadInputEl.addEventListener("change", () => {
      const file = pdfUploadInputEl.files && pdfUploadInputEl.files[0];
      if (!file) {
        setPdfExtractSummary(PDF_UPLOAD_HELP_TEXT, "neutral");
        updateZkControls();
        return;
      }

      const looksLikePdf = /\.pdf$/i.test(file.name) || file.type === "application/pdf";
      setPdfExtractSummary(
        looksLikePdf
          ? `Selected PDF: ${file.name} (${formatFileSize(file.size)}). Click "Extract PDF Text" to run local layout-aware extraction and safety gating.`
          : `Selected file is not a PDF: ${file.name}`,
        looksLikePdf ? "neutral" : "error",
      );
      updateZkControls();
    });
  }

  if (pdfExtractBtn) {
    pdfExtractBtn.addEventListener("click", () => {
      const file = pdfUploadInputEl?.files?.[0];
      extractSelectedPdfIntoEditor(file);
    });
  }

  setCameraCapabilityState();
  resetCameraModalUi();

  if (cameraScanBtn) {
    cameraScanBtn.addEventListener("click", () => {
      openCameraModal();
      updateZkControls();
    });
  }

  if (cameraStartBtn) {
    cameraStartBtn.addEventListener("click", () => {
      startCameraPreview();
    });
  }

  if (cameraSceneHintSelectEl) {
    cameraSceneHintSelectEl.addEventListener("change", () => {
      updateCameraGuideHint();
      scheduleCameraGuideFrameUpdate();
    });
  }

  if (cameraPreviewEl) {
    cameraPreviewEl.addEventListener("loadedmetadata", () => {
      scheduleCameraGuideFrameUpdate();
    });
    cameraPreviewEl.addEventListener("resize", () => {
      scheduleCameraGuideFrameUpdate();
    });
  }

  window.addEventListener("resize", () => {
    if (!cameraModalOpen) return;
    scheduleCameraGuideFrameUpdate();
  });
  window.addEventListener("orientationchange", () => {
    if (!cameraModalOpen) return;
    scheduleCameraGuideFrameUpdate();
  });

  if (cameraCaptureBtn) {
    cameraCaptureBtn.addEventListener("click", () => {
      captureCameraPage();
    });
  }

  if (cameraRetakeBtn) {
    cameraRetakeBtn.addEventListener("click", () => {
      const removed = cameraQueue.retakeLast();
      if (!removed) {
        setCameraStatus("No captured pages to retake.");
      } else {
        if (cameraQueue.pages.length > 0) {
          cameraSelectedCropPageIndex = clamp(
            cameraSelectedCropPageIndex,
            0,
            cameraQueue.pages.length - 1,
          );
          loadCameraCropControlsFromPage(cameraSelectedCropPageIndex);
        } else {
          cameraSelectedCropPageIndex = 0;
          cameraCropDragState = null;
          cameraCropShowZoomPreview = false;
          writeCameraCropMarginsToInputs({ top: 0, right: 0, bottom: 0, left: 0 });
        }
        setCameraStatus(`Removed last capture. ${cameraQueue.pages.length} page(s) remaining.`);
      }
      renderCameraWarnings(collectCameraWarnings(cameraQueue.pages));
      renderCameraThumbnails();
      updateCameraControls();
    });
  }

  if (cameraClearBtn) {
    cameraClearBtn.addEventListener("click", () => {
      const count = clearCapturedCameraPages();
      setCameraStatus(count ? `Cleared ${count} captured page(s).` : "No captured pages to clear.");
      updateCameraControls();
    });
  }

  if (cameraCropPageSelectEl) {
    cameraCropPageSelectEl.addEventListener("change", () => {
      const nextIndex = Number(cameraCropPageSelectEl.value);
      loadCameraCropControlsFromPage(nextIndex);
      renderCameraThumbnails();
      updateCameraControls();
    });
  }

  const cameraCropRangeInputs = [
    cameraCropTopRangeEl,
    cameraCropRightRangeEl,
    cameraCropBottomRangeEl,
    cameraCropLeftRangeEl,
  ].filter(Boolean);
  for (const input of cameraCropRangeInputs) {
    input.addEventListener("input", () => {
      updateCameraCropPreviewFromInputs();
    });
  }

  if (cameraCropPreviewImgEl) {
    cameraCropPreviewImgEl.addEventListener("load", () => {
      renderCameraCropPreview();
    });
  }

  if (cameraCropPreviewStageEl) {
    cameraCropPreviewStageEl.addEventListener("pointerdown", (event) => {
      beginCameraCropDrag(event);
    });
    cameraCropPreviewStageEl.addEventListener("pointermove", (event) => {
      updateCameraCropDrag(event);
    });
    cameraCropPreviewStageEl.addEventListener("pointerup", (event) => {
      endCameraCropDrag(event);
    });
    cameraCropPreviewStageEl.addEventListener("pointercancel", (event) => {
      endCameraCropDrag(event);
    });
    cameraCropPreviewStageEl.addEventListener("lostpointercapture", () => {
      cameraCropDragState = null;
    });
  }

  if (cameraCropApplyBtn) {
    cameraCropApplyBtn.addEventListener("click", () => {
      applyCurrentCropToSelectedPage();
    });
  }

  if (cameraCropApplyAllBtn) {
    cameraCropApplyAllBtn.addEventListener("click", () => {
      applyCurrentCropToAllPages();
    });
  }

  if (cameraCropResetBtn) {
    cameraCropResetBtn.addEventListener("click", () => {
      resetSelectedPageCrop();
    });
  }

  if (cameraCropResetAllBtn) {
    cameraCropResetAllBtn.addEventListener("click", () => {
      resetAllPageCrops();
    });
  }

  if (cameraRunOcrBtn) {
    cameraRunOcrBtn.addEventListener("click", () => {
      runCameraOcrAndLoadEditor();
    });
  }

  if (cameraScanModalEl) {
    cameraScanModalEl.addEventListener("cancel", (event) => {
      event.preventDefault();
      closeCameraModal();
    });
    cameraScanModalEl.addEventListener("close", () => {
      if (cameraModalOpen) closeCameraModal();
    });
    cameraScanModalEl.addEventListener("click", () => {
      scheduleCameraGuideFrameUpdate();
    });
  }

  if (cameraCloseBtn) {
    cameraCloseBtn.addEventListener("click", () => {
      closeCameraModal();
    });
  }

  privacyShieldTeardown = initPrivacyShield({
    shieldEl: privacyShieldEl,
    shouldActivate: () => cameraModalOpen || Boolean(cameraStream) || extractingCamera,
    onBackground: () => {
      stopCameraPreviewStream();
      cancelCameraOcrIfRunning();
      if (cameraModalOpen) {
        setCameraStatus("Privacy shield active. Tap to resume and restart camera if needed.");
      }
      updateCameraControls();
    },
    onResumeRequested: () => {
      if (cameraModalOpen) {
        setCameraStatus("Resumed. Tap Start Camera to restore live preview.");
      }
      updateCameraControls();
    },
  });

  let detections = [];
  let detectionsById = new Map();
  let excluded = new Set();
  let decorations = [];

  let currentSelection = null;

  // Track selection changes for manual redaction
  if (!usingPlainEditor && editor) {
    editor.onDidChangeCursorSelection((e) => {
      const selection = e.selection;
      const hasSelection = !selection.isEmpty();

      currentSelection = hasSelection ? selection : null;

      // Only enable Add button if we have a selection and aren't running detection
      if (addRedactionBtn) {
        addRedactionBtn.disabled = !hasSelection || running;
      }
    });
  } else if (fallbackTextarea) {
    const updateSelection = () => {
      const start = fallbackTextarea.selectionStart;
      const end = fallbackTextarea.selectionEnd;
      const hasSelection = Number.isFinite(start) && Number.isFinite(end) && end > start;

      currentSelection = hasSelection ? { start, end } : null;

      if (addRedactionBtn) {
        addRedactionBtn.disabled = !hasSelection || running;
      }
    };

    fallbackTextarea.addEventListener("select", updateSelection);
    fallbackTextarea.addEventListener("mouseup", updateSelection);
    fallbackTextarea.addEventListener("keyup", updateSelection);
    updateSelection();
  }

  function setScrubbedConfirmed(value) {
    scrubbedConfirmed = value;
    submitBtn.disabled = !scrubbedConfirmed || running;
    if (completenessOpenReporterBtn) completenessOpenReporterBtn.disabled = !scrubbedConfirmed || running;
    // Update button title for better UX
    if (submitBtn.disabled) {
      if (running) {
        submitBtn.title = "Wait for detection to complete";
      } else if (!scrubbedConfirmed) {
        submitBtn.title = "Click 'Apply redactions' first";
      }
    } else {
      submitBtn.title = "Submit the scrubbed note to the server";
    }
    updateZkControls();
  }

  function updateZkControls() {
    const busy = running || bundleBusy || extractingPdf || extractingCamera;
    const hasText = Boolean(model.getValue().trim());
    const hasPdfSelected = Boolean(pdfUploadInputEl?.files && pdfUploadInputEl.files.length > 0);
    const cameraSupport = canUseCameraScan();
    if (chronoPreviewBtn) chronoPreviewBtn.disabled = busy || !hasRunDetection;
    if (clearCurrentNoteBtn) clearCurrentNoteBtn.disabled = busy || !hasText;
    if (genBundleIdsBtn) genBundleIdsBtn.disabled = busy;
    if (addToBundleBtn) addToBundleBtn.disabled = busy || !scrubbedConfirmed;
    const hasBundleDocs = Array.isArray(bundleDocs) && bundleDocs.length > 0;
    if (submitBundleBtn) submitBundleBtn.disabled = busy || !hasBundleDocs;
    if (clearBundleBtn) clearBundleBtn.disabled = busy || !hasBundleDocs;
    if (pdfUploadInputEl) pdfUploadInputEl.disabled = busy;
    if (pdfOcrQualitySelectEl) pdfOcrQualitySelectEl.disabled = busy;
    if (pdfOcrMaskSelectEl) pdfOcrMaskSelectEl.disabled = busy;
    if (pdfExtractBtn) pdfExtractBtn.disabled = busy || !hasPdfSelected;
    if (cameraScanBtn) cameraScanBtn.disabled = busy || !cameraSupport.ok;
    updateCameraControls();
  }

  function clearDetections() {
    detections = [];
    detectionsById = new Map();
    excluded = new Set();
    if (!usingPlainEditor && editor) {
      decorations = editor.deltaDecorations(decorations, []);
    } else {
      decorations = [];
    }
    detectionsListEl.innerHTML = "";
    detectionsCountEl.textContent = "0";
    applyBtn.disabled = true;
    revertBtn.disabled = true;
    lastServerResponse = null;
    if (exportBtn) exportBtn.disabled = true;
    clearResultsUi();
    updateZkControls();
  }

  function updateDecorations() {
    if (usingPlainEditor || !editor) return;

    const text = model.getValue();
    const lineStarts = buildLineStartOffsets(text);
    const textLength = text.length;

    const included = detections.filter((d) => !excluded.has(d.id));
    const newDecorations = included
      .filter((d) => Number.isFinite(d.start) && Number.isFinite(d.end) && d.end > d.start)
      .map((d) => {
        const startPos = offsetToPosition(d.start, lineStarts, textLength);
        const endPos = offsetToPosition(d.end, lineStarts, textLength);

        // Determine class and hover based on source (manual vs auto)
        const className = d.source === "manual"
          ? "phi-detection-manual"
          : "phi-detection";

        const hoverMessage = d.source === "manual"
          ? `**${d.label}** (Manual)`
          : `**${d.label}** (${d.source}, score ${formatScore(d.score)})`;

        return {
          range: new monaco.Range(
            startPos.lineNumber,
            startPos.column,
            endPos.lineNumber,
            endPos.column
          ),
          options: {
            inlineClassName: className,
            hoverMessage: { value: hoverMessage },
          },
        };
      });

    decorations = editor.deltaDecorations(decorations, newDecorations);
  }

  function renderDetections() {
    const text = model.getValue();
    detectionsCountEl.textContent = String(detections.length);
    detectionsListEl.innerHTML = "";

    const sorted = [...detections].sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return (b.score ?? 0) - (a.score ?? 0);
    });

    for (const d of sorted) {
      const checked = !excluded.has(d.id);
      const checkbox = el("input", {
        type: "checkbox",
        checked: checked ? "checked" : null,
        onChange: (ev) => {
          const on = ev.target.checked;
          if (!on) excluded.add(d.id);
          else excluded.delete(d.id);
          updateDecorations();
        },
      });
      checkbox.checked = checked;

      // Add conditional classes for manual detections
      const sourceClass = d.source === "manual" ? "pill source source-manual" : "pill source";
      const scoreText = d.source === "manual" ? "Manual" : `score ${formatScore(d.score)}`;

      const meta = el("div", { className: "detMeta" }, [
        el("span", { className: "pill label", text: d.label }),
        el("span", { className: sourceClass, text: d.source }),
        el("span", { className: "pill score", text: scoreText }),
        el("span", { className: "pill", text: `${d.start}–${d.end}` }),
      ]);

      const snippet = el("div", {
        className: "snippet",
        text: safeSnippet(text, d.start, d.end),
      });

      detectionsListEl.appendChild(
        el("div", { className: "detRow" }, [
          checkbox,
          el("div", {}, [meta, snippet]),
        ])
      );
    }

    if (detections.length === 0 && hasRunDetection && !running) {
      detectionsListEl.innerHTML = '<div class="subtle" style="padding: 1rem; text-align: center;">No PHI detected. Click "Apply redactions" to enable submit.</div>';
    }

    updateDecorations();
    // Enable apply button if detection has completed (even with 0 detections)
    applyBtn.disabled = running || !hasRunDetection;
    if (applyBtn.disabled) {
      if (running) {
        applyBtn.title = "Wait for detection to complete";
      } else if (!hasRunDetection) {
        applyBtn.title = "Click 'Run detection' first";
      }
    } else {
      applyBtn.title = "Apply redactions to enable submit button";
    }
    revertBtn.disabled = running || originalText === model.getValue();
  }

  if (!usingPlainEditor && typeof model?.onDidChangeContent === "function") {
    model.onDidChangeContent(() => {
      if (suppressDirtyFlag) return;
      setScrubbedConfirmed(false);
      revertBtn.disabled = running || originalText === model.getValue();
    });
  } else if (fallbackTextarea) {
    fallbackTextarea.addEventListener("input", () => {
      if (suppressDirtyFlag) return;
      setScrubbedConfirmed(false);
      revertBtn.disabled = running || originalText === model.getValue();
    });
  }

  let worker = null;
  let workerReady = false;
  let lastWorkerMessageAt = Date.now();
  let aiModelReady = false;
  let aiModelFailed = false;
  let aiModelError = null;
  let legacyFallbackAttempted = false;
  let usingLegacyWorker = false;
  let workerInitTimer = null;

  function clearWorkerInitTimer() {
    if (!workerInitTimer) return;
    clearTimeout(workerInitTimer);
    workerInitTimer = null;
  }

  function shouldForceLegacyWorker() {
    const params = new URLSearchParams(location.search);
    if (params.get("legacy") === "1") return true;
    return isMobileSafariBrowser;
  }

  function buildWorkerUrl(name) {
    return `/ui/${name}?v=${Date.now()}`;
  }

  function startWorker({ forceLegacy = false } = {}) {
    if (worker) {
      try {
        worker.terminate();
      } catch (err) {
        // ignore
      }
    }

    workerReady = false;
    aiModelReady = false;
    aiModelFailed = false;
    aiModelError = null;

    let nextWorker = null;
    let nextIsLegacy = forceLegacy;

    if (!forceLegacy) {
      try {
        nextWorker = new Worker(buildWorkerUrl("redactor.worker.js"), { type: "module" });
        nextIsLegacy = false;
      } catch (err) {
        legacyFallbackAttempted = true;
        nextIsLegacy = true;
        setStatus("Module worker unsupported; falling back to legacy worker…");
      }
    }

    if (!nextWorker) {
      nextWorker = new Worker(buildWorkerUrl("redactor.worker.legacy.js"));
      nextIsLegacy = true;
    }

    worker = nextWorker;
    usingLegacyWorker = nextIsLegacy;
    attachWorkerHandlers(worker);
    worker.postMessage({ type: "init", debug: WORKER_CONFIG.debug, config: WORKER_CONFIG });
    clearWorkerInitTimer();
    workerInitTimer = setTimeout(() => {
      if (!workerReady) {
        setStatus(
          "Worker initializing… (first load can take a few minutes). If this stalls, check DevTools."
        );
      }
    }, 8000);
  }

  function attachWorkerHandlers(activeWorker) {
    activeWorker.addEventListener("error", (ev) => {
      clearWorkerInitTimer();
      if (!usingLegacyWorker && !legacyFallbackAttempted) {
        legacyFallbackAttempted = true;
        setStatus("Module worker failed to load; falling back to legacy worker…");
        setProgress("");
        running = false;
        cancelBtn.disabled = true;
        runBtn.disabled = true;
        applyBtn.disabled = true;
        updateZkControls();
        startWorker({ forceLegacy: true });
        return;
      }
      setStatus(`Worker error: ${ev.message || "failed to load"}`);
      setProgress("");
      running = false;
      cancelBtn.disabled = true;
      runBtn.disabled = true;
      applyBtn.disabled = true;
      updateZkControls();
    });

    activeWorker.addEventListener("messageerror", () => {
      clearWorkerInitTimer();
      setStatus("Worker message error (serialization failed)");
      setProgress("");
      running = false;
      cancelBtn.disabled = true;
      runBtn.disabled = true;
      applyBtn.disabled = true;
      updateZkControls();
    });

    activeWorker.onmessage = (e) => {
      const msg = e.data;
      if (!msg || typeof msg.type !== "string") return;
      lastWorkerMessageAt = Date.now();

      if (msg.type === "ready") {
        clearWorkerInitTimer();
        workerReady = true;
        setStatus("Ready (local model loaded)");
        setProgress("");
        runBtn.disabled = extractingPdf || extractingCamera || !workerReady;
        return;
      }

      if (msg.type === "progress") {
        const stage = msg.stage ? String(msg.stage) : null;
        if (stage) {
          if (stage.startsWith("AI model ready")) {
            aiModelReady = true;
            aiModelFailed = false;
            aiModelError = null;
            if (!running) setStatus("Ready (AI model loaded)");
          } else if (stage.startsWith("AI model failed")) {
            aiModelReady = false;
            aiModelFailed = true;
            aiModelError = stage.includes(":") ? stage.split(":").slice(1).join(":").trim() : null;
            const shortErr =
              aiModelError && aiModelError.length > 120
                ? `${aiModelError.slice(0, 117)}…`
                : aiModelError;
            if (!running) {
              setStatus(
                shortErr
                  ? `Ready (regex-only; AI failed: ${shortErr})`
                  : "Ready (regex-only; AI model failed)"
              );
            }
          }
          if (msg.windowCount && msg.windowIndex) {
            setProgress(`${stage} (${msg.windowIndex}/${msg.windowCount})`);
          } else {
            setProgress(stage);
          }
        } else {
          const percent = msg.windowCount
            ? Math.round((msg.windowIndex / msg.windowCount) * 100)
            : 0;
          setProgress(`Processing window ${msg.windowIndex}/${msg.windowCount} (${percent}%)`);
        }
        return;
      }

      if (msg.type === "detections_delta") {
        for (const det of msg.detections || []) detectionsById.set(det.id, det);
        detections = Array.from(detectionsById.values());
        renderDetections();
        return;
      }

      if (msg.type === "done") {
        running = false;
        cancelBtn.disabled = true;
        runBtn.disabled = extractingPdf || extractingCamera || !workerReady;
        applyBtn.disabled = false; // Enable even with 0 detections
        revertBtn.disabled = originalText === model.getValue();
        updateZkControls();

        detections = ensureDetectionIds(msg.detections);
        detectionsById = new Map(detections.map((d) => [d.id, d]));

        const detectionCount = detections.length;
        if (detectionCount === 0) {
          const modeNote = aiModelReady
            ? "AI+regex"
            : aiModelFailed
            ? "regex-only (AI failed)"
            : "regex-only (AI loading)";
          setStatus(`Done (0 detections) — ${modeNote}`);
        } else {
          const modeNote = aiModelReady
            ? "AI+regex"
            : aiModelFailed
            ? "regex-only (AI failed)"
            : "regex-only (AI loading)";
          setStatus(
            `Done (${detectionCount} detection${detectionCount === 1 ? "" : "s"}) — ${modeNote}`
          );
        }
        setProgress("");
        renderDetections();
        return;
      }

      if (msg.type === "error") {
        clearWorkerInitTimer();
        running = false;
        cancelBtn.disabled = true;
        runBtn.disabled = extractingPdf || extractingCamera || !workerReady;
        applyBtn.disabled = !hasRunDetection;
        updateZkControls();
        setStatus(`Error: ${msg.message || "unknown"}`);
        setProgress("");
        return;
      }
    };
  }

  const forceLegacy = shouldForceLegacyWorker();
  if (forceLegacy) {
    legacyFallbackAttempted = true;
    setStatus("Using legacy worker for Safari compatibility…");
  }
  startWorker({ forceLegacy });

  cancelBtn.addEventListener("click", () => {
    if (!running) return;
    worker.postMessage({ type: "cancel" });
    setStatus("Cancelling…");
  });

  runBtn.addEventListener("click", () => {
    if (running) return;
    if (!workerReady) {
      setStatus("Worker still loading… (first run may take minutes)");
      return;
    }
    hasRunDetection = true;
    setScrubbedConfirmed(false);

    originalText = model.getValue();
    clearDetections();

    running = true;
    runBtn.disabled = true;
    cancelBtn.disabled = false;
    applyBtn.disabled = true;
    revertBtn.disabled = false;
    submitBtn.disabled = true;
    updateZkControls();

    setStatus("Detecting… (client-side)");
    setProgress("");

        worker.postMessage({
          type: "start",
          text: originalText,
          config: buildWorkerConfigForRun(),
        });
      });

  applyBtn.addEventListener("click", () => {
    if (!hasRunDetection) return;

    const included = detections.filter((d) => !excluded.has(d.id));
    const spans = included
      .filter((d) => Number.isFinite(d.start) && Number.isFinite(d.end) && d.end > d.start)
      .sort((a, b) => b.start - a.start);

    const translateDates = Boolean(translateDatesToggleEl?.checked);
    const indexYmd = parseIsoDateInput(indexDateInputEl?.value);
    const baseText = model.getValue();

    suppressDirtyFlag = true;
    try {
      if (!usingPlainEditor && editor) {
        const lineStarts = buildLineStartOffsets(baseText);
        const textLength = baseText.length;

        const edits = spans.map((d) => {
          const startPos = offsetToPosition(d.start, lineStarts, textLength);
          const endPos = offsetToPosition(d.end, lineStarts, textLength);
          const raw = baseText.slice(d.start, d.end);
          const label = String(d.label || "").toUpperCase().replace(/^[BI]-/, "");
          const replacement =
            label === "DATE"
              ? buildDateRedactionReplacement(raw, { translateDates, indexYmd })
              : "[REDACTED]";
          return {
            range: new monaco.Range(
              startPos.lineNumber,
              startPos.column,
              endPos.lineNumber,
              endPos.column
            ),
            text: replacement,
          };
        });

        editor.executeEdits("phi-redactor", edits);
      } else {
        let text = baseText;
        // Apply replacements from the end to preserve offsets.
        for (const d of spans) {
          const start = clamp(d.start, 0, text.length);
          const end = clamp(d.end, 0, text.length);
          if (end <= start) continue;
          const raw = baseText.slice(start, end);
          const label = String(d.label || "").toUpperCase().replace(/^[BI]-/, "");
          const replacement =
            label === "DATE"
              ? buildDateRedactionReplacement(raw, { translateDates, indexYmd })
              : "[REDACTED]";
          text = `${text.slice(0, start)}${replacement}${text.slice(end)}`;
        }
        model.setValue(text);
      }
    } finally {
      suppressDirtyFlag = false;
    }

    setScrubbedConfirmed(true);
    setStatus("Redactions applied (scrubbed text ready to submit)");
    revertBtn.disabled = false;
  });

	  revertBtn.addEventListener("click", () => {
	    suppressDirtyFlag = true;
	    try {
	      if (!usingPlainEditor && editor) editor.setValue(originalText);
	      else model.setValue(originalText);
    } finally {
      suppressDirtyFlag = false;
    }
    clearDetections();
    hasRunDetection = false;
    setScrubbedConfirmed(false);
	    setStatus("Reverted to baseline");
	    setProgress("");
	  });

    function clearCurrentNote() {
      if (running || bundleBusy || extractingPdf || extractingCamera) return;
      setEditorText("");
      originalText = "";
      hasRunDetection = false;
      setScrubbedConfirmed(false);
      clearDetections();
      clearResultsUi();
      resetPdfUploadUi();
      resetFeedbackDraft();
      if (cameraModalOpen) closeCameraModal();
      else {
        clearCapturedCameraPages();
        stopCameraPreviewStream();
      }
      setStatus("Ready for new note");
      setProgress("");
      if (runBtn) runBtn.disabled = extractingPdf || extractingCamera || !workerReady;
      updateZkControls();
    }

    if (clearCurrentNoteBtn) {
      clearCurrentNoteBtn.addEventListener("click", () => {
        clearCurrentNote();
      });
    }

	    if (chronoPreviewBtn) {
	      chronoPreviewBtn.addEventListener("click", () => {
        if (!hasRunDetection) {
          setStatus("Run detection before previewing chronology");
          return;
        }

        const translateDates = Boolean(translateDatesToggleEl?.checked);
        const indexYmd = parseIsoDateInput(indexDateInputEl?.value);
        const docYmd = parseIsoDateInput(docDateInputEl?.value);
        const role = String(timepointRoleSelectEl?.value || "");
        const seq = Number(docSeqInputEl?.value || "0");

        const docOffsetDays = indexYmd && docYmd ? diffDaysUtcNoon(indexYmd, docYmd) : null;
        const systemHeader = docOffsetDays == null ? "" : buildSystemHeaderToken({ role, seq, docOffsetDays });

        const currentText = model.getValue();
        const leakCount = countDateLikeStringsForLeakScan(currentText);

        const included = detections.filter((d) => !excluded.has(d.id));
        const dateSpans = included
          .filter((d) => String(d.label || "").toUpperCase().replace(/^[BI]-/, "") === "DATE")
          .filter((d) => Number.isFinite(d.start) && Number.isFinite(d.end) && d.end > d.start)
          .sort((a, b) => a.start - b.start);

        const maxRows = 200;
        const truncated = dateSpans.length > maxRows;
        const rows = dateSpans.slice(0, maxRows).map((d) => {
          const raw = currentText.slice(d.start, d.end);
          const parsed = parseAbsoluteDateCandidate(raw);
          const replacement = buildDateRedactionReplacement(raw, { translateDates, indexYmd });
          const parsedText = parsed.ymd ? parsed.normalized : "—";
          const notes = parsed.warning || "";
          return `
            <tr>
              <td>${safeHtml(raw)}</td>
              <td>${safeHtml(parsedText)}</td>
              <td>${safeHtml(replacement)}</td>
              <td class="subtle">${safeHtml(notes)}</td>
            </tr>
          `;
        });

        const headerHtml = systemHeader
          ? `<div class="bundle-doc-meta"><strong>Header token:</strong> ${safeHtml(systemHeader)}</div>`
          : `<div class="bundle-doc-meta"><strong>Header token:</strong> (requires Index date + Document date)</div>`;

        const indexHtml = indexYmd
          ? `<div class="bundle-doc-meta"><strong>Index date:</strong> ${safeHtml(indexDateInputEl?.value || "")}</div>`
          : `<div class="bundle-doc-meta"><strong>Index date:</strong> (not set)</div>`;

        const docHtml = docYmd
          ? `<div class="bundle-doc-meta"><strong>Document date:</strong> ${safeHtml(docDateInputEl?.value || "")}</div>`
          : `<div class="bundle-doc-meta"><strong>Document date:</strong> (not set)</div>`;

        const translateHtml = `<div class="bundle-doc-meta"><strong>Translate dates:</strong> ${
          translateDates ? "ON" : "OFF"
        }</div>`;

        const leakHtml = `<div class="bundle-doc-meta"><strong>Date-like strings detected (pre-submit guardrail):</strong> ${leakCount}</div>`;

        const tableHtml = `
          <table class="dash-table" style="margin-top:10px;">
            <thead>
              <tr>
                <th width="24%">Detected</th>
                <th width="18%">Parsed</th>
                <th width="28%">Replacement</th>
                <th width="30%">Notes</th>
              </tr>
            </thead>
            <tbody>
              ${rows.join("") || `<tr><td colspan="4" class="subtle">No DATE detections.</td></tr>`}
            </tbody>
          </table>
          ${truncated ? `<div class="subtle" style="margin-top:8px;">Showing first ${maxRows} of ${dateSpans.length} DATE spans.</div>` : ""}
        `;

        if (!chronoPreviewModalEl || typeof chronoPreviewModalEl.showModal !== "function") {
          setStatus("Preview unavailable (dialog unsupported)");
          return;
        }
        if (chronoPreviewBodyEl) {
          chronoPreviewBodyEl.innerHTML = `
            ${translateHtml}
            ${indexHtml}
            ${docHtml}
            ${headerHtml}
            ${leakHtml}
            ${tableHtml}
          `;
        }
	        chronoPreviewModalEl.showModal();
	      });
	    }

    function generateClientId(prefix) {
      const p = String(prefix || "id").replace(/[^a-z0-9_]/gi, "");
      try {
        if (globalThis.crypto && typeof globalThis.crypto.randomUUID === "function") {
          return `${p}_${globalThis.crypto.randomUUID()}`;
        }
      } catch {
        // ignore
      }
      return `${p}_${Math.random().toString(36).slice(2)}_${Date.now().toString(36)}`;
    }

    function hideBundleSummary() {
      if (!bundleSummaryHostEl) return;
      bundleSummaryHostEl.classList.add("hidden");
      bundleSummaryHostEl.innerHTML = "";
    }

    function renderBundleDocsList() {
      if (!bundleDocsHostEl) return;
      if (!Array.isArray(bundleDocs) || bundleDocs.length === 0) {
        bundleDocsHostEl.textContent = "No bundle docs yet.";
        return;
      }

      bundleDocsHostEl.innerHTML = "";
      const sorted = [...bundleDocs].sort((a, b) => Number(a.seq) - Number(b.seq));
      for (const doc of sorted) {
        const offset =
          Number.isFinite(Number(doc.doc_t_offset_days)) ? formatTOffset(Number(doc.doc_t_offset_days)) : "T?";
        const meta = el(
          "div",
          { className: "bundle-doc-meta" },
          [
            el("div", {
              text: `#${doc.seq} · ${doc.timepoint_role} · ${offset} DAYS · ${String(doc.text || "").length} chars`,
            }),
          ]
        );

        const removeBtn = el("button", {
          className: "secondary",
          text: "Remove",
          onClick: () => {
            bundleDocs = bundleDocs.filter((d) => d.id !== doc.id);
            lastBundleResponse = null;
            hideBundleSummary();
            renderBundleDocsList();
            updateZkControls();
            setStatus("Removed bundle document");
          },
        });
        removeBtn.disabled = running || bundleBusy;

        const row = el("div", { className: "bundle-doc-row" }, [meta, removeBtn]);
        bundleDocsHostEl.appendChild(row);
      }
    }

    function ensureBundleIds() {
      const zkValue = String(zkPatientIdInputEl?.value || "").trim();
      const epValue = String(episodeIdInputEl?.value || "").trim();
      const zk = zkValue || generateClientId("zk");
      const ep = epValue || generateClientId("ep");
      if (zkPatientIdInputEl && !zkValue) zkPatientIdInputEl.value = zk;
      if (episodeIdInputEl && !epValue) episodeIdInputEl.value = ep;
      return { zk, ep };
    }

    async function renderBundleSummary(bundleResp) {
      if (!bundleSummaryHostEl) return;
      if (!bundleResp) {
        hideBundleSummary();
        return;
      }

      bundleSummaryHostEl.classList.remove("hidden");
      bundleSummaryHostEl.innerHTML = "";

      const docs = Array.isArray(bundleResp.documents) ? bundleResp.documents : [];
      const timeline = bundleResp.timeline || {};

      const title = el("div", {
        text: `Bundle: ${bundleResp.zk_patient_id || "(missing)"} / ${bundleResp.episode_id || "(missing)"} (${docs.length} doc${docs.length === 1 ? "" : "s"})`,
      });
      title.style.fontWeight = "600";

      const offsetsByRole = timeline.doc_offsets_by_role || {};
      const offsetsText = Object.keys(offsetsByRole).length
        ? Object.entries(offsetsByRole)
            .map(([role, off]) => `${role}=${formatTOffset(Number(off))}`)
            .join(" · ")
        : "(no doc offsets parsed)";

      const offsetsRow = el("div", { className: "bundle-doc-meta", text: `Offsets: ${offsetsText}` });

      const selectLabel = el("div", { className: "bundle-doc-meta", text: "View doc results:" });

      const docSelect = el("select", { className: "param-select" });
      docs.forEach((doc, idx) => {
        const off = doc.doc_t_offset_days == null ? "T?" : formatTOffset(Number(doc.doc_t_offset_days));
        const opt = document.createElement("option");
        opt.value = String(idx);
        opt.textContent = `#${doc.seq} · ${doc.timepoint_role} · ${off} DAYS`;
        docSelect.appendChild(opt);
      });

      const exportBundleBtn = el("button", {
        className: "secondary",
        text: "Export Bundle JSON",
        onClick: () => {
          const payload = JSON.stringify(bundleResp, null, 2);
          const blob = new Blob([payload], { type: "application/json" });
          const url = URL.createObjectURL(blob);
          const anchor = document.createElement("a");
          anchor.href = url;
          anchor.download = `procedure_suite_bundle_${Date.now()}.json`;
          document.body.appendChild(anchor);
          anchor.click();
          anchor.remove();
          URL.revokeObjectURL(url);
          setStatus("Exported bundle JSON");
        },
      });

      const controlsRow = el("div", { className: "bundle-actions" }, [selectLabel, docSelect, exportBundleBtn]);

      bundleSummaryHostEl.appendChild(title);
      bundleSummaryHostEl.appendChild(offsetsRow);
      bundleSummaryHostEl.appendChild(controlsRow);

      const showDocAt = async (idx) => {
        const safeIdx = clamp(Number(idx) || 0, 0, Math.max(0, docs.length - 1));
        const doc = docs[safeIdx];
        if (!doc || !doc.result) return;
        await renderResults(doc.result, { rawData: bundleResp });
      };

      docSelect.addEventListener("change", () => {
        showDocAt(docSelect.value).catch((err) => {
          console.error("Failed to render selected bundle doc", err);
        });
      });

      // Default to INDEX_PROCEDURE if present, else first doc.
      const defaultIdx = Math.max(
        0,
        docs.findIndex((d) => String(d.timepoint_role || "") === "INDEX_PROCEDURE")
      );
      docSelect.value = String(defaultIdx);
      await showDocAt(defaultIdx);
    }

    if (genBundleIdsBtn) {
      genBundleIdsBtn.addEventListener("click", () => {
        if (bundleBusy) return;
        const { zk, ep } = ensureBundleIds();
        setStatus(`Generated IDs: ${zk} / ${ep}`);
      });
    }

    if (clearBundleBtn) {
      clearBundleBtn.addEventListener("click", () => {
        if (running || bundleBusy) return;
        bundleDocs = [];
        lastBundleResponse = null;
        hideBundleSummary();
        renderBundleDocsList();
        updateZkControls();
        setStatus("Cleared bundle");
      });
    }

    if (addToBundleBtn) {
      addToBundleBtn.addEventListener("click", () => {
        if (running || bundleBusy) return;
        if (!scrubbedConfirmed) {
          setStatus("Apply redactions before adding to bundle");
          return;
        }

        const indexYmd = parseIsoDateInput(indexDateInputEl?.value);
        const docYmd = parseIsoDateInput(docDateInputEl?.value);
        const role = String(timepointRoleSelectEl?.value || "");
        const seq = Number(docSeqInputEl?.value || "0");

        if (!indexYmd || !docYmd) {
          if (bundlePanelEl) bundlePanelEl.open = true;
          setStatus("Bundle requires Index date (T=0) and Document date");
          return;
        }

        const docOffsetDays = diffDaysUtcNoon(indexYmd, docYmd);
        if (docOffsetDays == null) {
          setStatus("Failed to compute document offset (check dates)");
          return;
        }

        const safeSeq = Number.isFinite(seq) && seq > 0 ? Math.trunc(seq) : null;
        if (!safeSeq) {
          setStatus("Seq must be a positive integer");
          return;
        }
        if (bundleDocs.some((d) => Number(d.seq) === safeSeq)) {
          setStatus(`Seq ${safeSeq} already exists in bundle (remove or choose another seq)`);
          return;
        }

        const noteText = model.getValue();
        const systemHeader = buildSystemHeaderToken({ role, seq: safeSeq, docOffsetDays });
        const bundledText = `${systemHeader}\n${noteText}`;

        const leaks = countDateLikeStringsForLeakScan(bundledText);
        if (leaks) {
          setStatus(
            `Bundle blocked: found ${leaks} date-like string${leaks === 1 ? "" : "s"} (run detection + apply redactions again, and manually redact remaining dates).`
          );
          return;
        }

        bundleDocs.push({
          id: generateClientId("doc"),
          timepoint_role: role,
          seq: safeSeq,
          doc_t_offset_days: docOffsetDays,
          text: bundledText,
        });

        lastBundleResponse = null;
        hideBundleSummary();
        renderBundleDocsList();
        updateZkControls();

        const nextSeq = Math.max(...bundleDocs.map((d) => Number(d.seq) || 0)) + 1;
        if (docSeqInputEl) docSeqInputEl.value = String(nextSeq);

        setStatus(`Added doc to bundle: #${safeSeq} · ${role} · ${formatTOffset(docOffsetDays)} DAYS`);
      });
    }

    if (submitBundleBtn) {
      submitBundleBtn.addEventListener("click", async () => {
        if (running || bundleBusy) return;
        if (!Array.isArray(bundleDocs) || bundleDocs.length === 0) {
          setStatus("Add at least one scrubbed doc to the bundle first");
          return;
        }

        // Final guardrail: block if any date-like strings remain.
        const leakCount = bundleDocs.reduce((acc, d) => acc + countDateLikeStringsForLeakScan(d.text), 0);
        if (leakCount) {
          setStatus(
            `Bundle blocked: found ${leakCount} date-like string${leakCount === 1 ? "" : "s"} across documents.`
          );
          return;
        }

        const { zk, ep } = ensureBundleIds();
        bundleBusy = true;
        updateZkControls();
        if (submitBtn) submitBtn.disabled = true;
        setStatus("Submitting bundle…");
        if (serverResponseEl) serverResponseEl.textContent = "(submitting bundle...)";

        try {
          const payload = {
            zk_patient_id: zk,
            episode_id: ep,
            documents: [...bundleDocs]
              .sort((a, b) => Number(a.seq) - Number(b.seq))
              .map((d) => ({ timepoint_role: d.timepoint_role, seq: d.seq, text: d.text })),
            already_scrubbed: true,
            include_financials: false,
            explain: true,
            include_v3_event_log: false,
          };

          const res = await fetch("/api/v1/process_bundle", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });

          const bodyText = await res.text();
          let data;
          try {
            data = bodyText ? JSON.parse(bodyText) : null;
          } catch (parseErr) {
            console.error("Failed to parse bundle JSON response:", parseErr);
            data = { error: "Invalid JSON response", raw: bodyText };
          }

          if (!res.ok) {
            if (serverResponseEl) serverResponseEl.textContent = JSON.stringify(data, null, 2);
            setStatus(`Bundle submit failed (${res.status})`);
            return;
          }

          lastBundleResponse = data;
          await renderBundleSummary(lastBundleResponse);
          setStatus("Bundle submitted (scrubbed-only; not persisted)");
        } catch (err) {
          console.error("Bundle submit error:", err);
          if (serverResponseEl) {
            serverResponseEl.textContent = JSON.stringify(
              { error: String(err?.message || err), type: err?.name || "UnknownError" },
              null,
              2
            );
          }
          setStatus("Bundle submit error - check console for details");
        } finally {
          bundleBusy = false;
          updateZkControls();
          if (submitBtn) submitBtn.disabled = !scrubbedConfirmed || running;
        }
      });
    }

    renderBundleDocsList();
    hideBundleSummary();
    updateZkControls();

	  // Manual redaction: Add button click handler
	  if (addRedactionBtn) {
	    addRedactionBtn.addEventListener("click", () => {
	      if (!currentSelection) return;

      let startOffset = 0;
      let endOffset = 0;
      if (!usingPlainEditor) {
        if (typeof currentSelection.isEmpty === "function" && currentSelection.isEmpty()) return;
        startOffset = model.getOffsetAt(currentSelection.getStartPosition());
        endOffset = model.getOffsetAt(currentSelection.getEndPosition());
      } else {
        startOffset = Number(currentSelection.start) || 0;
        endOffset = Number(currentSelection.end) || 0;
        if (endOffset <= startOffset) return;
      }

      const selectedText = model.getValue().slice(startOffset, endOffset);
      const entityType = entityTypeSelect ? entityTypeSelect.value : "OTHER";

      // Create new detection object
      const newDetection = {
        id: `manual_${Date.now()}_${Math.random().toString(36).slice(2)}`,
        label: entityType,
        text: selectedText,
        start: startOffset,
        end: endOffset,
        score: 1.0,
        source: "manual"  // Critical for styling distinction
      };

      // Add to global state (manual additions "win" by being at end)
      detections.push(newDetection);
      detectionsById.set(newDetection.id, newDetection);

      // Re-render sidebar and editor highlights
      renderDetections();

      // Reset UI: Clear selection and disable button
      if (!usingPlainEditor && editor) {
        editor.setSelection(new monaco.Selection(0, 0, 0, 0));
      } else if (fallbackTextarea) {
        fallbackTextarea.focus();
        fallbackTextarea.setSelectionRange(0, 0);
      }
      currentSelection = null;
      addRedactionBtn.disabled = true;

      setStatus(`Added manual redaction: ${entityType}`);
    });
  }

  submitBtn.addEventListener("click", async () => {
    if (!scrubbedConfirmed) {
      console.warn("Submit blocked: redactions not confirmed. Click 'Apply redactions' first.");
      setStatus("Error: Apply redactions before submitting");
      return;
    }
    submitBtn.disabled = true;
    if (newNoteBtn) newNoteBtn.disabled = true;
    clearResultsUi();
    setStatus("Submitting scrubbed note…");
    serverResponseEl.textContent = "(submitting...)";

    try {
      const submitterName = getSubmitterName();
      const noteText = model.getValue();

	      const processBody = {
	        note: noteText,
	        already_scrubbed: true,
	      };
	      // Force backend to return evidence spans
	      processBody.explain = true;
	      processBody.include_evidence = true;
	      processBody.return_explain = true;

      const shouldAttemptPersistence = !TESTER_MODE || submitterName.length > 0;

      if (shouldAttemptPersistence) {
        const confirmed = await confirmPhiRemoval();
        if (!confirmed) {
          serverResponseEl.textContent = "(cancelled)";
          setStatus("Submit cancelled");
          return;
        }

        const persistBody = {
          ...processBody,
          submitter_name: submitterName || null,
        };

        console.log("Submitting to /api/v1/registry/runs", { noteLength: noteText.length });
        const persistRes = await fetch("/api/v1/registry/runs", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(persistBody),
        });

        const persistText = await persistRes.text();
        let persistData;
        try {
          persistData = persistText ? JSON.parse(persistText) : null;
        } catch (parseErr) {
          console.error("Failed to parse JSON response:", parseErr);
          persistData = { error: "Invalid JSON response", raw: persistText };
        }

	        if (persistRes.ok) {
	          console.log("Persist success:", persistData);
	          const runId = persistData?.run_id;
	          const result = persistData?.result;
	          const resp = result;
	          console.log("[debug] processResponse keys:", Object.keys(resp || {}));
	          console.log("[debug] has explain?", !!(resp?.explain || resp?.explanation));
	          console.log("[debug] evidence candidates:", {
	            explain: resp?.explain,
	            evidence: resp?.evidence,
	            registryEvidence: resp?.registry?.evidence,
	            registryExplain: resp?.registry?.explain,
	          });
	          if (!runId || !result) {
	            throw new Error("Registry runs response missing run_id/result");
	          }
          setRunId(runId);
          setFeedbackStatus("Run persisted. You can submit feedback and save corrections.");
          await renderResults(result);
          setStatus("Submitted + persisted (scrubbed text only)");
          return;
        }

        if (persistRes.status === 400) {
          serverResponseEl.textContent = JSON.stringify(
            { error: persistData, status: persistRes.status, statusText: persistRes.statusText },
            null,
            2
          );
          setFeedbackStatus("Persistence rejected. Re-check redaction and retry.");
          setStatus("Persistence rejected (PHI risk)");
          return;
        }

        const detailText = String(persistData?.detail || "").toLowerCase();
        const persistenceDisabled =
          persistRes.status === 404 ||
          persistRes.status === 403 ||
          (persistRes.status === 503 && detailText.includes("persistence") && detailText.includes("disabled"));

        if (!persistenceDisabled) {
          console.error("Persistence request failed:", persistRes.status, persistData);
          serverResponseEl.textContent = JSON.stringify(
            { error: persistData, status: persistRes.status, statusText: persistRes.statusText },
            null,
            2
          );
          setStatus(`Submit failed (${persistRes.status})`);
          return;
        }

        console.warn("Persistence unavailable; falling back to /api/v1/process", persistRes.status);
        setFeedbackStatus(
          "Persistence unavailable (REGISTRY_RUNS_PERSIST_ENABLED may be off). Falling back to stateless /api/v1/process."
        );
      } else {
        setFeedbackStatus("Tester mode: enter your name to enable persistence.");
      }

      console.log("Submitting to /api/v1/process", { noteLength: noteText.length });

      const res = await fetch("/api/v1/process", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(processBody),
      });

      console.log("Response status:", res.status, res.statusText);

      const bodyText = await res.text();
      let data;
      try {
        data = bodyText ? JSON.parse(bodyText) : null;
      } catch (parseErr) {
        console.error("Failed to parse JSON response:", parseErr);
        data = { error: "Invalid JSON response", raw: bodyText };
      }

      if (!res.ok) {
        console.error("Request failed:", res.status, data);
        serverResponseEl.textContent = JSON.stringify(
          { error: data, status: res.status, statusText: res.statusText },
          null,
          2
        );
        setStatus(`Submit failed (${res.status})`);
        return;
	      }

	      console.log("Success:", data);
	      const resp = data;
	      console.log("[debug] processResponse keys:", Object.keys(resp || {}));
	      console.log("[debug] has explain?", !!(resp?.explain || resp?.explanation));
	      console.log("[debug] evidence candidates:", {
	        explain: resp?.explain,
	        evidence: resp?.evidence,
	        registryEvidence: resp?.registry?.evidence,
	        registryExplain: resp?.registry?.explain,
	      });
	      await renderResults(data);
	      setStatus("Submitted (scrubbed text only; not persisted)");
	    } catch (err) {
      console.error("Submit error:", err);
      serverResponseEl.textContent = JSON.stringify(
        { error: String(err?.message || err), type: err?.name || "UnknownError" },
        null,
        2
      );
      setStatus("Submit error - check console for details");
    } finally {
      submitBtn.disabled = false;
      if (newNoteBtn) newNoteBtn.disabled = running;
    }
  });

  if (submitFeedbackBtn) {
    submitFeedbackBtn.addEventListener("click", async () => {
      if (!currentRunId) return;
      const name = getSubmitterName();
      if (!name) {
        setFeedbackStatus("Name is required to submit feedback.");
        updateFeedbackButtons();
        return;
      }

      const rating = Number.parseInt(String(feedbackRatingEl?.value || "0"), 10);
      const comment = String(feedbackCommentEl?.value || "").trim() || null;

      setFeedbackStatus("Submitting feedback…");

      try {
        const res = await fetch(`/api/v1/registry/runs/${currentRunId}/feedback`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ reviewer_name: name, rating, comment }),
        });

        const text = await res.text();
        let data;
        try {
          data = text ? JSON.parse(text) : null;
        } catch {
          data = { raw: text };
        }

        if (res.status === 409) {
          feedbackSubmitted = true;
          updateFeedbackButtons();
          setFeedbackStatus("Feedback already submitted for this run.");
          return;
        }

        if (!res.ok) {
          setFeedbackStatus(`Feedback submit failed (${res.status}).`);
          console.error("Feedback submit failed:", res.status, data);
          return;
        }

        feedbackSubmitted = true;
        updateFeedbackButtons();
        setFeedbackStatus("Feedback submitted. Thank you.");
      } catch (err) {
        console.error("Feedback submit error:", err);
        setFeedbackStatus("Feedback submit error - check console for details.");
      }
    });
  }

  if (saveCorrectionsBtn) {
    saveCorrectionsBtn.addEventListener("click", async () => {
      if (!currentRunId) return;
      if (!editedPayload) {
        setFeedbackStatus("No review payload yet. Make table edits or flag fields first.");
        updateFeedbackButtons();
        return;
      }

      const name = getSubmitterName();
      const editedTablesSnapshot = getTablesForExport();

      setFeedbackStatus("Saving corrections…");

      try {
        const res = await fetch(`/api/v1/registry/runs/${currentRunId}/correction`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            corrected_response_json: editedPayload,
            edited_tables_json: editedTablesSnapshot,
            editor_name: name || null,
          }),
        });

        const text = await res.text();
        let data;
        try {
          data = text ? JSON.parse(text) : null;
        } catch {
          data = { raw: text };
        }

        if (!res.ok) {
          setFeedbackStatus(`Save corrections failed (${res.status}).`);
          console.error("Save corrections failed:", res.status, data);
          return;
        }

        setFeedbackStatus("Corrections saved.");
      } catch (err) {
        console.error("Save corrections error:", err);
        setFeedbackStatus("Save corrections error - check console for details.");
      }
    });
  }

  if (exportBtn) {
    exportBtn.addEventListener("click", () => {
      if (!lastServerResponse) {
        setStatus("No results to export yet");
        return;
      }

      const payload = JSON.stringify(lastServerResponse, null, 2);
      const blob = new Blob([payload], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `procedure_suite_response_${Date.now()}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      setStatus("Exported results");
    });
  }

  if (exportEditedBtn) {
    exportEditedBtn.addEventListener("click", () => {
      if (!editedPayload) {
        setStatus("No edited payload to export yet");
        return;
      }

      const payload = JSON.stringify(editedPayload, null, 2);
      const blob = new Blob([payload], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `procedure_suite_edited_${Date.now()}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      setStatus("Exported edited payload");
    });
  }

  if (exportPatchBtn) {
    exportPatchBtn.addEventListener("click", () => {
      const patch = registryGridEdits?.edited_patch;
      if (!Array.isArray(patch) || patch.length === 0) {
        setStatus("No registry patch to export yet");
        return;
      }

      const exportObj = {
        edited_source: "ui_registry_grid",
        edited_patch: patch,
        edited_fields: Array.isArray(registryGridEdits?.edited_fields) ? registryGridEdits.edited_fields : [],
      };

      const payload = JSON.stringify(exportObj, null, 2);
      const blob = new Blob([payload], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = `procedure_suite_registry_patch_${Date.now()}.json`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
      setStatus("Exported registry patch");
    });
  }

  if (exportTablesBtn) {
    exportTablesBtn.addEventListener("click", () => {
      exportTablesToExcel();
    });
  }

	  if (newNoteBtn) {
	    newNoteBtn.addEventListener("click", () => {
	      if (running || extractingPdf || extractingCamera) return;
	      setEditorText("");
	      originalText = "";
	      hasRunDetection = false;
	      setScrubbedConfirmed(false);
	      clearDetections();
	      clearResultsUi();
	      resetPdfUploadUi();
	      resetFeedbackDraft();
	      if (cameraModalOpen) closeCameraModal();
	      else {
	        clearCapturedCameraPages();
	        stopCameraPreviewStream();
	      }
	      setStatus("Ready for new note");
	      setProgress("");
	      if (runBtn) runBtn.disabled = extractingPdf || extractingCamera || !workerReady;
	    });
	  }

  // Optional: service worker (local assets only)
  if ("serviceWorker" in navigator && new URL(location.href).searchParams.get("sw") === "1") {
    try {
      await navigator.serviceWorker.register("./sw.js");
    } catch {
      // ignore
    }
  }

  window.addEventListener("beforeunload", () => {
    try {
      privacyShieldTeardown();
    } catch {
      // ignore
    }
    stopCameraPreviewStream();
    cameraQueue.clearAll();
    if (cameraWorker) {
      try {
        cameraWorker.terminate();
      } catch {
        // ignore
      }
      cameraWorker = null;
    }
  });

  setStatus("Initializing local PHI model (first load downloads ONNX)…");

  setInterval(() => {
    if (!running) return;
    const quietMs = Date.now() - lastWorkerMessageAt;
    if (quietMs > 15_000) {
      setProgress("Still working… (model download/inference can take a while)");
    }
  }, 2_000);
}

main().catch((e) => {
  console.error(e);
  statusTextEl.textContent = `Init failed: ${e?.message || e}`;
});
