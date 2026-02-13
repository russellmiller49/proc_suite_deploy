const statusTextEl = document.getElementById("statusText");
const actionBannerEl = document.getElementById("actionBanner");
const seedTextEl = document.getElementById("seedText");
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
      phiDetections = detections
        .filter((d) => Number.isFinite(d?.start) && Number.isFinite(d?.end) && d.end > d.start)
        .map((d) => ({
          start: Number(d.start),
          end: Number(d.end),
          label: String(d.label || "PHI"),
        }));
      phiHasRunDetection = true;
      phiScrubbedConfirmed = false;
      setPhiStatus("Detection complete. Apply redactions to confirm scrubbed text.");
      setPhiProgress("");
      setPhiSummary(`${phiDetections.length} PHI span${phiDetections.length === 1 ? "" : "s"} detected`);
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
    "granular_data.navigation_targets[*].air_bronchogram_present": { type: "boolean" },
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
  const requiresPhi = Boolean(phiRunBtn && phiApplyBtn && phiStatusTextEl);
  const phiOk = !requiresPhi || phiScrubbedConfirmed;
  seedBtn.disabled = state.busy || !phiOk;
  seedBtn.title = !phiOk ? "Run PHI detection and apply redactions first" : "";
  refreshBtn.disabled = state.busy || !hasBundle;
  clearBtn.disabled = state.busy;
  applyPatchBtn.disabled = state.busy || !hasBundle || !hasQuestions;
  if (transferToDashboardBtn) transferToDashboardBtn.disabled = state.busy || !hasTransferNote;
  if (completenessInsertBtn) completenessInsertBtn.disabled = state.busy || !completenessPrompts.length;
  if (completenessCopyBtn) completenessCopyBtn.disabled = state.busy || !buildCompletenessAddendumBlock();
  strictToggleEl.disabled = state.busy;

  if (phiRunBtn) phiRunBtn.disabled = state.busy || phiRunning || !phiWorkerReady;
  if (phiApplyBtn) phiApplyBtn.disabled = state.busy || phiRunning || !phiHasRunDetection;
  if (phiRevertBtn) phiRevertBtn.disabled = state.busy || phiRunning || !phiOriginalText;
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

  if (!issues.length && !warnings.length && !suggestions.length && !notes.length) {
    validationHostEl.innerHTML = '<div class="empty-state">No validation output yet.</div>';
    return;
  }

  let html = "";
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
    };
    state.verify = {
      bundle: seed.bundle,
      issues: seed.issues || [],
      warnings: seed.warnings || [],
      inference_notes: seed.inference_notes || [],
      suggestions: seed.suggestions || [],
      questions: seed.questions || [],
    };
    state.bundle = seed.bundle;
    state.questions = seed.questions || [];
    state.lastPatch = [];
    renderAll();

    showBanner(
      "success",
      `Bundle seeded. ${state.questions.length} follow-up question${state.questions.length === 1 ? "" : "s"} generated.`,
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

    showBanner(
      "success",
      `Questions refreshed. ${state.questions.length} question${state.questions.length === 1 ? "" : "s"} remaining.`,
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

    showBanner(
      "success",
      `Patch applied (${patchOps.length} op${patchOps.length === 1 ? "" : "s"}). ${state.questions.length} question${
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
  seedTextEl.value = "";
  phiDetections = [];
  phiHasRunDetection = false;
  phiScrubbedConfirmed = false;
  phiOriginalText = "";
  setPhiSummary("");
  setPhiProgress("");
  if (phiWorkerReady) setPhiStatus("Ready (local model loaded)");
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
seedTextEl.addEventListener("input", () => {
  if (!phiRunBtn) {
    updateControls();
    return;
  }

  // Any text edit invalidates prior detection + scrub confirmation.
  if (phiHasRunDetection || phiScrubbedConfirmed) {
    phiHasRunDetection = false;
    phiScrubbedConfirmed = false;
    phiDetections = [];
    phiOriginalText = "";
    setPhiSummary("");
    setPhiProgress("");
    setPhiStatus("Text changed. Run detection to confirm redactions.");
  }

  updateControls();
});

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
    phiOriginalText = String(seedTextEl.value || "");
    setPhiStatus("Detecting… (client-side)");
    setPhiProgress("");
    setPhiSummary("");
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

    const spans = (Array.isArray(phiDetections) ? phiDetections : [])
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
    setPhiStatus("Redactions applied (scrubbed text ready to seed)");
    setPhiProgress("");
    setPhiSummary(`${spans.length} span${spans.length === 1 ? "" : "s"} redacted`);
    updateControls();
  });
}

if (phiRevertBtn) {
  phiRevertBtn.addEventListener("click", () => {
    if (!phiOriginalText) return;
    seedTextEl.value = phiOriginalText;
    phiDetections = [];
    phiHasRunDetection = false;
    phiScrubbedConfirmed = false;
    phiOriginalText = "";
    setPhiSummary("");
    setPhiProgress("");
    setPhiStatus("Reverted. Run detection to confirm redactions.");
    updateControls();
  });
}

const dashboardTransfer = consumeDashboardTransferPayload();
if (dashboardTransfer?.note) {
  seedTextEl.value = dashboardTransfer.note;
  if (phiRunBtn) {
    phiDetections = [];
    phiHasRunDetection = false;
    phiScrubbedConfirmed = false;
    phiOriginalText = "";
    setPhiSummary("");
    setPhiProgress("");
    setPhiStatus("Note loaded. Run detection and apply redactions before seeding.");
  }
  showBanner("success", "Loaded note from dashboard. Run PHI detection, apply redactions, then seed.");
}

renderAll();
