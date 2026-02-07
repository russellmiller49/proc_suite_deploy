const statusTextEl = document.getElementById("statusText");
const actionBannerEl = document.getElementById("actionBanner");
const seedTextEl = document.getElementById("seedText");
const strictToggleEl = document.getElementById("strictToggle");
const seedBtn = document.getElementById("seedBtn");
const refreshBtn = document.getElementById("refreshBtn");
const clearBtn = document.getElementById("clearBtn");
const applyPatchBtn = document.getElementById("applyPatchBtn");
const transferToDashboardBtn = document.getElementById("transferToDashboardBtn");

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

const state = {
  bundle: null,
  seed: null,
  verify: null,
  render: null,
  questions: [],
  lastPatch: [],
  busy: false,
};

function setStatus(text) {
  statusTextEl.textContent = text;
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
  if (renderedMarkdown) return renderedMarkdown;
  return String(seedTextEl.value || "").trim();
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
  seedBtn.disabled = state.busy;
  refreshBtn.disabled = state.busy || !hasBundle;
  clearBtn.disabled = state.busy;
  applyPatchBtn.disabled = state.busy || !hasBundle || !hasQuestions;
  if (transferToDashboardBtn) transferToDashboardBtn.disabled = state.busy || !hasTransferNote;
  strictToggleEl.disabled = state.busy;
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
      metadata: {},
      strict,
    });

    state.seed = seed;
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
  seedTextEl.value = "";
  hideBanner();
  renderAll();
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
seedTextEl.addEventListener("input", updateControls);

renderAll();
