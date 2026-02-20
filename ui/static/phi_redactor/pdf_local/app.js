import { extractPdfAdaptive, buildPdfDocumentModel } from "./pdf/pipeline.js";
import { resolvePageSource } from "./pdf/pageClassifier.js";
import { detectRegexPhi } from "./redaction/regexDetectors.js";
import { mergeRedactionSpans } from "./redaction/spanMerge.js";
import { applyRedactions } from "./redaction/applyRedactions.js";

const statusEl = document.getElementById("pdfLocalStatus");
const progressEl = document.getElementById("pdfLocalProgress");
const fileInputEl = document.getElementById("pdfFileInput");
const forceOcrAllEl = document.getElementById("forceOcrAll");
const runRegexBtn = document.getElementById("runRegexBtn");
const pagesTbodyEl = document.getElementById("pdfPagesBody");
const pageDecisionWarningEl = document.getElementById("pageDecisionWarning");
const nativePreviewEl = document.getElementById("nativePreview");
const redactedPreviewEl = document.getElementById("redactedPreview");
const redactionSummaryEl = document.getElementById("redactionSummary");

const state = {
  file: null,
  rawPages: [],
  pageOverrides: new Map(),
  forceOcrAll: false,
  documentModel: null,
  extractionRunning: false,
};

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.classList.toggle("error", Boolean(isError));
}

function setProgress(message) {
  progressEl.textContent = message || "";
}

function resetOutput() {
  state.rawPages = [];
  state.pageOverrides.clear();
  state.documentModel = null;
  pagesTbodyEl.innerHTML = "";
  pageDecisionWarningEl.textContent = "";
  nativePreviewEl.textContent = "";
  redactedPreviewEl.textContent = "";
  redactionSummaryEl.textContent = "";
  runRegexBtn.disabled = true;
}

function buildDocument() {
  if (!state.file) return;

  const pagesWithOverrides = state.rawPages.map((page) => ({
    ...page,
    userOverride: state.pageOverrides.get(page.pageIndex),
  }));

  state.documentModel = buildPdfDocumentModel(state.file, pagesWithOverrides, {
    forceOcrAll: state.forceOcrAll,
    ocr: {
      available: true,
      enabled: true,
      lang: "eng",
      qualityMode: "fast",
    },
    gate: {
      minCompletenessConfidence: 0.72,
      maxContaminationScore: 0.24,
      hardBlockWhenUnsafeWithoutOcr: true,
    },
  });

  renderPageTable();
  renderNativePreview();
  runRegexBtn.disabled = state.documentModel.pages.length === 0 || Boolean(state.documentModel.blocked);
}

function renderPageTable() {
  pagesTbodyEl.innerHTML = "";
  if (!state.documentModel || !state.documentModel.pages.length) {
    return;
  }

  let ocrNeededCount = 0;

  for (const page of state.documentModel.pages) {
    const resolved = resolvePageSource(page, { forceOcrAll: state.forceOcrAll });
    if (resolved.source === "ocr") {
      ocrNeededCount += 1;
    }

    const row = document.createElement("tr");

    const pageCell = document.createElement("td");
    pageCell.textContent = String(page.pageIndex + 1);

    const statsCell = document.createElement("td");
    statsCell.textContent = `chars=${page.stats.charCount}, items=${page.stats.itemCount}`;

    const decisionCell = document.createElement("td");
    decisionCell.textContent = `${page.sourceDecision.toUpperCase()} (${Math.round(page.classification.confidence * 100)}%)`;

    const reasonCell = document.createElement("td");
    reasonCell.textContent = page.sourceReason || page.classification.reason;

    const overrideCell = document.createElement("td");
    const select = document.createElement("select");
    select.className = "param-select";
    select.dataset.pageIndex = String(page.pageIndex);

    const autoOpt = document.createElement("option");
    autoOpt.value = "auto";
    autoOpt.textContent = "Auto";

    const nativeOpt = document.createElement("option");
    nativeOpt.value = "force_native";
    nativeOpt.textContent = "Force native";

    const ocrOpt = document.createElement("option");
    ocrOpt.value = "force_ocr";
    ocrOpt.textContent = "Force OCR";

    select.append(autoOpt, nativeOpt, ocrOpt);

    const override = state.pageOverrides.get(page.pageIndex);
    select.value = override || "auto";

    overrideCell.appendChild(select);

    row.append(pageCell, statsCell, decisionCell, reasonCell, overrideCell);
    pagesTbodyEl.appendChild(row);
  }

  if (state.documentModel.blocked) {
    pageDecisionWarningEl.textContent =
      `Blocked: ${state.documentModel.blockReason || "unsafe native extraction without OCR."}`;
  } else if (ocrNeededCount > 0) {
    pageDecisionWarningEl.textContent =
      `${ocrNeededCount} page(s) are classified as OCR-needed/hybrid. ` +
      "OCR recovery is enabled; review source decisions and confidence before redaction.";
  } else {
    pageDecisionWarningEl.textContent = "All pages classified as native text.";
  }
}

function renderNativePreview() {
  nativePreviewEl.textContent = state.documentModel ? state.documentModel.fullText : "";
}

function renderRedactionSummary(spans) {
  if (!spans.length) {
    redactionSummaryEl.textContent = "No regex PHI spans detected.";
    return;
  }

  const counts = new Map();
  for (const span of spans) {
    counts.set(span.type, (counts.get(span.type) || 0) + 1);
  }

  const parts = [];
  for (const [type, count] of [...counts.entries()].sort((a, b) => a[0].localeCompare(b[0]))) {
    parts.push(`${type}:${count}`);
  }

  redactionSummaryEl.textContent = `Detected ${spans.length} PHI spans (${parts.join(", ")}).`;
}

async function handleFileSelected(file) {
  if (!file) return;

  resetOutput();
  state.file = file;
  state.extractionRunning = true;

    setStatus(`Running layout-aware extraction + OCR for ${file.name}...`);
  setProgress("Starting worker...");

  try {
    let completedPages = 0;
    let totalPages = 0;
    let builtDocument = null;
    let lastOcrError = "";

    for await (const event of extractPdfAdaptive(file, {
      ocr: {
        available: true,
        enabled: true,
        lang: "eng",
        qualityMode: "fast",
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
        setProgress(`Processed ${completedPages}/${totalPages} pages`);
      } else if (event.kind === "ocr_progress") {
        setProgress(`OCR ${event.completedPages}/${event.totalPages} pages`);
      } else if (event.kind === "ocr_status") {
        const pct = Number.isFinite(event.progress) ? Math.round(event.progress * 100) : 0;
        setProgress(`OCR ${event.status || "processing"} (${pct}%)`);
      } else if (event.kind === "ocr_error") {
        lastOcrError = String(event.error || "");
        setStatus(`OCR failed: ${event.error}`, true);
      } else if (event.kind === "stage") {
        setProgress(`PDF ${String(event.stage || "").replace(/_/g, " ")} (${event.pageIndex + 1}/${event.totalPages})`);
      } else if (event.kind === "page") {
        state.rawPages[event.page.pageIndex] = event.page;
      } else if (event.kind === "done") {
        builtDocument = event.document || null;
      }
    }

    state.rawPages = state.rawPages.filter(Boolean).sort((a, b) => a.pageIndex - b.pageIndex);
    if (builtDocument) {
      state.documentModel = builtDocument;
      renderPageTable();
      renderNativePreview();
      runRegexBtn.disabled = state.documentModel.pages.length === 0 || Boolean(state.documentModel.blocked);
    } else {
      buildDocument();
    }

    if (state.documentModel?.blocked) {
      const ocrSuffix = lastOcrError ? ` OCR error: ${lastOcrError}` : "";
      setStatus(`Extraction blocked: ${state.documentModel.blockReason || "unsafe native extraction"}${ocrSuffix}`, true);
    } else {
      setStatus(`Adaptive extraction complete (${state.rawPages.length} pages).`);
    }
    if (totalPages > 0) {
      setProgress(`Processed ${completedPages}/${totalPages} pages`);
    }
  } catch (error) {
    setStatus(`Extraction failed: ${error instanceof Error ? error.message : String(error)}`, true);
    setProgress("");
  } finally {
    state.extractionRunning = false;
  }
}

function runRegexDetection() {
  if (!state.documentModel) return;
  if (state.documentModel.blocked) {
    setStatus("Extraction is blocked by safety gate. OCR path is required before PHI detection.", true);
    return;
  }

  const regexSpans = detectRegexPhi(state.documentModel.fullText);
  const mergedSpans = mergeRedactionSpans(regexSpans);
  const viewModel = applyRedactions(state.documentModel.fullText, mergedSpans);

  redactedPreviewEl.textContent = viewModel.redactedText;
  renderRedactionSummary(mergedSpans);
  setStatus("Regex PHI detection complete. Redacted preview rendered locally.");
}

fileInputEl.addEventListener("change", () => {
  const file = fileInputEl.files && fileInputEl.files[0];
  if (!file) return;
  if (!/\.pdf$/i.test(file.name) && file.type !== "application/pdf") {
    setStatus("Please upload a PDF file.", true);
    return;
  }
  handleFileSelected(file);
});

forceOcrAllEl.addEventListener("change", () => {
  state.forceOcrAll = forceOcrAllEl.checked;
  if (!state.documentModel || state.extractionRunning) return;
  buildDocument();
});

pagesTbodyEl.addEventListener("change", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLSelectElement)) return;
  const pageIndex = Number(target.dataset.pageIndex);
  if (!Number.isFinite(pageIndex)) return;

  if (target.value === "auto") {
    state.pageOverrides.delete(pageIndex);
  } else {
    state.pageOverrides.set(pageIndex, target.value);
  }

  buildDocument();
});

runRegexBtn.addEventListener("click", runRegexDetection);

setStatus("Upload a PDF to begin local extraction/redaction.");
setProgress("");
