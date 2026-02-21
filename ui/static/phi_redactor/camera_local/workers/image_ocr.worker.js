import Tesseract from "../../vendor/tesseract/tesseract.esm.min.js";
import { computeOcrTextMetrics } from "../../pdf_local/pdf/ocrMetrics.js";
import {
  applyClinicalOcrHeuristics,
  composeOcrPageText,
  filterOcrLinesDetailed,
} from "../../pdf_local/pdf/ocrPostprocess.js";
import { preprocessCanvasForOcr } from "../imagePreprocess.js";

const DEFAULT_OPTIONS = Object.freeze({
  lang: "eng",
  mode: "fast",
  sceneHint: "auto",
  warningProfile: "default",
  preprocess: {
    mode: "auto",
  },
});

const OCR_PASS_LIMITS = Object.freeze({
  fast: 3,
  high_accuracy: 4,
});
const DEFAULT_MIN_WORD_CONFIDENCE = 45;

let tesseractWorker = null;
let tesseractConfigKey = "";
let activePsm = "";
let activeProfileKey = "";
let activeJobId = "";
const cancelledJobs = new Set();

function hasFunction(value) {
  return typeof value === "function";
}

function safeNumber(value, fallback = 0) {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function clamp01(value) {
  const n = safeNumber(value, 0);
  return Math.max(0, Math.min(1, n));
}

function resolvePreprocessMode(value) {
  if (value === "bw_high_contrast") return "bw_high_contrast";
  if (value === "grayscale") return "grayscale";
  if (value === "off") return "off";
  return "auto";
}

function resolveOptions(input = {}) {
  const qualityMode = input?.mode === "high_accuracy" ? "high_accuracy" : "fast";
  const preprocessMode = resolvePreprocessMode(input?.preprocess?.mode);

  return {
    ...DEFAULT_OPTIONS,
    ...input,
    lang: typeof input?.lang === "string" && input.lang.trim() ? input.lang.trim() : "eng",
    mode: qualityMode,
    sceneHint: input?.sceneHint === "monitor"
      ? "monitor"
      : input?.sceneHint === "document"
        ? "document"
        : "auto",
    warningProfile: input?.warningProfile === "ios_safari" ? "ios_safari" : "default",
    psm: "6",
    fallbackPsm: "3",
    sparsePsm: "11",
    maxDim: qualityMode === "high_accuracy" ? 3200 : 2600,
    preprocess: {
      mode: preprocessMode,
    },
  };
}

function toAssetUrl(path) {
  return new URL(`../../vendor/tesseract/${path}`, import.meta.url).toString();
}

async function getTesseractWorker(options) {
  const configKey = `${options.lang}`;
  if (tesseractWorker && configKey === tesseractConfigKey) {
    return tesseractWorker;
  }

  if (tesseractWorker) {
    try {
      await tesseractWorker.terminate();
    } catch {
      // ignore best-effort cleanup
    }
    tesseractWorker = null;
    tesseractConfigKey = "";
    activePsm = "";
    activeProfileKey = "";
  }

  const worker = await Tesseract.createWorker(
    options.lang,
    Tesseract.OEM.LSTM_ONLY,
    {
      workerPath: toAssetUrl("worker.min.js"),
      corePath: toAssetUrl("tesseract-core-simd.wasm.js"),
      langPath: toAssetUrl("tessdata/"),
      workerBlobURL: false,
      gzip: false,
      logger: (message) => {
        if (!message || typeof message !== "object") return;
        if (!activeJobId) return;
        const progress = Number(message.progress);
        if (!Number.isFinite(progress)) return;
        self.postMessage({
          type: "camera_ocr_progress",
          jobId: activeJobId,
          stage: "recognize",
          pct: progress,
          status: String(message.status || "recognize"),
        });
      },
    },
  );

  tesseractWorker = worker;
  tesseractConfigKey = configKey;
  activePsm = "";
  activeProfileKey = "";
  return worker;
}

async function setWorkerPsm(worker, psm) {
  const normalized = String(psm || "6");
  if (normalized === activePsm) return;
  await worker.setParameters({
    tessedit_pageseg_mode: normalized,
  });
  activePsm = normalized;
}

async function setWorkerRecognitionProfile(worker, options) {
  const profileKey = `${options.mode}:${options.lang}`;
  if (profileKey === activeProfileKey) return;
  await worker.setParameters({
    preserve_interword_spaces: "1",
    user_defined_dpi: options.mode === "high_accuracy" ? "330" : "300",
  });
  activeProfileKey = profileKey;
}

function coerceConfidence(value) {
  const conf = safeNumber(value, Number.NaN);
  if (!Number.isFinite(conf) || conf < 0) return null;
  return Math.max(0, Math.min(100, conf));
}

function normalizeBbox(rawBBox = {}) {
  if (!rawBBox || typeof rawBBox !== "object") {
    return { x: 0, y: 0, width: 0, height: 0 };
  }

  const x0 = Number.isFinite(rawBBox.x0) ? Number(rawBBox.x0) : safeNumber(rawBBox.x, 0);
  const y0 = Number.isFinite(rawBBox.y0) ? Number(rawBBox.y0) : safeNumber(rawBBox.y, 0);
  const x1 = Number.isFinite(rawBBox.x1)
    ? Number(rawBBox.x1)
    : Number.isFinite(rawBBox.w)
      ? x0 + Number(rawBBox.w)
      : x0;
  const y1 = Number.isFinite(rawBBox.y1)
    ? Number(rawBBox.y1)
    : Number.isFinite(rawBBox.h)
      ? y0 + Number(rawBBox.h)
      : y0;

  const left = Math.min(x0, x1);
  const top = Math.min(y0, y1);
  return {
    x: left,
    y: top,
    width: Math.max(0, Math.abs(x1 - x0)),
    height: Math.max(0, Math.abs(y1 - y0)),
  };
}

function cleanCameraOcrLineText(value) {
  let text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return "";
  text = text
    .replace(/^[~|_*=+`]+/g, "")
    .replace(/[~|_*=+`]+$/g, "")
    .replace(/[|~_]{2,}/g, " ")
    .replace(/\s{2,}/g, " ")
    .trim();
  return text;
}

function rebuildLineTextFromWords(rawWords, minWordConfidence = DEFAULT_MIN_WORD_CONFIDENCE) {
  const words = Array.isArray(rawWords) ? rawWords : [];
  if (!words.length) return { text: "", keptWords: 0, totalWords: 0 };

  const keptTokens = [];
  let totalWords = 0;
  let keptWords = 0;
  for (const rawWord of words) {
    const token = cleanCameraOcrLineText(rawWord?.text);
    if (!token) continue;
    totalWords += 1;
    const conf = coerceConfidence(rawWord?.confidence ?? rawWord?.conf);
    const tokenLower = token.toLowerCase();
    const shortMedicalToken = /^(?:ii|iii|iv|v|vi|ml|mg|mm|cm|cc|%|fr)$/i.test(tokenLower);
    const dosageLikeToken = /^[0-9]+(?:\.[0-9]+)?(?:mg|ml|mm|cm|%|fr)$/i.test(tokenLower);
    if (!Number.isFinite(conf) || conf >= minWordConfidence || shortMedicalToken || dosageLikeToken) {
      keptTokens.push(token);
      keptWords += 1;
    }
  }

  return {
    text: cleanCameraOcrLineText(keptTokens.join(" ")),
    keptWords,
    totalWords,
  };
}

function extractLines(recognizedData, pageIndex, opts = {}) {
  const rawLines = Array.isArray(recognizedData?.lines) ? recognizedData.lines : [];
  const minWordConfidence = Number.isFinite(opts?.minWordConfidence)
    ? Number(opts.minWordConfidence)
    : DEFAULT_MIN_WORD_CONFIDENCE;
  const minKeepRatio = Number.isFinite(opts?.minWordKeepRatio)
    ? clamp01(Number(opts.minWordKeepRatio))
    : 0.4;
  const preferWordRebuild = opts?.preferWordRebuild === true;
  const trustedLinePattern = /\b(?:patient|procedure|mrn|gender|age|date|findings|anesthesia|incision|biopsy|pleura|ultrasound|lidocaine|university|texas|anderson|cancer|center)\b/i;
  const lines = [];

  for (const raw of rawLines) {
    const lineConfidence = coerceConfidence(raw?.confidence ?? raw?.conf);
    const directText = cleanCameraOcrLineText(raw?.text);
    const rebuilt = rebuildLineTextFromWords(raw?.words, minWordConfidence);
    const keepRatio = rebuilt.totalWords > 0 ? rebuilt.keptWords / rebuilt.totalWords : 0;
    const preferWordText = Boolean(rebuilt.text) && (
      preferWordRebuild ||
      !directText ||
      (lineConfidence !== null && lineConfidence < 65) ||
      (rebuilt.totalWords >= 2 && keepRatio >= minKeepRatio)
    );
    const text = cleanCameraOcrLineText(preferWordText ? rebuilt.text : directText || rebuilt.text);
    if (!text) continue;

    if (
      preferWordRebuild &&
      rebuilt.totalWords >= 4 &&
      keepRatio < Math.max(0.32, minKeepRatio * 0.72) &&
      !trustedLinePattern.test(directText)
    ) {
      continue;
    }

    lines.push({
      text,
      confidence: lineConfidence,
      bbox: normalizeBbox(raw?.bbox || raw),
      words: Array.isArray(raw?.words) ? raw.words : [],
      pageIndex,
    });
  }

  return lines;
}

function computeWordConfidenceStats(rawWords) {
  const words = Array.isArray(rawWords) ? rawWords : [];
  let sumConf = 0;
  let count = 0;
  let lowConf = 0;

  for (const word of words) {
    const rawText = String(word?.text || "").trim();
    if (!rawText) continue;
    const conf = coerceConfidence(word?.confidence ?? word?.conf);
    if (!Number.isFinite(conf)) continue;
    sumConf += conf;
    count += 1;
    if (conf < 50) lowConf += 1;
  }

  return {
    wordCount: count,
    meanWordConf: count ? sumConf / count : null,
    lowConfWordFrac: count ? lowConf / count : null,
  };
}

function cleanCameraOcrPageText(text) {
  const lines = String(text || "")
    .split(/\r?\n/)
    .map((line) => cleanCameraOcrLineText(line))
    .filter((line) => {
      if (!line) return false;
      if (/^[~|_*=+`]+$/.test(line)) return false;
      if (/^[^A-Za-z0-9]{0,3}$/.test(line)) return false;
      return true;
    });

  return lines
    .join("\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

async function buildOcrInputFromCanvas(canvas) {
  if (hasFunction(canvas?.convertToBlob)) {
    try {
      const blob = await canvas.convertToBlob({ type: "image/png" });
      if (blob) return blob;
    } catch {
      // fallback to canvas
    }
  }
  return canvas;
}

function normalizeCropBox(crop) {
  if (!crop || typeof crop !== "object") return null;
  const x0 = Number(crop.x0);
  const y0 = Number(crop.y0);
  const x1 = Number(crop.x1);
  const y1 = Number(crop.y1);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;

  const left = clamp01(Math.min(x0, x1));
  const right = clamp01(Math.max(x0, x1));
  const top = clamp01(Math.min(y0, y1));
  const bottom = clamp01(Math.max(y0, y1));
  const minSpan = 0.05;
  if (right - left < minSpan || bottom - top < minSpan) return null;
  return { x0: left, y0: top, x1: right, y1: bottom };
}

function resolveCropPixels(crop, width, height) {
  const safeWidth = Math.max(1, Math.floor(safeNumber(width, 1)));
  const safeHeight = Math.max(1, Math.floor(safeNumber(height, 1)));
  const normalized = normalizeCropBox(crop);
  if (!normalized) {
    return {
      applied: false,
      x: 0,
      y: 0,
      width: safeWidth,
      height: safeHeight,
      areaRatio: 1,
    };
  }

  // Add light padding so tight manual crops don't clip leading/trailing characters.
  const spanX = Math.max(0.05, normalized.x1 - normalized.x0);
  const spanY = Math.max(0.05, normalized.y1 - normalized.y0);
  const padFrac = Math.min(0.03, Math.max(0.012, Math.min(spanX, spanY) * 0.06));
  const expanded = {
    x0: clamp01(normalized.x0 - padFrac),
    y0: clamp01(normalized.y0 - padFrac),
    x1: clamp01(normalized.x1 + padFrac),
    y1: clamp01(normalized.y1 + padFrac),
  };

  const x = Math.max(0, Math.min(safeWidth - 1, Math.floor(expanded.x0 * safeWidth)));
  const y = Math.max(0, Math.min(safeHeight - 1, Math.floor(expanded.y0 * safeHeight)));
  const rightPx = Math.max(x + 1, Math.min(safeWidth, Math.ceil(expanded.x1 * safeWidth)));
  const bottomPx = Math.max(y + 1, Math.min(safeHeight, Math.ceil(expanded.y1 * safeHeight)));
  const cropWidth = Math.max(1, rightPx - x);
  const cropHeight = Math.max(1, bottomPx - y);
  const areaRatio = (cropWidth * cropHeight) / Math.max(1, safeWidth * safeHeight);

  return {
    applied: true,
    x,
    y,
    width: cropWidth,
    height: cropHeight,
    areaRatio: clamp01(areaRatio),
  };
}

function buildWorkingCanvasFromCrop(baseCanvas, cropMeta) {
  if (!cropMeta?.applied) return baseCanvas;
  const out = new OffscreenCanvas(Math.max(1, cropMeta.width), Math.max(1, cropMeta.height));
  const ctx = out.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) return baseCanvas;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, out.width, out.height);
  ctx.drawImage(
    baseCanvas,
    cropMeta.x,
    cropMeta.y,
    cropMeta.width,
    cropMeta.height,
    0,
    0,
    out.width,
    out.height,
  );
  return out;
}

function estimateLuma(r, g, b) {
  return Math.round(0.299 * r + 0.587 * g + 0.114 * b);
}

function trimDarkEdgeBars(canvas, enabled) {
  if (!enabled) {
    return { canvas, trimmed: false, areaRatio: 1 };
  }

  const width = Math.max(1, Math.floor(safeNumber(canvas?.width, 1)));
  const height = Math.max(1, Math.floor(safeNumber(canvas?.height, 1)));
  if (width < 80 || height < 80) {
    return { canvas, trimmed: false, areaRatio: 1 };
  }

  const ctx = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) return { canvas, trimmed: false, areaRatio: 1 };

  let imageData = null;
  try {
    imageData = ctx.getImageData(0, 0, width, height);
  } catch {
    return { canvas, trimmed: false, areaRatio: 1 };
  }
  const data = imageData?.data;
  if (!data) return { canvas, trimmed: false, areaRatio: 1 };

  const darkThreshold = 24;
  const minNonDarkFracCol = 0.035;
  const minNonDarkFracRow = 0.03;
  const stepY = Math.max(1, Math.floor(height / 280));
  const stepX = Math.max(1, Math.floor(width / 280));

  const colRatio = (x) => {
    let nonDark = 0;
    let count = 0;
    for (let y = 0; y < height; y += stepY) {
      const idx = (y * width + x) * 4;
      const luma = estimateLuma(data[idx], data[idx + 1], data[idx + 2]);
      if (luma > darkThreshold) nonDark += 1;
      count += 1;
    }
    return nonDark / Math.max(1, count);
  };

  let left = 0;
  while (left < width - 1 && colRatio(left) < minNonDarkFracCol) left += 1;
  let right = width - 1;
  while (right > left && colRatio(right) < minNonDarkFracCol) right -= 1;

  const rowRatio = (y) => {
    let nonDark = 0;
    let count = 0;
    for (let x = left; x <= right; x += stepX) {
      const idx = (y * width + x) * 4;
      const luma = estimateLuma(data[idx], data[idx + 1], data[idx + 2]);
      if (luma > darkThreshold) nonDark += 1;
      count += 1;
    }
    return nonDark / Math.max(1, count);
  };

  let top = 0;
  while (top < height - 1 && rowRatio(top) < minNonDarkFracRow) top += 1;
  let bottom = height - 1;
  while (bottom > top && rowRatio(bottom) < minNonDarkFracRow) bottom -= 1;

  const rawTrimWidth = Math.max(1, right - left + 1);
  const rawTrimHeight = Math.max(1, bottom - top + 1);
  if (rawTrimWidth < width * 0.45 || rawTrimHeight < height * 0.45) {
    return { canvas, trimmed: false, areaRatio: 1 };
  }

  const padX = Math.max(2, Math.round(width * 0.01));
  const padY = Math.max(2, Math.round(height * 0.01));
  const x0 = Math.max(0, left - padX);
  const y0 = Math.max(0, top - padY);
  const x1 = Math.min(width - 1, right + padX);
  const y1 = Math.min(height - 1, bottom + padY);
  const trimWidth = Math.max(1, x1 - x0 + 1);
  const trimHeight = Math.max(1, y1 - y0 + 1);
  const areaRatio = (trimWidth * trimHeight) / Math.max(1, width * height);
  if (areaRatio > 0.985) {
    return { canvas, trimmed: false, areaRatio: 1 };
  }

  const out = new OffscreenCanvas(trimWidth, trimHeight);
  const outCtx = out.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!outCtx) return { canvas, trimmed: false, areaRatio: 1 };
  outCtx.fillStyle = "#ffffff";
  outCtx.fillRect(0, 0, trimWidth, trimHeight);
  outCtx.drawImage(canvas, x0, y0, trimWidth, trimHeight, 0, 0, trimWidth, trimHeight);

  return {
    canvas: out,
    trimmed: true,
    areaRatio: clamp01(areaRatio),
  };
}

function maybeUpscaleCroppedCanvas(canvas, cropMeta, options) {
  if (!cropMeta?.applied) {
    return { canvas, upscaled: false, scale: 1 };
  }

  const width = Math.max(1, Math.floor(safeNumber(canvas?.width, 1)));
  const height = Math.max(1, Math.floor(safeNumber(canvas?.height, 1)));
  const longEdge = Math.max(width, height);
  const shortEdge = Math.max(1, Math.min(width, height));
  const maxLong = Math.max(1, Math.floor(safeNumber(options?.maxDim, longEdge)));
  const targetLong = options?.mode === "high_accuracy" ? 2800 : 2200;
  const targetShort = options?.mode === "high_accuracy" ? 1600 : 1200;
  const maxScale = options?.mode === "high_accuracy" ? 3.2 : 2.8;

  const desiredScale = Math.max(targetLong / longEdge, targetShort / shortEdge);
  const scale = Math.min(maxScale, maxLong / longEdge, desiredScale);
  if (!Number.isFinite(scale) || scale <= 1.06) {
    return { canvas, upscaled: false, scale: 1 };
  }

  const outWidth = Math.max(1, Math.round(width * scale));
  const outHeight = Math.max(1, Math.round(height * scale));
  const out = new OffscreenCanvas(outWidth, outHeight);
  const ctx = out.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) return { canvas, upscaled: false, scale: 1 };

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, outWidth, outHeight);
  ctx.drawImage(canvas, 0, 0, width, height, 0, 0, outWidth, outHeight);
  return {
    canvas: out,
    upscaled: true,
    scale: outWidth / Math.max(1, width),
  };
}

function summarizePageMetrics(metrics, preprocessResult, selectedPass, passCount, cropMeta = null, inputMeta = null) {
  return {
    charCount: Number(metrics.charCount) || 0,
    alphaRatio: Number(metrics.alphaRatio) || 0,
    meanConf: Number.isFinite(metrics.meanLineConf) ? Number(metrics.meanLineConf) : null,
    meanWordConf: Number.isFinite(metrics.meanWordConf) ? Number(metrics.meanWordConf) : null,
    lowConfFrac: Number.isFinite(metrics.lowConfLineFrac) ? Number(metrics.lowConfLineFrac) : null,
    lowConfWordFrac: Number.isFinite(metrics.lowConfWordFrac) ? Number(metrics.lowConfWordFrac) : null,
    numLines: Number(metrics.numLines) || 0,
    wordCount: Number(metrics.wordCount) || 0,
    medianTokenLen: Number(metrics.medianTokenLen) || 0,
    junkScore: Number(metrics.junkScore) || 0,
    textSource: String(metrics.textSource || "lines"),
    blurVariance: Number(preprocessResult?.metrics?.blurVariance) || 0,
    overexposureRatio: Number(preprocessResult?.metrics?.overexposureRatio) || 0,
    underexposureRatio: Number(preprocessResult?.metrics?.underexposureRatio) || 0,
    dynamicRange: Number(preprocessResult?.metrics?.dynamicRange) || 0,
    selectedPass: String(selectedPass || "primary"),
    passCount: Number(passCount) || 1,
    preprocessMode: String(preprocessResult?.plan?.resolvedMode || preprocessResult?.plan?.mode || "off"),
    cropApplied: Boolean(cropMeta?.applied),
    cropAreaRatio: Number.isFinite(cropMeta?.areaRatio) ? Number(cropMeta.areaRatio) : 1,
    ocrInputWidth: Number(inputMeta?.width) || 0,
    ocrInputHeight: Number(inputMeta?.height) || 0,
    cropUpscaled: Boolean(inputMeta?.upscaled),
    cropUpscaleFactor: Number.isFinite(inputMeta?.upscaleFactor) ? Number(inputMeta.upscaleFactor) : 1,
    cropEdgeTrimmed: Boolean(inputMeta?.trimmed),
  };
}

function scoreOcrCandidate(metrics) {
  const charScore = clamp01((safeNumber(metrics?.charCount, 0) - 40) / 520);
  const alphaScore = clamp01((safeNumber(metrics?.alphaRatio, 0) - 0.25) / 0.55);
  const confScore = Number.isFinite(metrics?.meanLineConf)
    ? clamp01((Number(metrics.meanLineConf) - 20) / 55)
    : 0.35;
  const wordConfScore = Number.isFinite(metrics?.meanWordConf)
    ? clamp01((Number(metrics.meanWordConf) - 20) / 55)
    : 0.35;
  const lineScore = clamp01(safeNumber(metrics?.numLines, 0) / 26);
  const junkPenalty = clamp01(safeNumber(metrics?.junkScore, 0));
  const lowConfPenalty = Number.isFinite(metrics?.lowConfLineFrac)
    ? clamp01(metrics.lowConfLineFrac)
    : 0.45;
  const lowConfWordPenalty = Number.isFinite(metrics?.lowConfWordFrac)
    ? clamp01(metrics.lowConfWordFrac)
    : 0.45;

  return (
    charScore * 0.3 +
    alphaScore * 0.17 +
    confScore * 0.17 +
    wordConfScore * 0.14 +
    lineScore * 0.1 -
    junkPenalty * 0.17 -
    lowConfPenalty * 0.11 -
    lowConfWordPenalty * 0.08
  );
}

function isWeakOcrMetrics(metrics) {
  const charCount = Number(metrics?.charCount) || 0;
  const alphaRatio = Number(metrics?.alphaRatio) || 0;
  const meanLineConf = Number.isFinite(metrics?.meanLineConf) ? Number(metrics.meanLineConf) : null;
  const meanWordConf = Number.isFinite(metrics?.meanWordConf) ? Number(metrics.meanWordConf) : null;
  const lowConfLineFrac = Number.isFinite(metrics?.lowConfLineFrac) ? Number(metrics.lowConfLineFrac) : null;
  const lowConfWordFrac = Number.isFinite(metrics?.lowConfWordFrac) ? Number(metrics.lowConfWordFrac) : null;
  const wordCount = Number(metrics?.wordCount) || 0;
  const junkScore = Number(metrics?.junkScore) || 0;

  if (charCount < 90) return true;
  if (alphaRatio < 0.4) return true;
  if (Number.isFinite(meanLineConf) && meanLineConf < 38) return true;
  if (Number.isFinite(meanWordConf) && meanWordConf < 46) return true;
  if (Number.isFinite(lowConfLineFrac) && lowConfLineFrac > 0.62) return true;
  if (wordCount >= 12 && Number.isFinite(lowConfWordFrac) && lowConfWordFrac > 0.55) return true;
  if (junkScore > 0.34) return true;
  return false;
}

function dedupePassSpecs(specs) {
  const out = [];
  const seen = new Set();
  for (const spec of Array.isArray(specs) ? specs : []) {
    if (!spec) continue;
    const key = `${spec.preprocessMode}|${spec.psm}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(spec);
  }
  return out;
}

function buildFallbackPassSpecs(options, primaryResolvedMode) {
  const mode = String(primaryResolvedMode || options.preprocess.mode || "auto");
  const alternateMode = mode === "bw_high_contrast" ? "grayscale" : "bw_high_contrast";
  const specs = [
    {
      label: "fallback_dense",
      preprocessMode: mode,
      psm: options.fallbackPsm,
    },
    {
      label: "fallback_sparse",
      preprocessMode: mode,
      psm: options.sparsePsm,
    },
    {
      label: `alt_${alternateMode}`,
      preprocessMode: alternateMode,
      psm: options.psm,
    },
  ];

  if (options.mode === "high_accuracy") {
    specs.push({
      label: "raw_off",
      preprocessMode: "off",
      psm: options.fallbackPsm,
    });
  }

  const deduped = dedupePassSpecs(specs);
  const maxTotalPasses = Number(OCR_PASS_LIMITS[options.mode] || OCR_PASS_LIMITS.fast);
  const maxFallbackPasses = Math.max(0, maxTotalPasses - 1);
  return deduped.slice(0, maxFallbackPasses);
}

async function runSingleOcrPass(worker, baseCanvas, pageIndex, options, passSpec) {
  const preprocessResult = preprocessCanvasForOcr(baseCanvas, {
    mode: passSpec.preprocessMode,
    maxDim: options.maxDim,
    sceneHint: options.sceneHint,
    warningProfile: options.warningProfile,
  });

  await setWorkerPsm(worker, passSpec.psm);
  const ocrInput = await buildOcrInputFromCanvas(preprocessResult.canvas);
  const recognized = await worker.recognize(ocrInput);

  const monitorCapture = options.sceneHint === "monitor";
  const rawLines = extractLines(recognized?.data, pageIndex, {
    minWordConfidence: monitorCapture
      ? options.mode === "high_accuracy"
        ? 50
        : 54
      : options.mode === "high_accuracy"
        ? 42
        : 45,
    minWordKeepRatio: monitorCapture ? 0.5 : 0.38,
    preferWordRebuild: monitorCapture,
  });
  const filtered = filterOcrLinesDetailed(rawLines, [], {
    disableFigureOverlap: true,
    dropCaptions: false,
    dropBoilerplate: true,
    shortLowConfThreshold: options.mode === "high_accuracy" ? 26 : 30,
  });

  const lines = Array.isArray(filtered?.lines) ? filtered.lines : [];
  const lineText = cleanCameraOcrPageText(composeOcrPageText(lines));
  const rawText = cleanCameraOcrPageText(applyClinicalOcrHeuristics(String(recognized?.data?.text || "")));

  let text = lineText;
  let metrics = computeOcrTextMetrics({ text: lineText, lines });
  const rawMetrics = computeOcrTextMetrics({ text: rawText });
  const lineScore = scoreOcrCandidate(metrics);
  const rawScore = scoreOcrCandidate(rawMetrics);
  const rawClearlyBetter =
    rawText.trim() &&
    (rawScore > lineScore + 0.08 || !lineText.trim()) &&
    Number(rawMetrics.junkScore || 0) <= Number(metrics.junkScore || 0) + 0.04 &&
    Number(rawMetrics.footerBoilerplateHits || 0) <= Number(metrics.footerBoilerplateHits || 0);
  if (rawClearlyBetter) {
    text = rawText;
    metrics = rawMetrics;
    metrics.textSource = "raw";
  } else {
    metrics.textSource = "lines";
  }

  const wordStats = computeWordConfidenceStats(recognized?.data?.words);
  metrics.meanWordConf = Number.isFinite(wordStats.meanWordConf) ? Number(wordStats.meanWordConf) : null;
  metrics.lowConfWordFrac = Number.isFinite(wordStats.lowConfWordFrac) ? Number(wordStats.lowConfWordFrac) : null;
  metrics.wordCount = Number(wordStats.wordCount) || 0;

  return {
    label: passSpec.label,
    preprocessResult,
    preprocessModeRequested: passSpec.preprocessMode,
    preprocessModeResolved: String(
      preprocessResult?.plan?.resolvedMode || preprocessResult?.plan?.mode || passSpec.preprocessMode,
    ),
    psm: String(passSpec.psm || options.psm),
    text,
    lines,
    metrics,
    score: scoreOcrCandidate(metrics),
    droppedCount: Array.isArray(filtered?.dropped) ? filtered.dropped.length : 0,
  };
}

function isCancelled(jobId) {
  return cancelledJobs.has(String(jobId || ""));
}

function closeBitmap(bitmap) {
  if (!bitmap || !hasFunction(bitmap.close)) return;
  try {
    bitmap.close();
  } catch {
    // ignore
  }
}

function collectWarnings(bestPass, passResults, cropMeta = null, inputMeta = null) {
  const warningSet = new Set();
  for (const warning of Array.isArray(bestPass?.preprocessResult?.warnings) ? bestPass.preprocessResult.warnings : []) {
    const text = String(warning || "").trim();
    if (text) warningSet.add(text);
  }

  if (Array.isArray(passResults) && passResults.length > 1 && bestPass && passResults[0] && bestPass !== passResults[0]) {
    warningSet.add("OCR auto-corrected low-quality initial pass.");
  }

  if (bestPass && isWeakOcrMetrics(bestPass.metrics)) {
    warningSet.add("OCR confidence is low; retake closer, flatter photos for better text recovery.");
  }
  if (cropMeta?.applied) {
    const keptPct = Math.round((Number(cropMeta.areaRatio) || 0) * 100);
    warningSet.add(`Crop applied before OCR (${keptPct}% of page area kept).`);
  }
  if (inputMeta?.trimmed) {
    warningSet.add("Trimmed dark crop borders before OCR for better text focus.");
  }
  if (inputMeta?.upscaled && Number.isFinite(inputMeta?.upscaleFactor) && inputMeta.upscaleFactor > 1.06) {
    warningSet.add(`Upscaled cropped text ${inputMeta.upscaleFactor.toFixed(2)}x for OCR readability.`);
  }

  return [...warningSet];
}

async function runJob(data) {
  const jobId = String(data.jobId || "");
  if (!jobId) {
    self.postMessage({ type: "camera_ocr_error", jobId, error: "Missing jobId" });
    return;
  }

  const options = resolveOptions(data.options || {});
  const pages = (Array.isArray(data.pages) ? data.pages : [])
    .filter((page) => page && page.bitmap && Number.isFinite(page.pageIndex))
    .sort((a, b) => Number(a.pageIndex) - Number(b.pageIndex));

  if (!pages.length) {
    self.postMessage({ type: "camera_ocr_error", jobId, error: "No pages provided" });
    return;
  }

  activeJobId = jobId;

  try {
    const worker = await getTesseractWorker(options);
    await setWorkerRecognitionProfile(worker, options);
    const outPages = [];

    for (let idx = 0; idx < pages.length; idx += 1) {
      if (isCancelled(jobId)) {
        self.postMessage({ type: "camera_ocr_cancelled", jobId });
        return;
      }

      const page = pages[idx];
      const pageIndex = Number(page.pageIndex);
      self.postMessage({
        type: "camera_ocr_progress",
        jobId,
        pageIndex,
        stage: "preprocess",
        pct: idx / Math.max(1, pages.length),
      });

      const baseCanvas = new OffscreenCanvas(
        Math.max(1, Math.floor(safeNumber(page.width, page.bitmap.width || 1))),
        Math.max(1, Math.floor(safeNumber(page.height, page.bitmap.height || 1))),
      );
      const baseCtx = baseCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
      if (!baseCtx) {
        throw new Error("Unable to acquire 2D context for camera OCR");
      }

      baseCtx.fillStyle = "#ffffff";
      baseCtx.fillRect(0, 0, baseCanvas.width, baseCanvas.height);
      baseCtx.drawImage(page.bitmap, 0, 0, baseCanvas.width, baseCanvas.height);
      const cropMeta = resolveCropPixels(page.crop, baseCanvas.width, baseCanvas.height);
      const croppedCanvas = buildWorkingCanvasFromCrop(baseCanvas, cropMeta);
      const trimResult = trimDarkEdgeBars(croppedCanvas, cropMeta.applied);
      const upscaleResult = maybeUpscaleCroppedCanvas(trimResult.canvas, cropMeta, options);
      const workingCanvas = upscaleResult.canvas;
      const inputMeta = {
        width: Number(workingCanvas?.width) || 0,
        height: Number(workingCanvas?.height) || 0,
        trimmed: Boolean(trimResult.trimmed),
        trimAreaRatio: Number.isFinite(trimResult.areaRatio) ? Number(trimResult.areaRatio) : 1,
        upscaled: Boolean(upscaleResult.upscaled),
        upscaleFactor: Number.isFinite(upscaleResult.scale) ? Number(upscaleResult.scale) : 1,
      };

      const passResults = [];
      const primarySpec = {
        label: "primary",
        preprocessMode: options.preprocess.mode,
        psm: options.psm,
      };

      self.postMessage({
        type: "camera_ocr_progress",
        jobId,
        pageIndex,
        stage: "recognize_primary",
        pct: idx / Math.max(1, pages.length),
      });

      let bestPass = await runSingleOcrPass(worker, workingCanvas, pageIndex, options, primarySpec);
      passResults.push(bestPass);

      if (isCancelled(jobId)) {
        closeBitmap(page.bitmap);
        self.postMessage({ type: "camera_ocr_cancelled", jobId });
        return;
      }

      if (isWeakOcrMetrics(bestPass.metrics)) {
        const fallbackSpecs = buildFallbackPassSpecs(options, bestPass.preprocessModeResolved);
        for (let retryIndex = 0; retryIndex < fallbackSpecs.length; retryIndex += 1) {
          if (isCancelled(jobId)) {
            closeBitmap(page.bitmap);
            self.postMessage({ type: "camera_ocr_cancelled", jobId });
            return;
          }

          const spec = fallbackSpecs[retryIndex];
          self.postMessage({
            type: "camera_ocr_progress",
            jobId,
            pageIndex,
            stage: `recognize_retry_${retryIndex + 1}`,
            pct: idx / Math.max(1, pages.length),
          });

          const candidate = await runSingleOcrPass(worker, workingCanvas, pageIndex, options, spec);
          passResults.push(candidate);
          if (candidate.score > bestPass.score) {
            bestPass = candidate;
          }

          // Early stop once we get a clearly better and stable fallback.
          if (!isWeakOcrMetrics(bestPass.metrics) && bestPass.score >= passResults[0].score + 0.02) {
            break;
          }
        }
      }

      if (isCancelled(jobId)) {
        closeBitmap(page.bitmap);
        self.postMessage({ type: "camera_ocr_cancelled", jobId });
        return;
      }

      self.postMessage({
        type: "camera_ocr_progress",
        jobId,
        pageIndex,
        stage: "postprocess",
        pct: idx / Math.max(1, pages.length),
      });

      const lines = Array.isArray(bestPass.lines) ? bestPass.lines : [];
      const pageResult = {
        pageIndex,
        text: String(bestPass.text || ""),
        lines: lines.map((line) => ({
          text: String(line.text || ""),
          confidence: coerceConfidence(line.confidence),
          bbox: normalizeBbox(line.bbox),
        })),
        metrics: summarizePageMetrics(
          bestPass.metrics,
          bestPass.preprocessResult,
          bestPass.label,
          passResults.length,
          cropMeta,
          inputMeta,
        ),
        warnings: collectWarnings(bestPass, passResults, cropMeta, inputMeta),
      };

      outPages.push(pageResult);
      self.postMessage({ type: "camera_ocr_page", jobId, page: pageResult });
      closeBitmap(page.bitmap);
    }

    self.postMessage({
      type: "camera_ocr_progress",
      jobId,
      stage: "postprocess",
      pct: 1,
      completedPages: pages.length,
      totalPages: pages.length,
    });
    self.postMessage({ type: "camera_ocr_done", jobId, pages: outPages });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ type: "camera_ocr_error", jobId, error: message });
  } finally {
    cancelledJobs.delete(jobId);
    if (activeJobId === jobId) {
      activeJobId = "";
    }
  }
}

self.onmessage = async (event) => {
  const data = event?.data || {};
  if (data.type === "camera_ocr_cancel") {
    const jobId = String(data.jobId || "");
    if (jobId) cancelledJobs.add(jobId);
    return;
  }

  if (data.type === "camera_ocr_run") {
    await runJob(data);
  }
};
