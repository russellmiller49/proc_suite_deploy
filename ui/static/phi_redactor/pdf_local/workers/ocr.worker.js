import "../../vendor/pdfjs/pdf.worker.mjs";
import * as pdfjs from "../../vendor/pdfjs/pdf.mjs";
import Tesseract from "../../vendor/tesseract/tesseract.esm.min.js";
import { clamp01, mergeRegions, normalizeRect, rectArea } from "../pdf/layoutAnalysis.js";
import {
  computeHeaderZoneColumns,
  computeOcrCropRect,
  computeProvationDiagramSkipRegions,
} from "../pdf/ocrRegions.js";
import { computeOcrTextMetrics } from "../pdf/ocrMetrics.js";
import {
  composeOcrPageText,
  dedupeConsecutiveLines,
  filterOcrLinesDetailed,
} from "../pdf/ocrPostprocess.js";

pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "../../vendor/pdfjs/pdf.worker.mjs",
  import.meta.url,
).toString();

const DEFAULT_OCR_OPTIONS = Object.freeze({
  lang: "eng",
  qualityMode: "fast",
  scale: 2,
  psm: "6",
  maskImages: "auto",
  maskMarginPx: 6,
  maxMaskRegions: 12,
  cropMode: "auto",
  cropPaddingPx: 14,
  headerBandFrac: 0.25,
  headerScaleBoost: 1.8,
  headerRetryPsms: ["6", "4", "11"],
  figureOverlapThreshold: 0.35,
  shortLowConfThreshold: 30,
  backfillScaleBoost: 1.7,
  backfillPreprocess: true,
  backfillThresholdBias: -8,
  backfillDilate: true,
});

let activeJobId = 0;
let tesseractWorker = null;
let tesseractConfigKey = "";

function resolveOcrOptions(options = {}) {
  const merged = {
    ...DEFAULT_OCR_OPTIONS,
    ...(options && typeof options === "object" ? options : {}),
  };

  const qualityMode = merged.qualityMode === "high_accuracy" ? "high_accuracy" : "fast";
  const qualityScale = qualityMode === "high_accuracy" ? 3.1 : 2.05;
  const configuredScale = Number.isFinite(merged.scale) ? Number(merged.scale) : qualityScale;
  const maskImages = merged.maskImages === "off" || merged.maskImages === "none" || merged.maskImages === false
    ? "off"
    : merged.maskImages === "on" || merged.maskImages === true
      ? "on"
      : "auto";
  const maskMarginPx = Number.isFinite(merged.maskMarginPx) ? Math.max(0, Number(merged.maskMarginPx)) : 6;
  const maxMaskRegions = Number.isFinite(merged.maxMaskRegions)
    ? Math.max(0, Math.min(40, Math.floor(Number(merged.maxMaskRegions))))
    : 12;
  const cropMode = merged.cropMode === "off" || merged.cropMode === false
    ? "off"
    : merged.cropMode === "on" || merged.cropMode === true
      ? "on"
      : "auto";
  const cropPaddingPx = Number.isFinite(merged.cropPaddingPx)
    ? Math.max(0, Number(merged.cropPaddingPx))
    : 14;
  const headerBandFrac = Number.isFinite(merged.headerBandFrac)
    ? clamp01(Number(merged.headerBandFrac))
    : DEFAULT_OCR_OPTIONS.headerBandFrac;
  const headerScaleBoost = Number.isFinite(merged.headerScaleBoost)
    ? Math.max(1, Math.min(3, Number(merged.headerScaleBoost)))
    : DEFAULT_OCR_OPTIONS.headerScaleBoost;
  const rawHeaderRetryPsms = Array.isArray(merged.headerRetryPsms)
    ? merged.headerRetryPsms
    : DEFAULT_OCR_OPTIONS.headerRetryPsms;
  const headerRetryPsms = [...new Set(rawHeaderRetryPsms
    .map((value) => String(value || "").trim())
    .filter(Boolean))]
    .slice(0, 6);
  const figureOverlapThreshold = Number.isFinite(merged.figureOverlapThreshold)
    ? clamp01(Number(merged.figureOverlapThreshold))
    : DEFAULT_OCR_OPTIONS.figureOverlapThreshold;
  const shortLowConfThreshold = Number.isFinite(merged.shortLowConfThreshold)
    ? Math.max(0, Math.min(100, Number(merged.shortLowConfThreshold)))
    : DEFAULT_OCR_OPTIONS.shortLowConfThreshold;
  const backfillScaleBoost = Number.isFinite(merged.backfillScaleBoost)
    ? Math.max(1, Math.min(2.4, Number(merged.backfillScaleBoost)))
    : DEFAULT_OCR_OPTIONS.backfillScaleBoost;
  const backfillThresholdBias = Number.isFinite(merged.backfillThresholdBias)
    ? Math.max(-40, Math.min(40, Number(merged.backfillThresholdBias)))
    : DEFAULT_OCR_OPTIONS.backfillThresholdBias;

  return {
    lang: typeof merged.lang === "string" && merged.lang.trim() ? merged.lang.trim() : "eng",
    qualityMode,
    scale: Math.max(1.1, Math.min(4, configuredScale)),
    psm: String(merged.psm || "6"),
    maskImages,
    maskMarginPx,
    maxMaskRegions,
    cropMode,
    cropPaddingPx,
    headerBandFrac: Math.max(0.2, Math.min(0.35, headerBandFrac || DEFAULT_OCR_OPTIONS.headerBandFrac)),
    headerScaleBoost,
    headerRetryPsms: headerRetryPsms.length ? headerRetryPsms : [...DEFAULT_OCR_OPTIONS.headerRetryPsms],
    figureOverlapThreshold,
    shortLowConfThreshold,
    backfillScaleBoost,
    backfillPreprocess: merged.backfillPreprocess !== false,
    backfillThresholdBias,
    backfillDilate: merged.backfillDilate !== false,
  };
}

function toAssetUrl(path) {
  return new URL(`../../vendor/tesseract/${path}`, import.meta.url).toString();
}

async function getTesseractWorker(options, pageIndex, totalPages) {
  const configKey = `${options.lang}`;
  if (tesseractWorker && configKey === tesseractConfigKey) {
    return tesseractWorker;
  }

  if (tesseractWorker) {
    try {
      await tesseractWorker.terminate();
    } catch {
      // Best-effort cleanup.
    }
  }

  self.postMessage({
    type: "ocr_stage",
    stage: "ocr_loading_assets",
    pageIndex,
    totalPages,
  });

  const workerPath = toAssetUrl("worker.min.js");
  const corePath = toAssetUrl("tesseract-core-simd.wasm.js");
  const langPath = toAssetUrl("tessdata/");

  const worker = await Tesseract.createWorker(
    options.lang,
    Tesseract.OEM.LSTM_ONLY,
    {
      workerPath,
      corePath,
      langPath,
      // Keep worker loading same-origin only; avoids CSP failures from blob workers.
      workerBlobURL: false,
      gzip: false,
      logger: (message) => {
        if (!message || typeof message !== "object") return;
        const progress = Number(message.progress);
        if (!Number.isFinite(progress)) return;
        self.postMessage({
          type: "ocr_status",
          status: String(message.status || "ocr"),
          progress,
        });
      },
    },
  );

  await worker.setParameters({
    preserve_interword_spaces: "1",
  });

  tesseractWorker = worker;
  tesseractConfigKey = configKey;
  activePsm = "";
  return worker;
}

function getViewportScale(page, baseScale) {
  const unscaled = page.getViewport({ scale: 1 });
  const maxDimension = Math.max(unscaled.width, unscaled.height);
  if (maxDimension <= 0) return baseScale;
  if (maxDimension * baseScale <= 2800) return baseScale;
  return Math.max(1.2, 2800 / maxDimension);
}

const OFFSCREEN_CANVAS_FACTORY = {
  create(width, height) {
    const safeWidth = Math.max(1, Math.floor(Number(width) || 0));
    const safeHeight = Math.max(1, Math.floor(Number(height) || 0));
    const canvas = new OffscreenCanvas(safeWidth, safeHeight);
    const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
    if (!context) {
      throw new Error("Unable to acquire 2D context for PDF render canvas.");
    }
    return { canvas, context };
  },

  reset(canvasAndContext, width, height) {
    if (!canvasAndContext?.canvas) {
      return this.create(width, height);
    }
    canvasAndContext.canvas.width = Math.max(1, Math.floor(Number(width) || 0));
    canvasAndContext.canvas.height = Math.max(1, Math.floor(Number(height) || 0));
    return canvasAndContext;
  },

  destroy(canvasAndContext) {
    if (!canvasAndContext?.canvas) return;
    canvasAndContext.canvas.width = 0;
    canvasAndContext.canvas.height = 0;
    canvasAndContext.canvas = null;
    canvasAndContext.context = null;
  },
};

class WorkerCanvasFactory {
  constructor({ enableHWA = false } = {}) {
    this._enableHWA = Boolean(enableHWA);
  }

  create(width, height) {
    const safeWidth = Math.max(1, Math.floor(Number(width) || 0));
    const safeHeight = Math.max(1, Math.floor(Number(height) || 0));
    const canvas = new OffscreenCanvas(safeWidth, safeHeight);
    const context = canvas.getContext("2d", { willReadFrequently: !this._enableHWA });
    if (!context) {
      throw new Error("Unable to acquire 2D context for WorkerCanvasFactory.");
    }
    return { canvas, context };
  }

  reset(canvasAndContext, width, height) {
    if (!canvasAndContext?.canvas) {
      return this.create(width, height);
    }
    canvasAndContext.canvas.width = Math.max(1, Math.floor(Number(width) || 0));
    canvasAndContext.canvas.height = Math.max(1, Math.floor(Number(height) || 0));
    return canvasAndContext;
  }

  destroy(canvasAndContext) {
    if (!canvasAndContext?.canvas) return;
    canvasAndContext.canvas.width = 0;
    canvasAndContext.canvas.height = 0;
    canvasAndContext.canvas = null;
    canvasAndContext.context = null;
  }
}

class WorkerFilterFactory {
  addFilter() {
    return "none";
  }

  addHCMFilter() {
    return "none";
  }

  addAlphaFilter() {
    return "none";
  }

  addLuminosityFilter() {
    return "none";
  }

  destroy() {
    // no-op for worker fallback filter factory.
  }
}

const HEADER_RETRY_PSMS = Object.freeze(["6", "4", "11"]);

function sortByReadingOrder(a, b) {
  const ay = Number(a?.bbox?.y) || 0;
  const by = Number(b?.bbox?.y) || 0;
  if (Math.abs(ay - by) > 4) return ay - by;
  const ax = Number(a?.bbox?.x) || 0;
  const bx = Number(b?.bbox?.x) || 0;
  return ax - bx;
}

function coerceConfidence(value) {
  if (!Number.isFinite(value)) return null;
  const conf = Number(value);
  if (!Number.isFinite(conf)) return null;
  if (conf < 0) return null;
  return Math.max(0, Math.min(100, conf));
}

function normalizeBboxFromTesseract(rawBBox, offsetX = 0, offsetY = 0, scaleFactor = 1) {
  const scale = Number.isFinite(scaleFactor) && scaleFactor > 0 ? scaleFactor : 1;
  const invScale = 1 / scale;
  if (!rawBBox || typeof rawBBox !== "object") {
    return { x: offsetX, y: offsetY, width: 0, height: 0 };
  }

  const x0 = Number.isFinite(rawBBox.x0) ? Number(rawBBox.x0) : Number(rawBBox.x) || 0;
  const y0 = Number.isFinite(rawBBox.y0) ? Number(rawBBox.y0) : Number(rawBBox.y) || 0;
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

  return normalizeRect({
    x: offsetX + Math.min(x0, x1) * invScale,
    y: offsetY + Math.min(y0, y1) * invScale,
    width: Math.abs(x1 - x0) * invScale,
    height: Math.abs(y1 - y0) * invScale,
  });
}

function extractWordsFromLine(rawLine, offsetX, offsetY, scaleFactor) {
  const words = [];
  for (const rawWord of Array.isArray(rawLine?.words) ? rawLine.words : []) {
    const text = String(rawWord?.text || "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    words.push({
      text,
      conf: coerceConfidence(rawWord?.confidence ?? rawWord?.conf),
      bbox: normalizeBboxFromTesseract(rawWord?.bbox || rawWord, offsetX, offsetY, scaleFactor),
    });
  }
  return words;
}

function groupWordsIntoLines(words = []) {
  const buckets = new Map();
  for (const rawWord of Array.isArray(words) ? words : []) {
    const text = String(rawWord?.text || "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    const block = Number(rawWord?.block_num) || 0;
    const paragraph = Number(rawWord?.par_num) || 0;
    const line = Number(rawWord?.line_num) || 0;
    const key = `${block}:${paragraph}:${line}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(rawWord);
  }
  return [...buckets.values()];
}

function buildLineFromWords(rawWords, pageIndex, offsetX, offsetY, scaleFactor) {
  const words = [];
  let bbox = null;
  const confValues = [];
  for (const rawWord of Array.isArray(rawWords) ? rawWords : []) {
    const text = String(rawWord?.text || "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    const wordBBox = normalizeBboxFromTesseract(rawWord?.bbox || rawWord, offsetX, offsetY, scaleFactor);
    bbox = bbox
      ? normalizeRect({
        x: Math.min(bbox.x, wordBBox.x),
        y: Math.min(bbox.y, wordBBox.y),
        width: Math.max(bbox.x + bbox.width, wordBBox.x + wordBBox.width) - Math.min(bbox.x, wordBBox.x),
        height: Math.max(bbox.y + bbox.height, wordBBox.y + wordBBox.height) - Math.min(bbox.y, wordBBox.y),
      })
      : wordBBox;

    const conf = coerceConfidence(rawWord?.confidence ?? rawWord?.conf);
    if (Number.isFinite(conf)) confValues.push(conf);
    words.push({ text, conf, bbox: wordBBox });
  }

  const text = words.map((word) => word.text).join(" ").replace(/\s+/g, " ").trim();
  if (!text) return null;
  const confidence = confValues.length
    ? confValues.reduce((acc, value) => acc + value, 0) / confValues.length
    : null;

  return {
    text,
    confidence,
    bbox: bbox || { x: offsetX, y: offsetY, width: 0, height: 0 },
    words,
    pageIndex,
  };
}

function extractStructuredOcrLines(data, opts = {}) {
  const pageIndex = Number.isFinite(opts.pageIndex) ? Number(opts.pageIndex) : 0;
  const offsetX = Number(opts.offsetX) || 0;
  const offsetY = Number(opts.offsetY) || 0;
  const scaleFactor = Number.isFinite(opts.scaleFactor) && opts.scaleFactor > 0 ? opts.scaleFactor : 1;

  const out = [];
  const rawLines = Array.isArray(data?.lines) ? data.lines : [];

  for (const rawLine of rawLines) {
    const text = String(rawLine?.text || "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    const words = extractWordsFromLine(rawLine, offsetX, offsetY, scaleFactor);
    const wordConf = words.map((word) => word.conf).filter((value) => Number.isFinite(value));
    const confidence = Number.isFinite(rawLine?.confidence)
      ? coerceConfidence(rawLine.confidence)
      : Number.isFinite(rawLine?.conf)
        ? coerceConfidence(rawLine.conf)
        : (wordConf.length
          ? wordConf.reduce((acc, value) => acc + value, 0) / wordConf.length
          : null);
    out.push({
      text,
      confidence,
      bbox: normalizeBboxFromTesseract(rawLine?.bbox || rawLine, offsetX, offsetY, scaleFactor),
      words,
      pageIndex,
    });
  }

  if (!out.length && Array.isArray(data?.words) && data.words.length) {
    const grouped = groupWordsIntoLines(data.words);
    for (const group of grouped) {
      const line = buildLineFromWords(group, pageIndex, offsetX, offsetY, scaleFactor);
      if (line) out.push(line);
    }
  }

  out.sort(sortByReadingOrder);
  return dedupeConsecutiveLines(out);
}

function cropCanvas(sourceCanvas, rect) {
  const normalized = normalizeRect(rect);
  const x = Math.max(0, Math.floor(normalized.x));
  const y = Math.max(0, Math.floor(normalized.y));
  const maxWidth = Math.max(1, sourceCanvas?.width || 1);
  const maxHeight = Math.max(1, sourceCanvas?.height || 1);
  if (x >= maxWidth || y >= maxHeight) return null;
  const unclampedWidth = Math.max(1, Math.ceil(normalized.width));
  const unclampedHeight = Math.max(1, Math.ceil(normalized.height));
  const width = Math.max(1, Math.min(unclampedWidth, maxWidth - x));
  const height = Math.max(1, Math.min(unclampedHeight, maxHeight - y));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) return null;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(sourceCanvas, x, y, width, height, 0, 0, width, height);
  return canvas;
}

function scaleCanvas(sourceCanvas, multiplier) {
  const factor = Number.isFinite(multiplier) ? Math.max(1, Number(multiplier)) : 1;
  if (factor <= 1.01) {
    return { canvas: sourceCanvas, scaleFactor: 1 };
  }
  const width = Math.max(1, Math.floor(sourceCanvas.width * factor));
  const height = Math.max(1, Math.floor(sourceCanvas.height * factor));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) {
    return { canvas: sourceCanvas, scaleFactor: 1 };
  }
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(sourceCanvas, 0, 0, sourceCanvas.width, sourceCanvas.height, 0, 0, width, height);
  return { canvas, scaleFactor: factor };
}

function computeOtsuThreshold(histogram, totalCount) {
  const hist = histogram instanceof Uint32Array ? histogram : new Uint32Array(256);
  const total = Math.max(1, Number(totalCount) || 1);
  let sum = 0;
  for (let i = 0; i < 256; i += 1) {
    sum += i * hist[i];
  }

  let sumBg = 0;
  let weightBg = 0;
  let bestThreshold = 127;
  let bestVariance = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < 256; i += 1) {
    weightBg += hist[i];
    if (!weightBg) continue;
    const weightFg = total - weightBg;
    if (!weightFg) break;

    sumBg += i * hist[i];
    const meanBg = sumBg / weightBg;
    const meanFg = (sum - sumBg) / weightFg;
    const diff = meanBg - meanFg;
    const variance = weightBg * weightFg * diff * diff;
    if (variance > bestVariance) {
      bestVariance = variance;
      bestThreshold = i;
    }
  }

  return bestThreshold;
}

function preprocessBackfillCanvas(sourceCanvas, options = {}) {
  const enabled = options?.backfillPreprocess !== false;
  if (!enabled || !sourceCanvas) {
    return {
      canvas: sourceCanvas,
      applied: false,
      reason: enabled ? "missing_source_canvas" : "disabled",
    };
  }

  const width = Math.max(1, Math.floor(Number(sourceCanvas.width) || 0));
  const height = Math.max(1, Math.floor(Number(sourceCanvas.height) || 0));
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) {
    return { canvas: sourceCanvas, applied: false, reason: "no_2d_context" };
  }

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.drawImage(sourceCanvas, 0, 0, width, height);

  let imageData;
  try {
    imageData = ctx.getImageData(0, 0, width, height);
  } catch {
    return { canvas: sourceCanvas, applied: false, reason: "image_data_unavailable" };
  }

  const data = imageData.data;
  const pixelCount = Math.floor(data.length / 4);
  if (!pixelCount) {
    return { canvas: sourceCanvas, applied: false, reason: "empty_canvas" };
  }

  const gray = new Uint8ClampedArray(pixelCount);
  const rawHistogram = new Uint32Array(256);
  let minGray = 255;
  let maxGray = 0;

  let p = 0;
  for (let i = 0; i < data.length; i += 4) {
    const value = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    gray[p] = value;
    rawHistogram[value] += 1;
    if (value < minGray) minGray = value;
    if (value > maxGray) maxGray = value;
    p += 1;
  }

  const dynamicRange = Math.max(1, maxGray - minGray);
  if (dynamicRange < 8) {
    return { canvas: sourceCanvas, applied: false, reason: "low_dynamic_range" };
  }

  const normalized = new Uint8ClampedArray(pixelCount);
  const normalizedHistogram = new Uint32Array(256);
  for (let i = 0; i < pixelCount; i += 1) {
    const stretched = Math.round(((gray[i] - minGray) * 255) / dynamicRange);
    const value = Math.max(0, Math.min(255, stretched));
    normalized[i] = value;
    normalizedHistogram[value] += 1;
  }

  const thresholdBias = Number.isFinite(options?.backfillThresholdBias)
    ? Number(options.backfillThresholdBias)
    : 0;
  const otsu = computeOtsuThreshold(normalizedHistogram, pixelCount);
  const threshold = Math.max(0, Math.min(255, Math.round(otsu + thresholdBias)));

  const binary = new Uint8ClampedArray(pixelCount);
  for (let i = 0; i < pixelCount; i += 1) {
    binary[i] = normalized[i] <= threshold ? 0 : 255;
  }

  let finalBinary = binary;
  const dilatedEnabled = options?.backfillDilate !== false;
  if (dilatedEnabled) {
    const dilated = new Uint8ClampedArray(pixelCount);
    dilated.fill(255);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = y * width + x;
        if (binary[idx] !== 0) continue;
        for (let ny = Math.max(0, y - 1); ny <= Math.min(height - 1, y + 1); ny += 1) {
          for (let nx = Math.max(0, x - 1); nx <= Math.min(width - 1, x + 1); nx += 1) {
            dilated[ny * width + nx] = 0;
          }
        }
      }
    }
    finalBinary = dilated;
  }

  p = 0;
  for (let i = 0; i < data.length; i += 4) {
    const value = finalBinary[p];
    data[i] = value;
    data[i + 1] = value;
    data[i + 2] = value;
    data[i + 3] = 255;
    p += 1;
  }

  ctx.putImageData(imageData, 0, 0);
  return {
    canvas,
    applied: true,
    reason: "grayscale_threshold",
    threshold,
    thresholdBias,
    dilated: dilatedEnabled,
    dynamicRange,
  };
}

async function buildOcrInputFromCanvas(canvas) {
  let ocrInput = canvas;
  if (typeof canvas.convertToBlob === "function") {
    try {
      ocrInput = await canvas.convertToBlob({ type: "image/png" });
    } catch {
      ocrInput = canvas;
    }
  }
  return { ocrInput, fallbackOcrInput: canvas };
}

function detectFigureRegionsFromCanvas(canvas) {
  if (!canvas || canvas.width < 2 || canvas.height < 2) return [];
  const sourceWidth = Math.max(1, canvas.width);
  const sourceHeight = Math.max(1, canvas.height);
  const targetWidth = Math.max(64, Math.min(256, sourceWidth));
  const scale = targetWidth / sourceWidth;
  const targetHeight = Math.max(64, Math.floor(sourceHeight * scale));

  const sampleCanvas = new OffscreenCanvas(targetWidth, targetHeight);
  const sampleCtx = sampleCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!sampleCtx) return [];
  sampleCtx.fillStyle = "#ffffff";
  sampleCtx.fillRect(0, 0, targetWidth, targetHeight);
  sampleCtx.drawImage(canvas, 0, 0, sourceWidth, sourceHeight, 0, 0, targetWidth, targetHeight);

  let imageData;
  try {
    imageData = sampleCtx.getImageData(0, 0, targetWidth, targetHeight);
  } catch {
    return [];
  }

  const data = imageData.data;
  const gray = new Uint8Array(targetWidth * targetHeight);
  let idx = 0;
  for (let i = 0; i < data.length; i += 4) {
    gray[idx] = Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    idx += 1;
  }

  const cellSize = 8;
  const gridW = Math.max(1, Math.ceil(targetWidth / cellSize));
  const gridH = Math.max(1, Math.ceil(targetHeight / cellSize));
  const edgeCounts = new Uint16Array(gridW * gridH);
  const midCounts = new Uint16Array(gridW * gridH);
  const totals = new Uint16Array(gridW * gridH);

  for (let y = 1; y < targetHeight - 1; y += 1) {
    for (let x = 1; x < targetWidth - 1; x += 1) {
      const p = y * targetWidth + x;
      const gx = Math.abs(gray[p + 1] - gray[p - 1]);
      const gy = Math.abs(gray[p + targetWidth] - gray[p - targetWidth]);
      const gradient = gx + gy;
      const cellX = Math.min(gridW - 1, Math.floor(x / cellSize));
      const cellY = Math.min(gridH - 1, Math.floor(y / cellSize));
      const cellIndex = cellY * gridW + cellX;
      totals[cellIndex] += 1;
      if (gradient >= 54) edgeCounts[cellIndex] += 1;
      const intensity = gray[p];
      if (intensity >= 35 && intensity <= 225) midCounts[cellIndex] += 1;
    }
  }

  const candidates = new Uint8Array(gridW * gridH);
  for (let i = 0; i < candidates.length; i += 1) {
    const total = Math.max(1, totals[i]);
    const edgeDensity = edgeCounts[i] / total;
    const midRatio = midCounts[i] / total;
    if (edgeDensity >= 0.16 && midRatio >= 0.32) {
      candidates[i] = 1;
    }
  }

  const visited = new Uint8Array(candidates.length);
  const out = [];
  const queue = [];
  for (let i = 0; i < candidates.length; i += 1) {
    if (!candidates[i] || visited[i]) continue;
    visited[i] = 1;
    queue.length = 0;
    queue.push(i);
    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    let pixelCells = 0;

    while (queue.length) {
      const current = queue.pop();
      const x = current % gridW;
      const y = Math.floor(current / gridW);
      pixelCells += 1;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);

      const neighbors = [
        [x - 1, y],
        [x + 1, y],
        [x, y - 1],
        [x, y + 1],
      ];
      for (const [nx, ny] of neighbors) {
        if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) continue;
        const next = ny * gridW + nx;
        if (visited[next] || !candidates[next]) continue;
        visited[next] = 1;
        queue.push(next);
      }
    }

    const cellAreaRatio = pixelCells / Math.max(1, gridW * gridH);
    if (cellAreaRatio < 0.05) continue;

    const sampleRect = normalizeRect({
      x: minX * cellSize,
      y: minY * cellSize,
      width: (maxX - minX + 1) * cellSize,
      height: (maxY - minY + 1) * cellSize,
    });
    const widthRatio = sampleRect.width / Math.max(1, targetWidth);
    const heightRatio = sampleRect.height / Math.max(1, targetHeight);
    if (widthRatio < 0.2 && heightRatio < 0.2) continue;

    out.push(normalizeRect({
      x: sampleRect.x / scale,
      y: sampleRect.y / scale,
      width: sampleRect.width / scale,
      height: sampleRect.height / scale,
    }));
  }

  return mergeRegions(out, { mergeGap: 8 });
}

function evaluateHeaderFieldQuality(text) {
  const source = String(text || "");
  const dobValid = /\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/.test(source);
  const ageToken = source.match(/\bAge\b[:\s]*([0-9]{1,3})\b/i) ||
    source.match(/\b([0-9]{1,3})\s*(?:y\/?o|years?\s*old)\b/i);
  const ageValue = ageToken ? Number(ageToken[1]) : null;
  const ageHasLabel = /\bAge\b/i.test(source);
  const ageValid = ageToken ? ageValue >= 0 && ageValue <= 120 : !ageHasLabel;
  return {
    dobValid,
    ageValid,
    retryNeeded: !dobValid || !ageValid,
  };
}

let activePsm = "";

async function setWorkerPsm(worker, psm) {
  const normalizedPsm = String(psm || "6");
  if (normalizedPsm === activePsm) return;
  await worker.setParameters({
    tessedit_pageseg_mode: normalizedPsm,
  });
  activePsm = normalizedPsm;
}

async function recognizeCanvas(worker, canvas, psm) {
  await setWorkerPsm(worker, psm);
  const { ocrInput, fallbackOcrInput } = await buildOcrInputFromCanvas(canvas);

  try {
    return await worker.recognize(ocrInput);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const canRetryWithCanvas = fallbackOcrInput && fallbackOcrInput !== ocrInput;
    if (!canRetryWithCanvas || !/createElement/i.test(message)) {
      throw error;
    }
    return worker.recognize(fallbackOcrInput);
  }
}

function toViewportRect(region, viewport) {
  const normalized = normalizeRect(region);
  const x1 = normalized.x;
  const y1 = normalized.y;
  const x2 = normalized.x + normalized.width;
  const y2 = normalized.y + normalized.height;
  if (![x1, y1, x2, y2].every((value) => Number.isFinite(value))) return null;

  const rect = viewport.convertToViewportRectangle([x1, y1, x2, y2]);
  if (!Array.isArray(rect) || rect.length < 4) return null;
  const left = Math.min(rect[0], rect[2]);
  const right = Math.max(rect[0], rect[2]);
  const top = Math.min(rect[1], rect[3]);
  const bottom = Math.max(rect[1], rect[3]);
  const width = right - left;
  const height = bottom - top;
  if (!Number.isFinite(width) || !Number.isFinite(height)) return null;
  if (width <= 1 || height <= 1) return null;
  return normalizeRect({ x: left, y: top, width, height });
}

function clampRectToCanvas(rect, canvasWidth, canvasHeight) {
  const normalized = normalizeRect(rect);
  const left = Math.max(0, normalized.x);
  const top = Math.max(0, normalized.y);
  const right = Math.min(canvasWidth, normalized.x + normalized.width);
  const bottom = Math.min(canvasHeight, normalized.y + normalized.height);
  const width = Math.max(0, right - left);
  const height = Math.max(0, bottom - top);
  if (width <= 1 || height <= 1) return null;
  return { x: left, y: top, width, height };
}

function expandRectPx(rect, marginPx) {
  const normalized = normalizeRect(rect);
  const m = Math.max(0, Number(marginPx) || 0);
  return {
    x: normalized.x - m,
    y: normalized.y - m,
    width: normalized.width + m * 2,
    height: normalized.height + m * 2,
  };
}

function buildViewportHintRegions(viewport, canvasWidth, canvasHeight, hint, options) {
  const imageRegions = [];
  const rawImageRegions = Array.isArray(hint?.imageRegions) ? hint.imageRegions : [];
  for (const region of rawImageRegions) {
    const vr = toViewportRect(region, viewport);
    if (!vr) continue;
    const clamped = clampRectToCanvas(vr, canvasWidth, canvasHeight);
    if (!clamped) continue;
    imageRegions.push(clamped);
  }

  const textRegions = [];
  const rawTextRegions = Array.isArray(hint?.textRegions) ? hint.textRegions : [];
  for (const region of rawTextRegions) {
    const vr = toViewportRect(region, viewport);
    if (!vr) continue;
    const clamped = clampRectToCanvas(vr, canvasWidth, canvasHeight);
    if (!clamped) continue;
    textRegions.push(clamped);
  }

  const lineBandRegions = [];
  const rawLineBandRegions = Array.isArray(hint?.lineBandRegions) ? hint.lineBandRegions : [];
  for (const region of rawLineBandRegions) {
    const vr = toViewportRect(region, viewport);
    if (!vr) continue;
    const clamped = clampRectToCanvas(vr, canvasWidth, canvasHeight);
    if (!clamped) continue;
    lineBandRegions.push(clamped);
  }

  const maskRegions = imageRegions
    .map((region) => expandRectPx(region, options.maskMarginPx))
    .map((region) => clampRectToCanvas(region, canvasWidth, canvasHeight))
    .filter(Boolean);

  return {
    imageRegions: mergeRegions(imageRegions, { mergeGap: 2 }),
    textRegions: mergeRegions(textRegions, { mergeGap: 2 }),
    maskRegions: mergeRegions(maskRegions, { mergeGap: 4 }),
    lineBandRegions: mergeRegions(lineBandRegions, { mergeGap: 4 }),
  };
}

function computeCoverageRatio(rects, pageWidth, pageHeight) {
  const pageArea = Math.max(1, pageWidth * pageHeight);
  const merged = mergeRegions(rects, { mergeGap: 2 });
  let total = 0;
  for (const rect of merged) {
    total += rectArea(rect);
  }
  return clamp01(total / pageArea);
}

function computeRegionSampleMetrics(imageData) {
  const data = imageData?.data;
  if (!data || typeof data.length !== "number" || data.length < 4) {
    return {
      colorfulness: 0,
      whiteRatio: 0,
      darkRatio: 0,
    };
  }

  let colorDiffSum = 0;
  let white = 0;
  let dark = 0;
  const pixelCount = Math.floor(data.length / 4);

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const brightness = (r + g + b) / 3;
    if (brightness >= 240) white += 1;
    else if (brightness <= 50) dark += 1;
    colorDiffSum += Math.abs(r - g) + Math.abs(r - b) + Math.abs(g - b);
  }

  const denom = Math.max(1, pixelCount);
  const colorfulness = clamp01(colorDiffSum / (denom * 510));
  return {
    colorfulness,
    whiteRatio: clamp01(white / denom),
    darkRatio: clamp01(dark / denom),
  };
}

function shouldMaskRegion(metrics, areaRatio) {
  const whiteRatio = clamp01(metrics.whiteRatio);
  const darkRatio = clamp01(metrics.darkRatio);
  const midRatio = clamp01(1 - whiteRatio - darkRatio);
  const colorfulness = clamp01(metrics.colorfulness);

  const looksLikeTextScan = (colorfulness < 0.09 && whiteRatio > 0.72) || whiteRatio > 0.88;
  const looksLikePhoto = midRatio > 0.55 ||
    (colorfulness > 0.12 && whiteRatio < 0.78 && midRatio > 0.25) ||
    (colorfulness > 0.18 && whiteRatio < 0.88);

  if (areaRatio >= 0.92) return { mask: false, reason: "full_page_image" };
  if (looksLikePhoto && !looksLikeTextScan) return { mask: true, reason: "photo_like" };
  return { mask: false, reason: looksLikeTextScan ? "text_like" : "uncertain" };
}

function selectMaskRects(canvas, hint, options, viewportHintRegions = null) {
  const mode = options.maskImages;
  if (mode === "off") {
    return { rects: [], meta: { applied: false, mode, reason: "disabled" } };
  }

  const viewportRegions = Array.isArray(viewportHintRegions?.maskRegions)
    ? viewportHintRegions.maskRegions
    : [];
  if (!viewportRegions.length) {
    return { rects: [], meta: { applied: false, mode, reason: "no_image_regions" } };
  }

  const width = Math.max(1, canvas.width);
  const height = Math.max(1, canvas.height);
  const pageArea = width * height;

  const mergedRegions = mergeRegions(viewportRegions, { mergeGap: 4 });
  const coverageRatio = computeCoverageRatio(mergedRegions, width, height);
  const maxAreaRatio = Math.max(
    0,
    ...mergedRegions.map((rect) => rectArea(rect) / Math.max(1, pageArea)),
  );
  const nativeCharCount = Number(hint?.stats?.charCount) || 0;

  if (mode === "auto" && maxAreaRatio >= 0.92 && nativeCharCount < 200) {
    return {
      rects: [],
      meta: {
        applied: false,
        mode,
        reason: "likely_scanned_page",
        coverageRatio,
        nativeCharCount,
      },
    };
  }

  const minAreaRatio = 0.002;
  const candidates = mergedRegions
    .map((rect) => ({
      rect,
      areaRatio: rectArea(rect) / Math.max(1, pageArea),
    }))
    .filter((entry) => entry.areaRatio >= minAreaRatio)
    .sort((a, b) => b.areaRatio - a.areaRatio)
    .slice(0, options.maxMaskRegions);

  if (!candidates.length) {
    return {
      rects: [],
      meta: {
        applied: false,
        mode,
        reason: "regions_too_small",
        coverageRatio,
      },
    };
  }

  const sampleSize = 64;
  const sampleCanvas = new OffscreenCanvas(sampleSize, sampleSize);
  const sampleCtx = sampleCanvas.getContext("2d", { alpha: false, willReadFrequently: true });

  const rects = [];
  let maskedPhotoLike = 0;
  let keptTextLike = 0;
  let keptUncertain = 0;

  for (const entry of candidates) {
    const rect = entry.rect;
    if (!sampleCtx) {
      rects.push(rect);
      maskedPhotoLike += 1;
      continue;
    }

    try {
      sampleCtx.fillStyle = "#ffffff";
      sampleCtx.fillRect(0, 0, sampleSize, sampleSize);
      sampleCtx.drawImage(
        canvas,
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        0,
        0,
        sampleSize,
        sampleSize,
      );
      const metrics = computeRegionSampleMetrics(sampleCtx.getImageData(0, 0, sampleSize, sampleSize));
      const decision = shouldMaskRegion(metrics, entry.areaRatio);
      if (decision.mask) {
        rects.push(rect);
        maskedPhotoLike += 1;
      } else if (decision.reason === "text_like" || decision.reason === "full_page_image") {
        keptTextLike += 1;
      } else {
        keptUncertain += 1;
        if (mode === "on") {
          rects.push(rect);
          maskedPhotoLike += 1;
        }
      }
    } catch {
      if (mode === "on") {
        rects.push(rect);
        maskedPhotoLike += 1;
      } else {
        keptUncertain += 1;
      }
    }
  }

  return {
    rects,
    meta: {
      applied: rects.length > 0,
      mode,
      reason: rects.length ? "masked_image_regions" : "no_photo_like_regions",
      coverageRatio,
      nativeCharCount,
      candidateCount: candidates.length,
      maskedCount: rects.length,
      maskedPhotoLike,
      keptTextLike,
      keptUncertain,
    },
  };
}

async function renderPageImageForOcr(page, pageIndex, requestedScale, hint, options) {
  if (typeof OffscreenCanvas === "undefined") {
    throw new Error("OffscreenCanvas is unavailable in this browser; OCR rendering cannot run in worker.");
  }

  const safeScale = getViewportScale(page, requestedScale);
  const viewport = page.getViewport({ scale: safeScale });
  const baseViewport = page.getViewport({ scale: 1 });
  const width = Math.max(1, Math.ceil(viewport.width));
  const height = Math.max(1, Math.ceil(viewport.height));

  const canvas = new OffscreenCanvas(width, height);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!context) {
    throw new Error("Unable to acquire 2D context for OCR rendering.");
  }

  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);

  await page.render({
    canvasContext: context,
    viewport,
    canvasFactory: OFFSCREEN_CANVAS_FACTORY,
  }).promise;

  const viewportHintRegions = buildViewportHintRegions(viewport, width, height, hint, options);
  const heuristicFigureRegions = detectFigureRegionsFromCanvas(canvas);
  const diagramSkip = computeProvationDiagramSkipRegions(
    {
      canvasWidth: width,
      canvasHeight: height,
      viewportWidth: baseViewport.width * safeScale,
      viewportHeight: baseViewport.height * safeScale,
      devicePixelRatio: 1,
      figureRegions: [
        ...viewportHintRegions.imageRegions,
        ...heuristicFigureRegions,
      ],
      textRegions: viewportHintRegions.textRegions,
      nativeCharCount: Number(hint?.stats?.charCount) || 0,
      pageIndex,
    },
  );
  const figureRegions = mergeRegions([
    ...viewportHintRegions.imageRegions,
    ...heuristicFigureRegions,
    ...(Array.isArray(diagramSkip.regions) ? diagramSkip.regions : []),
  ], { mergeGap: 8 });

  const masking = selectMaskRects(canvas, hint, options, viewportHintRegions);
  if (masking.rects.length) {
    context.fillStyle = "#ffffff";
    for (const rect of masking.rects) {
      context.fillRect(rect.x, rect.y, rect.width, rect.height);
    }
  }
  if (diagramSkip.regions?.length) {
    context.fillStyle = "#ffffff";
    for (const rect of diagramSkip.regions) {
      context.fillRect(rect.x, rect.y, rect.width, rect.height);
    }
  }

  const crop = computeOcrCropRect(
    {
      canvasWidth: width,
      canvasHeight: height,
      viewportWidth: baseViewport.width * safeScale,
      viewportHeight: baseViewport.height * safeScale,
      devicePixelRatio: 1,
      textRegions: viewportHintRegions.textRegions,
      imageRegions: viewportHintRegions.imageRegions,
      nativeCharCount: Number(hint?.stats?.charCount) || 0,
    },
    {
      mode: options.cropMode,
      paddingPx: options.cropPaddingPx,
    },
  );
  const appliedCropRect = crop.rect && crop.meta?.applied
    ? normalizeRect(crop.rect)
    : null;
  const workingRect = appliedCropRect
    ? normalizeRect(appliedCropRect)
    : normalizeRect({ x: 0, y: 0, width, height });
  const headerLayout = computeHeaderZoneColumns(
    {
      canvasWidth: width,
      canvasHeight: height,
      textRegions: viewportHintRegions.textRegions,
      workingRect,
    },
    {
      topFraction: options.headerBandFrac,
    },
  );
  const headerRect = normalizeRect(headerLayout.headerZoneRect);
  const headerZones = (Array.isArray(headerLayout.columns) && headerLayout.columns.length
    ? headerLayout.columns
    : [{ id: "header_full", order: 0, rect: headerRect }])
    .map((zone) => ({
      ...zone,
      rect: normalizeRect(zone.rect),
    }))
    .sort((a, b) => (Number(a.order) || 0) - (Number(b.order) || 0))
    .map((zone) => ({
      ...zone,
      canvas: cropCanvas(canvas, zone.rect) || cropCanvas(canvas, headerRect) || canvas,
    }));

  const bodyRect = normalizeRect(headerLayout.bodyRect);
  const bodyCanvas = cropCanvas(canvas, bodyRect) || cropCanvas(canvas, workingRect);

  return {
    pageCanvas: canvas,
    headerZones,
    bodyCanvas: bodyCanvas || canvas,
    width,
    height,
    scale: safeScale,
    masking: {
      ...masking.meta,
      chosenMode: masking.meta?.mode || options.maskImages,
      selectedRegionCount: masking.rects.length,
    },
    crop: crop.meta,
    figureRegions,
    figureCoverageRatio: computeCoverageRatio(figureRegions, width, height),
    diagramSkip: diagramSkip.meta,
    headerRect,
    headerLayout: headerLayout.meta,
    bodyRect,
    workingRect,
    lineBandRegions: Array.isArray(viewportHintRegions.lineBandRegions)
      ? viewportHintRegions.lineBandRegions
      : [],
    lineBandMeta: hint?.lineBandMeta && typeof hint.lineBandMeta === "object"
      ? hint.lineBandMeta
      : { applied: false, reason: "not_provided" },
    ocrMode: hint?.ocrMode === "backfill" ? "backfill" : "full",
    preMaskFigureRegionCount: figureRegions.length,
    nativeCharCount: Number(hint?.stats?.charCount) || 0,
  };
}

function summarizeDropReasons(droppedLines) {
  const out = {};
  for (const dropped of Array.isArray(droppedLines) ? droppedLines : []) {
    const reason = String(dropped?.reason || "unknown");
    out[reason] = (out[reason] || 0) + 1;
  }
  return out;
}

function chooseFigureFilterMode(renderInfo, options) {
  const figureCoverageRatio = Number(renderInfo?.figureCoverageRatio) || 0;
  const nativeCharCount = Number(renderInfo?.nativeCharCount) || 0;
  const maskReason = String(renderInfo?.masking?.reason || "");
  const scannedLike = maskReason === "likely_scanned_page" ||
    (figureCoverageRatio >= 0.74 && nativeCharCount <= 260);

  return {
    disableFigureOverlap: scannedLike,
    reason: scannedLike ? "scanned_like_page" : "normal",
  };
}

function buildHeaderPsmAttempts(options) {
  const configured = Array.isArray(options?.headerRetryPsms) && options.headerRetryPsms.length
    ? options.headerRetryPsms
    : HEADER_RETRY_PSMS;
  const basePsm = String(options?.psm || configured[0] || "6");
  const psms = [...new Set([basePsm, ...configured.map((value) => String(value || "").trim()).filter(Boolean)])];
  const attempts = psms.map((psm, index) => ({
    psm,
    scale: index === 0 ? Math.max(1, Number(options?.headerScaleBoost) || 1.8) : Math.max(1.8, Number(options?.headerScaleBoost) || 1.8),
  }));
  attempts.push({
    psm: basePsm,
    scale: Math.max(2.2, Number(options?.headerScaleBoost) || 1.8),
  });
  return attempts;
}

async function ocrHeaderThenBody(worker, renderInfo, pageIndex, options, hint = null) {
  const backfillMode = hint?.ocrMode === "backfill" &&
    Array.isArray(renderInfo?.lineBandRegions) &&
    renderInfo.lineBandRegions.length > 0;
  const headerAttempts = buildHeaderPsmAttempts(options);
  const headerZones = (Array.isArray(renderInfo?.headerZones) && renderInfo.headerZones.length
    ? renderInfo.headerZones
    : [{
      id: "header_full",
      order: 0,
      rect: renderInfo?.headerRect || { x: 0, y: 0, width: renderInfo?.width || 0, height: renderInfo?.height || 0 },
      canvas: renderInfo?.pageCanvas || null,
    }])
    .sort((a, b) => (Number(a?.order) || 0) - (Number(b?.order) || 0));

  let attemptsUsed = 0;
  const headerZoneSummaries = [];
  const headerLines = [];
  const backfillBandSummaries = [];
  let bodyRecognized = null;
  let bodyLines = [];

  if (backfillMode) {
    const bandRegions = [...renderInfo.lineBandRegions]
      .map((region) => normalizeRect(region))
      .filter((region) => region.width > 1 && region.height > 1)
      .sort((a, b) => {
        const dy = b.y - a.y;
        if (Math.abs(dy) > 2) return dy;
        return a.x - b.x;
      });

    for (let i = 0; i < bandRegions.length; i += 1) {
      const bandRect = bandRegions[i];
      const bandCanvas = cropCanvas(renderInfo.pageCanvas, bandRect);
      if (!bandCanvas) continue;
      const scaled = scaleCanvas(bandCanvas, options.backfillScaleBoost);
      const preprocessed = preprocessBackfillCanvas(scaled.canvas, options);
      const ocrCanvas = preprocessed.canvas || scaled.canvas;
      bodyRecognized = await recognizeCanvas(worker, ocrCanvas, options.psm);
      const lines = extractStructuredOcrLines(bodyRecognized?.data, {
        pageIndex,
        offsetX: bandRect.x,
        offsetY: bandRect.y,
        scaleFactor: scaled.scaleFactor,
      }).map((line) => ({
        ...line,
        zoneId: `backfill_band_${i + 1}`,
        zoneOrder: i,
      }));
      bodyLines.push(...lines);
      backfillBandSummaries.push({
        id: `backfill_band_${i + 1}`,
        order: i,
        charCount: composeOcrPageText(lines).length,
        box: [
          Math.floor(bandRect.x),
          Math.floor(bandRect.y),
          Math.floor(bandRect.x + bandRect.width),
          Math.floor(bandRect.y + bandRect.height),
        ],
        scaleBoost: scaled.scaleFactor,
        preprocess: {
          applied: Boolean(preprocessed.applied),
          reason: String(preprocessed.reason || ""),
          threshold: Number.isFinite(preprocessed.threshold) ? Number(preprocessed.threshold) : null,
          thresholdBias: Number.isFinite(preprocessed.thresholdBias) ? Number(preprocessed.thresholdBias) : null,
          dynamicRange: Number.isFinite(preprocessed.dynamicRange) ? Number(preprocessed.dynamicRange) : null,
          dilated: Boolean(preprocessed.dilated),
        },
      });
    }
  } else {
    for (const zone of headerZones) {
      const zoneRect = normalizeRect(zone?.rect || renderInfo?.headerRect || { x: 0, y: 0, width: 0, height: 0 });
      const zoneCanvas = zone?.canvas || cropCanvas(renderInfo.pageCanvas, zoneRect) || renderInfo.pageCanvas;
      let attemptsForZone = 0;
      let bestHeader = {
        lines: [],
        text: "",
        score: Number.NEGATIVE_INFINITY,
        retryNeeded: true,
        psm: String(options.psm || "6"),
        scale: 1,
      };

      for (const attempt of headerAttempts) {
        attemptsUsed += 1;
        attemptsForZone += 1;
        const scaled = scaleCanvas(zoneCanvas, attempt.scale);
        const recognized = await recognizeCanvas(worker, scaled.canvas, attempt.psm);
        const lines = extractStructuredOcrLines(recognized?.data, {
          pageIndex,
          offsetX: zoneRect.x,
          offsetY: zoneRect.y,
          scaleFactor: scaled.scaleFactor,
        }).map((line) => ({
          ...line,
          zoneId: String(zone?.id || "header"),
          zoneOrder: Number.isFinite(zone?.order) ? Number(zone.order) : 0,
        }));
        const text = composeOcrPageText(lines);
        const quality = evaluateHeaderFieldQuality(text);
        const metrics = computeOcrTextMetrics({ text, lines });
        const score = (quality.dobValid ? 2 : 0) +
          (quality.ageValid ? 1 : 0) +
          (Number.isFinite(metrics.meanLineConf) ? metrics.meanLineConf / 100 : 0) +
          Math.min(0.9, metrics.charCount / 1400);

        if (score > bestHeader.score) {
          bestHeader = {
            lines,
            text,
            score,
            retryNeeded: quality.retryNeeded,
            psm: attempt.psm,
            scale: attempt.scale,
          };
        }
        const confidentEnough = metrics.charCount >= 28 &&
          Number.isFinite(metrics.meanLineConf) &&
          metrics.meanLineConf >= 62;
        if ((!quality.retryNeeded && metrics.charCount >= 16) || confidentEnough) break;
      }

      headerZoneSummaries.push({
        id: String(zone?.id || "header"),
        order: Number.isFinite(zone?.order) ? Number(zone.order) : 0,
        psmUsed: bestHeader.psm,
        scaleUsed: bestHeader.scale,
        retries: attemptsForZone,
        retryNeeded: bestHeader.retryNeeded,
        charCount: bestHeader.text.length,
      });
      headerLines.push(...bestHeader.lines);
    }

    bodyRecognized = await recognizeCanvas(worker, renderInfo.bodyCanvas, options.psm);
    bodyLines = extractStructuredOcrLines(bodyRecognized?.data, {
      pageIndex,
      offsetX: renderInfo.bodyRect.x,
      offsetY: renderInfo.bodyRect.y,
      scaleFactor: 1,
    }).map((line) => ({
      ...line,
      zoneId: "body",
      zoneOrder: headerZones.length,
    }));
  }

  const rawLines = dedupeConsecutiveLines([
    ...headerLines,
    ...bodyLines,
  ]);
  const preFilterText = composeOcrPageText(rawLines);
  const preFilterMetrics = computeOcrTextMetrics({ text: preFilterText, lines: rawLines });

  const filterMode = chooseFigureFilterMode(renderInfo, options);
  let filtered = filterOcrLinesDetailed(rawLines, renderInfo.figureRegions, {
    overlapThreshold: options.figureOverlapThreshold,
    shortLowConfThreshold: options.shortLowConfThreshold,
    dropCaptions: true,
    dropBoilerplate: true,
    disableFigureOverlap: filterMode.disableFigureOverlap,
  });
  let filteredLines = dedupeConsecutiveLines(filtered.lines);
  let filteredText = composeOcrPageText(filteredLines);
  let postFilterMetrics = computeOcrTextMetrics({ text: filteredText, lines: filteredLines });

  // Safety fallback: if figure suppression erased OCR output, retry without overlap suppression.
  const needsFallback = !filterMode.disableFigureOverlap &&
    postFilterMetrics.charCount < 32 &&
    preFilterMetrics.charCount >= 140;
  if (needsFallback) {
    const relaxed = filterOcrLinesDetailed(rawLines, [], {
      overlapThreshold: options.figureOverlapThreshold,
      shortLowConfThreshold: options.shortLowConfThreshold,
      dropCaptions: true,
      dropBoilerplate: true,
      disableFigureOverlap: true,
    });
    const relaxedLines = dedupeConsecutiveLines(relaxed.lines);
    const relaxedText = composeOcrPageText(relaxedLines);
    const relaxedMetrics = computeOcrTextMetrics({ text: relaxedText, lines: relaxedLines });
    if (relaxedMetrics.charCount > postFilterMetrics.charCount * 1.8) {
      filtered = relaxed;
      filteredLines = relaxedLines;
      filteredText = relaxedText;
      postFilterMetrics = relaxedMetrics;
      filterMode.reason = "fallback_disable_overlap";
      filterMode.disableFigureOverlap = true;
    }
  }

  const meanLineConf = Number.isFinite(postFilterMetrics.meanLineConf)
    ? postFilterMetrics.meanLineConf
      : Number.isFinite(bodyRecognized?.data?.confidence)
        ? Number(bodyRecognized.data.confidence)
        : null;
  const firstHeaderZone = headerZoneSummaries[0] || {};

  return {
    text: filteredText,
    lines: filteredLines,
    rawLines,
    droppedLines: filtered.dropped,
    confidence: meanLineConf,
    header: {
      psmUsed: String(firstHeaderZone.psmUsed || options.psm || "6"),
      scaleUsed: Number.isFinite(firstHeaderZone.scaleUsed) ? Number(firstHeaderZone.scaleUsed) : 1,
      retries: attemptsUsed,
      retryNeeded: headerZoneSummaries.some((zone) => Boolean(zone.retryNeeded)),
      zoneCount: headerZoneSummaries.length,
      zones: headerZoneSummaries,
    },
    backfill: {
      enabled: backfillMode,
      bandCount: backfillBandSummaries.length,
      bands: backfillBandSummaries,
      lineBandMeta: renderInfo?.lineBandMeta || { applied: false, reason: "not_provided" },
    },
    metrics: {
      preMask: preFilterMetrics,
      preFilter: preFilterMetrics,
      postOcr: preFilterMetrics,
      postFilter: postFilterMetrics,
    },
    filterMode,
  };
}

async function runOcrForPages(pdfBytes, pageIndexes, pageHints, options, jobId) {
  const loadingTask = pdfjs.getDocument({
    data: pdfBytes,
    isEvalSupported: false,
    useWorkerFetch: false,
    // Worker-safe rendering: avoid DOM factories and font-face paths that call `document.createElement`.
    CanvasFactory: WorkerCanvasFactory,
    FilterFactory: WorkerFilterFactory,
    disableFontFace: true,
  });
  const doc = await loadingTask.promise;

  const hintMap = new Map();
  for (const hint of Array.isArray(pageHints) ? pageHints : []) {
    const pageIndex = Number(hint?.pageIndex);
    if (!Number.isFinite(pageIndex)) continue;
    hintMap.set(pageIndex, hint);
  }

  const uniquePageIndexes = [...new Set(
    (Array.isArray(pageIndexes) ? pageIndexes : [])
      .map((value) => Number(value))
      .filter((value) => Number.isFinite(value) && value >= 0 && value < doc.numPages),
  )].sort((a, b) => a - b);

  const totalPages = uniquePageIndexes.length;
  const results = [];

  if (!totalPages) {
    await doc.destroy();
    return results;
  }

  const worker = await getTesseractWorker(options, 0, totalPages);

  for (let i = 0; i < uniquePageIndexes.length; i += 1) {
    if (jobId !== activeJobId) {
      break;
    }

    const pageIndex = uniquePageIndexes[i];
    const page = await doc.getPage(pageIndex + 1);

    self.postMessage({
      type: "ocr_stage",
      stage: "ocr_rendering",
      pageIndex: i,
      totalPages,
      sourcePageIndex: pageIndex,
    });

    const hint = hintMap.get(pageIndex) || null;
    const render = await renderPageImageForOcr(page, pageIndex, options.scale, hint, options);

    self.postMessage({
      type: "ocr_stage",
      stage: "ocr_recognizing",
      pageIndex: i,
      totalPages,
      sourcePageIndex: pageIndex,
    });

    const startedAt = Date.now();
    const structured = await ocrHeaderThenBody(worker, render, pageIndex, options, hint);
    const durationMs = Date.now() - startedAt;
    const text = structured.text;
    const confidence = structured.confidence;

    const pageResult = {
      pageIndex,
      text,
      meta: {
        confidence,
        source: "ocr",
        width: render.width,
        height: render.height,
        scale: render.scale,
        durationMs,
        masking: render.masking,
        crop: render.crop,
        header: structured.header,
        headerLayout: render.headerLayout,
        backfill: structured.backfill,
        diagramSkip: render.diagramSkip,
        headerColumns: (Array.isArray(render.headerZones) ? render.headerZones : []).map((zone) => ({
          id: zone.id,
          order: zone.order,
          rect: zone.rect,
        })),
        filterMode: structured.filterMode,
        metrics: structured.metrics,
        lines: structured.lines,
        droppedLineSummary: summarizeDropReasons(structured.droppedLines),
        droppedLines: (Array.isArray(structured.droppedLines) ? structured.droppedLines : [])
          .slice(0, 64)
          .map((entry) => ({
            reason: entry.reason,
            text: String(entry.line?.text || "").slice(0, 220),
            confidence: Number.isFinite(entry.line?.confidence) ? Number(entry.line.confidence) : null,
            overlapRatio: Number.isFinite(entry.overlapRatio) ? Number(entry.overlapRatio) : null,
          })),
        figureRegions: render.figureRegions,
      },
    };

    results.push(pageResult);

    self.postMessage({
      type: "ocr_page",
      page: pageResult,
    });
    self.postMessage({
      type: "ocr_progress",
      completedPages: i + 1,
      totalPages,
    });
  }

  await doc.destroy();
  return results;
}

self.onmessage = async (event) => {
  const data = event.data || {};
  if (data.type !== "ocr_extract") return;

  const jobId = activeJobId + 1;
  activeJobId = jobId;

  try {
    const options = resolveOcrOptions(data.options || {});
    const pages = await runOcrForPages(
      data.pdfBytes,
      data.pageIndexes,
      data.pageHints,
      options,
      jobId,
    );

    if (jobId !== activeJobId) return;
    self.postMessage({ type: "ocr_done", pages });
  } catch (error) {
    if (jobId !== activeJobId) return;
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ type: "error", error: message });
  }
};
