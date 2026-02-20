import { arbitratePageText } from "./fusion.js";
import { clamp01, mergeRegions, rectArea } from "./layoutAnalysis.js";
import { computeOcrTextMetrics } from "./ocrMetrics.js";
import { getLineBandRegions } from "./ocrRegions.js";
import { classifyPage, isUnsafeNativePage, resolvePageSource } from "./pageClassifier.js";

const DEFAULT_GATE_OPTIONS = Object.freeze({
  minCompletenessConfidence: 0.72,
  maxContaminationScore: 0.24,
  hardBlockWhenUnsafeWithoutOcr: true,
});

const DEFAULT_OCR_OPTIONS = Object.freeze({
  available: true,
  enabled: true,
  lang: "eng",
  qualityMode: "fast",
  scale: 2.05,
  psm: "6",
  maskImages: "auto",
  cropMode: "auto",
  cropPaddingPx: 14,
  headerBandFrac: 0.25,
  headerScaleBoost: 1.8,
  headerRetryPsms: ["6", "4", "11"],
  figureOverlapThreshold: 0.35,
  shortLowConfThreshold: 30,
  backfillScaleBoost: 1.7,
  backfillBandPaddingPx: 14,
  backfillBandPaddingRatio: 0.04,
  backfillBandMinWidthRatio: 0.6,
  backfillBandMaxRegions: 3,
  backfillPreprocess: true,
  backfillThresholdBias: -8,
  backfillDilate: true,
});

function normalizeGateOptions(gate) {
  const merged = {
    ...DEFAULT_GATE_OPTIONS,
    ...(gate && typeof gate === "object" ? gate : {}),
  };

  return {
    minCompletenessConfidence: clamp01(Number(merged.minCompletenessConfidence)),
    maxContaminationScore: clamp01(Number(merged.maxContaminationScore)),
    hardBlockWhenUnsafeWithoutOcr: Boolean(merged.hardBlockWhenUnsafeWithoutOcr),
  };
}

function normalizeOcrOptions(opts = {}) {
  const ocr = opts.ocr && typeof opts.ocr === "object" ? opts.ocr : {};
  const qualityMode = ocr.qualityMode === "high_accuracy" ? "high_accuracy" : "fast";
  const defaultScale = qualityMode === "high_accuracy" ? 3.1 : 2.05;
  const scale = Number.isFinite(ocr.scale) ? Math.max(1.1, Math.min(4, Number(ocr.scale))) : defaultScale;
  const maskImages = ocr.maskImages === "off" || ocr.maskImages === "none"
    ? "off"
    : ocr.maskImages === "on"
      ? "on"
      : "auto";
  const cropMode = ocr.cropMode === "off" || ocr.cropMode === false
    ? "off"
    : ocr.cropMode === "on" || ocr.cropMode === true
      ? "on"
      : "auto";
  const cropPaddingPx = Number.isFinite(ocr.cropPaddingPx)
    ? Math.max(0, Math.min(120, Number(ocr.cropPaddingPx)))
    : DEFAULT_OCR_OPTIONS.cropPaddingPx;
  const headerBandFrac = Number.isFinite(ocr.headerBandFrac)
    ? Math.max(0.2, Math.min(0.35, Number(ocr.headerBandFrac)))
    : DEFAULT_OCR_OPTIONS.headerBandFrac;
  const headerScaleBoost = Number.isFinite(ocr.headerScaleBoost)
    ? Math.max(1, Math.min(3, Number(ocr.headerScaleBoost)))
    : DEFAULT_OCR_OPTIONS.headerScaleBoost;
  const rawHeaderRetryPsms = Array.isArray(ocr.headerRetryPsms)
    ? ocr.headerRetryPsms
    : DEFAULT_OCR_OPTIONS.headerRetryPsms;
  const headerRetryPsms = [...new Set(rawHeaderRetryPsms
    .map((value) => String(value || "").trim())
    .filter(Boolean))]
    .slice(0, 6);
  const figureOverlapThreshold = Number.isFinite(ocr.figureOverlapThreshold)
    ? Math.max(0, Math.min(1, Number(ocr.figureOverlapThreshold)))
    : DEFAULT_OCR_OPTIONS.figureOverlapThreshold;
  const shortLowConfThreshold = Number.isFinite(ocr.shortLowConfThreshold)
    ? Math.max(0, Math.min(100, Number(ocr.shortLowConfThreshold)))
    : DEFAULT_OCR_OPTIONS.shortLowConfThreshold;
  const backfillScaleBoost = Number.isFinite(ocr.backfillScaleBoost)
    ? Math.max(1, Math.min(2.4, Number(ocr.backfillScaleBoost)))
    : DEFAULT_OCR_OPTIONS.backfillScaleBoost;
  const backfillBandPaddingPx = Number.isFinite(ocr.backfillBandPaddingPx)
    ? Math.max(0, Math.min(80, Number(ocr.backfillBandPaddingPx)))
    : DEFAULT_OCR_OPTIONS.backfillBandPaddingPx;
  const backfillBandPaddingRatio = Number.isFinite(ocr.backfillBandPaddingRatio)
    ? Math.max(0, Math.min(0.2, Number(ocr.backfillBandPaddingRatio)))
    : DEFAULT_OCR_OPTIONS.backfillBandPaddingRatio;
  const backfillBandMinWidthRatio = Number.isFinite(ocr.backfillBandMinWidthRatio)
    ? Math.max(0.2, Math.min(1, Number(ocr.backfillBandMinWidthRatio)))
    : DEFAULT_OCR_OPTIONS.backfillBandMinWidthRatio;
  const backfillBandMaxRegions = Number.isFinite(ocr.backfillBandMaxRegions)
    ? Math.max(1, Math.min(16, Math.floor(Number(ocr.backfillBandMaxRegions))))
    : DEFAULT_OCR_OPTIONS.backfillBandMaxRegions;
  const backfillThresholdBias = Number.isFinite(ocr.backfillThresholdBias)
    ? Math.max(-40, Math.min(40, Number(ocr.backfillThresholdBias)))
    : DEFAULT_OCR_OPTIONS.backfillThresholdBias;

  return {
    ...DEFAULT_OCR_OPTIONS,
    ...ocr,
    available: ocr.available !== false,
    enabled: ocr.enabled !== false,
    lang: typeof ocr.lang === "string" && ocr.lang.trim() ? ocr.lang.trim() : DEFAULT_OCR_OPTIONS.lang,
    qualityMode,
    scale,
    psm: String(ocr.psm || DEFAULT_OCR_OPTIONS.psm),
    maskImages,
    cropMode,
    cropPaddingPx,
    headerBandFrac,
    headerScaleBoost,
    headerRetryPsms: headerRetryPsms.length ? headerRetryPsms : [...DEFAULT_OCR_OPTIONS.headerRetryPsms],
    figureOverlapThreshold,
    shortLowConfThreshold,
    backfillScaleBoost,
    backfillBandPaddingPx,
    backfillBandPaddingRatio,
    backfillBandMinWidthRatio,
    backfillBandMaxRegions,
    backfillPreprocess: ocr.backfillPreprocess !== false,
    backfillThresholdBias,
    backfillDilate: ocr.backfillDilate !== false,
    workerUrl: ocr.workerUrl,
  };
}

function isFileLike(value) {
  return typeof File !== "undefined" && value instanceof File;
}

function sourceHeaderLabel(page) {
  if (page.sourceDecision === "hybrid") return "HYBRID";
  if (page.sourceDecision === "ocr" && !page.ocrText) return "OCR_REQUIRED";
  return page.sourceDecision.toUpperCase();
}

function getProcessingMode(page) {
  const qualityFlags = new Set(Array.isArray(page?.classification?.qualityFlags) ? page.classification.qualityFlags : []);
  if (qualityFlags.has("NATIVE_DENSE_TEXT") && page?.sourceDecision === "native") {
    return "native_dense_bypass";
  }
  if (page?.classification?.needsOcrBackfill) return "backfill_roi";
  if (page?.sourceDecision === "ocr" || page?.sourceDecision === "hybrid") return "full_ocr";
  return "native_only";
}

function boxAreaFromArray(box) {
  if (!Array.isArray(box) || box.length < 4) return 0;
  const x0 = Number(box[0]);
  const y0 = Number(box[1]);
  const x1 = Number(box[2]);
  const y1 = Number(box[3]);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return 0;
  return Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
}

function rectFromBox(box) {
  if (!Array.isArray(box) || box.length < 4) return null;
  const x0 = Number(box[0]);
  const y0 = Number(box[1]);
  const x1 = Number(box[2]);
  const y1 = Number(box[3]);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;
  const width = Math.max(0, x1 - x0);
  const height = Math.max(0, y1 - y0);
  if (width <= 0 || height <= 0) return null;
  return { x: x0, y: y0, width, height };
}

function resolveOcrAreaToNativeFactor(page, ocrMeta) {
  const pageArea = Math.max(1, Number(page?.stats?.pageArea) || 0);
  const renderWidth = Number(ocrMeta?.width) || 0;
  const renderHeight = Number(ocrMeta?.height) || 0;
  const renderArea = renderWidth > 0 && renderHeight > 0 ? renderWidth * renderHeight : 0;
  if (pageArea > 0 && renderArea > 0) {
    return Math.max(0.01, Math.min(1, pageArea / renderArea));
  }

  const renderScale = Number(ocrMeta?.scale) || 0;
  if (renderScale > 0) {
    return 1 / Math.max(1, renderScale * renderScale);
  }

  return 1;
}

function computeOcrRoiMetrics(page, sourceDecision) {
  const ocrMeta = page?.ocrMeta && typeof page.ocrMeta === "object" ? page.ocrMeta : null;
  const pageArea = Math.max(1, Number(page?.stats?.pageArea) || 0);
  const areaToNativeFactor = resolveOcrAreaToNativeFactor(page, ocrMeta);
  const backfillBands = Array.isArray(ocrMeta?.backfill?.bands) ? ocrMeta.backfill.bands : [];
  if (Boolean(ocrMeta?.backfill?.enabled) && backfillBands.length) {
    const mergedRects = mergeRegions(
      backfillBands
        .map((band) => rectFromBox(band?.box))
        .filter(Boolean),
      { mergeGap: 2 },
    );
    const roiAreaPx = mergedRects.reduce((sum, rect) => sum + rectArea(rect), 0) * areaToNativeFactor;
    return {
      roiKind: "backfill_bands",
      roiCount: backfillBands.length,
      roiAreaPx,
      roiAreaRatio: pageArea > 0 ? clamp01(roiAreaPx / pageArea) : 0,
    };
  }

  const cropBox = Array.isArray(ocrMeta?.crop?.box) ? ocrMeta.crop.box : null;
  if (ocrMeta?.crop?.applied && cropBox) {
    const roiAreaPx = boxAreaFromArray(cropBox) * areaToNativeFactor;
    return {
      roiKind: "crop_rect",
      roiCount: roiAreaPx > 0 ? 1 : 0,
      roiAreaPx,
      roiAreaRatio: pageArea > 0 ? clamp01(roiAreaPx / pageArea) : 0,
    };
  }

  if (sourceDecision === "ocr" || sourceDecision === "hybrid") {
    return {
      roiKind: "full_page",
      roiCount: 1,
      roiAreaPx: pageArea,
      roiAreaRatio: 1,
    };
  }

  return {
    roiKind: "none",
    roiCount: 0,
    roiAreaPx: 0,
    roiAreaRatio: 0,
  };
}

function normalizeMetricValue(value, fallback = null) {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function looksLowConfidenceByFinalMetrics(metrics) {
  if (!metrics || typeof metrics !== "object") return true;
  const charCount = Number(metrics.charCount) || 0;
  const alphaRatio = Number(metrics.alphaRatio) || 0;
  const meanLineConf = Number.isFinite(metrics.meanLineConf) ? Number(metrics.meanLineConf) : null;
  const lowConfLineFrac = Number.isFinite(metrics.lowConfLineFrac) ? Number(metrics.lowConfLineFrac) : null;

  if (charCount < 80) return true;
  if (alphaRatio < 0.38) return true;
  if (Number.isFinite(meanLineConf) && meanLineConf < 38) return true;
  if (Number.isFinite(lowConfLineFrac) && lowConfLineFrac > 0.62) return true;
  return false;
}

function buildPageQualityMetrics(pageText, rawPage, sourceDecision) {
  const safeText = typeof pageText === "string" ? pageText : "";
  const ocrMeta = rawPage?.ocrMeta && typeof rawPage.ocrMeta === "object"
    ? rawPage.ocrMeta
    : null;
  const ocrLines = Array.isArray(ocrMeta?.lines) ? ocrMeta.lines : [];
  const stageMetrics = ocrMeta?.metrics && typeof ocrMeta.metrics === "object"
    ? ocrMeta.metrics
    : {};

  const finalMetrics = computeOcrTextMetrics({ text: safeText });
  const postFilter = stageMetrics.postFilter && typeof stageMetrics.postFilter === "object"
    ? stageMetrics.postFilter
    : null;

  if (sourceDecision !== "native" && postFilter) {
    finalMetrics.meanLineConf = normalizeMetricValue(postFilter.meanLineConf, finalMetrics.meanLineConf);
    finalMetrics.lowConfLineFrac = normalizeMetricValue(postFilter.lowConfLineFrac, finalMetrics.lowConfLineFrac);
  }

  return {
    ...finalMetrics,
    lowConfidence: looksLowConfidenceByFinalMetrics(finalMetrics),
    stages: {
      preMask: stageMetrics.preMask,
      preFilter: stageMetrics.preFilter,
      postOcr: stageMetrics.postOcr,
      postFilter: stageMetrics.postFilter,
      postHybrid: finalMetrics,
    },
  };
}

function pageUnsafeReason(page, gateOptions) {
  if (!page) return "";

  const reasons = [];
  if (page.classification?.needsOcr) {
    reasons.push(page.classification.reason);
  } else {
    if ((page.stats?.contaminationScore || 0) >= gateOptions.maxContaminationScore) {
      reasons.push(`contamination score ${page.stats.contaminationScore.toFixed(2)}`);
    }
    if ((page.stats?.completenessConfidence || 0) < gateOptions.minCompletenessConfidence) {
      reasons.push(`completeness confidence ${page.stats.completenessConfidence.toFixed(2)}`);
    }
  }

  return reasons.join(", ");
}

function summarizeBlockedPages(pages) {
  const blockedPages = pages.filter((page) => page.blockedReason);
  if (!blockedPages.length) return undefined;

  const details = blockedPages
    .slice(0, 3)
    .map((page) => `p${page.pageIndex + 1}: ${page.blockedReason}`);

  if (blockedPages.length > 3) {
    details.push(`+${blockedPages.length - 3} more page(s)`);
  }

  return details.join(" | ");
}

function normalizeRawPages(rawPages) {
  return (Array.isArray(rawPages) ? rawPages : [])
    .filter((page) => page && Number.isFinite(page.pageIndex))
    .sort((a, b) => a.pageIndex - b.pageIndex);
}

/**
 * Build the client-side canonical document object from extracted pages.
 *
 * @param {{name:string}} file
 * @param {Array<{pageIndex:number,text:string,rawText?:string,stats:object,userOverride?:'force_native'|'force_ocr',layoutBlocks?:Array,imageRegions?:Array,textRegions?:Array,contaminatedSpans?:Array,qualityFlags?:Array,ocrText?:string,ocrMeta?:object}>} rawPages
 * @param {{forceOcrAll?:boolean,ocr?:object,gate?:object,classifier?:object}} [opts]
 */
export function buildPdfDocumentModel(file, rawPages, opts = {}) {
  const gateOptions = normalizeGateOptions(opts.gate);
  const ocrOptions = normalizeOcrOptions(opts);
  const forceOcrAll = Boolean(opts.forceOcrAll);

  const pages = [];
  const pageStartOffsets = [];
  let fullText = "";

  let lowConfidencePages = 0;
  let contaminatedPages = 0;
  const pageMetricLines = [];

  for (const rawPage of normalizeRawPages(rawPages)) {
    const rawText = typeof rawPage.rawText === "string" ? rawPage.rawText : (rawPage.text || "");
    const nativeText = typeof rawPage.text === "string" ? rawPage.text : rawText;
    const pageStats = { ...(rawPage.stats || {}) };

    const classification = classifyPage(pageStats, nativeText, {
      thresholds: opts.classifier,
    });
    pageStats.completenessConfidence = classification.completenessConfidence;

    if ((pageStats.contaminationScore || 0) >= gateOptions.maxContaminationScore) {
      contaminatedPages += 1;
    }

    const provisionalPage = {
      classification,
      userOverride: rawPage.userOverride,
    };
    const resolvedSource = resolvePageSource(provisionalPage, { forceOcrAll });

    const fusion = arbitratePageText({
      nativeText,
      ocrText: rawPage.ocrText,
      requestedSource: resolvedSource.source,
      ocrAvailable: ocrOptions.available && ocrOptions.enabled,
      mergeMode: classification.needsOcrBackfill ? "repair_only" : "augment",
      classification,
      stats: pageStats,
    });

    let sourceDecision = fusion.sourceDecision;
    const hasAnyOcrText = typeof rawPage.ocrText === "string";
    const hasOcrText = typeof rawPage.ocrText === "string" && rawPage.ocrText.trim().length > 0;
    if (resolvedSource.source === "ocr" && !hasAnyOcrText) {
      sourceDecision = "ocr";
    }

    const unsafeEval = isUnsafeNativePage(pageStats, nativeText, {
      minCompletenessConfidence: gateOptions.minCompletenessConfidence,
      maxContaminationScore: gateOptions.maxContaminationScore,
      thresholds: opts.classifier,
    });

    let blockedReason;
    if (!ocrOptions.available && gateOptions.hardBlockWhenUnsafeWithoutOcr && unsafeEval.unsafe) {
      blockedReason = `unsafe native extraction (${pageUnsafeReason({ stats: pageStats, classification }, gateOptions)})`;
    }

    const qualityFlags = new Set([
      ...(Array.isArray(rawPage.qualityFlags) ? rawPage.qualityFlags : []),
      ...(Array.isArray(classification.qualityFlags) ? classification.qualityFlags : []),
    ]);
    if (blockedReason) qualityFlags.add("BLOCKED_UNSAFE_NATIVE");

    const pageText = sourceDecision === "ocr" && hasOcrText
      ? rawPage.ocrText
      : fusion.text || nativeText;
    const nativeMetrics = computeOcrTextMetrics({ text: nativeText });
    const qualityMetrics = buildPageQualityMetrics(pageText, rawPage, sourceDecision);
    qualityMetrics.junkScoreBeforeMerge = Number(nativeMetrics.junkScore) || 0;
    qualityMetrics.junkScoreAfterMerge = Number(qualityMetrics.junkScore) || 0;
    qualityMetrics.junkScoreDelta = qualityMetrics.junkScoreAfterMerge - qualityMetrics.junkScoreBeforeMerge;
    const backfillSignals = Object.entries(classification.backfill?.signals || {})
      .filter(([, enabled]) => Boolean(enabled))
      .map(([name]) => name);
    const roiMetrics = computeOcrRoiMetrics(
      {
        stats: pageStats,
        ocrMeta: rawPage.ocrMeta,
      },
      sourceDecision,
    );
    const processingMode = getProcessingMode({
      classification,
      sourceDecision,
      ocrText: rawPage.ocrText,
    });
    if (qualityMetrics.lowConfidence) {
      lowConfidencePages += 1;
    }
    const meanLineConf = Number.isFinite(qualityMetrics.meanLineConf)
      ? qualityMetrics.meanLineConf.toFixed(1)
      : "n/a";
    const lowConfLineFrac = Number.isFinite(qualityMetrics.lowConfLineFrac)
      ? qualityMetrics.lowConfLineFrac.toFixed(2)
      : "n/a";
    const densityText = Number(classification.nativeTextDensity || pageStats.nativeTextDensity || 0).toFixed(4);
    const roiAreaPct = Math.round((Number(roiMetrics.roiAreaRatio) || 0) * 100);
    const junkBefore = (Number(qualityMetrics.junkScoreBeforeMerge) || 0).toFixed(3);
    const junkAfter = (Number(qualityMetrics.junkScoreAfterMerge) || 0).toFixed(3);
    pageMetricLines.push(
      `p${rawPage.pageIndex + 1}: mode=${processingMode}, dens=${densityText}, backfill=${classification.backfill?.votes || 0}/${backfillSignals.length ? backfillSignals.join("+") : "none"}, needsBackfill=${Boolean(classification.needsOcrBackfill)}, roi=${roiMetrics.roiCount}/${roiAreaPct}%, junk=${junkBefore}->${junkAfter}, chars=${qualityMetrics.charCount}, alpha=${qualityMetrics.alphaRatio.toFixed(2)}, conf=${meanLineConf}, lowConf=${lowConfLineFrac}, lines=${qualityMetrics.numLines}, medTok=${qualityMetrics.medianTokenLen.toFixed(1)}, footer=${qualityMetrics.footerBoilerplateHits}`,
    );

    const page = {
      pageIndex: rawPage.pageIndex,
      text: pageText,
      rawText,
      stats: pageStats,
      classification,
      userOverride: rawPage.userOverride,
      source: sourceDecision === "native" ? "native" : "ocr",
      sourceDecision,
      sourceReason: resolvedSource.reason,
      layoutBlocks: Array.isArray(rawPage.layoutBlocks) ? rawPage.layoutBlocks : [],
      imageRegions: Array.isArray(rawPage.imageRegions) ? rawPage.imageRegions : [],
      textRegions: Array.isArray(rawPage.textRegions) ? rawPage.textRegions : [],
      contaminatedSpans: Array.isArray(rawPage.contaminatedSpans) ? rawPage.contaminatedSpans : [],
      qualityFlags: [...qualityFlags],
      blockedReason,
      ocrText: typeof rawPage.ocrText === "string" ? rawPage.ocrText : undefined,
      ocrMeta: rawPage.ocrMeta && typeof rawPage.ocrMeta === "object" ? rawPage.ocrMeta : undefined,
      qualityMetrics,
      extractionMetrics: {
        mode: processingMode,
        nativeTextDensity: Number(classification.nativeTextDensity || pageStats.nativeTextDensity || 0),
        needsOcrBackfill: Boolean(classification.needsOcrBackfill),
        backfillVotes: Number(classification.backfill?.votes) || 0,
        backfillStrongVotes: Number(classification.backfill?.strongVotes) || 0,
        backfillScore: Number(classification.backfill?.severityScore) || 0,
        backfillSignals,
        roiKind: roiMetrics.roiKind,
        ocrRoiCount: Number(roiMetrics.roiCount) || 0,
        ocrRoiAreaPx: Number(roiMetrics.roiAreaPx) || 0,
        ocrRoiAreaRatio: Number(roiMetrics.roiAreaRatio) || 0,
        junkScoreBeforeMerge: Number(qualityMetrics.junkScoreBeforeMerge) || 0,
        junkScoreAfterMerge: Number(qualityMetrics.junkScoreAfterMerge) || 0,
      },
    };

    const header = `\n===== PAGE ${page.pageIndex + 1} (${sourceHeaderLabel(page)}) =====\n`;
    pageStartOffsets.push(fullText.length + header.length);
    fullText += header;
    fullText += page.text || "";
    fullText += "\n";

    pages.push(page);
  }

  const requiresOcr = pages.some((page) =>
    page.classification?.needsOcrBackfill ||
    page.sourceDecision !== "native" ||
    page.classification?.needsOcr,
  );
  const blockReason = summarizeBlockedPages(pages);
  const blocked = Boolean(blockReason);
  const gate = {
    status: blocked ? "blocked" : "pass",
    blocked,
    reason: blockReason,
    ocrAvailable: ocrOptions.available && ocrOptions.enabled,
    requiresOcr,
    thresholds: gateOptions,
  };

  return {
    fileName: file.name,
    pages,
    fullText,
    pageStartOffsets,
    requiresOcr,
    blocked,
    blockReason,
    qualitySummary: {
      lowConfidencePages,
      contaminatedPages,
      pageMetrics: pageMetricLines,
    },
    gate,
  };
}

/**
 * Select page indices that require OCR from the current document decision state.
 *
 * @param {{pages:Array<{pageIndex:number,sourceDecision?:string,classification?:{needsOcr?:boolean}}>}|null} documentModel
 * @param {{forceOcrAll?:boolean}} [opts]
 */
export function selectPagesForOcr(documentModel, opts = {}) {
  const pages = Array.isArray(documentModel?.pages) ? documentModel.pages : [];
  if (!pages.length) return [];
  if (opts.forceOcrAll) {
    return pages.map((page) => page.pageIndex).filter((index) => Number.isFinite(index));
  }

  return pages
    .filter((page) => page.sourceDecision === "ocr" || page.classification?.needsOcrBackfill)
    .map((page) => page.pageIndex)
    .filter((index) => Number.isFinite(index));
}

export function applyOcrResultsToRawPages(rawPages, ocrPages) {
  const map = new Map();
  for (const page of normalizeRawPages(rawPages)) {
    map.set(page.pageIndex, { ...page });
  }

  for (const ocrPage of Array.isArray(ocrPages) ? ocrPages : []) {
    if (!Number.isFinite(ocrPage?.pageIndex)) continue;
    const existing = map.get(ocrPage.pageIndex);
    if (!existing) continue;
    const ocrText = typeof ocrPage.text === "string" ? ocrPage.text : "";
    map.set(ocrPage.pageIndex, {
      ...existing,
      ocrText,
      ocrMeta: ocrPage.meta && typeof ocrPage.meta === "object" ? ocrPage.meta : undefined,
    });
  }

  return [...map.values()].sort((a, b) => a.pageIndex - b.pageIndex);
}

async function runOcrPass(file, pageIndexes, opts, push) {
  if (!pageIndexes.length) return [];

  const ocrOptions = normalizeOcrOptions(opts);
  const workerUrl = ocrOptions.workerUrl || new URL("../workers/ocr.worker.js", import.meta.url);
  const worker = new Worker(workerUrl, { type: "module" });
  const pdfBytes = await file.arrayBuffer();
  const rawPages = normalizeRawPages(opts.rawPages);
  const rawPageByIndex = new Map(rawPages.map((page) => [page.pageIndex, page]));
  const decisionPages = Array.isArray(opts.decisionPages) ? opts.decisionPages : [];
  const decisionByIndex = new Map(decisionPages.map((page) => [page.pageIndex, page]));
  const pageHints = pageIndexes.map((pageIndex) => {
    const rawPage = rawPageByIndex.get(pageIndex);
    if (!rawPage) return { pageIndex };
    const decision = decisionByIndex.get(pageIndex);
    const needsBackfill = Boolean(decision?.classification?.needsOcrBackfill);

    let lineBandRegions = [];
    let lineBandMeta = { applied: false, reason: "disabled" };
    if (needsBackfill) {
      const lineBands = getLineBandRegions(
        {
          layoutBlocks: Array.isArray(rawPage.layoutBlocks) ? rawPage.layoutBlocks : [],
          canvasWidth: rawPage.stats?.pageWidth,
          canvasHeight: rawPage.stats?.pageHeight,
          viewportWidth: rawPage.stats?.pageWidth,
          viewportHeight: rawPage.stats?.pageHeight,
        },
        {
          yPaddingPx: ocrOptions.backfillBandPaddingPx,
          xPaddingRatio: ocrOptions.backfillBandPaddingRatio,
          minBandWidthRatio: ocrOptions.backfillBandMinWidthRatio,
          maxBands: ocrOptions.backfillBandMaxRegions,
        },
      );
      lineBandRegions = Array.isArray(lineBands?.regions) ? lineBands.regions : [];
      lineBandMeta = lineBands?.meta && typeof lineBands.meta === "object"
        ? lineBands.meta
        : lineBandMeta;
    }

    return {
      pageIndex,
      stats: rawPage.stats && typeof rawPage.stats === "object" ? rawPage.stats : undefined,
      imageRegions: Array.isArray(rawPage.imageRegions) ? rawPage.imageRegions : [],
      textRegions: Array.isArray(rawPage.textRegions) ? rawPage.textRegions : [],
      lineBandRegions,
      lineBandMeta,
      ocrMode: needsBackfill && lineBandRegions.length ? "backfill" : "full",
    };
  });

  return new Promise((resolve, reject) => {
    let done = false;
    const results = [];

    const finish = (error) => {
      if (done) return;
      done = true;
      worker.terminate();
      if (error) reject(error);
      else resolve(results);
    };

    worker.onmessage = (event) => {
      const data = event.data || {};

      if (data.type === "ocr_stage") {
        push({
          kind: "stage",
          stage: data.stage,
          pageIndex: Number.isFinite(data.sourcePageIndex)
            ? Number(data.sourcePageIndex)
            : Number(data.pageIndex) || 0,
          totalPages: Number(data.totalPages) || pageIndexes.length,
        });
        return;
      }
      if (data.type === "ocr_progress") {
        push({
          kind: "ocr_progress",
          completedPages: Number(data.completedPages) || 0,
          totalPages: Number(data.totalPages) || pageIndexes.length,
        });
        return;
      }
      if (data.type === "ocr_status") {
        push({
          kind: "ocr_status",
          status: data.status,
          progress: Number.isFinite(data.progress) ? data.progress : 0,
        });
        return;
      }
      if (data.type === "ocr_page") {
        const page = data.page || {};
        results.push(page);
        push({
          kind: "ocr_page",
          page,
        });
        return;
      }
      if (data.type === "ocr_done") {
        const pages = Array.isArray(data.pages) ? data.pages : [];
        if (pages.length > results.length) {
          results.length = 0;
          results.push(...pages);
        }
        finish();
        return;
      }
      if (data.type === "error") {
        finish(new Error(data.error || "OCR worker failed"));
      }
    };

    worker.onerror = (event) => {
      const error = event instanceof ErrorEvent && event.error
        ? event.error
        : new Error(event.message || "OCR worker error");
      finish(error);
    };

    worker.postMessage({
      type: "ocr_extract",
      pdfBytes,
      pageIndexes,
      pageHints,
      options: {
        lang: ocrOptions.lang,
        qualityMode: ocrOptions.qualityMode,
        scale: ocrOptions.scale,
        psm: ocrOptions.psm,
        maskImages: ocrOptions.maskImages,
        cropMode: ocrOptions.cropMode,
        cropPaddingPx: ocrOptions.cropPaddingPx,
        headerBandFrac: ocrOptions.headerBandFrac,
        headerScaleBoost: ocrOptions.headerScaleBoost,
        headerRetryPsms: ocrOptions.headerRetryPsms,
        figureOverlapThreshold: ocrOptions.figureOverlapThreshold,
        shortLowConfThreshold: ocrOptions.shortLowConfThreshold,
        backfillScaleBoost: ocrOptions.backfillScaleBoost,
        backfillPreprocess: ocrOptions.backfillPreprocess,
        backfillThresholdBias: ocrOptions.backfillThresholdBias,
        backfillDilate: ocrOptions.backfillDilate,
      },
    }, [pdfBytes]);
  });
}

async function* runWorkerExtraction(file, opts = {}, messageType = "extract_adaptive") {
  if (!isFileLike(file)) {
    throw new Error("PDF extraction expects a File");
  }

  const workerUrl = opts.workerUrl || new URL("../workers/pdf.worker.js", import.meta.url);
  const worker = new Worker(workerUrl, { type: "module" });
  const pdfBytes = await file.arrayBuffer();

  const queue = [];
  let wake = null;
  let finished = false;
  let fatalError = null;

  const push = (event) => {
    queue.push(event);
    if (wake) {
      const next = wake;
      wake = null;
      next();
    }
  };

  worker.onmessage = (event) => {
    const data = event.data || {};
    if (data.type === "stage") {
      push({
        kind: "stage",
        stage: data.stage,
        pageIndex: Number(data.pageIndex) || 0,
        totalPages: Number(data.totalPages) || 0,
      });
      return;
    }
    if (data.type === "progress") {
      push({
        kind: "progress",
        completedPages: data.completedPages,
        totalPages: data.totalPages,
      });
      return;
    }
    if (data.type === "page") {
      push({ kind: "page", page: data.page });
      return;
    }
    if (data.type === "done") {
      (async () => {
        try {
          let pages = normalizeRawPages(data.pages);
          let document = buildPdfDocumentModel(file, pages, opts);

          const ocrOptions = normalizeOcrOptions(opts);
          const shouldRunOcr = messageType === "extract_adaptive" && ocrOptions.available && ocrOptions.enabled;
          const ocrTargets = shouldRunOcr ? selectPagesForOcr(document, { forceOcrAll: opts.forceOcrAll }) : [];

          if (ocrTargets.length) {
            push({
              kind: "stage",
              stage: "ocr_prepare",
              pageIndex: 0,
              totalPages: ocrTargets.length,
            });
            try {
              const ocrPages = await runOcrPass(file, ocrTargets, {
                ...opts,
                rawPages: pages,
                decisionPages: document.pages,
              }, push);
              pages = applyOcrResultsToRawPages(pages, ocrPages);
              document = buildPdfDocumentModel(file, pages, {
                ...opts,
                ocr: {
                  ...ocrOptions,
                  available: true,
                  enabled: true,
                },
              });
            } catch (ocrError) {
              push({
                kind: "stage",
                stage: "ocr_failed",
                pageIndex: 0,
                totalPages: ocrTargets.length,
              });
              const message = ocrError instanceof Error ? ocrError.message : String(ocrError);
              push({
                kind: "ocr_error",
                error: message,
              });
              document = buildPdfDocumentModel(file, pages, {
                ...opts,
                ocr: {
                  ...ocrOptions,
                  available: false,
                  enabled: false,
                },
              });
            }
          }

          push({
            kind: "done",
            pages,
            document,
            gate: document.gate,
          });
        } catch (error) {
          fatalError = error instanceof Error ? error : new Error(String(error));
        } finally {
          finished = true;
          push({ kind: "__closed__" });
        }
      })();
      return;
    }
    if (data.type === "error") {
      fatalError = new Error(data.error || "PDF extraction worker failed");
      finished = true;
      push({ kind: "__closed__" });
    }
  };

  worker.onerror = (event) => {
    fatalError = event instanceof ErrorEvent && event.error
      ? event.error
      : new Error(event.message || "Worker error");
    finished = true;
    push({ kind: "__closed__" });
  };

  worker.postMessage({
    type: messageType,
    pdfBytes,
    options: {
      lineYTolerance: Number.isFinite(opts.lineYTolerance) ? opts.lineYTolerance : undefined,
      imageRegionMargin: Number.isFinite(opts.imageRegionMargin) ? opts.imageRegionMargin : undefined,
      dropContaminatedNumericTokens: opts.dropContaminatedNumericTokens !== false,
    },
  }, [pdfBytes]);

  try {
    while (!finished || queue.length) {
      if (!queue.length) {
        await new Promise((resolve) => {
          wake = resolve;
        });
      }

      while (queue.length) {
        const next = queue.shift();
        if (!next || next.kind === "__closed__") continue;
        yield next;
      }
    }

    if (fatalError) throw fatalError;
  } finally {
    worker.terminate();
  }
}

/**
 * Adaptive extraction pipeline with layout-aware analysis + OCR and safety gate output.
 *
 * @param {File} file
 * @param {{workerUrl?:URL,forceOcrAll?:boolean,ocr?:object,gate?:object}} [opts]
 */
export async function* extractPdfAdaptive(file, opts = {}) {
  yield* runWorkerExtraction(file, opts, "extract_adaptive");
}

/**
 * Legacy native extraction generator. Kept for compatibility.
 *
 * @param {File} file
 * @param {object} [opts]
 */
export async function* extractPdfNative(file, opts = {}) {
  const nativeOpts = {
    ...opts,
    ocr: {
      ...normalizeOcrOptions(opts),
      ...(opts.ocr || {}),
      available: false,
      enabled: false,
    },
    gate: {
      ...normalizeGateOptions(opts.gate),
      hardBlockWhenUnsafeWithoutOcr: false,
    },
  };

  for await (const event of runWorkerExtraction(file, nativeOpts, "extract")) {
    if (event.kind === "stage") continue;
    if (event.kind === "done") {
      yield { kind: "done", pages: event.pages };
      continue;
    }
    yield event;
  }
}
