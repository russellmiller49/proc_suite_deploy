import "../../vendor/pdfjs/pdf.worker.mjs";
import * as pdfjs from "../../vendor/pdfjs/pdf.mjs";
import {
  assembleTextFromBlocks,
  buildLayoutBlocks,
  buildTextRegionsFromBlocks,
  clamp01,
  computeOverlapRatio,
  computeTextStats,
  estimateCompletenessConfidence,
  markContaminatedItems,
  mergeRegions,
  normalizeRect,
  scoreContamination,
} from "../pdf/layoutAnalysis.js";

// Ensure PDF.js has a same-origin worker source fallback.
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "../../vendor/pdfjs/pdf.worker.mjs",
  import.meta.url,
).toString();

function extractPositionedItems(textContent) {
  const out = [];
  let index = 0;

  for (const item of textContent.items || []) {
    if (!item || typeof item.str !== "string") continue;
    const transform = Array.isArray(item.transform) ? item.transform : [1, 0, 0, 1, 0, 0];
    const x = Number(transform[4]) || 0;
    const y = Number(transform[5]) || 0;
    const width = Math.max(0.1, Number(item.width) || 0);
    const height = Math.max(0.1, Math.abs(Number(item.height) || Math.abs(Number(transform[3]) || 0)));
    out.push({ itemIndex: index, str: item.str, x, y, width, height });
    index += 1;
  }

  return out;
}

function mulTransform(m1, m2) {
  return [
    m1[0] * m2[0] + m1[2] * m2[1],
    m1[1] * m2[0] + m1[3] * m2[1],
    m1[0] * m2[2] + m1[2] * m2[3],
    m1[1] * m2[2] + m1[3] * m2[3],
    m1[0] * m2[4] + m1[2] * m2[5] + m1[4],
    m1[1] * m2[4] + m1[3] * m2[5] + m1[5],
  ];
}

function transformPoint(m, x, y) {
  return {
    x: m[0] * x + m[2] * y + m[4],
    y: m[1] * x + m[3] * y + m[5],
  };
}

function deriveImageSizeFromArgs(args) {
  if (!Array.isArray(args)) return { width: 1, height: 1 };

  if (Number.isFinite(args[1]) && Number.isFinite(args[2])) {
    return {
      width: Math.max(1, Number(args[1])),
      height: Math.max(1, Number(args[2])),
    };
  }

  const first = args[0];
  if (first && typeof first === "object") {
    const width = Number(first.width);
    const height = Number(first.height);
    if (Number.isFinite(width) && Number.isFinite(height)) {
      return {
        width: Math.max(1, width),
        height: Math.max(1, height),
      };
    }
  }

  return { width: 1, height: 1 };
}

function estimateImageRegion(ctm, args) {
  const { width, height } = deriveImageSizeFromArgs(args);
  const points = [
    transformPoint(ctm, 0, 0),
    transformPoint(ctm, width, 0),
    transformPoint(ctm, 0, height),
    transformPoint(ctm, width, height),
  ];

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);

  const x1 = Math.min(...xs);
  const x2 = Math.max(...xs);
  const y1 = Math.min(...ys);
  const y2 = Math.max(...ys);

  const region = normalizeRect({
    x: x1,
    y: y1,
    width: x2 - x1,
    height: y2 - y1,
  });

  if (region.width < 0.5 || region.height < 0.5) return null;
  return region;
}

async function computeOperatorStats(page) {
  try {
    const opList = await page.getOperatorList();
    const fnArray = Array.isArray(opList?.fnArray) ? opList.fnArray : [];
    const argsArray = Array.isArray(opList?.argsArray) ? opList.argsArray : [];

    let imageOpCount = 0;
    let textShowOpCount = 0;
    let ctm = [1, 0, 0, 1, 0, 0];
    const ctmStack = [];
    const imageRegions = [];

    for (let i = 0; i < fnArray.length; i += 1) {
      const fn = fnArray[i];
      const args = argsArray[i];

      if (fn === pdfjs.OPS.save) {
        ctmStack.push(ctm.slice());
        continue;
      }
      if (fn === pdfjs.OPS.restore) {
        ctm = ctmStack.length ? ctmStack.pop() : [1, 0, 0, 1, 0, 0];
        continue;
      }
      if (fn === pdfjs.OPS.transform && Array.isArray(args) && args.length >= 6) {
        ctm = mulTransform(ctm, args);
        continue;
      }

      if (fn === pdfjs.OPS.showText || fn === pdfjs.OPS.showSpacedText) {
        textShowOpCount += 1;
      }

      if (
        fn === pdfjs.OPS.paintImageXObject ||
        fn === pdfjs.OPS.paintInlineImageXObject ||
        fn === pdfjs.OPS.paintImageMaskXObject
      ) {
        imageOpCount += 1;
        const region = estimateImageRegion(ctm, args);
        if (region) imageRegions.push(region);
      }
    }

    return {
      imageOpCount,
      textShowOpCount,
      imageRegions: mergeRegions(imageRegions, { mergeGap: 1.5 }),
    };
  } catch {
    return {
      imageOpCount: 0,
      textShowOpCount: 0,
      imageRegions: [],
    };
  }
}

function summarizeLayoutBlocks(blocks) {
  return (Array.isArray(blocks) ? blocks : []).map((block) => ({
    id: block.id,
    bbox: block.bbox,
    lineCount: Number(block.lineCount) || 0,
    textPreview: (block.text || "").slice(0, 140),
    lines: (Array.isArray(block.lines) ? block.lines : [])
      .map((line) => ({
        bbox: line?.bbox,
        baselineY: Number.isFinite(line?.baselineY) ? Number(line.baselineY) : null,
        text: String(line?.text || "").slice(0, 220),
      }))
      .filter((line) => line.text && line.bbox),
  }));
}

function computeQualityFlags(stats, assembly) {
  const flags = [];
  if ((stats.imageOpCount || 0) > 0) flags.push("IMAGE_CONTENT_PRESENT");
  if ((stats.imageOpCount || 0) >= 5) flags.push("IMAGE_HEAVY");
  if ((stats.overlapRatio || 0) >= 0.08) flags.push("IMAGE_TEXT_OVERLAP");
  if ((stats.contaminationScore || 0) >= 0.24) flags.push("CONTAMINATION_RISK");
  if ((stats.completenessConfidence || 0) < 0.72) flags.push("LOW_COMPLETENESS");
  if ((stats.charCount || 0) < 80) flags.push("SPARSE_TEXT");
  if ((assembly?.excludedTokenCount || 0) > 0) flags.push("EXCLUDED_IMAGE_LABELS");
  return flags;
}

async function analyzePage(page, pageIndex, totalPages, options = {}) {
  self.postMessage({ type: "stage", stage: "layout_analysis", pageIndex, totalPages });

  const viewport = page.getViewport({ scale: 1 });
  const pageWidth = Math.max(1, Math.abs(Number(viewport?.width) || 0));
  const pageHeight = Math.max(1, Math.abs(Number(viewport?.height) || 0));
  const pageArea = Math.max(1, pageWidth * pageHeight);

  const textContent = await page.getTextContent({
    includeMarkedContent: false,
    disableNormalization: false,
  });
  const items = extractPositionedItems(textContent);
  const blocks = buildLayoutBlocks(items, {
    lineYTolerance: Number.isFinite(options.lineYTolerance) ? options.lineYTolerance : undefined,
  });
  const textRegions = buildTextRegionsFromBlocks(blocks);
  const opStats = await computeOperatorStats(page);

  self.postMessage({ type: "stage", stage: "contamination_detection", pageIndex, totalPages });

  const overlapRatio = computeOverlapRatio(textRegions, opStats.imageRegions, {
    margin: Number.isFinite(options.imageRegionMargin) ? options.imageRegionMargin : 3,
  });
  const contamination = markContaminatedItems(items, opStats.imageRegions, {
    margin: Number.isFinite(options.imageRegionMargin) ? options.imageRegionMargin : 3,
    minOverlapRatio: Number.isFinite(options.itemOverlapThreshold) ? clamp01(options.itemOverlapThreshold) : 0.12,
  });

  self.postMessage({ type: "stage", stage: "adaptive_assembly", pageIndex, totalPages });

  const assembly = assembleTextFromBlocks(blocks, contamination, {
    dropContaminatedNumericTokens: options.dropContaminatedNumericTokens !== false,
    lineYTolerance: Number.isFinite(options.lineYTolerance) ? Number(options.lineYTolerance) : undefined,
  });

  const textStats = computeTextStats(items, assembly.text);
  const contaminationScore = scoreContamination({
    overlapRatio,
    contaminatedRatio: contamination.contaminatedRatio,
    shortContaminatedRatio: contamination.shortContaminatedRatio,
    excludedTokenRatio: assembly.excludedTokenRatio,
  });

  const stats = {
    ...textStats,
    pageWidth,
    pageHeight,
    pageArea,
    nativeTextDensity: textStats.charCount / pageArea,
    imageOpCount: opStats.imageOpCount,
    textShowOpCount: opStats.textShowOpCount,
    layoutBlockCount: blocks.length,
    overlapRatio,
    contaminationScore,
    completenessConfidence: 0,
    excludedTokenRatio: assembly.excludedTokenRatio,
  };

  stats.completenessConfidence = estimateCompletenessConfidence(stats, {
    excludedTokenRatio: assembly.excludedTokenRatio,
  });

  const qualityFlags = computeQualityFlags(stats, assembly);

  return {
    pageIndex,
    text: assembly.text,
    rawText: assembly.rawText,
    stats,
    source: "native",
    sourceDecision: "native",
    layoutBlocks: summarizeLayoutBlocks(blocks),
    imageRegions: opStats.imageRegions,
    textRegions,
    contaminatedSpans: assembly.contaminatedSpans,
    qualityFlags,
  };
}

async function extractPages(pdfBytes, options = {}) {
  const loadingTask = pdfjs.getDocument({
    data: pdfBytes,
    isEvalSupported: false,
    useWorkerFetch: false,
  });

  const doc = await loadingTask.promise;
  const pages = [];

  for (let i = 1; i <= doc.numPages; i += 1) {
    const page = await doc.getPage(i);
    const pageResult = await analyzePage(page, i - 1, doc.numPages, options);
    pages.push(pageResult);

    self.postMessage({ type: "page", page: pageResult });
    self.postMessage({
      type: "progress",
      completedPages: i,
      totalPages: doc.numPages,
    });
  }

  self.postMessage({
    type: "stage",
    stage: "native_extraction_done",
    pageIndex: Math.max(0, doc.numPages - 1),
    totalPages: doc.numPages,
  });

  await doc.destroy();
  return pages;
}

self.onmessage = async (event) => {
  const data = event.data || {};
  if (data.type !== "extract" && data.type !== "extract_adaptive") return;

  try {
    const pages = await extractPages(data.pdfBytes, data.options || {});
    self.postMessage({ type: "done", pages });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    self.postMessage({ type: "error", error: message });
  }
};
