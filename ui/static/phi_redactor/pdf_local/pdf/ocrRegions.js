import {
  clamp01,
  deriveVerticalSplitFromRects,
  mergeRegions,
  normalizeRect,
  rectArea,
} from "./layoutAnalysis.js";

const DEFAULT_CROP_OPTIONS = Object.freeze({
  mode: "auto",
  paddingPx: 14,
  minWidthRatio: 0.42,
  minHeightRatio: 0.5,
  minTextRegionCount: 6,
  rightMarginSafetyRatio: 0.12,
  rightMarginTextRegionMin: 2,
});

const DEFAULT_HEADER_ZONE_OPTIONS = Object.freeze({
  topFraction: 0.25,
  minColumnWidthRatio: 0.24,
  minGapPx: 24,
});

const DEFAULT_PROVATION_SKIP_ZONE_OPTIONS = Object.freeze({
  rightStartRatio: 0.62,
  topRatio: 0.2,
  heightRatio: 0.58,
  minFigureCoverageRatio: 0.18,
  maxTextCoverageRatio: 0.12,
  minNativeCharCount: 140,
});

const DEFAULT_LINE_BAND_OPTIONS = Object.freeze({
  shortLineCharMax: 28,
  orphanCharMax: 36,
  maxBands: 8,
  yPaddingPx: 12,
  yPaddingScale: 0.34,
  maxYPaddingRatio: 0.055,
  xPaddingRatio: 0.03,
  xPaddingPxMin: 12,
  minBandWidthRatio: 0.72,
  minBandHeightPx: 26,
  maxBandHeightRatio: 0.12,
  maxLineHeightRatio: 0.065,
  contextLineWindow: 1,
  contextLineDistancePx: 34,
  mergeGapPx: 8,
});

function clamp(value, min, max) {
  if (!Number.isFinite(value)) return min;
  if (value < min) return min;
  if (value > max) return max;
  return value;
}

function quantile(values, q) {
  const list = (Array.isArray(values) ? values : [])
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!list.length) return null;
  if (list.length === 1) return list[0];
  const qq = clamp(q, 0, 1);
  const position = (list.length - 1) * qq;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return list[lower];
  const weight = position - lower;
  return list[lower] * (1 - weight) + list[upper] * weight;
}

function resolveCanvasSize(input = {}) {
  const canvasWidth = Number(input.canvasWidth ?? input.canvas?.width) || 0;
  const canvasHeight = Number(input.canvasHeight ?? input.canvas?.height) || 0;
  const viewportWidth = Number(input.viewportWidth) || 0;
  const viewportHeight = Number(input.viewportHeight) || 0;
  const dpr = Number(input.devicePixelRatio) || 1;
  const pixelRatio = Number.isFinite(dpr) && dpr > 0 ? dpr : 1;
  const viewportPixelWidth = viewportWidth > 0 ? viewportWidth * pixelRatio : 0;
  const viewportPixelHeight = viewportHeight > 0 ? viewportHeight * pixelRatio : 0;
  return {
    width: Math.max(1, Math.ceil(Math.max(canvasWidth, viewportPixelWidth))),
    height: Math.max(1, Math.ceil(Math.max(canvasHeight, viewportPixelHeight))),
  };
}

function normalizeRegions(regions, width, height) {
  return mergeRegions(
    (Array.isArray(regions) ? regions : [])
      .map(normalizeRect)
      .filter((rect) => rect.width > 1 && rect.height > 1)
      .map((rect) => {
        const x0 = clamp(rect.x, 0, width);
        const y0 = clamp(rect.y, 0, height);
        const x1 = clamp(rect.x + rect.width, 0, width);
        const y1 = clamp(rect.y + rect.height, 0, height);
        return normalizeRect({
          x: x0,
          y: y0,
          width: Math.max(0, x1 - x0),
          height: Math.max(0, y1 - y0),
        });
      })
      .filter((rect) => rect.width > 1 && rect.height > 1),
    { mergeGap: 2 },
  );
}

function clampRectToCanvas(rect, width, height) {
  const normalized = normalizeRect(rect || { x: 0, y: 0, width: 0, height: 0 });
  const left = clamp(normalized.x, 0, width);
  const top = clamp(normalized.y, 0, height);
  const right = clamp(normalized.x + normalized.width, 0, width);
  const bottom = clamp(normalized.y + normalized.height, 0, height);
  const clamped = normalizeRect({
    x: left,
    y: top,
    width: Math.max(0, right - left),
    height: Math.max(0, bottom - top),
  });
  return clamped.width > 1 && clamped.height > 1 ? clamped : null;
}

function intersectRect(a, b) {
  const ra = normalizeRect(a);
  const rb = normalizeRect(b);
  const left = Math.max(ra.x, rb.x);
  const top = Math.max(ra.y, rb.y);
  const right = Math.min(ra.x + ra.width, rb.x + rb.width);
  const bottom = Math.min(ra.y + ra.height, rb.y + rb.height);
  const width = Math.max(0, right - left);
  const height = Math.max(0, bottom - top);
  if (width <= 1 || height <= 1) return null;
  return { x: left, y: top, width, height };
}

function sumOverlapArea(regions, target) {
  let area = 0;
  for (const region of Array.isArray(regions) ? regions : []) {
    const overlap = intersectRect(region, target);
    if (!overlap) continue;
    area += rectArea(overlap);
  }
  return area;
}

function buildBox(rect) {
  return [
    Math.floor(rect.x),
    Math.floor(rect.y),
    Math.floor(rect.x + rect.width),
    Math.floor(rect.y + rect.height),
  ];
}

function isLikelyOrphanContinuationLine(text, options) {
  const clean = String(text || "").trim();
  if (!clean) return false;
  if (clean.length < 3 || clean.length > options.orphanCharMax) return false;
  if (/[:;]/.test(clean)) return false;
  if (!/[.?!]$/.test(clean)) return false;
  if (!/^[a-z0-9]/.test(clean)) return false;

  const words = clean.split(/\s+/).filter(Boolean);
  if (!words.length || words.length > 7) return false;
  if (/\b(?:mg|ml|mcg|ug|mm|cm|hours?|mins?|days?|weeks?|nose|mouth)\.$/i.test(clean)) return true;
  if (words.length <= 2) return true;
  return words.length <= 4;
}

function lineLooksRowFragment(prevText, lineText, options) {
  const prev = String(prevText || "").trim();
  const current = String(lineText || "").trim();
  if (!prev || !current) return false;
  if (/[.?!:]$/.test(prev)) return false;
  if (prev.length < 12 || prev.length > 78) return false;
  if (!/[A-Za-z]$/.test(prev)) return false;
  if (current.length > options.shortLineCharMax + 12) return false;
  if (!/^[a-z0-9]/.test(current)) return false;
  return true;
}

/**
 * Derive OCR band regions around likely truncated native lines.
 * Regions are full-width horizontal strips to recover missing prefixes/suffixes.
 *
 * @param {{layoutBlocks?:Array,canvasWidth?:number,canvasHeight?:number,viewportWidth?:number,viewportHeight?:number,canvas?:{width:number,height:number}}} input
 * @param {{shortLineCharMax?:number,orphanCharMax?:number,maxBands?:number,yPaddingPx?:number,yPaddingScale?:number,maxYPaddingRatio?:number,xPaddingRatio?:number,xPaddingPxMin?:number,minBandWidthRatio?:number,minBandHeightPx?:number,maxBandHeightRatio?:number,maxLineHeightRatio?:number,contextLineWindow?:number,contextLineDistancePx?:number,mergeGapPx?:number}} [options]
 */
export function getLineBandRegions(input = {}, options = {}) {
  const { width, height } = resolveCanvasSize(input);
  const merged = {
    ...DEFAULT_LINE_BAND_OPTIONS,
    ...(options && typeof options === "object" ? options : {}),
  };
  const yPaddingPx = Math.max(0, Number(merged.yPaddingPx) || DEFAULT_LINE_BAND_OPTIONS.yPaddingPx);
  const yPaddingScale = clamp(
    Number(merged.yPaddingScale) || DEFAULT_LINE_BAND_OPTIONS.yPaddingScale,
    0.15,
    1.4,
  );
  const maxYPaddingRatio = clamp(
    Number(merged.maxYPaddingRatio) || DEFAULT_LINE_BAND_OPTIONS.maxYPaddingRatio,
    0.01,
    0.2,
  );
  const xPaddingRatio = clamp(
    Number(merged.xPaddingRatio) || DEFAULT_LINE_BAND_OPTIONS.xPaddingRatio,
    0,
    0.18,
  );
  const xPaddingPxMin = Math.max(0, Number(merged.xPaddingPxMin) || DEFAULT_LINE_BAND_OPTIONS.xPaddingPxMin);
  const minBandWidthRatio = clamp(
    Number(merged.minBandWidthRatio) || DEFAULT_LINE_BAND_OPTIONS.minBandWidthRatio,
    0.22,
    1,
  );
  const minBandHeightPx = Math.max(10, Number(merged.minBandHeightPx) || DEFAULT_LINE_BAND_OPTIONS.minBandHeightPx);
  const maxBandHeightRatio = clamp(
    Number(merged.maxBandHeightRatio) || DEFAULT_LINE_BAND_OPTIONS.maxBandHeightRatio,
    0.08,
    0.6,
  );
  const maxLineHeightRatio = clamp(
    Number(merged.maxLineHeightRatio) || DEFAULT_LINE_BAND_OPTIONS.maxLineHeightRatio,
    0.02,
    0.16,
  );
  const contextLineWindow = Math.max(
    0,
    Math.min(4, Math.floor(Number(merged.contextLineWindow) || DEFAULT_LINE_BAND_OPTIONS.contextLineWindow)),
  );
  const contextLineDistancePx = Math.max(
    12,
    Number(merged.contextLineDistancePx) || DEFAULT_LINE_BAND_OPTIONS.contextLineDistancePx,
  );
  const maxBands = Math.max(1, Math.floor(Number(merged.maxBands) || DEFAULT_LINE_BAND_OPTIONS.maxBands));
  const mergeGapPx = Math.max(0, Number(merged.mergeGapPx) || DEFAULT_LINE_BAND_OPTIONS.mergeGapPx);

  const rawLines = [];
  for (const block of Array.isArray(input.layoutBlocks) ? input.layoutBlocks : []) {
    const blockLines = Array.isArray(block?.lines) ? block.lines : [];
    for (const line of blockLines) {
      const text = String(line?.text || "").replace(/\s+/g, " ").trim();
      const bbox = clampRectToCanvas(line?.bbox, width, height);
      if (!text || !bbox) continue;
      rawLines.push({ text, bbox });
    }
  }

  if (!rawLines.length) {
    return {
      regions: [],
      meta: {
        applied: false,
        reason: "no_line_geometry",
        candidateCount: 0,
      },
    };
  }

  const lineHeights = rawLines
    .map((entry) => Number(entry?.bbox?.height) || 0)
    .filter((value) => value > 0)
    .sort((a, b) => a - b);
  const medianLineHeight = lineHeights.length
    ? lineHeights[Math.floor(lineHeights.length / 2)]
    : Math.max(14, height * 0.018);
  const maxLineHeightPx = Math.max(
    16,
    Math.min(height * maxLineHeightRatio, medianLineHeight * 2.8),
  );

  const lines = rawLines.map((entry) => {
    const box = normalizeRect(entry.bbox);
    if (box.height <= maxLineHeightPx) return entry;
    const centerY = box.y + box.height / 2;
    const boundedHeight = maxLineHeightPx;
    return {
      text: entry.text,
      bbox: normalizeRect({
        x: box.x,
        y: centerY - boundedHeight / 2,
        width: box.width,
        height: boundedHeight,
      }),
    };
  });

  if (!lines.length) {
    return {
      regions: [],
      meta: {
        applied: false,
        reason: "no_line_geometry",
        candidateCount: 0,
      },
    };
  }

  lines.sort((a, b) => {
    const dy = b.bbox.y - a.bbox.y;
    if (Math.abs(dy) > 4) return dy;
    return a.bbox.x - b.bbox.x;
  });

  const bands = [];
  let orphanCount = 0;
  let rowFragmentCount = 0;

  for (let i = 0; i < lines.length; i += 1) {
    const current = lines[i];
    const previous = i > 0 ? lines[i - 1] : null;
    const shortLine = current.text.length <= merged.shortLineCharMax;
    const orphan = isLikelyOrphanContinuationLine(current.text, merged);
    const nearPreviousRow = previous &&
      Math.abs(previous.bbox.y - current.bbox.y) <= (Math.max(previous.bbox.height, current.bbox.height) * 2.6 + 10);
    const rowFragment = shortLine && previous && nearPreviousRow &&
      lineLooksRowFragment(previous.text, current.text, merged);
    if (!orphan && !rowFragment) continue;

    if (orphan) orphanCount += 1;
    if (rowFragment) rowFragmentCount += 1;

    let xLeft = current.bbox.x;
    let xRight = current.bbox.x + current.bbox.width;
    let yTop = current.bbox.y;
    let yBottom = current.bbox.y + current.bbox.height;
    if (rowFragment && previous) {
      xLeft = Math.min(xLeft, previous.bbox.x);
      xRight = Math.max(xRight, previous.bbox.x + previous.bbox.width);
      yTop = Math.min(yTop, previous.bbox.y);
      yBottom = Math.max(yBottom, previous.bbox.y + previous.bbox.height);
    } else if (orphan && previous) {
      // Orphans often trail a longer line in the same row: widen capture around both lines.
      const nearSameRow = Math.abs(previous.bbox.y - current.bbox.y) <= Math.max(previous.bbox.height, current.bbox.height) * 2.4;
      if (nearSameRow) {
        xLeft = Math.min(xLeft, previous.bbox.x);
        xRight = Math.max(xRight, previous.bbox.x + previous.bbox.width);
        yTop = Math.min(yTop, previous.bbox.y);
        yBottom = Math.max(yBottom, previous.bbox.y + previous.bbox.height);
      }
    }

    if (contextLineWindow > 0) {
      for (let offset = -contextLineWindow; offset <= contextLineWindow; offset += 1) {
        if (!offset) continue;
        const context = lines[i + offset];
        if (!context) continue;
        const withinDistance = Math.abs(context.bbox.y - current.bbox.y) <= contextLineDistancePx;
        if (!withinDistance) continue;
        xLeft = Math.min(xLeft, context.bbox.x);
        xRight = Math.max(xRight, context.bbox.x + context.bbox.width);
        yTop = Math.min(yTop, context.bbox.y);
        yBottom = Math.max(yBottom, context.bbox.y + context.bbox.height);
      }
    }

    const unclampedHeight = Math.max(minBandHeightPx, yBottom - yTop);
    const maxBandHeightPx = Math.max(minBandHeightPx, height * maxBandHeightRatio);
    const desiredHeight = Math.min(unclampedHeight, maxBandHeightPx);
    const localHeight = Math.max(current.bbox.height, yBottom - yTop);
    const maxYPaddingPx = Math.max(yPaddingPx, height * maxYPaddingRatio, desiredHeight * 0.65);
    const localYPadding = Math.min(maxYPaddingPx, Math.max(yPaddingPx, localHeight * yPaddingScale));
    const center = (yTop + yBottom) / 2;
    const adjustedTop = center - desiredHeight / 2;
    const horizontalPad = Math.max(xPaddingPxMin, width * xPaddingRatio);
    const desiredWidth = Math.max(width * minBandWidthRatio, (xRight - xLeft) + horizontalPad * 2);
    const centerX = (xLeft + xRight) / 2;
    const adjustedLeft = centerX - desiredWidth / 2;
    const bandRect = clampRectToCanvas({
      x: adjustedLeft,
      y: adjustedTop - localYPadding,
      width: desiredWidth,
      height: desiredHeight + localYPadding * 2,
    }, width, height);
    if (!bandRect) continue;
    bands.push(bandRect);
  }

  if (!bands.length) {
    return {
      regions: [],
      meta: {
        applied: false,
        reason: "no_suspect_lines",
        candidateCount: 0,
      },
    };
  }

  const mergedBands = mergeRegions(bands, { mergeGap: mergeGapPx })
    .sort((a, b) => b.y - a.y)
    .slice(0, maxBands);

  return {
    regions: mergedBands,
    meta: {
      applied: mergedBands.length > 0,
      reason: mergedBands.length ? "line_backfill_bands" : "no_suspect_lines",
      candidateCount: bands.length,
      orphanCount,
      rowFragmentCount,
      regionCount: mergedBands.length,
    },
  };
}

/**
 * Build a strict top-of-page header OCR zone and split it into vertical columns.
 * Column OCR can be queued independently to prevent cross-column bleeding.
 *
 * @param {{canvasWidth?:number,canvasHeight?:number,canvas?:{width:number,height:number},textRegions?:Array,workingRect?:{x:number,y:number,width:number,height:number}}} input
 * @param {{topFraction?:number,minColumnWidthRatio?:number,minGapPx?:number}} [options]
 */
export function computeHeaderZoneColumns(input = {}, options = {}) {
  const { width, height } = resolveCanvasSize(input);
  const merged = {
    ...DEFAULT_HEADER_ZONE_OPTIONS,
    ...(options && typeof options === "object" ? options : {}),
  };
  const topFraction = clamp(
    Number(merged.topFraction) || DEFAULT_HEADER_ZONE_OPTIONS.topFraction,
    0.16,
    0.4,
  );
  const minColumnWidthRatio = clamp(
    Number(merged.minColumnWidthRatio) || DEFAULT_HEADER_ZONE_OPTIONS.minColumnWidthRatio,
    0.18,
    0.45,
  );
  const minGapPx = Math.max(8, Number(merged.minGapPx) || DEFAULT_HEADER_ZONE_OPTIONS.minGapPx);

  const pageRect = normalizeRect({ x: 0, y: 0, width, height });
  const workingRect = clampRectToCanvas(input.workingRect || pageRect, width, height) || pageRect;
  const headerHeight = Math.max(24, Math.floor(height * topFraction));
  const headerZoneRect = normalizeRect({
    x: 0,
    y: 0,
    width,
    height: Math.min(height, headerHeight),
  });

  const headerTextRegions = normalizeRegions(input.textRegions, width, height)
    .map((rect) => intersectRect(rect, headerZoneRect))
    .filter(Boolean);

  const split = deriveVerticalSplitFromRects(headerTextRegions, headerZoneRect, {
    minGapPx,
    minColumnWidthPx: headerZoneRect.width * minColumnWidthRatio,
  });
  const splitX = clamp(split.splitX, headerZoneRect.x + 1, headerZoneRect.x + headerZoneRect.width - 1);

  const leftRect = normalizeRect({
    x: headerZoneRect.x,
    y: headerZoneRect.y,
    width: Math.max(1, splitX - headerZoneRect.x),
    height: headerZoneRect.height,
  });
  const rightRect = normalizeRect({
    x: splitX,
    y: headerZoneRect.y,
    width: Math.max(1, headerZoneRect.x + headerZoneRect.width - splitX),
    height: headerZoneRect.height,
  });
  const columns = [leftRect, rightRect]
    .map((rect, index) => ({
      id: index === 0 ? "header_left" : "header_right",
      order: index,
      rect,
    }))
    .filter((entry) => entry.rect.width > 8 && entry.rect.height > 8);

  const workingBottom = Math.min(height, workingRect.y + workingRect.height);
  let bodyTop = Math.max(workingRect.y, headerZoneRect.y + headerZoneRect.height);
  if (workingBottom - bodyTop < 28) {
    bodyTop = workingRect.y;
  }
  const bodyRect = normalizeRect({
    x: workingRect.x,
    y: bodyTop,
    width: workingRect.width,
    height: Math.max(1, workingBottom - bodyTop),
  });

  return {
    headerZoneRect,
    columns,
    bodyRect,
    workingRect,
    meta: {
      topFraction,
      headerTextRegionCount: headerTextRegions.length,
      splitX,
      splitGapPx: split.gapPx,
      splitUsedSignal: split.usedSignal,
      box: buildBox(headerZoneRect),
    },
  };
}

/**
 * Detect predictable right-side tracheobronchial diagram zones and mark them
 * as OCR-skip regions to reduce injected graphical artifacts.
 */
export function computeProvationDiagramSkipRegions(input = {}, options = {}) {
  const { width, height } = resolveCanvasSize(input);
  const pageArea = Math.max(1, width * height);
  const merged = {
    ...DEFAULT_PROVATION_SKIP_ZONE_OPTIONS,
    ...(options && typeof options === "object" ? options : {}),
  };
  const nativeCharCount = Math.max(0, Number(input.nativeCharCount) || 0);
  const pageIndex = Number.isFinite(input.pageIndex) ? Number(input.pageIndex) : 0;

  const knownZone = normalizeRect({
    x: width * clamp(merged.rightStartRatio, 0.45, 0.85),
    y: height * clamp(merged.topRatio, 0.05, 0.7),
    width: width * (1 - clamp(merged.rightStartRatio, 0.45, 0.85)),
    height: height * clamp(merged.heightRatio, 0.2, 0.8),
  });

  const figureRegions = normalizeRegions(
    input.figureRegions || input.imageRegions,
    width,
    height,
  );
  const textRegions = normalizeRegions(input.textRegions, width, height);

  if (!figureRegions.length) {
    return {
      regions: [],
      meta: {
        applied: false,
        reason: "no_figure_regions",
        pageIndex,
      },
    };
  }

  const knownZoneArea = Math.max(1, rectArea(knownZone));
  const figureCoverage = clamp01(sumOverlapArea(figureRegions, knownZone) / knownZoneArea);
  const textCoverage = clamp01(sumOverlapArea(textRegions, knownZone) / knownZoneArea);
  const rightMidFigureArea = figureRegions
    .filter((rect) => {
      const centerX = rect.x + rect.width / 2;
      const centerY = rect.y + rect.height / 2;
      return centerX >= width * 0.58 &&
        centerY >= height * 0.15 &&
        centerY <= height * 0.85;
    })
    .reduce((sum, rect) => sum + rectArea(rect), 0);
  const rightMidFigureRatio = clamp01(rightMidFigureArea / pageArea);
  const earlyPage = pageIndex <= 1;

  const likelyDiagram = nativeCharCount >= Math.max(40, Number(merged.minNativeCharCount) || 140) &&
    textCoverage <= clamp01(Number(merged.maxTextCoverageRatio) || DEFAULT_PROVATION_SKIP_ZONE_OPTIONS.maxTextCoverageRatio) &&
    (
      figureCoverage >= clamp01(Number(merged.minFigureCoverageRatio) || DEFAULT_PROVATION_SKIP_ZONE_OPTIONS.minFigureCoverageRatio) ||
      rightMidFigureRatio >= 0.09
    ) &&
    (earlyPage || figureCoverage >= 0.33 || rightMidFigureRatio >= 0.14);

  if (!likelyDiagram) {
    return {
      regions: [],
      meta: {
        applied: false,
        reason: "diagram_not_detected",
        figureCoverage,
        textCoverage,
        rightMidFigureRatio,
        pageIndex,
      },
    };
  }

  const candidateRegions = figureRegions
    .filter((rect) => intersectRect(rect, knownZone))
    .map((rect) => intersectRect(rect, knownZone))
    .filter(Boolean);
  const regions = mergeRegions([knownZone, ...candidateRegions], { mergeGap: 10 });

  return {
    regions,
    meta: {
      applied: regions.length > 0,
      reason: regions.length ? "provation_tree_diagram" : "diagram_not_detected",
      figureCoverage,
      textCoverage,
      rightMidFigureRatio,
      pageIndex,
      regionCount: regions.length,
    },
  };
}

/**
 * Compute a left-column OCR crop rectangle from text/image geometry.
 * Input regions are expected in viewport pixel space.
 */
export function computeOcrCropRect(input = {}, options = {}) {
  const { width, height } = resolveCanvasSize(input);
  const pageArea = Math.max(1, width * height);

  const merged = {
    ...DEFAULT_CROP_OPTIONS,
    ...(options && typeof options === "object" ? options : {}),
  };
  const mode = merged.mode === "on" ? "on" : merged.mode === "off" ? "off" : "auto";
  const paddingPx = Math.max(0, Number(merged.paddingPx) || DEFAULT_CROP_OPTIONS.paddingPx);
  const minWidthRatio = clamp01(Number(merged.minWidthRatio) || DEFAULT_CROP_OPTIONS.minWidthRatio);
  const minHeightRatio = clamp01(Number(merged.minHeightRatio) || DEFAULT_CROP_OPTIONS.minHeightRatio);
  const minTextRegionCount = Math.max(2, Math.floor(Number(merged.minTextRegionCount) || DEFAULT_CROP_OPTIONS.minTextRegionCount));
  const rightMarginSafetyRatio = clamp(
    Number(merged.rightMarginSafetyRatio) || DEFAULT_CROP_OPTIONS.rightMarginSafetyRatio,
    0.06,
    0.2,
  );
  const rightMarginTextRegionMin = Math.max(
    1,
    Math.floor(Number(merged.rightMarginTextRegionMin) || DEFAULT_CROP_OPTIONS.rightMarginTextRegionMin),
  );

  if (mode === "off") {
    return {
      rect: null,
      meta: {
        applied: false,
        mode,
        reason: "disabled",
      },
    };
  }

  const textRegions = normalizeRegions(input.textRegions, width, height);
  const imageRegions = normalizeRegions(input.imageRegions, width, height);
  const nativeCharCount = Math.max(0, Number(input.nativeCharCount) || 0);

  if (!textRegions.length && !imageRegions.length) {
    return {
      rect: null,
      meta: {
        applied: false,
        mode,
        reason: "no_regions",
      },
    };
  }

  if (mode === "auto" && textRegions.length < minTextRegionCount && nativeCharCount < 220) {
    return {
      rect: null,
      meta: {
        applied: false,
        mode,
        reason: "low_text_signal",
        nativeCharCount,
        textRegionCount: textRegions.length,
      },
    };
  }

  const rightEdges = textRegions.map((rect) => rect.x + rect.width);
  const x1FromText = quantile(rightEdges, 0.85);
  const rightMostTextEdge = rightEdges.length ? Math.max(...rightEdges) : null;
  const rightMarginTextRegions = textRegions.filter(
    (rect) => rect.x + rect.width >= width * (1 - rightMarginSafetyRatio),
  );
  const hasStrongRightMarginText = rightMarginTextRegions.length >= rightMarginTextRegionMin;

  const rightStripRegions = imageRegions.filter((rect) => {
    const tallEnough = rect.height / Math.max(1, height) >= 0.18;
    const largeEnough = rectArea(rect) / pageArea >= 0.035;
    const rightSide = rect.x >= width * 0.45;
    return rightSide && (tallEnough || largeEnough);
  });
  const x1FromImageStrip = rightStripRegions.length
    ? Math.min(...rightStripRegions.map((rect) => rect.x))
    : null;
  const rightImageBarrierStrong = Number.isFinite(x1FromImageStrip) && x1FromImageStrip <= width * 0.9;

  if (mode === "auto" && hasStrongRightMarginText && !rightImageBarrierStrong) {
    return {
      rect: null,
      meta: {
        applied: false,
        mode,
        reason: "right_margin_text",
        rightMarginTextRegionCount: rightMarginTextRegions.length,
        textRegionCount: textRegions.length,
      },
    };
  }

  let targetX1 = null;
  if (Number.isFinite(x1FromText) && Number.isFinite(x1FromImageStrip)) {
    targetX1 = Math.min(Number(x1FromText), Number(x1FromImageStrip)) + paddingPx;
  } else if (Number.isFinite(x1FromText)) {
    targetX1 = Number(x1FromText) + paddingPx;
  } else if (mode === "on" && Number.isFinite(x1FromImageStrip)) {
    targetX1 = Number(x1FromImageStrip) + paddingPx;
  }

  if (!Number.isFinite(targetX1)) {
    return {
      rect: null,
      meta: {
        applied: false,
        mode,
        reason: "no_column_signal",
        textRegionCount: textRegions.length,
        imageRegionCount: imageRegions.length,
      },
    };
  }

  const minWidthPx = Math.floor(width * Math.max(0.25, minWidthRatio));
  const minSafeRightX1 = Number.isFinite(rightMostTextEdge) &&
    rightMostTextEdge >= width * (1 - rightMarginSafetyRatio)
    ? Math.ceil(rightMostTextEdge + paddingPx)
    : null;
  const effectiveTargetX1 = Number.isFinite(minSafeRightX1)
    ? Math.max(targetX1, minSafeRightX1)
    : targetX1;
  const croppedX1 = clamp(Math.ceil(effectiveTargetX1), minWidthPx, width);

  const yStarts = textRegions.map((rect) => rect.y);
  const yEnds = textRegions.map((rect) => rect.y + rect.height);
  let y0 = 0;
  let y1 = height;
  if (yStarts.length && yEnds.length) {
    y0 = clamp(Math.floor(Math.min(...yStarts) - paddingPx), 0, height);
    y1 = clamp(Math.ceil(Math.max(...yEnds) + paddingPx), 0, height);
    if (y1 - y0 < height * Math.max(0.25, minHeightRatio)) {
      y0 = 0;
      y1 = height;
    }
  }

  const rect = normalizeRect({
    x: 0,
    y: y0,
    width: croppedX1,
    height: Math.max(1, y1 - y0),
  });
  const applied = rect.width < width - 2 || rect.height < height - 2;

  return {
    rect: applied ? rect : null,
    meta: {
      applied,
      mode,
      reason: applied ? "left_column_crop" : "full_page",
      textRegionCount: textRegions.length,
      imageRegionCount: imageRegions.length,
      nativeCharCount,
      rightMarginTextRegionCount: rightMarginTextRegions.length,
      x1FromText: Number.isFinite(x1FromText) ? Number(x1FromText) : null,
      x1FromImageStrip: Number.isFinite(x1FromImageStrip) ? Number(x1FromImageStrip) : null,
      box: buildBox(rect),
    },
  };
}
