const DEFAULT_LINE_Y_TOLERANCE = 5.5;
const DEFAULT_SEGMENT_GAP_MIN = 14;
const DEFAULT_REGION_MARGIN = 3;

export function clamp01(value) {
  if (!Number.isFinite(value)) return 0;
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

export function normalizeRect(rect) {
  const rawX = Number(rect?.x) || 0;
  const rawY = Number(rect?.y) || 0;
  const rawWidth = Number(rect?.width) || 0;
  const rawHeight = Number(rect?.height) || 0;

  const x = rawWidth >= 0 ? rawX : rawX + rawWidth;
  const y = rawHeight >= 0 ? rawY : rawY + rawHeight;
  const width = Math.abs(rawWidth);
  const height = Math.abs(rawHeight);

  return { x, y, width, height };
}

export function rectArea(rect) {
  const normalized = normalizeRect(rect);
  return Math.max(0, normalized.width) * Math.max(0, normalized.height);
}

export function intersectionArea(a, b) {
  const ra = normalizeRect(a);
  const rb = normalizeRect(b);

  const left = Math.max(ra.x, rb.x);
  const top = Math.max(ra.y, rb.y);
  const right = Math.min(ra.x + ra.width, rb.x + rb.width);
  const bottom = Math.min(ra.y + ra.height, rb.y + rb.height);

  const width = Math.max(0, right - left);
  const height = Math.max(0, bottom - top);
  return width * height;
}

export function expandRect(rect, margin = DEFAULT_REGION_MARGIN) {
  const normalized = normalizeRect(rect);
  const m = Math.max(0, Number(margin) || 0);
  return {
    x: normalized.x - m,
    y: normalized.y - m,
    width: normalized.width + m * 2,
    height: normalized.height + m * 2,
  };
}

function clampRectToBounds(rect, bounds) {
  const r = normalizeRect(rect);
  const b = normalizeRect(bounds);
  const left = Math.max(b.x, r.x);
  const top = Math.max(b.y, r.y);
  const right = Math.min(b.x + b.width, r.x + r.width);
  const bottom = Math.min(b.y + b.height, r.y + r.height);
  const width = Math.max(0, right - left);
  const height = Math.max(0, bottom - top);
  return { x: left, y: top, width, height };
}

/**
 * Find a vertical split for two-column header OCR regions.
 *
 * @param {Array<{x:number,y:number,width:number,height:number}>} rects
 * @param {{x:number,y:number,width:number,height:number}} bounds
 * @param {{minGapPx?:number,minColumnWidthPx?:number}} [opts]
 * @returns {{splitX:number,gapPx:number,usedSignal:boolean,minGapPx:number,minColumnWidthPx:number,rectCount:number}}
 */
export function deriveVerticalSplitFromRects(rects, bounds, opts = {}) {
  const normalizedBounds = normalizeRect(bounds || { x: 0, y: 0, width: 0, height: 0 });
  const fallbackSplit = normalizedBounds.x + normalizedBounds.width / 2;
  if (normalizedBounds.width <= 2 || normalizedBounds.height <= 2) {
    return {
      splitX: fallbackSplit,
      gapPx: 0,
      usedSignal: false,
      minGapPx: 0,
      minColumnWidthPx: 0,
      rectCount: 0,
    };
  }

  const normalizedRects = (Array.isArray(rects) ? rects : [])
    .map((rect) => clampRectToBounds(rect, normalizedBounds))
    .filter((rect) => rect.width > 1 && rect.height > 1)
    .sort((a, b) => a.x - b.x);

  const minGapPx = Number.isFinite(opts.minGapPx)
    ? Math.max(8, Number(opts.minGapPx))
    : Math.max(24, normalizedBounds.width * 0.04);
  const minColumnWidthPx = Number.isFinite(opts.minColumnWidthPx)
    ? Math.max(16, Number(opts.minColumnWidthPx))
    : Math.max(32, normalizedBounds.width * 0.24);

  let splitX = fallbackSplit;
  let gapPx = 0;
  let usedSignal = false;

  if (normalizedRects.length >= 2) {
    for (let i = 0; i < normalizedRects.length - 1; i += 1) {
      const leftRightEdge = normalizedRects[i].x + normalizedRects[i].width;
      const rightLeftEdge = normalizedRects[i + 1].x;
      const gap = rightLeftEdge - leftRightEdge;
      if (gap <= gapPx) continue;
      gapPx = gap;
      splitX = leftRightEdge + gap / 2;
    }

    if (gapPx >= minGapPx) {
      usedSignal = true;
    } else {
      splitX = fallbackSplit;
      gapPx = 0;
    }
  }

  const minSplit = normalizedBounds.x + minColumnWidthPx;
  const maxSplit = normalizedBounds.x + normalizedBounds.width - minColumnWidthPx;
  if (minSplit < maxSplit) {
    splitX = Math.min(maxSplit, Math.max(minSplit, splitX));
  } else {
    splitX = fallbackSplit;
  }

  return {
    splitX,
    gapPx,
    usedSignal,
    minGapPx,
    minColumnWidthPx,
    rectCount: normalizedRects.length,
  };
}

function rectsTouchOrOverlap(a, b, gap = 0) {
  const ra = normalizeRect(a);
  const rb = normalizeRect(b);
  const g = Math.max(0, Number(gap) || 0);

  return !(
    ra.x + ra.width + g < rb.x ||
    rb.x + rb.width + g < ra.x ||
    ra.y + ra.height + g < rb.y ||
    rb.y + rb.height + g < ra.y
  );
}

function unionRect(a, b) {
  const ra = normalizeRect(a);
  const rb = normalizeRect(b);

  const x1 = Math.min(ra.x, rb.x);
  const y1 = Math.min(ra.y, rb.y);
  const x2 = Math.max(ra.x + ra.width, rb.x + rb.width);
  const y2 = Math.max(ra.y + ra.height, rb.y + rb.height);

  return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
}

function horizontalOverlapRatio(a, b) {
  const ra = normalizeRect(a);
  const rb = normalizeRect(b);

  const overlap = Math.max(0, Math.min(ra.x + ra.width, rb.x + rb.width) - Math.max(ra.x, rb.x));
  const minWidth = Math.max(1, Math.min(ra.width, rb.width));
  return overlap / minWidth;
}

function verticalGap(a, b) {
  const ra = normalizeRect(a);
  const rb = normalizeRect(b);
  if (ra.y + ra.height < rb.y) return rb.y - (ra.y + ra.height);
  if (rb.y + rb.height < ra.y) return ra.y - (rb.y + rb.height);
  return 0;
}

function rectCenterX(rect) {
  const normalized = normalizeRect(rect);
  return normalized.x + normalized.width / 2;
}

function lineSort(a, b, yTolerance = DEFAULT_LINE_Y_TOLERANCE) {
  const dy = b.y - a.y;
  if (Math.abs(dy) > yTolerance) return dy;
  return a.x - b.x;
}

function normalizeToken(value) {
  if (typeof value !== "string") return "";
  return value.replace(/\s+/g, " ").trim();
}

function startsWithPunctuation(token) {
  return /^[,.;:!?)]/.test(token);
}

function appendTokenWithGap(buffer, prevItem, item, token) {
  let output = buffer;

  if (prevItem) {
    const prevRight = prevItem.x + prevItem.width;
    const gap = item.x - prevRight;
    const avgCharWidth = item.width > 0 ? item.width / Math.max(1, token.length) : 1;
    const likelyWordBoundary = gap > Math.max(2, avgCharWidth * 0.5);
    if (likelyWordBoundary && !startsWithPunctuation(token) && !output.endsWith(" ")) {
      output += " ";
    }
  }

  const start = output.length;
  output += token;
  const end = output.length;

  return { text: output, start, end };
}

function itemToRect(item) {
  const width = Math.max(0.1, Number(item?.width) || 0);
  const height = Math.max(0.1, Math.abs(Number(item?.height) || 0));
  const x = Number(item?.x) || 0;
  const baselineY = Number(item?.y) || 0;
  return normalizeRect({ x, y: baselineY - height, width, height });
}

function shouldBreakSegment(previousItem, item, token) {
  if (!previousItem) return false;
  const gap = item.x - (previousItem.x + previousItem.width);
  const avgCharWidth = item.width > 0 ? item.width / Math.max(1, token.length) : 1;
  return gap > Math.max(DEFAULT_SEGMENT_GAP_MIN, avgCharWidth * 2.5);
}

function buildLineText(items) {
  let text = "";
  let prev = null;

  for (const item of items) {
    const token = normalizeToken(item.str);
    if (!token) continue;
    const next = appendTokenWithGap(text, prev, item, token);
    text = next.text;
    prev = item;
  }

  return text;
}

export function mergeRegions(regions, opts = {}) {
  const mergeGap = Number.isFinite(opts.mergeGap) ? Math.max(0, opts.mergeGap) : 2;
  const normalized = (Array.isArray(regions) ? regions : [])
    .map(normalizeRect)
    .filter((rect) => rect.width > 0 && rect.height > 0)
    .sort((a, b) => {
      if (a.y !== b.y) return a.y - b.y;
      return a.x - b.x;
    });

  if (!normalized.length) return [];

  const merged = [];

  for (const region of normalized) {
    let mergedInto = false;

    for (const current of merged) {
      if (rectsTouchOrOverlap(current, region, mergeGap)) {
        const union = unionRect(current, region);
        current.x = union.x;
        current.y = union.y;
        current.width = union.width;
        current.height = union.height;
        mergedInto = true;
        break;
      }
    }

    if (!mergedInto) {
      merged.push({ ...region });
    }
  }

  return merged;
}

export function buildLayoutBlocks(items, opts = {}) {
  const yTolerance = Number.isFinite(opts.lineYTolerance) ? Math.max(0.5, opts.lineYTolerance) : DEFAULT_LINE_Y_TOLERANCE;
  const normalizedItems = (Array.isArray(items) ? items : [])
    .map((item, index) => ({
      ...item,
      itemIndex: Number.isFinite(item?.itemIndex) ? item.itemIndex : index,
      width: Math.max(0.1, Number(item?.width) || 0),
      height: Math.max(0.1, Math.abs(Number(item?.height) || 0)),
      x: Number(item?.x) || 0,
      y: Number(item?.y) || 0,
    }))
    .sort((a, b) => lineSort(a, b, yTolerance));

  const itemHeights = normalizedItems
    .map((item) => item.height)
    .filter((height) => Number.isFinite(height) && height > 0)
    .sort((a, b) => a - b);
  const medianItemHeight = itemHeights.length
    ? (itemHeights.length % 2
      ? itemHeights[Math.floor(itemHeights.length / 2)]
      : (itemHeights[itemHeights.length / 2 - 1] + itemHeights[itemHeights.length / 2]) / 2)
    : 0;

  function resolveDynamicLineTolerance(itemHeight, line) {
    const fromBase = yTolerance;
    const fromItem = Number.isFinite(itemHeight) && itemHeight > 0
      ? Math.max(fromBase, Math.min(10, itemHeight * 0.55))
      : fromBase;
    const fromMedian = medianItemHeight > 0
      ? Math.max(fromBase, Math.min(10, medianItemHeight * 0.5))
      : fromBase;
    const fromLine = Number.isFinite(line?.avgHeight) && line.avgHeight > 0
      ? Math.max(fromBase, Math.min(10, line.avgHeight * 0.55))
      : fromBase;
    return Math.max(fromBase, fromItem, fromMedian, fromLine);
  }

  const lines = [];
  for (const item of normalizedItems) {
    let bestLine = null;
    let bestDistance = Number.POSITIVE_INFINITY;

    for (const line of lines) {
      const tolerance = resolveDynamicLineTolerance(item.height, line);
      const distance = Math.abs(line.y - item.y);
      if (distance > tolerance) continue;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestLine = line;
      }
    }

    if (!bestLine) {
      lines.push({
        y: item.y,
        sampleCount: 1,
        heightSamples: 1,
        avgHeight: item.height,
        items: [item],
      });
      continue;
    }

    bestLine.items.push(item);
    bestLine.sampleCount += 1;
    bestLine.heightSamples += 1;
    // Baseline clustering keeps each row stable when glyph y-values jitter.
    bestLine.y = ((bestLine.y * (bestLine.sampleCount - 1)) + item.y) / bestLine.sampleCount;
    bestLine.avgHeight = ((bestLine.avgHeight * (bestLine.heightSamples - 1)) + item.height) / bestLine.heightSamples;
  }

  const segments = [];
  let segmentId = 1;

  for (const line of lines) {
    line.items.sort((a, b) => a.x - b.x);
    let current = [];

    for (const item of line.items) {
      const token = normalizeToken(item.str);
      if (!token) continue;

      if (current.length && shouldBreakSegment(current[current.length - 1], item, token)) {
        segments.push(buildSegment(current, segmentId++));
        current = [];
      }
      current.push(item);
    }

    if (current.length) {
      segments.push(buildSegment(current, segmentId++));
    }
  }

  segments.sort((a, b) => {
    const dy = b.baselineY - a.baselineY;
    if (Math.abs(dy) > yTolerance) return dy;
    return a.bbox.x - b.bbox.x;
  });

  const blocks = [];
  for (const segment of segments) {
    let best = null;
    let bestScore = Number.POSITIVE_INFINITY;

    for (const block of blocks) {
      const vGap = verticalGap(block.bbox, segment.bbox);
      const xOverlap = horizontalOverlapRatio(block.bbox, segment.bbox);
      const xDistance = Math.abs(rectCenterX(block.bbox) - rectCenterX(segment.bbox));
      const allowAttach = vGap <= 14 && (xOverlap >= 0.15 || xDistance <= 38);
      if (!allowAttach) continue;

      const score = vGap + (1 - Math.min(1, xOverlap)) * 4 + xDistance * 0.02;
      if (score < bestScore) {
        bestScore = score;
        best = block;
      }
    }

    if (!best) {
      blocks.push({
        id: `block-${blocks.length + 1}`,
        bbox: { ...segment.bbox },
        lines: [segment],
        itemIndices: [...segment.itemIndices],
      });
    } else {
      best.lines.push(segment);
      best.itemIndices.push(...segment.itemIndices);
      best.bbox = unionRect(best.bbox, segment.bbox);
    }
  }

  for (const block of blocks) {
    block.lines.sort((a, b) => {
      const dy = b.baselineY - a.baselineY;
      if (Math.abs(dy) > yTolerance) return dy;
      return a.bbox.x - b.bbox.x;
    });
    block.text = block.lines.map((line) => line.text).filter(Boolean).join("\n");
    block.lineCount = block.lines.length;
  }

  blocks.sort((a, b) => {
    const dy = b.bbox.y - a.bbox.y;
    if (Math.abs(dy) > 8) return dy;
    return a.bbox.x - b.bbox.x;
  });

  return blocks;
}

function buildSegment(items, id) {
  const itemIndices = [];
  let bbox = null;

  for (const item of items) {
    const rect = itemToRect(item);
    bbox = bbox ? unionRect(bbox, rect) : rect;
    itemIndices.push(item.itemIndex);
  }

  return {
    id: `segment-${id}`,
    items: [...items],
    itemIndices,
    bbox: bbox || { x: 0, y: 0, width: 0, height: 0 },
    baselineY: items[0]?.y || 0,
    text: buildLineText(items),
  };
}

export function buildTextRegionsFromBlocks(blocks) {
  const out = [];
  for (const block of Array.isArray(blocks) ? blocks : []) {
    for (const line of block.lines || []) {
      out.push(normalizeRect(line.bbox));
    }
  }
  return mergeRegions(out, { mergeGap: 0 });
}

export function computeOverlapRatio(textRegions, imageRegions, opts = {}) {
  const margin = Number.isFinite(opts.margin) ? Math.max(0, opts.margin) : DEFAULT_REGION_MARGIN;
  const textRects = Array.isArray(textRegions) ? textRegions.map(normalizeRect) : [];
  const imageRects = mergeRegions(
    (Array.isArray(imageRegions) ? imageRegions : []).map((region) => expandRect(region, margin)),
    { mergeGap: 0 },
  );

  let totalTextArea = 0;
  let totalOverlapArea = 0;

  for (const textRect of textRects) {
    const area = rectArea(textRect);
    if (area <= 0) continue;
    totalTextArea += area;

    let overlap = 0;
    for (const imageRect of imageRects) {
      overlap += intersectionArea(textRect, imageRect);
    }
    totalOverlapArea += Math.min(area, overlap);
  }

  return totalTextArea > 0 ? clamp01(totalOverlapArea / totalTextArea) : 0;
}

export function markContaminatedItems(items, imageRegions, opts = {}) {
  const margin = Number.isFinite(opts.margin) ? Math.max(0, opts.margin) : DEFAULT_REGION_MARGIN;
  const minOverlapRatio = Number.isFinite(opts.minOverlapRatio) ? clamp01(opts.minOverlapRatio) : 0.12;

  const expandedImageRegions = mergeRegions(
    (Array.isArray(imageRegions) ? imageRegions : []).map((region) => expandRect(region, margin)),
    { mergeGap: 2 },
  );

  const contaminatedByItemIndex = new Set();
  let contaminatedItemCount = 0;
  let shortContaminatedCount = 0;

  for (const item of Array.isArray(items) ? items : []) {
    const itemIndex = Number.isFinite(item?.itemIndex) ? item.itemIndex : null;
    if (itemIndex === null) continue;

    const rect = itemToRect(item);
    const rectSize = rectArea(rect);
    if (rectSize <= 0) continue;

    let overlapArea = 0;
    for (const imageRegion of expandedImageRegions) {
      overlapArea += intersectionArea(rect, imageRegion);
    }
    const overlapRatio = clamp01(overlapArea / rectSize);
    if (overlapRatio < minOverlapRatio) continue;

    contaminatedByItemIndex.add(itemIndex);
    contaminatedItemCount += 1;

    const token = normalizeToken(item.str);
    const shortSymbolic = token.length <= 3 && !/[A-Za-z]/.test(token);
    if (shortSymbolic) shortContaminatedCount += 1;
  }

  const itemCount = Array.isArray(items) ? items.length : 0;
  return {
    contaminatedByItemIndex,
    contaminatedItemCount,
    shortContaminatedCount,
    contaminatedRatio: itemCount ? contaminatedItemCount / itemCount : 0,
    shortContaminatedRatio: itemCount ? shortContaminatedCount / itemCount : 0,
    expandedImageRegions,
  };
}

function shouldExcludeContaminatedToken(token, contaminated) {
  if (!contaminated) return false;
  if (!token) return true;
  if (token.length <= 2) return true;
  if (/^[\d]+$/.test(token) && token.length <= 4) return true;
  if (/^[^A-Za-z0-9]+$/.test(token)) return true;
  return false;
}

function mergeSpanList(spans) {
  if (!spans.length) return [];
  const sorted = [...spans].sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    return a.end - b.end;
  });

  const out = [];
  for (const span of sorted) {
    const last = out[out.length - 1];
    if (!last || span.start > last.end + 1) {
      out.push({ ...span });
      continue;
    }
    last.end = Math.max(last.end, span.end);
  }
  return out;
}

export function assembleTextFromBlocks(blocks, contamination, opts = {}) {
  const dropContaminatedNumericTokens = opts.dropContaminatedNumericTokens !== false;
  const contaminatedByItemIndex = contamination?.contaminatedByItemIndex || new Set();

  function median(values) {
    const nums = (Array.isArray(values) ? values : []).filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
    if (!nums.length) return 0;
    const mid = Math.floor(nums.length / 2);
    return nums.length % 2 ? nums[mid] : (nums[mid - 1] + nums[mid]) / 2;
  }

  function buildSegmentLine(segment) {
    const segmentItems = [...(segment?.items || [])].sort((a, b) => a.x - b.x);
    let filteredLine = "";
    let rawLine = "";
    let filteredPrev = null;
    let rawPrev = null;
    const lineContaminatedSpans = [];

    for (const item of segmentItems) {
      const token = normalizeToken(item.str);
      if (!token) continue;
      totalTokenCount += 1;

      const rawNext = appendTokenWithGap(rawLine, rawPrev, item, token);
      rawLine = rawNext.text;
      rawPrev = item;

      const contaminated = contaminatedByItemIndex.has(item.itemIndex);
      const exclude = dropContaminatedNumericTokens && shouldExcludeContaminatedToken(token, contaminated);
      if (exclude) {
        excludedTokenCount += 1;
        continue;
      }

      const filteredNext = appendTokenWithGap(filteredLine, filteredPrev, item, token);
      filteredLine = filteredNext.text;
      filteredPrev = item;

      if (contaminated) {
        keptContaminatedCount += 1;
        lineContaminatedSpans.push({
          start: filteredNext.start,
          end: filteredNext.end,
        });
      }
    }

    return {
      bbox: segment?.bbox || { x: 0, y: 0, width: 0, height: 0 },
      baselineY: Number(segment?.baselineY) || 0,
      filteredText: filteredLine,
      rawText: rawLine,
      contaminatedSpans: lineContaminatedSpans,
    };
  }

  function rowJoiner(previous, current, currentText) {
    if (!previous) return "";

    const prevText = String(currentText || "");
    if (prevText.endsWith(":") || prevText.endsWith(": ")) return " ";

    const prevRight = (Number(previous.bbox?.x) || 0) + (Number(previous.bbox?.width) || 0);
    const nextLeft = Number(current.bbox?.x) || 0;
    const gap = nextLeft - prevRight;
    if (gap > 56) return "  ";
    return " ";
  }

  function isLabelLike(text) {
    const trimmed = String(text || "").trim();
    if (!trimmed) return false;
    if (!/:$/.test(trimmed)) return false;
    const tokens = trimmed.split(/\s+/).filter(Boolean);
    if (!tokens.length || tokens.length > 7) return false;
    if (trimmed.length > 64) return false;
    return true;
  }

  function countLabelSegments(row) {
    return row.segments.filter((segment) => isLabelLike(segment.filteredText || segment.rawText)).length;
  }

  function countValueSegments(row) {
    return row.segments.filter((segment) => !isLabelLike(segment.filteredText || segment.rawText)).length;
  }

  function shouldZipLabelValueRows(row, nextRow, typicalRowGap) {
    if (!row || !nextRow) return false;

    const rowGap = (Number(row.baselineY) || 0) - (Number(nextRow.baselineY) || 0);
    if (!(rowGap > 0 && rowGap <= Math.max(18, typicalRowGap * 1.6))) return false;

    const labelCount = countLabelSegments(row);
    if (labelCount < 2) return false;
    if (labelCount / Math.max(1, row.segments.length) < 0.6) return false;

    const nextLabelCount = countLabelSegments(nextRow);
    if (nextLabelCount / Math.max(1, nextRow.segments.length) > 0.34) return false;

    if (nextRow.segments.length < row.segments.length) return false;
    return true;
  }

  let text = "";
  let rawText = "";
  const contaminatedSpans = [];
  let excludedTokenCount = 0;
  let keptContaminatedCount = 0;
  let totalTokenCount = 0;

  const segments = [];
  for (const block of Array.isArray(blocks) ? blocks : []) {
    for (const segment of block?.lines || []) {
      if (!segment || !Array.isArray(segment.items) || !segment.items.length) continue;
      segments.push(segment);
    }
  }

  const baseYTolerance = Number.isFinite(opts.lineYTolerance)
    ? Math.max(1.5, Number(opts.lineYTolerance))
    : DEFAULT_LINE_Y_TOLERANCE;
  const segmentHeights = segments
    .map((segment) => Number(segment?.bbox?.height) || 0)
    .filter((height) => height > 0);
  const medianSegmentHeight = median(segmentHeights);
  const adaptiveYTolerance = medianSegmentHeight > 0
    ? Math.max(baseYTolerance, Math.min(8, medianSegmentHeight * 0.55))
    : baseYTolerance;
  const yTolerance = adaptiveYTolerance;
  segments.sort((a, b) => {
    const dy = (Number(b?.baselineY) || 0) - (Number(a?.baselineY) || 0);
    if (Math.abs(dy) > yTolerance) return dy;
    return (Number(a?.bbox?.x) || 0) - (Number(b?.bbox?.x) || 0);
  });

  const rows = [];
  for (const segment of segments) {
    const last = rows[rows.length - 1];
    const baselineY = Number(segment?.baselineY) || 0;
    if (!last || Math.abs(last.baselineY - baselineY) > yTolerance) {
      rows.push({ baselineY, sampleCount: 1, segments: [buildSegmentLine(segment)] });
    } else {
      last.segments.push(buildSegmentLine(segment));
      last.sampleCount += 1;
      // Same-row drift handling for native text layers with tiny y jitter.
      last.baselineY = ((last.baselineY * (last.sampleCount - 1)) + baselineY) / last.sampleCount;
    }
  }

  for (const row of rows) {
    row.segments.sort((a, b) => (Number(a.bbox?.x) || 0) - (Number(b.bbox?.x) || 0));
  }

  const rowGaps = [];
  for (let i = 1; i < rows.length; i += 1) {
    const gap = (Number(rows[i - 1].baselineY) || 0) - (Number(rows[i].baselineY) || 0);
    if (gap > 0) rowGaps.push(gap);
  }
  const typicalRowGap = median(rowGaps) || 12;
  const paragraphBreakThreshold = Math.max(18, typicalRowGap * 1.8);

  let prevBaseline = null;
  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    const nextRow = rows[rowIndex + 1];

    if (text) {
      const gap = prevBaseline !== null ? prevBaseline - row.baselineY : 0;
      const separator = gap > paragraphBreakThreshold ? "\n\n" : "\n";
      text += separator;
      rawText += separator;
    }

    prevBaseline = row.baselineY;

    if (shouldZipLabelValueRows(row, nextRow, typicalRowGap)) {
      const labelSegments = row.segments;
      const valueSegments = nextRow.segments;
      const zippedCount = Math.min(labelSegments.length, valueSegments.length);

      function appendSegmentAsLine(segment) {
        if (!segment) return;
        const segmentText = String(segment.filteredText || "").trim();
        const segmentRaw = String(segment.rawText || "").trim();
        if (!segmentText && !segmentRaw) return;
        if (text) {
          text += "\n";
          rawText += "\n";
        }
        const lineStart = text.length;
        text += segmentText || segmentRaw;
        rawText += segmentRaw || segmentText;
        for (const span of segment.contaminatedSpans) {
          contaminatedSpans.push({
            start: lineStart + span.start,
            end: lineStart + span.end,
            kind: "image_overlap",
          });
        }
      }

      for (let i = 0; i < zippedCount; i += 1) {
        if (i > 0) {
          text += "\n";
          rawText += "\n";
        }

        const label = labelSegments[i];
        const value = valueSegments[i];
        const labelText = String(label.filteredText || label.rawText || "").trim();
        const valueText = String(value.filteredText || value.rawText || "").trim();

        const joiner = labelText.endsWith(":") ? " " : ": ";
        const combined = `${labelText}${joiner}${valueText}`.trim();
        const lineStart = text.length;
        text += combined;
        rawText += combined;

        for (const span of label.contaminatedSpans) {
          contaminatedSpans.push({
            start: lineStart + span.start,
            end: lineStart + span.end,
            kind: "image_overlap",
          });
        }
        const valueOffset = labelText.length + joiner.length;
        for (const span of value.contaminatedSpans) {
          contaminatedSpans.push({
            start: lineStart + valueOffset + span.start,
            end: lineStart + valueOffset + span.end,
            kind: "image_overlap",
          });
        }
      }

      // Keep any overflow segments instead of silently dropping them.
      for (let i = zippedCount; i < labelSegments.length; i += 1) {
        appendSegmentAsLine(labelSegments[i]);
      }
      for (let i = zippedCount; i < valueSegments.length; i += 1) {
        appendSegmentAsLine(valueSegments[i]);
      }

      rowIndex += 1;
      prevBaseline = nextRow.baselineY;
      continue;
    }

    const rowTextParts = [];
    const rowRawParts = [];
    const rowContaminatedSpans = [];
    let rowTextCursor = 0;

    let prevSegment = null;
    let currentRowText = "";

    for (const segment of row.segments) {
      const segmentText = String(segment.filteredText || "").trim();
      const segmentRaw = String(segment.rawText || "").trim();
      if (!segmentText && !segmentRaw) continue;

      const joiner = rowTextParts.length ? rowJoiner(prevSegment, segment, currentRowText) : "";
      if (segmentText) {
        currentRowText += joiner;
        const segStart = currentRowText.length;
        currentRowText += segmentText;

        for (const span of segment.contaminatedSpans) {
          rowContaminatedSpans.push({
            start: segStart + span.start,
            end: segStart + span.end,
          });
        }
      }

      if (segmentRaw) {
        const rawJoiner = rowRawParts.length ? joiner : "";
        rowRawParts.push(`${rawJoiner}${segmentRaw}`);
      }

      prevSegment = segment;
      rowTextCursor = currentRowText.length;
      rowTextParts.push(segmentText);
    }

    const finalRowText = currentRowText.trimEnd();
    const finalRowRaw = rowRawParts.join("").trimEnd();

    const lineStart = text.length;
    text += finalRowText;
    rawText += finalRowRaw || finalRowText;

    for (const span of rowContaminatedSpans) {
      contaminatedSpans.push({
        start: lineStart + span.start,
        end: lineStart + span.end,
        kind: "image_overlap",
      });
    }
  }

  return {
    text: text.trimEnd(),
    rawText: (rawText || text).trimEnd(),
    contaminatedSpans: mergeSpanList(contaminatedSpans),
    excludedTokenCount,
    keptContaminatedCount,
    totalTokenCount,
    excludedTokenRatio: totalTokenCount ? excludedTokenCount / totalTokenCount : 0,
  };
}

export function scoreContamination({
  overlapRatio = 0,
  contaminatedRatio = 0,
  shortContaminatedRatio = 0,
  excludedTokenRatio = 0,
}) {
  const score =
    clamp01(overlapRatio) * 0.55 +
    clamp01(contaminatedRatio) * 0.2 +
    clamp01(shortContaminatedRatio) * 0.15 +
    clamp01(excludedTokenRatio) * 0.1;
  return clamp01(score);
}

export function computeTextStats(items, text) {
  const sourceText = typeof text === "string"
    ? text
    : (Array.isArray(items) ? items.map((item) => item?.str || "").join("") : "");

  const chars = [...sourceText];
  let nonPrintableCount = 0;
  for (const ch of chars) {
    if (/[\x00-\x08\x0E-\x1F\x7F-\x9F]/.test(ch)) {
      nonPrintableCount += 1;
    }
  }

  let singleCharItems = 0;
  const itemList = Array.isArray(items) ? items : [];
  for (const item of itemList) {
    if (normalizeToken(item?.str).length === 1) {
      singleCharItems += 1;
    }
  }

  const charCount = chars.length;
  const itemCount = itemList.length;
  return {
    charCount,
    itemCount,
    nonPrintableRatio: charCount ? nonPrintableCount / charCount : 0,
    singleCharItemRatio: itemCount ? singleCharItems / itemCount : 0,
  };
}

export function estimateCompletenessConfidence(stats, opts = {}) {
  const safeStats = stats || {};
  const excludedTokenRatio = Number.isFinite(opts.excludedTokenRatio)
    ? clamp01(opts.excludedTokenRatio)
    : clamp01(safeStats.excludedTokenRatio || 0);

  let confidence = 1;
  const charCount = Number(safeStats.charCount) || 0;
  const singleCharItemRatio = Number(safeStats.singleCharItemRatio) || 0;
  const nonPrintableRatio = Number(safeStats.nonPrintableRatio) || 0;
  const imageOpCount = Number(safeStats.imageOpCount) || 0;
  const layoutBlockCount = Number(safeStats.layoutBlockCount) || 0;
  const overlapRatio = Number(safeStats.overlapRatio) || 0;
  const contaminationScore = Number(safeStats.contaminationScore) || 0;

  if (charCount < 80) confidence -= 0.35;
  if (singleCharItemRatio >= 0.55) confidence -= 0.17;
  if (nonPrintableRatio >= 0.08) confidence -= 0.12;
  if (imageOpCount >= 5 && charCount <= 1800) confidence -= 0.2;
  if (layoutBlockCount <= 1 && imageOpCount >= 4) confidence -= 0.12;

  confidence -= Math.min(0.32, clamp01(overlapRatio) * 0.6);
  confidence -= Math.min(0.28, clamp01(contaminationScore) * 0.55);
  confidence -= Math.min(0.2, excludedTokenRatio * 0.6);

  return clamp01(confidence);
}
