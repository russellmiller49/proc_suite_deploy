const TYPE_PRIORITY = Object.freeze({
  SSN: 100,
  MRN: 95,
  ACCOUNT: 90,
  EMAIL: 85,
  PHONE: 80,
  DATE: 70,
  IP: 60,
  URL: 50,
});

function typePriority(type) {
  return TYPE_PRIORITY[type] || 0;
}

function strongerSpan(a, b) {
  if ((a.confidence || 0) !== (b.confidence || 0)) {
    return (a.confidence || 0) > (b.confidence || 0) ? a : b;
  }

  const aPriority = typePriority(a.type);
  const bPriority = typePriority(b.type);
  if (aPriority !== bPriority) {
    return aPriority > bPriority ? a : b;
  }

  const aLen = a.end - a.start;
  const bLen = b.end - b.start;
  return aLen >= bLen ? a : b;
}

/**
 * Merge overlapping spans while retaining the strongest type/confidence.
 *
 * @param {Array<{start:number,end:number,type:string,confidence:number,source:string}>} spans
 * @returns {Array<{start:number,end:number,type:string,confidence:number,source:string}>}
 */
export function mergeRedactionSpans(spans) {
  if (!Array.isArray(spans) || spans.length === 0) return [];

  const sorted = [...spans]
    .filter((span) => Number.isFinite(span.start) && Number.isFinite(span.end) && span.end > span.start)
    .sort((a, b) => {
      if (a.start !== b.start) return a.start - b.start;
      return b.end - a.end;
    });

  if (!sorted.length) return [];

  const merged = [];

  for (const span of sorted) {
    const last = merged[merged.length - 1];

    if (!last || span.start > last.end) {
      merged.push({ ...span });
      continue;
    }

    const strongest = strongerSpan(last, span);
    last.start = Math.min(last.start, span.start);
    last.end = Math.max(last.end, span.end);
    last.type = strongest.type;
    last.confidence = Math.max(last.confidence || 0, span.confidence || 0);
    last.source = strongest.source;
  }

  return merged;
}
