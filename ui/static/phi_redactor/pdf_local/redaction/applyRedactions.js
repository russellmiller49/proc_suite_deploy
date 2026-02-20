/**
 * Apply redaction spans to text and keep reversible mapping metadata.
 *
 * @param {string} text
 * @param {Array<{start:number,end:number,type:string,confidence:number,source:string}>} spans
 * @param {{tokenFormatter?:(span:object, originalText:string)=>string}} [opts]
 */
export function applyRedactions(text, spans, opts = {}) {
  const tokenFormatter =
    typeof opts.tokenFormatter === "function"
      ? opts.tokenFormatter
      : (span) => `[REDACTED:${span.type}]`;

  const safeText = typeof text === "string" ? text : "";
  const ordered = Array.isArray(spans)
    ? [...spans]
        .filter((span) => Number.isFinite(span.start) && Number.isFinite(span.end) && span.end > span.start)
        .sort((a, b) => {
          if (a.start !== b.start) return a.start - b.start;
          return b.end - a.end;
        })
    : [];

  const segments = [];
  let redactedText = "";
  let cursor = 0;

  for (const span of ordered) {
    const spanStart = Math.max(cursor, Math.max(0, span.start));
    const spanEnd = Math.max(spanStart, Math.min(safeText.length, span.end));
    if (spanEnd <= spanStart) continue;

    if (spanStart > cursor) {
      const visibleText = safeText.slice(cursor, spanStart);
      const visibleRedactedStart = redactedText.length;
      redactedText += visibleText;
      segments.push({
        kind: "plain",
        originalStart: cursor,
        originalEnd: spanStart,
        redactedStart: visibleRedactedStart,
        redactedEnd: redactedText.length,
        text: visibleText,
      });
    }

    const originalText = safeText.slice(spanStart, spanEnd);
    const replacement = String(tokenFormatter(span, originalText));
    const redactionStart = redactedText.length;
    redactedText += replacement;

    segments.push({
      kind: "redaction",
      type: span.type,
      confidence: span.confidence,
      source: span.source,
      originalStart: spanStart,
      originalEnd: spanEnd,
      redactedStart: redactionStart,
      redactedEnd: redactedText.length,
      originalText,
      replacement,
    });

    cursor = spanEnd;
  }

  if (cursor < safeText.length) {
    const visibleText = safeText.slice(cursor);
    const visibleRedactedStart = redactedText.length;
    redactedText += visibleText;
    segments.push({
      kind: "plain",
      originalStart: cursor,
      originalEnd: safeText.length,
      redactedStart: visibleRedactedStart,
      redactedEnd: redactedText.length,
      text: visibleText,
    });
  }

  return {
    redactedText,
    spans: ordered,
    segments,
  };
}

/**
 * Reconstruct original text from the redaction view model.
 *
 * @param {{segments:Array<object>}} viewModel
 */
export function restoreOriginalText(viewModel) {
  if (!viewModel || !Array.isArray(viewModel.segments)) return "";
  return viewModel.segments
    .map((segment) =>
      segment.kind === "redaction" ? segment.originalText || "" : segment.text || "",
    )
    .join("");
}
