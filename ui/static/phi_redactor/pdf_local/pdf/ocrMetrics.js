import { clamp01 } from "./layoutAnalysis.js";

const DEFAULT_LOW_CONFIDENCE_THRESHOLD = 30;

export const OCR_BOILERPLATE_PATTERNS = Object.freeze([
  /\bPowered\s+by\s+Provation\b/i,
  /\bProvation\s+MD\b/i,
  /\bProvation\b.*\b(?:Suite|Road|Street|Drive|Avenue|Blvd|Lane|Court|Way)\b/i,
  /^\s*Page\s+\d+\s+of\s+\d+\s*$/i,
  /^\s*Page\s*[0-9Il]{1,3}\s*[o0]f\s*[0-9Il]{1,3}\s*$/i,
  /\bAMA\b.*\bcopyright\b/i,
  /\bAmerican\s+Medical\s+Association\b/i,
]);

function normalizeText(value) {
  return typeof value === "string" ? value : "";
}

function normalizeLineList(lines, fallbackText = "") {
  if (Array.isArray(lines) && lines.length) {
    return lines
      .map((line) => {
        if (!line) return null;
        return {
          text: normalizeText(line.text).trim(),
          confidence: Number.isFinite(line.confidence)
            ? Number(line.confidence)
            : Number.isFinite(line.conf)
              ? Number(line.conf)
              : null,
        };
      })
      .filter((line) => line && line.text.length > 0);
  }

  return normalizeText(fallbackText)
    .split(/\r?\n/)
    .map((text) => text.trim())
    .filter(Boolean)
    .map((text) => ({ text, confidence: null }));
}

function tokenize(text) {
  return normalizeText(text)
    .split(/\s+/)
    .map((token) => token.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, ""))
    .filter(Boolean);
}

function median(values) {
  const list = (Array.isArray(values) ? values : [])
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!list.length) return 0;
  const mid = Math.floor(list.length / 2);
  if (list.length % 2 === 1) return list[mid];
  return (list[mid - 1] + list[mid]) / 2;
}

function computeJunkScore(text, lines) {
  const joined = normalizeText(text);
  if (!joined.trim()) return 0;
  const tokenSource = Array.isArray(lines) && lines.length
    ? lines.map((line) => normalizeText(line.text)).join(" ")
    : joined;
  const tokens = tokenSource.split(/\s+/).filter(Boolean);
  if (!tokens.length) return 0;

  const weirdTokenCount = tokens.filter((token) => {
    const trimmed = token.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, "");
    if (!trimmed) return true;
    const alpha = (trimmed.match(/[A-Za-z]/g) || []).length;
    const digits = (trimmed.match(/[0-9]/g) || []).length;
    if (trimmed.length >= 4 && alpha <= 1) return true;
    if (trimmed.length >= 5 && digits >= 3 && alpha <= 2) return true;
    if (/[O0]{3,}/.test(trimmed) && alpha <= 2) return true;
    return false;
  }).length;

  const symbolCount = (joined.match(/[^A-Za-z0-9\s]/g) || []).length;
  const nonAlnumRatio = symbolCount / Math.max(1, joined.length);
  const weirdTokenRatio = weirdTokenCount / Math.max(1, tokens.length);
  return clamp01(nonAlnumRatio * 0.55 + weirdTokenRatio * 0.45);
}

export function countFooterBoilerplateHits(lines) {
  let hits = 0;
  for (const line of Array.isArray(lines) ? lines : []) {
    const text = normalizeText(line?.text || line).trim();
    if (!text) continue;
    if (OCR_BOILERPLATE_PATTERNS.some((pattern) => pattern.test(text))) {
      hits += 1;
    }
  }
  return hits;
}

/**
 * Build compact OCR quality metrics used for gate/debug reporting.
 */
export function computeOcrTextMetrics(input = {}, opts = {}) {
  const lowConfidenceThreshold = Number.isFinite(opts.lowConfidenceThreshold)
    ? Number(opts.lowConfidenceThreshold)
    : DEFAULT_LOW_CONFIDENCE_THRESHOLD;

  const text = normalizeText(input.text);
  const lines = normalizeLineList(input.lines, text);
  const joinedText = text || lines.map((line) => line.text).join("\n");

  const charCount = joinedText.length;
  const alphaCount = (joinedText.match(/[A-Za-z]/g) || []).length;
  const alphaRatio = charCount ? clamp01(alphaCount / charCount) : 0;

  const confValues = lines
    .map((line) => line.confidence)
    .filter((value) => Number.isFinite(value));
  const meanLineConf = confValues.length
    ? confValues.reduce((acc, value) => acc + value, 0) / confValues.length
    : null;
  const lowConfLineFrac = confValues.length
    ? clamp01(confValues.filter((value) => value < lowConfidenceThreshold).length / confValues.length)
    : null;

  const tokenLens = tokenize(joinedText).map((token) => token.length);
  const medianTokenLen = tokenLens.length ? median(tokenLens) : 0;
  const junkScore = computeJunkScore(joinedText, lines);

  return {
    charCount,
    alphaRatio,
    junkScore,
    meanLineConf,
    lowConfLineFrac,
    numLines: lines.length,
    medianTokenLen,
    footerBoilerplateHits: countFooterBoilerplateHits(lines),
  };
}
