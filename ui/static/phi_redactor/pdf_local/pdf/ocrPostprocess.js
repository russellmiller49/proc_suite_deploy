import { intersectionArea, normalizeRect, rectArea } from "./layoutAnalysis.js";
import { OCR_BOILERPLATE_PATTERNS } from "./ocrMetrics.js";

function safeText(value) {
  return typeof value === "string" ? value : "";
}

const EXACT_CLINICAL_REPLACEMENTS = Object.freeze([
  { re: /\bLidocaine\s+49\%(?=$|[\s,.;:])/gi, replace: "Lidocaine 4%" },
  { re: /\bAtropine\s+9\.5\s+mg\b/gi, replace: "Atropine 0.5 mg" },
  { re: /\bfrom the mouth\s+nose\b/gi, replace: "from the mouth or nose" },
  { re: /\bmouth\s+nose\b/gi, replace: "mouth or nose" },
  { re: /\blyrnphadenopathy\b/gi, replace: "lymphadenopathy" },
  { re: /\bhytnph/gi, replace: "lymph" },
]);

const CLINICAL_TERM_DICTIONARY = Object.freeze([
  "tracheobronchial",
  "subsegmental",
  "lymphadenopathy",
  "endobronchial",
  "mediastinal",
  "bronchoscopy",
  "bronchoscopic",
  "bronchus",
  "mainstem",
  "carina",
  "trachea",
  "biopsy",
  "aspiration",
  "lavage",
  "sedation",
  "anesthesia",
  "atelectasis",
  "hemoptysis",
  "parenchymal",
  "bronchiolitis",
  "pneumothorax",
  "pneumomediastinum",
  "subcarinal",
  "paratracheal",
  "hilar",
  "ultrasound",
  "transbronchial",
  "cytology",
  "histology",
]);

const CLINICAL_TERM_SET = new Set(CLINICAL_TERM_DICTIONARY);
const ANATOMY_CAPTION_TOKENS = new Set([
  "left",
  "right",
  "upper",
  "lower",
  "middle",
  "lobe",
  "lobar",
  "mainstem",
  "entrance",
  "segment",
  "bronchus",
  "airway",
  "carina",
  "trachea",
  "lingula",
  "lul",
  "lll",
  "rul",
  "rml",
  "rll",
]);

const CAPTION_VERB_RE = /\b(?:is|are|was|were|be|been|being|shows?|showed|noted?|seen|performed|placed|inserted|advanced|removed|biops(?:y|ied)|lavage|aspirat(?:e|ed)|examined)\b/i;
const HEADER_DOB_BIRCH_RE = /\b(?:data|date)\s+(?:of|nf)\s+birch\b/i;
const HEADER_ACCOUNT_LABEL_RE = /\b(?:account|acct)\b/i;

function levenshteinDistance(a, b) {
  const left = String(a || "");
  const right = String(b || "");
  if (!left) return right.length;
  if (!right) return left.length;
  if (left === right) return 0;

  const prev = new Array(right.length + 1);
  const curr = new Array(right.length + 1);
  for (let j = 0; j <= right.length; j += 1) prev[j] = j;

  for (let i = 1; i <= left.length; i += 1) {
    curr[0] = i;
    const lc = left.charCodeAt(i - 1);
    for (let j = 1; j <= right.length; j += 1) {
      const rc = right.charCodeAt(j - 1);
      const cost = lc === rc ? 0 : 1;
      curr[j] = Math.min(
        prev[j] + 1,
        curr[j - 1] + 1,
        prev[j - 1] + cost,
      );
    }
    for (let j = 0; j <= right.length; j += 1) prev[j] = curr[j];
  }

  return prev[right.length];
}

function applyCaseTemplate(source, replacement) {
  const raw = String(source || "");
  const target = String(replacement || "");
  if (!raw) return target;
  if (/^[A-Z]+$/.test(raw)) return target.toUpperCase();
  if (/^[A-Z]/.test(raw) && raw.slice(1) === raw.slice(1).toLowerCase()) {
    return target.charAt(0).toUpperCase() + target.slice(1);
  }
  return target;
}

function correctLongClinicalToken(token) {
  const source = String(token || "");
  if (!/^[A-Za-z]{9,}$/.test(source)) return source;
  const lower = source.toLowerCase();
  if (CLINICAL_TERM_SET.has(lower)) return source;

  let bestTerm = "";
  let bestDistance = Number.POSITIVE_INFINITY;
  const maxDistance = lower.length >= 14 ? 3 : 2;

  for (const term of CLINICAL_TERM_DICTIONARY) {
    if (Math.abs(term.length - lower.length) > 2) continue;
    if (term[0] !== lower[0]) continue;
    const distance = levenshteinDistance(lower, term);
    if (distance > maxDistance) continue;
    const similarity = 1 - distance / Math.max(lower.length, term.length);
    if (similarity < 0.72) continue;
    if (distance < bestDistance) {
      bestDistance = distance;
      bestTerm = term;
    }
  }

  return bestTerm ? applyCaseTemplate(source, bestTerm) : source;
}

export function applyClinicalOcrHeuristics(text) {
  let out = safeText(text);
  if (!out) return "";

  for (const rule of EXACT_CLINICAL_REPLACEMENTS) {
    out = out.replace(rule.re, rule.replace);
  }

  out = out.replace(/[A-Za-z]{9,}/g, (token) => correctLongClinicalToken(token));
  return out;
}

function normalizeAlphaToken(value) {
  return safeText(value).toLowerCase().replace(/[^a-z]/g, "");
}

function tokenSimilarity(a, b) {
  const left = normalizeAlphaToken(a);
  const right = normalizeAlphaToken(b);
  if (!left || !right) return 0;
  const distance = levenshteinDistance(left, right);
  return 1 - (distance / Math.max(left.length, right.length));
}

function isHeaderLineContext(line) {
  const zoneId = safeText(line?.zoneId).toLowerCase();
  if (zoneId.includes("header")) return true;
  const zoneOrder = Number(line?.zoneOrder);
  return Number.isFinite(zoneOrder) && zoneOrder === 0;
}

function looksLikeGarbledAccountToken(token) {
  const normalized = normalizeAlphaToken(token);
  if (!normalized || normalized.length < 6) return false;
  if (normalized.startsWith("aeecrnimt")) return true;
  return tokenSimilarity(normalized, "aeecrnimt") >= 0.72;
}

function looksLikeAccountToken(token) {
  const normalized = normalizeAlphaToken(token);
  if (!normalized || normalized.length < 5) return false;
  return tokenSimilarity(normalized, "account") >= 0.66;
}

function isCorruptDobValue(text) {
  const source = safeText(text);
  if (!source) return false;
  if (/\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b/.test(source)) return false;
  return /[0-9][A-Za-z]/.test(source) || /[A-Za-z][0-9]/.test(source);
}

function shouldDropProvationHeaderNoise(text, line) {
  const source = safeText(text).replace(/\s+/g, " ").trim();
  if (!source) return false;

  const tokens = source.split(/\s+/).filter(Boolean);
  const firstToken = tokens[0] || "";
  if (looksLikeGarbledAccountToken(firstToken)) return true;

  const headerLine = isHeaderLineContext(line);
  if (headerLine) {
    if (HEADER_ACCOUNT_LABEL_RE.test(source)) return true;
    if (looksLikeAccountToken(firstToken) && /#|number|num|acct/i.test(source)) return true;
  }

  if (HEADER_DOB_BIRCH_RE.test(source) && isCorruptDobValue(source)) {
    return true;
  }

  return false;
}

function normalizeBBox(bbox) {
  const normalized = normalizeRect(bbox || { x: 0, y: 0, width: 0, height: 0 });
  if (!Number.isFinite(normalized.x) || !Number.isFinite(normalized.y)) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }
  if (!Number.isFinite(normalized.width) || !Number.isFinite(normalized.height)) {
    return { x: normalized.x, y: normalized.y, width: 0, height: 0 };
  }
  return normalized;
}

function normalizeLine(line) {
  const text = applyClinicalOcrHeuristics(safeText(line?.text)).replace(/\s+/g, " ").trim();
  const cleanedText = shouldDropProvationHeaderNoise(text, line) ? "" : text;
  return {
    text: cleanedText,
    confidence: Number.isFinite(line?.confidence)
      ? Number(line.confidence)
      : Number.isFinite(line?.conf)
        ? Number(line.conf)
        : null,
    bbox: normalizeBBox(line?.bbox),
    words: Array.isArray(line?.words) ? line.words : [],
    pageIndex: Number.isFinite(line?.pageIndex) ? Number(line.pageIndex) : 0,
    zoneId: typeof line?.zoneId === "string" ? line.zoneId : undefined,
    zoneOrder: Number.isFinite(line?.zoneOrder) ? Number(line.zoneOrder) : null,
  };
}

function normalizeLineKey(text) {
  return safeText(text)
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

function isBoilerplateLine(text) {
  if (!text) return false;
  return OCR_BOILERPLATE_PATTERNS.some((pattern) => pattern.test(text));
}

function isLikelyCaptionLine(text) {
  if (!text || text.length > 58) return false;
  if (/[.,;:!?]/.test(text)) return false;
  if (CAPTION_VERB_RE.test(text)) return false;

  const rawTokens = text.split(/\s+/).filter(Boolean);
  const hasNumericPrefix = /^[#(]?\d+[).:-]?$/.test(rawTokens[0] || "");
  if (/\d/.test(text) && !hasNumericPrefix) return false;

  const tokens = rawTokens
    .slice(hasNumericPrefix ? 1 : 0)
    .map((token) => token.replace(/[^A-Za-z]/g, "").toLowerCase())
    .filter(Boolean);
  if (tokens.length < 2 || tokens.length > 5) return false;
  const anatomyCount = tokens.filter((token) => ANATOMY_CAPTION_TOKENS.has(token)).length;
  return anatomyCount >= 2 && anatomyCount / tokens.length >= 0.5;
}

function maxOverlapRatio(lineBBox, regions) {
  const lineRect = normalizeBBox(lineBBox);
  const lineArea = Math.max(1, rectArea(lineRect));
  let maxRatio = 0;

  for (const region of Array.isArray(regions) ? regions : []) {
    const overlap = intersectionArea(lineRect, normalizeBBox(region));
    if (overlap <= 0) continue;
    const ratio = overlap / lineArea;
    if (ratio > maxRatio) maxRatio = ratio;
  }

  return maxRatio;
}

export function dedupeConsecutiveLines(lines) {
  const out = [];
  let prevKey = "";

  for (const rawLine of Array.isArray(lines) ? lines : []) {
    const line = normalizeLine(rawLine);
    if (!line.text) continue;
    const key = normalizeLineKey(line.text);
    if (key && key === prevKey) continue;
    prevKey = key;
    out.push(line);
  }

  return out;
}

/**
 * Filter OCR lines using detected figure regions and confidence gates.
 *
 * @param {Array<{text:string,confidence?:number,conf?:number,bbox?:{x:number,y:number,width:number,height:number},words?:Array,pageIndex?:number}>} lines
 * @param {Array<{x:number,y:number,width:number,height:number}>} figureRegions
 * @param {{overlapThreshold?:number,shortLowConfThreshold?:number,dropCaptions?:boolean,dropBoilerplate?:boolean}} [opts]
 * @returns {{lines:Array,dropped:Array<{line:object,reason:string,overlapRatio?:number}>}}
 */
export function filterOcrLinesDetailed(lines, figureRegions, opts = {}) {
  const overlapThreshold = Number.isFinite(opts.overlapThreshold) ? Number(opts.overlapThreshold) : 0.35;
  const shortLowConfThreshold = Number.isFinite(opts.shortLowConfThreshold)
    ? Number(opts.shortLowConfThreshold)
    : 30;
  const disableFigureOverlap = opts.disableFigureOverlap === true;
  const dropCaptions = opts.dropCaptions !== false;
  const dropBoilerplate = opts.dropBoilerplate !== false;

  const normalizedRegions = (Array.isArray(figureRegions) ? figureRegions : []).map(normalizeBBox);
  const dropped = [];
  const kept = [];

  for (const rawLine of dedupeConsecutiveLines(lines)) {
    const line = normalizeLine(rawLine);
    if (!line.text) continue;

    if (dropBoilerplate && isBoilerplateLine(line.text)) {
      dropped.push({ line, reason: "boilerplate" });
      continue;
    }

    if (dropCaptions && isLikelyCaptionLine(line.text)) {
      dropped.push({ line, reason: "caption" });
      continue;
    }

    if (!disableFigureOverlap) {
      const overlapRatio = maxOverlapRatio(line.bbox, normalizedRegions);
      if (overlapRatio > overlapThreshold) {
        dropped.push({ line, reason: "figure_overlap", overlapRatio });
        continue;
      }
    }

    const textLen = line.text.length;
    if (Number.isFinite(line.confidence) && line.confidence < shortLowConfThreshold && textLen < 6) {
      dropped.push({ line, reason: "low_conf_short" });
      continue;
    }

    kept.push(line);
  }

  return {
    lines: dedupeConsecutiveLines(kept),
    dropped,
  };
}

export function filterOcrLines(lines, figureRegions, opts = {}) {
  return filterOcrLinesDetailed(lines, figureRegions, opts).lines;
}

export function composeOcrPageText(lines) {
  const normalized = dedupeConsecutiveLines(lines);
  const out = [];
  let previousZone = null;

  for (const line of normalized) {
    const text = applyClinicalOcrHeuristics(line.text).trim();
    if (!text) continue;

    const zoneKey = Number.isFinite(line.zoneOrder)
      ? `zone:${line.zoneOrder}`
      : (typeof line.zoneId === "string" && line.zoneId.trim())
        ? `zone:${line.zoneId}`
        : null;
    if (out.length && zoneKey && previousZone && zoneKey !== previousZone) {
      out.push("");
    }
    out.push(text);
    if (zoneKey) previousZone = zoneKey;
  }

  return out.join("\n");
}
