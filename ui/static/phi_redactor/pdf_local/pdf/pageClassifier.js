import { clamp01, estimateCompletenessConfidence } from "./layoutAnalysis.js";

export const DEFAULT_CLASSIFIER_THRESHOLDS = Object.freeze({
  charCount: 80,
  singleCharItemRatio: 0.55,
  nonPrintableRatio: 0.08,
  alphaRatioMin: 0.38,
  medianTokenLenMin: 2.2,
  imageOpCount: 5,
  imageTextCharMax: 1800,
  overlapRatio: 0.08,
  contaminationScore: 0.24,
  completenessConfidence: 0.72,
  classifierDecisionScore: 0.5,
  nativeTextDensityBypass: 0.0022,
  nativeTextDensityCharFloor: 900,
  nativeTextDensityAlphaFloor: 0.55,
  fragmentMinLineCount: 8,
  fragmentLineLenMax: 30,
  fragmentWordCountMax: 6,
  fragmentOrphanLenMax: 14,
  fragmentCandidateMin: 2,
  fragmentCandidateRatio: 0.03,
  fragmentOrphanMin: 1,
  fragmentHighCandidateMin: 3,
  fragmentHighCandidateRatio: 0.06,
  backfillMinLineCount: 8,
  backfillShortLineCharMax: 25,
  backfillShortLineRatio: 0.3,
  backfillOrphanCharMax: 34,
  backfillOrphanSignatureMin: 1,
  backfillRowFragmentMin: 2,
  backfillRowFragmentRatio: 0.06,
  backfillAbruptLineCharMax: 72,
  backfillModerateDensityMin: 0.0011,
  backfillModerateDensityMax: 0.0021,
  backfillModerateShortRatio: 0.28,
  backfillStrongSignalMin: 2,
  backfillTotalSignalMin: 3,
  backfillScoreThreshold: 3.15,
});

const BACKFILL_CONTINUATION_SUFFIX_RE = /\b(?:mg|ml|mcg|ug|mm|cm|hr|hrs|hour|hours|day|days|week|weeks|nose|mouth|care|pain|drink|drinks)\.$/i;

function computeTextQualityStats(text) {
  const trimmed = typeof text === "string" ? text.trim() : "";
  if (!trimmed) {
    return {
      alphaRatio: 0,
      medianTokenLen: 0,
    };
  }

  const tokens = trimmed.split(/\s+/).filter(Boolean);
  const chars = [...trimmed];
  const alphaCount = chars.filter((ch) => /[A-Za-z]/.test(ch)).length;

  const tokenLens = tokens
    .map((token) => token.replace(/^[^A-Za-z0-9]+|[^A-Za-z0-9]+$/g, ""))
    .filter(Boolean)
    .map((token) => token.length)
    .sort((a, b) => a - b);
  const mid = Math.floor(tokenLens.length / 2);
  const medianTokenLen = tokenLens.length
    ? (tokenLens.length % 2 ? tokenLens[mid] : (tokenLens[mid - 1] + tokenLens[mid]) / 2)
    : 0;

  return {
    alphaRatio: chars.length ? clamp01(alphaCount / chars.length) : 0,
    medianTokenLen,
  };
}

function addTextQualitySignals(scoreState, textStats, thresholds) {
  const { reasons, qualityFlags } = scoreState;
  let score = scoreState.score;

  if (textStats.alphaRatio < thresholds.alphaRatioMin) {
    score += 0.15;
    reasons.push(`low alpha ratio (${textStats.alphaRatio.toFixed(2)})`);
    qualityFlags.push("LOW_ALPHA_RATIO");
  }

  if (textStats.medianTokenLen > 0 && textStats.medianTokenLen < thresholds.medianTokenLenMin) {
    score += 0.1;
    reasons.push(`short median token length (${textStats.medianTokenLen.toFixed(1)})`);
    qualityFlags.push("SHORT_TOKENS");
  }

  return score;
}

function computeNativeFragmentationStats(text, thresholds) {
  const lines = typeof text === "string"
    ? text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
    : [];
  if (lines.length < thresholds.fragmentMinLineCount) {
    return {
      lineCount: lines.length,
      candidateCount: 0,
      orphanCount: 0,
      bridgeCount: 0,
      candidateRatio: 0,
      likelyFragmented: false,
    };
  }

  let candidateCount = 0;
  let orphanCount = 0;
  let bridgeCount = 0;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (!/^[a-z]/.test(line)) continue;
    if (!/[.?!]$/.test(line)) continue;
    if (/[:;]/.test(line)) continue;

    const words = line.split(/\s+/).filter(Boolean);
    const wordCount = words.length;
    const len = line.length;
    if (wordCount > thresholds.fragmentWordCountMax) continue;
    if (len > thresholds.fragmentLineLenMax) continue;

    candidateCount += 1;
    if (wordCount <= 2 && len <= thresholds.fragmentOrphanLenMax) {
      orphanCount += 1;
    }

    const prevLine = i > 0 ? lines[i - 1] : "";
    const nextLine = i < lines.length - 1 ? lines[i + 1] : "";
    const prevEndsSentence = /[.?!:]$/.test(prevLine);
    const nextStartsUpper = /^[A-Z]/.test(nextLine);
    if (prevEndsSentence && nextStartsUpper) {
      bridgeCount += 1;
    }
  }

  const candidateRatio = lines.length ? candidateCount / lines.length : 0;
  const likelyFragmented = (
    candidateCount >= thresholds.fragmentCandidateMin &&
    orphanCount >= thresholds.fragmentOrphanMin &&
    candidateRatio >= thresholds.fragmentCandidateRatio
  ) || (
    candidateCount >= thresholds.fragmentHighCandidateMin &&
    candidateRatio >= thresholds.fragmentHighCandidateRatio
  ) || (
    bridgeCount >= 2 && candidateCount >= 2
  );

  return {
    lineCount: lines.length,
    candidateCount,
    orphanCount,
    bridgeCount,
    candidateRatio,
    likelyFragmented,
  };
}

function isLikelyOrphanContinuationLine(line, thresholds) {
  const text = String(line || "").trim();
  if (!text) return false;
  if (text.length < 3 || text.length > thresholds.backfillOrphanCharMax) return false;
  if (/[:;]/.test(text)) return false;
  if (!/[.?!]$/.test(text)) return false;
  if (!/^[a-z0-9]/.test(text)) return false;

  const words = text.split(/\s+/).filter(Boolean);
  if (!words.length || words.length > 7) return false;
  if (BACKFILL_CONTINUATION_SUFFIX_RE.test(text)) return true;
  if (words.length <= 2) return true;
  return words.length <= 4 && /^[a-z]/.test(text);
}

function isLikelyRowFragmentBreak(currentLine, nextLine, thresholds) {
  const left = String(currentLine || "").trim();
  const right = String(nextLine || "").trim();
  if (!left || !right) return false;

  if (/[.?!:]$/.test(left)) return false;
  if (left.length < 12 || left.length > thresholds.backfillAbruptLineCharMax) return false;
  if (!/[A-Za-z]$/.test(left)) return false;

  const leftWords = left.split(/\s+/).filter(Boolean);
  if (leftWords.length < 3) return false;

  if (right.length > thresholds.backfillShortLineCharMax + 10) return false;
  if (!/^[a-z0-9]/.test(right)) return false;
  if (isLikelyOrphanContinuationLine(right, thresholds)) return true;
  const rightWords = right.split(/\s+/).filter(Boolean);
  if (rightWords.length > 4) return false;
  return /[.?!]$/.test(right);
}

function computeLineShapeStats(text, thresholds) {
  const lines = typeof text === "string"
    ? text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean)
    : [];
  if (!lines.length) {
    return {
      lineCount: 0,
      shortLineCount: 0,
      shortLineRatio: 0,
      orphanSignatureCount: 0,
      orphanSignatureRatio: 0,
      rowFragmentCount: 0,
      rowFragmentRatio: 0,
    };
  }

  let shortLineCount = 0;
  let orphanSignatureCount = 0;
  let rowFragmentCount = 0;

  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.length <= thresholds.backfillShortLineCharMax) {
      shortLineCount += 1;
    }
    if (isLikelyOrphanContinuationLine(line, thresholds)) {
      orphanSignatureCount += 1;
    }

    const nextLine = i < lines.length - 1 ? lines[i + 1] : "";
    if (isLikelyRowFragmentBreak(line, nextLine, thresholds)) {
      rowFragmentCount += 1;
    }
  }

  return {
    lineCount: lines.length,
    shortLineCount,
    shortLineRatio: lines.length ? shortLineCount / lines.length : 0,
    orphanSignatureCount,
    orphanSignatureRatio: lines.length ? orphanSignatureCount / lines.length : 0,
    rowFragmentCount,
    rowFragmentRatio: lines.length ? rowFragmentCount / lines.length : 0,
  };
}

function computeBackfillSignals({
  nativeTextDensity,
  nativeFragmentation,
  lineShape,
  thresholds,
}) {
  const fragmentationSignal = nativeFragmentation.likelyFragmented ||
    nativeFragmentation.candidateCount >= thresholds.fragmentCandidateMin;
  const orphanSignal = nativeFragmentation.orphanCount >= 1 ||
    lineShape.orphanSignatureCount >= thresholds.backfillOrphanSignatureMin;
  const shortLineSignal = lineShape.shortLineRatio >= thresholds.backfillShortLineRatio;
  const rowFragmentSignal = lineShape.rowFragmentCount >= thresholds.backfillRowFragmentMin ||
    lineShape.rowFragmentRatio >= thresholds.backfillRowFragmentRatio;
  const moderateMessyDensitySignal = nativeTextDensity >= thresholds.backfillModerateDensityMin &&
    nativeTextDensity <= thresholds.backfillModerateDensityMax &&
    lineShape.shortLineRatio >= thresholds.backfillModerateShortRatio &&
    (
      nativeFragmentation.candidateRatio >= thresholds.fragmentCandidateRatio ||
      lineShape.rowFragmentRatio >= thresholds.backfillRowFragmentRatio
    );

  const signals = {
    fragmentationSignal,
    orphanSignal,
    shortLineSignal,
    rowFragmentSignal,
    moderateMessyDensitySignal,
  };

  const enabledSignals = Object.entries(signals).filter(([, enabled]) => Boolean(enabled));
  const votes = enabledSignals.length;
  const signalWeights = {
    orphanSignal: 2.3,
    fragmentationSignal: 1.8,
    rowFragmentSignal: 1.3,
    shortLineSignal: 0.8,
    moderateMessyDensitySignal: 0.6,
  };
  const strongSignalNames = new Set(["orphanSignal", "fragmentationSignal"]);
  const strongVotes = enabledSignals.filter(([name]) => strongSignalNames.has(name)).length;
  const severityScore = enabledSignals.reduce(
    (acc, [name]) => acc + (signalWeights[name] || 0),
    0,
  );
  const reasonFlags = enabledSignals.map(([name]) => {
    if (name === "fragmentationSignal") return "BACKFILL_FRAGMENT_SIGNATURE";
    if (name === "orphanSignal") return "BACKFILL_ORPHAN_SIGNATURE";
    if (name === "shortLineSignal") return "BACKFILL_SHORT_LINE_RATIO";
    if (name === "rowFragmentSignal") return "BACKFILL_ROW_FRAGMENTATION";
    if (name === "moderateMessyDensitySignal") return "BACKFILL_MODERATE_MESSY_DENSITY";
    return `BACKFILL_${name.toUpperCase()}`;
  });

  const needsOcrBackfill = lineShape.lineCount >= thresholds.backfillMinLineCount &&
    severityScore >= thresholds.backfillScoreThreshold &&
    (
      strongVotes >= thresholds.backfillStrongSignalMin ||
      votes >= thresholds.backfillTotalSignalMin
    );

  return {
    needsOcrBackfill,
    votes,
    strongVotes,
    severityScore,
    signals,
    reasonFlags,
  };
}

function mergeThresholds(override) {
  if (!override || typeof override !== "object") return DEFAULT_CLASSIFIER_THRESHOLDS;
  return {
    ...DEFAULT_CLASSIFIER_THRESHOLDS,
    ...override,
  };
}

/**
 * Decide whether a page likely requires OCR.
 *
 * @param {{charCount:number,itemCount:number,nonPrintableRatio:number,singleCharItemRatio:number,imageOpCount?:number,overlapRatio?:number,contaminationScore?:number,completenessConfidence?:number,excludedTokenRatio?:number,pageArea?:number,nativeTextDensity?:number}} stats
 * @param {string} text
 * @param {{thresholds?:Partial<typeof DEFAULT_CLASSIFIER_THRESHOLDS>}} [opts]
 * @returns {{needsOcr:boolean,needsOcrBackfill:boolean,reason:string,confidence:number,qualityFlags:string[],completenessConfidence:number,nativeTextDensity:number}}
 */
export function classifyPage(stats, text, opts = {}) {
  const thresholds = mergeThresholds(opts.thresholds);
  const safeStats = stats || {};

  const charCount = Number(safeStats.charCount) || 0;
  const singleCharItemRatio = clamp01(Number(safeStats.singleCharItemRatio) || 0);
  const nonPrintableRatio = clamp01(Number(safeStats.nonPrintableRatio) || 0);
  const imageOpCount = Math.max(0, Number(safeStats.imageOpCount) || 0);
  const overlapRatio = clamp01(Number(safeStats.overlapRatio) || 0);
  const contaminationScore = clamp01(Number(safeStats.contaminationScore) || 0);
  const pageArea = Math.max(0, Number(safeStats.pageArea) || 0);
  const nativeTextDensity = Number.isFinite(safeStats.nativeTextDensity)
    ? Math.max(0, Number(safeStats.nativeTextDensity))
    : (pageArea > 0 ? Math.max(0, charCount / pageArea) : 0);
  const textStats = computeTextQualityStats(text);
  const nativeFragmentation = computeNativeFragmentationStats(text, thresholds);
  const lineShape = computeLineShapeStats(text, thresholds);
  const completenessConfidence = Number.isFinite(safeStats.completenessConfidence)
    ? clamp01(safeStats.completenessConfidence)
    : estimateCompletenessConfidence(safeStats, {
      excludedTokenRatio: Number(safeStats.excludedTokenRatio) || 0,
    });
  const backfill = computeBackfillSignals({
    nativeTextDensity,
    nativeFragmentation,
    lineShape,
    thresholds,
  });

  const nativeDensityBypass = pageArea > 0 &&
    charCount >= thresholds.nativeTextDensityCharFloor &&
    textStats.alphaRatio >= thresholds.nativeTextDensityAlphaFloor &&
    nativeTextDensity >= thresholds.nativeTextDensityBypass &&
    !nativeFragmentation.likelyFragmented &&
    !backfill.needsOcrBackfill;
  if (nativeDensityBypass) {
    return {
      needsOcr: false,
      needsOcrBackfill: false,
      reason: `high native text density (${nativeTextDensity.toFixed(4)} chars/unit^2)`,
      confidence: clamp01(Math.max(0.85, completenessConfidence)),
      qualityFlags: ["NATIVE_DENSE_TEXT"],
      completenessConfidence,
      nativeTextDensity,
      nativeFragmentation,
      lineShape,
      backfill,
    };
  }

  let score = 0;
  const reasons = [];
  const qualityFlags = [];

  if (charCount < thresholds.charCount) {
    score += 0.35;
    reasons.push(`low char count (${charCount})`);
    qualityFlags.push("SPARSE_TEXT");
  }

  if (singleCharItemRatio >= thresholds.singleCharItemRatio) {
    score += 0.23;
    reasons.push(`high single-char item ratio (${singleCharItemRatio.toFixed(2)})`);
    qualityFlags.push("CHAR_FRAGMENTATION");
  }

  if (nonPrintableRatio >= thresholds.nonPrintableRatio) {
    score += 0.2;
    reasons.push(`high non-printable ratio (${nonPrintableRatio.toFixed(2)})`);
    qualityFlags.push("NON_PRINTABLE_TEXT");
  }

  score = addTextQualitySignals({ score, reasons, qualityFlags }, textStats, thresholds);

  if (nativeFragmentation.likelyFragmented) {
    score += 0.58;
    reasons.push(
      `fragmented native lines (${nativeFragmentation.candidateCount}/${nativeFragmentation.lineCount}, orphan=${nativeFragmentation.orphanCount})`,
    );
    qualityFlags.push("FRAGMENTED_NATIVE_LINES");
  }

  if (backfill.needsOcrBackfill) {
    score += 0.52;
    reasons.push(
      `native backfill signals ${backfill.votes}/5 (strong=${backfill.strongVotes}, score=${backfill.severityScore.toFixed(2)})`,
    );
    qualityFlags.push(...backfill.reasonFlags);
  }

  if (imageOpCount >= thresholds.imageOpCount && charCount <= thresholds.imageTextCharMax) {
    score += 0.32;
    reasons.push(`image-heavy page (${imageOpCount} image ops) with limited text (${charCount} chars)`);
    qualityFlags.push("IMAGE_HEAVY");
  }

  if (overlapRatio >= thresholds.overlapRatio) {
    score += 0.31;
    reasons.push(`image/text overlap ratio (${overlapRatio.toFixed(2)})`);
    qualityFlags.push("IMAGE_TEXT_OVERLAP");
  }

  if (contaminationScore >= thresholds.contaminationScore) {
    score += 0.33;
    reasons.push(`contamination score (${contaminationScore.toFixed(2)})`);
    qualityFlags.push("CONTAMINATION_RISK");
  }

  if (completenessConfidence < thresholds.completenessConfidence) {
    score += 0.45;
    reasons.push(`low completeness confidence (${completenessConfidence.toFixed(2)})`);
    qualityFlags.push("LOW_COMPLETENESS");
  }

  const needsOcr = score >= thresholds.classifierDecisionScore || backfill.needsOcrBackfill;
  const confidence = clamp01(needsOcr ? score : 1 - score);

  return {
    needsOcr,
    needsOcrBackfill: backfill.needsOcrBackfill,
    reason: reasons.length ? reasons.join(", ") : "layout-safe native text",
    confidence,
    qualityFlags: [...new Set(qualityFlags)],
    completenessConfidence,
    nativeTextDensity,
    nativeFragmentation,
    lineShape,
    backfill,
  };
}

/**
 * Evaluate whether native-only extraction should be considered unsafe.
 *
 * @param {object} stats
 * @param {string} text
 * @param {{minCompletenessConfidence?:number,maxContaminationScore?:number,thresholds?:Partial<typeof DEFAULT_CLASSIFIER_THRESHOLDS>}} [opts]
 */
export function isUnsafeNativePage(stats, text, opts = {}) {
  const thresholds = mergeThresholds(opts.thresholds);
  const classification = classifyPage(stats, text, { thresholds });
  if (
    classification.qualityFlags?.includes("NATIVE_DENSE_TEXT") &&
    !classification.qualityFlags?.includes("FRAGMENTED_NATIVE_LINES")
  ) {
    return {
      unsafe: false,
      classification,
      contaminationScore: clamp01(Number(stats?.contaminationScore) || 0),
      completenessConfidence: classification.completenessConfidence,
    };
  }
  const maxContaminationScore = Number.isFinite(opts.maxContaminationScore)
    ? clamp01(opts.maxContaminationScore)
    : thresholds.contaminationScore;
  const minCompletenessConfidence = Number.isFinite(opts.minCompletenessConfidence)
    ? clamp01(opts.minCompletenessConfidence)
    : thresholds.completenessConfidence;

  const contaminationScore = clamp01(Number(stats?.contaminationScore) || 0);
  const completenessConfidence = classification.completenessConfidence;

  const unsafe = Boolean(
    classification.needsOcr ||
    contaminationScore >= maxContaminationScore ||
    completenessConfidence < minCompletenessConfidence,
  );

  return {
    unsafe,
    classification,
    contaminationScore,
    completenessConfidence,
  };
}

/**
 * Resolve the requested source for a page after user/global overrides.
 *
 * @param {{classification:{needsOcr:boolean}, userOverride?:'force_native'|'force_ocr'}} page
 * @param {{forceOcrAll?:boolean}} [opts]
 * @returns {{source:'native'|'ocr', reason:string}}
 */
export function resolvePageSource(page, opts = {}) {
  if (opts.forceOcrAll) {
    return { source: "ocr", reason: "OCR all pages enabled" };
  }
  if (page.userOverride === "force_ocr") {
    return { source: "ocr", reason: "user override: force OCR" };
  }
  if (page.userOverride === "force_native") {
    return { source: "native", reason: "user override: force native" };
  }
  if (page.classification?.needsOcrBackfill) {
    return { source: "native", reason: "native extraction flagged for OCR backfill repair" };
  }
  if (page.classification?.needsOcr) {
    return { source: "ocr", reason: page.classification.reason };
  }
  return { source: "native", reason: page.classification?.reason || "layout-safe native text" };
}
