import { RedactionType } from "./types.js";

function clamp01(value) {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

function addSpan(spans, start, end, type, confidence) {
  if (!Number.isFinite(start) || !Number.isFinite(end)) return;
  if (end <= start) return;
  spans.push({
    start,
    end,
    type,
    confidence: clamp01(confidence),
    source: "regex",
  });
}

function pushMatchesFromCapture(spans, text, regex, groupIndex, type, confidence) {
  regex.lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    const groupText = match[groupIndex];
    if (!groupText) {
      if (regex.lastIndex === match.index) regex.lastIndex += 1;
      continue;
    }

    const offsetInMatch = match[0].indexOf(groupText);
    const start = match.index + Math.max(0, offsetInMatch);
    const end = start + groupText.length;
    addSpan(spans, start, end, type, confidence);

    if (regex.lastIndex === match.index) regex.lastIndex += 1;
  }
}

function pushMatches(spans, text, regex, type, confidence) {
  regex.lastIndex = 0;
  let match;
  while ((match = regex.exec(text)) !== null) {
    addSpan(spans, match.index, match.index + match[0].length, type, confidence);
    if (regex.lastIndex === match.index) regex.lastIndex += 1;
  }
}

const EMAIL_RE = /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;
const PHONE_RE = /(?:^|[^\d])((?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]?\d{3}[-.\s]?\d{4}(?:\s*(?:x|ext\.?|extension)\s*\d{1,6})?)(?=$|[^\d])/gim;
const SSN_DASHED_RE = /(?:^|[^\d])(\d{3}-\d{2}-\d{4})(?=$|[^\d])/g;
const SSN_ANCHORED_RE = /\b(?:SSN|SOCIAL SECURITY(?: NUMBER)?)\b[\s:#-]*(\d{9}|\d{3}-\d{2}-\d{4})\b/gi;
const DATE_NUMERIC_RE = /(?:^|[^\d])((?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{2}|\d{4}))(?=$|[^\d])/g;
const DATE_ISO_RE = /\b(\d{4}-\d{2}-\d{2})\b/g;
const DATE_MONTH_TEXT_RE = /\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4})\b/gi;
const MRN_RE = /\b(?:MRN|MEDICAL RECORD(?: NUMBER)?|PATIENT ID)\b[\s:#-]*([A-Z0-9][A-Z0-9-]{3,})\b/gi;
const ACCOUNT_RE = /\b(?:ACCOUNT(?: NUMBER| NO\.?)?|ACCT(?: NUMBER| NO\.?)?|ACCOUNT#|ACCT#)\b[\s:#-]*([A-Z0-9][A-Z0-9-]{3,})\b/gi;
const DOB_ANCHORED_RE = /\b(?:DOB|DATE OF BIRTH)\b[\s:#-]*((?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:\d{2}|\d{4})|\d{4}-\d{2}-\d{2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{2,4})\b/gi;
const URL_RE = /\b(?:https?:\/\/|www\.)[^\s<>"']+/gi;
const IPV4_RE = /(?:^|[^\d])((?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})(?=$|[^\d])/g;

/**
 * High-recall regex baseline PHI detector.
 *
 * @param {string} text
 * @returns {Array<{start:number,end:number,type:string,confidence:number,source:'regex'}>}
 */
export function detectRegexPhi(text) {
  if (typeof text !== "string" || !text.length) return [];

  const spans = [];

  pushMatches(spans, text, EMAIL_RE, RedactionType.EMAIL, 0.98);
  pushMatchesFromCapture(spans, text, PHONE_RE, 1, RedactionType.PHONE, 0.95);
  pushMatchesFromCapture(spans, text, SSN_DASHED_RE, 1, RedactionType.SSN, 0.98);
  pushMatchesFromCapture(spans, text, SSN_ANCHORED_RE, 1, RedactionType.SSN, 0.97);

  pushMatchesFromCapture(spans, text, DATE_NUMERIC_RE, 1, RedactionType.DATE, 0.83);
  pushMatchesFromCapture(spans, text, DATE_ISO_RE, 1, RedactionType.DATE, 0.82);
  pushMatchesFromCapture(spans, text, DATE_MONTH_TEXT_RE, 1, RedactionType.DATE, 0.82);
  pushMatchesFromCapture(spans, text, DOB_ANCHORED_RE, 1, RedactionType.DATE, 0.9);

  pushMatchesFromCapture(spans, text, MRN_RE, 1, RedactionType.MRN, 0.93);
  pushMatchesFromCapture(spans, text, ACCOUNT_RE, 1, RedactionType.ACCOUNT, 0.92);

  pushMatches(spans, text, URL_RE, RedactionType.URL, 0.95);
  pushMatchesFromCapture(spans, text, IPV4_RE, 1, RedactionType.IP, 0.9);

  spans.sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    return b.end - a.end;
  });

  return spans;
}
