#!/usr/bin/env node
/**
 * Test PHI Redaction using DistilBERT ONNX model (same as browser UI)
 *
 * This script mirrors the client-side redactor.worker.js pipeline:
 * 1. Load DistilBERT ONNX model via Transformers.js
 * 2. Run NER inference on procedure notes
 * 3. Apply regex detectors for headers/dates
 * 4. Apply veto layer (protectedVeto.js)
 * 5. Output comparison of original vs redacted
 *
 * Usage:
 *   cd ops/tools/phi_test_node
 *   npm install
 *   node test_phi_redaction.mjs [--count N] [--seed N] [--output FILE]
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { pipeline, env } from "@huggingface/transformers";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PROJECT_ROOT = path.resolve(__dirname, "../..");

// =============================================================================
// Configuration
// =============================================================================

const MODEL_PATH = path.join(
  PROJECT_ROOT,
  "ui/static/phi_redactor/vendor/phi_distilbert_ner"
);
const PROTECTED_TERMS_PATH = path.join(MODEL_PATH, "protected_terms.json");
const GOLDEN_DIR = path.join(PROJECT_ROOT, "data/knowledge/golden_extractions");

// Configure Transformers.js for local model
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = MODEL_PATH;

// =============================================================================
// Import veto layer (ES module)
// =============================================================================

const VETO_PATH = path.join(
  PROJECT_ROOT,
  "ui/static/phi_redactor/protectedVeto.js"
);

// Dynamic import for ES module
const { applyVeto } = await import(VETO_PATH);

// =============================================================================
// Regex Patterns (from redactor.worker.js)
// =============================================================================

// Requires colon/dash delimiter to avoid matching "patient went into" as a name
// Also matches "Pt: C. Rodriguez" (Initial. Lastname format) and "Pt: White, E." (Last, Initial format)
const PATIENT_HEADER_RE =
  /(?:Patient(?:\s+Name)?|Pt\.?(?:\s+Name)?|Pat\.?|Name|Subject)\s*[:\-]\s*([A-Z][a-z]+\s*,\s*[A-Z]\.?|[A-Z]\.?\s+[A-Z][a-z]+|[A-Z][a-z]+\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?|[A-Z][a-z]+\s+[A-Z]'?[A-Za-z]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)/gim;

// Greedy header scan: capture full name after Name/Patient labels until a terminator.
const HEADER_NAME_LABEL_RE =
  /\b(?:Patient(?:\s+Name)?|Pt\.?(?:\s+Name)?|Pat\.?|Name|Subject)\s*[:\-]/gi;
const HEADER_NAME_TERMINATOR_RE =
  /\b(?:DOB|DOD|Date\s+of\s+Birth|Birth\s*Date|Birthdate|Date|MRN|ID|EDIPI|Age|Sex|Gender|Phone|Address|Procedure|Indication(?:s)?|Findings|Impression|Assessment|Plan|History|Technique|Diagnosis)\b(?=\s*[:#-]|\s+\d)/i;
const HEADER_NAME_TOKENS_RE =
  /^\s*([A-Z][A-Za-z'.-]*(?:\s*,\s*[A-Z][A-Za-z'.-]*)?(?:\s+[A-Z][A-Za-z'.-]*)*)/;
const HEADER_NAME_STOPWORDS = new Set([
  "procedure", "indication", "indications", "findings", "impression",
  "assessment", "plan", "history", "technique", "diagnosis",
  "date", "dob", "dod", "mrn", "id", "age", "sex", "gender",
  "phone", "address", "medications", "medication", "anesthesia", "sedation", "general"
]);

// IMPORTANT: Must contain at least one digit to avoid matching medical acronyms like "rEBUS"
const MRN_RE =
  /\b(?:MRN|MR|Medical\s*Record|Patient\s*ID|ID|EDIPI|DOD\s*ID)\s*[:\#]?\s*([A-Z0-9\-]*\d[A-Z0-9\-]*)\b/gi;

// Matches: MRN with spaces like "A92 555" or "AB 123 456" (2-3 groups of alphanumerics)
const MRN_SPACED_RE =
  /\b(?:MRN|MR|Medical\s*Record|Patient\s*ID|ID)\s*[:\#]?\s*([A-Z0-9]{2,5}\s+[A-Z0-9]{2,5}(?:\s+[A-Z0-9]{2,5})?)\b/gi;

// IMPORTANT: Excludes "The patient", "A patient", etc. via negative lookahead
const INLINE_PATIENT_NAME_RE =
  /\b(?!(?:The|A|An)\s+(?:patient|subject|candidate|individual|person)\b)([A-Z][a-z]+(?:(?:\s+|,\s*)[A-Z]\.?)?(?:\s+|,\s*)[A-Z][a-z]+),?\s+(?:(?:is|was)\s+)?(?:a\s+)?(?:\d{1,3}\s*-?\s*(?:year|yr|y\/?o|yo)[\s-]*old|aged?\s+\d{1,3})\b/gi;

// Matches names after procedural verbs: "performed on Robert Chen", "procedure for Jane Doe"
// IMPORTANT: Case-sensitive (no 'i' flag) to avoid matching lowercase words like "the core"
const PROCEDURAL_NAME_RE =
  /\b(?:performed|completed|done|scheduled|procedure)\s+(?:on|for)\s+([A-Z][a-z]+(?:'[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:'[A-Z][a-z]+)?)\b/g;

const PT_NAME_MRN_RE = /\bpt\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+mrn\s+\d+/gi;

// Matches: "PT Name" standalone (without MRN) in unstructured text
// E.g., "PT James Wilson ID MRN..."
const PT_STANDALONE_RE =
  /\bPT\s+([A-Z][a-z]+(?:'[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:'[A-Z][a-z]+)?)?)\b/g;

// Matches: "Mr./Mrs./Ms./Miss Smith" or "Mr. John Smith" or "Mr. O'Brien"
// IMPORTANT: Case-sensitive for name capture to avoid consuming lowercase verbs like "underwent"
// Only matches names starting with capital letters, supports apostrophe surnames (O'Brien, D'Angelo)
const TITLE_NAME_RE =
  /\b(?:Mr|Mrs|Ms|Miss|Mister|Missus)\.?\s+([A-Z][a-z]*(?:'[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:'[A-Z][a-z]+)?)?)\b/g;

// Matches narrative names: "for [Name]" in context like "bronch for Logan Roy massive bleeding"
// Requires name to be followed by common clinical words to reduce false positives
const NARRATIVE_FOR_NAME_RE =
  /\bfor\s+([A-Z][a-z]+(?:'[A-Z][a-z]+)?(?:\s+[A-Z][a-z]+(?:'[A-Z][a-z]+)?)?)\s+(?:who|with|has|had|massive|severe|acute|chronic|presenting|underwent|scheduled|referred)\b/g;

// Matches de-identification placeholders/artifacts: "<PERSON>", "[Name]", "[REDACTED]", "***NAME***"
// These appear in pre-processed or dirty data and should be redacted to prevent leakage
const PLACEHOLDER_NAME_RE =
  /(?:Patient(?:\s+Name)?|Pt(?:\s+Name)?|Name|Subject)\s*[:\-]?\s*(<[A-Z]+>|\[[A-Za-z_]+\]|\*{2,}[A-Za-z_]+\*{2,})/gi;

// Matches "Patient 69F" shorthand pattern (no colon, alphanumeric identifier)
// Fallback for non-standard patient identifiers like age/gender clusters
const PATIENT_SHORTHAND_RE =
  /\bPatient\s+(\d{1,3}\s*[MF](?:emale)?)\b/gi;

// Matches names at sentence start followed by clinical verbs: "Robert Smith has a LLL nodule..."
// Must be at sentence start (after period/newline) and followed by clinical context
// Requires both first and last name to reduce false positives
const SENTENCE_START_NAME_RE =
  /(?:^|[.!?]\s+|\n\s*)([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:has|had|is|was|presents|presented|underwent|undergoing|needs|needed|required|denies|denied|reports|reported|complained|complains|notes|noted|states|stated|describes|described|exhibits|exhibited|demonstrates|demonstrated|developed|shows|showed|appears|appeared)\b/gm;

// Matches names at very start of line/document followed by period: "Kimberly Garcia. Ion Bronchoscopy."
// For notes that begin directly with patient name without a header label
const LINE_START_NAME_RE =
  /^([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[.,]/gm;

// Matches informal/lowercase names followed by "here for": "jason phillips here for right lung lavage"
// Case-insensitive to catch dictation notes where names aren't capitalized
// The "here for" phrase is a strong indicator of patient context in informal notes
const INFORMAL_NAME_HERE_RE =
  /^([a-z]+\s+[a-z]+)\s+here\s+for\b/gim;

// Matches underscore-wrapped template placeholders: "___Lisa Anderson___", "___BB-8472-K___", "___03/19/1961___"
// These appear in de-identification templates where PHI is wrapped in triple underscores
const UNDERSCORE_NAME_RE =
  /___([A-Za-z][A-Za-z\s]+[A-Za-z])___/g;
const UNDERSCORE_ID_RE =
  /___([A-Z0-9][A-Z0-9\-]+)___/gi;
const UNDERSCORE_DATE_RE =
  /___(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})___/g;

// Matches lowercase honorific + name patterns: "mrs lopez", "mr harris", "ms johnson"
// Case-insensitive to catch dictation/informal notes where names aren't capitalized
const TITLE_NAME_LOWERCASE_RE =
  /\b((?:mr|mrs|ms|miss|mister|missus)\.?\s+[a-z]+(?:\s+[a-z]+)?)\b/gi;

// Matches standalone first names followed by clinical verbs: "liam came in choking", "Frank underwent a procedure"
// Requires first name to be followed by a verb to reduce false positives
const FIRST_NAME_CLINICAL_RE =
  /\b([A-Z][a-z]+)\s+(?:came|went|underwent|presents|presented|needs|needed|required|complains|complained|reports|reported|developed|has|had|is|was)\b/g;

// Matches names after "Procedure note" header: "Procedure note Justin Fowler 71M."
// Common format in procedure documentation headers
const PROCEDURE_NOTE_NAME_RE =
  /\bProcedure\s+note\s+([A-Z][a-z]+\s+[A-Z][a-z]+)\b/gi;

// Matches alphanumeric MRN patterns without prefix: "LM-9283", "AB-1234-K"
// For identifiers that look like MRNs but lack the "MRN:" prefix
const STANDALONE_ALPHANUMERIC_ID_RE =
  /\b([A-Z]{2,3}-\d{3,6}(?:-[A-Z0-9])?)\b/g;

// Matches "pt [Name]" prefix patterns in informal notes: "pt Juan C R long term trach"
// Supports mixed case and middle initials
const PT_LOWERCASE_NAME_RE =
  /\bpt\s+([A-Za-z][a-z]*(?:\s+[A-Z])?(?:\s+[A-Z])?(?:\s+[A-Za-z][a-z]+)?)\b/gi;

// Matches lowercase full names at start of line followed by date: "oscar godsey 5/15/19"
// Common in dictation notes where names aren't capitalized
const LOWERCASE_NAME_DATE_RE =
  /^([a-z]+\s+[a-z]+)\s+\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}/gim;

// Matches lowercase names followed by age/gender: "barbara kim 69 female"
// Common format in informal/dictated notes
const LOWERCASE_NAME_AGE_GENDER_RE =
  /^([a-z]+\s+[a-z]+)\s+\d{1,3}\s*(?:year|yr|y\/?o|yo|male|female|m|f)\b/gim;

// Matches lowercase names followed by "here to" (variant of "here for"): "gilbert barkley here to get his stents out"
const INFORMAL_NAME_HERE_TO_RE =
  /^([a-z]+\s+[a-z]+)\s+here\s+to\b/gim;

// Matches lowercase names followed by "note": "michael foster note hard to read"
// Common in resident notes where patient name precedes "note"
const LOWERCASE_NAME_NOTE_RE =
  /^([a-z]+\s+[a-z]+)\s+note\b/gim;

const DATE_DDMMMYYYY_RE =
  /\b(\d{1,2}[-\s]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s]?\d{2,4})\b/gi;

const DATE_DDMMMYYYY_SPACED_RE =
  /\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\b/gi;

const DATE_SLASH_RE = /\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b/g;

const DATE_ISO_RE = /\b(\d{4}[-\/]\d{1,2}[-\/]\d{1,2})\b/g;

const DOB_HEADER_RE =
  /\b(?:DOB|Date\s+of\s+Birth|Birth\s*Date|Birthdate)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}[-\s]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s,]?\s*\d{1,2}[-,\s]+\d{2,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}[-,\s]+\d{2,4})\b/gi;

// Facility / institution patterns (treated as PHI → GEO)
const FACILITY_NAME_RE =
  /\b(?:The\s+)?(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*|&|of|the|and|for|at|de|la|st\.?|st|saint|mount|mt)){0,12}\s+(?:Medical\s+(?:Center|Centre|Pavilion)|Hospital\s+Center|Hospital|Hospitals|Clinic|Clinics|Health\s+(?:System|Center)|Cancer\s+Center|Institute|Clinical\s+Center)\b(?!\s+Review\b)(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)){0,2})?/g;

const FACILITY_CAMEL_HEALTH_RE =
  /\b[A-Z][A-Za-z]+Health\b(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)){0,6}\b/g;

const FACILITY_ENDING_HEALTH_RE =
  /\b(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*|&|of|the|and)){1,10}\s+(?:Health|Healthcare)\b/g;

const STATE_MEDICINE_RE =
  /\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming)\s+Medicine\b/g;

// =============================================================================
// Helpers
// =============================================================================

function resetRegex(...regexes) {
  for (const re of regexes) re.lastIndex = 0;
}

function isFollowedByCredentials(text, matchEnd) {
  const after = text.slice(matchEnd, Math.min(text.length, matchEnd + 40));
  return /^,?\s*(?:MD|DO|RN|RT|PA|NP|CRNA|PhD|FCCP|DAABIP)\b/i.test(after);
}

function isPrecededByProviderContext(text, matchStart) {
  const before = text.slice(Math.max(0, matchStart - 60), matchStart);
  return /\b(?:Dr\.?|Attending|Fellow|Proceduralist|Operator|Surgeon|Anesthesiologist|RN|RT|Assistant|Staff|Proctored\s+by|Supervised\s+by|Performed\s+by|Dictated\s+by|Reviewed\s+by)\s*:?\s*$/i.test(
    before
  );
}

// =============================================================================
// Regex Detection (from redactor.worker.js)
// =============================================================================

function runRegexDetectors(text) {
  const spans = [];

  resetRegex(
    PATIENT_HEADER_RE,
    HEADER_NAME_LABEL_RE,
    FACILITY_NAME_RE,
    FACILITY_CAMEL_HEALTH_RE,
    FACILITY_ENDING_HEALTH_RE,
    STATE_MEDICINE_RE,
    MRN_RE,
    MRN_SPACED_RE,
    INLINE_PATIENT_NAME_RE,
    PROCEDURAL_NAME_RE,
    PT_NAME_MRN_RE,
    PT_STANDALONE_RE,
    TITLE_NAME_RE,
    NARRATIVE_FOR_NAME_RE,
    PLACEHOLDER_NAME_RE,
    PATIENT_SHORTHAND_RE,
    SENTENCE_START_NAME_RE,
    LINE_START_NAME_RE,
    INFORMAL_NAME_HERE_RE,
    UNDERSCORE_NAME_RE,
    UNDERSCORE_ID_RE,
    UNDERSCORE_DATE_RE,
    TITLE_NAME_LOWERCASE_RE,
    FIRST_NAME_CLINICAL_RE,
    PROCEDURE_NOTE_NAME_RE,
    STANDALONE_ALPHANUMERIC_ID_RE,
    PT_LOWERCASE_NAME_RE,
    LOWERCASE_NAME_DATE_RE,
    LOWERCASE_NAME_AGE_GENDER_RE,
    INFORMAL_NAME_HERE_TO_RE,
    LOWERCASE_NAME_NOTE_RE,
    DATE_DDMMMYYYY_RE,
    DATE_DDMMMYYYY_SPACED_RE,
    DATE_SLASH_RE,
    DATE_ISO_RE,
    DOB_HEADER_RE
  );

  // 1) Patient header names
  for (const match of text.matchAll(PATIENT_HEADER_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        const start = match.index + groupOffset;
        const end = start + nameGroup.length;
        if (!isFollowedByCredentials(text, end)) {
          spans.push({
            start,
            end,
            label: "PATIENT",
            score: 1.0,
            source: "regex_patient_header",
          });
        }
      }
    }
  }

  // 1b) Greedy header scan: "Name: Booth Mark Johnson DOB: 1/2/1980"
  for (const match of text.matchAll(HEADER_NAME_LABEL_RE)) {
    if (match.index == null) continue;
    const labelEnd = match.index + match[0].length;
    const after = text.slice(labelEnd);
    const lineBreak = after.search(/[\r\n]/);
    const lineSlice = lineBreak === -1 ? after : after.slice(0, lineBreak);

    const termIdx = lineSlice.search(HEADER_NAME_TERMINATOR_RE);
    const candidate = termIdx === -1 ? lineSlice : lineSlice.slice(0, termIdx);
    if (!candidate) continue;

    const nameMatch = candidate.match(HEADER_NAME_TOKENS_RE);
    if (!nameMatch) continue;

    const nameGroup = nameMatch[1];
    const fullMatch = nameMatch[0];
    const groupOffset = fullMatch.indexOf(nameGroup);
    if (groupOffset !== -1) {
      spans.push({
        start: labelEnd + groupOffset,
        end: labelEnd + groupOffset + nameGroup.length,
        label: "PATIENT",
        score: 1.0,
        source: "regex_header_greedy",
      });
    }
  }

  // 1c) Placeholder/artifact names: "<PERSON>", "[Name]", "***NAME***"
  for (const match of text.matchAll(PLACEHOLDER_NAME_RE)) {
    const fullMatch = match[0];
    const placeholderGroup = match[1];
    const groupOffset = fullMatch.indexOf(placeholderGroup);
    if (groupOffset !== -1 && match.index != null) {
      spans.push({
        start: match.index + groupOffset,
        end: match.index + groupOffset + placeholderGroup.length,
        label: "PATIENT",
        score: 1.0,
        source: "regex_placeholder",
      });
    }
  }

  // 1d) Patient shorthand: "Patient 69F" (age/gender cluster as identifier)
  for (const match of text.matchAll(PATIENT_SHORTHAND_RE)) {
    const fullMatch = match[0];
    const shorthandGroup = match[1];
    const groupOffset = fullMatch.indexOf(shorthandGroup);
    if (groupOffset !== -1 && match.index != null) {
      spans.push({
        start: match.index + groupOffset,
        end: match.index + groupOffset + shorthandGroup.length,
        label: "PATIENT",
        score: 0.9,
        source: "regex_shorthand",
      });
    }
  }

  // 2) MRN patterns
  for (const match of text.matchAll(MRN_RE)) {
    const idGroup = match[1];
    const fullMatch = match[0];
    if (idGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(idGroup);
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + idGroup.length,
          label: "ID",
          score: 1.0,
          source: "regex_mrn",
        });
      }
    }
  }

  // 2a) MRN with spaces: "A92 555" or "AB 123 456"
  for (const match of text.matchAll(MRN_SPACED_RE)) {
    const idGroup = match[1];
    const fullMatch = match[0];
    if (idGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(idGroup);
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + idGroup.length,
          label: "ID",
          score: 1.0,
          source: "regex_mrn_spaced",
        });
      }
    }
  }

  // 3) Inline patient names
  for (const match of text.matchAll(INLINE_PATIENT_NAME_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const end = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, end) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end,
          label: "PATIENT",
          score: 0.95,
          source: "regex_inline_name",
        });
      }
    }
  }

  // 3a) Procedural verb + name: "performed on Robert Chen", "procedure for Jane Doe"
  for (const match of text.matchAll(PROCEDURAL_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        if (
          !isFollowedByCredentials(text, nameEnd) &&
          !isPrecededByProviderContext(text, match.index)
        ) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.9,
            source: "regex_procedural_name",
          });
        }
      }
    }
  }

  // 4) Pt Name MRN pattern
  for (const match of text.matchAll(PT_NAME_MRN_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.toLowerCase().indexOf(nameGroup.toLowerCase());
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + nameGroup.length,
          label: "PATIENT",
          score: 0.95,
          source: "regex_pt_name_mrn",
        });
      }
    }
  }

  // 5) Title names (Mr./Mrs./Ms.)
  for (const match of text.matchAll(TITLE_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const end = match.index + fullMatch.length;
      if (
        !isFollowedByCredentials(text, end) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        const groupOffset = fullMatch.indexOf(nameGroup);
        if (groupOffset !== -1) {
          spans.push({
            start: match.index + groupOffset,
            end: match.index + groupOffset + nameGroup.length,
            label: "PATIENT",
            score: 0.9,
            source: "regex_title_name",
          });
        }
      }
    }
  }

  // 5a) PT standalone: "PT James Wilson" (without MRN pattern)
  for (const match of text.matchAll(PT_STANDALONE_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        if (
          !isFollowedByCredentials(text, nameEnd) &&
          !isPrecededByProviderContext(text, match.index)
        ) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.9,
            source: "regex_pt_standalone",
          });
        }
      }
    }
  }

  // 5b) Narrative "for [Name]" pattern: "bronch for Logan Roy massive bleeding"
  for (const match of text.matchAll(NARRATIVE_FOR_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        if (
          !isFollowedByCredentials(text, nameEnd) &&
          !isPrecededByProviderContext(text, match.index)
        ) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.85,
            source: "regex_narrative_for",
          });
        }
      }
    }
  }

  // 5c) Sentence-start name pattern: "Robert Smith has a LLL nodule..."
  for (const match of text.matchAll(SENTENCE_START_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        if (
          !isFollowedByCredentials(text, nameEnd) &&
          !isPrecededByProviderContext(text, match.index)
        ) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.85,
            source: "regex_sentence_start",
          });
        }
      }
    }
  }

  // 5d) Line-start name pattern: "Kimberly Garcia. Ion Bronchoscopy." (name at very start of line)
  for (const match of text.matchAll(LINE_START_NAME_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, nameEnd) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.8,
          source: "regex_line_start",
        });
      }
    }
  }

  // 5e) Informal lowercase names: "jason phillips here for right lung lavage"
  for (const match of text.matchAll(INFORMAL_NAME_HERE_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, nameEnd) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.85,
          source: "regex_informal_here",
        });
      }
    }
  }

  // 5f) Underscore-wrapped template names: "___Lisa Anderson___"
  for (const match of text.matchAll(UNDERSCORE_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      spans.push({
        start: match.index,
        end: match.index + fullMatch.length,
        label: "PATIENT",
        score: 1.0,
        source: "regex_underscore_name",
      });
    }
  }

  // 5g) Underscore-wrapped IDs: "___BB-8472-K___"
  for (const match of text.matchAll(UNDERSCORE_ID_RE)) {
    const idGroup = match[1];
    const fullMatch = match[0];
    if (idGroup && match.index != null) {
      spans.push({
        start: match.index,
        end: match.index + fullMatch.length,
        label: "ID",
        score: 1.0,
        source: "regex_underscore_id",
      });
    }
  }

  // 5h) Underscore-wrapped dates: "___03/19/1961___"
  for (const match of text.matchAll(UNDERSCORE_DATE_RE)) {
    const dateGroup = match[1];
    const fullMatch = match[0];
    if (dateGroup && match.index != null) {
      spans.push({
        start: match.index,
        end: match.index + fullMatch.length,
        label: "DATE",
        score: 1.0,
        source: "regex_underscore_date",
      });
    }
  }

  // 5i) Lowercase honorific + name: "mrs lopez", "mr harris"
  for (const match of text.matchAll(TITLE_NAME_LOWERCASE_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const matchEnd = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, matchEnd) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end: matchEnd,
          label: "PATIENT",
          score: 0.85,
          source: "regex_title_lowercase",
        });
      }
    }
  }

  // 5j) Standalone first name followed by clinical verb: "liam came in", "Frank underwent"
  for (const match of text.matchAll(FIRST_NAME_CLINICAL_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, nameEnd) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.75,
          source: "regex_first_name_clinical",
        });
      }
    }
  }

  // 5k) Procedure note header names: "Procedure note Justin Fowler 71M."
  for (const match of text.matchAll(PROCEDURE_NOTE_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.toLowerCase().indexOf(nameGroup.toLowerCase());
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        if (!isFollowedByCredentials(text, nameEnd)) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.95,
            source: "regex_procedure_note",
          });
        }
      }
    }
  }

  // 5l) Standalone alphanumeric IDs: "LM-9283", "AB-1234-K"
  for (const match of text.matchAll(STANDALONE_ALPHANUMERIC_ID_RE)) {
    const idGroup = match[1];
    if (idGroup && match.index != null) {
      const ctx = text.slice(
        Math.max(0, match.index - 30),
        Math.min(text.length, match.index + idGroup.length + 30)
      ).toLowerCase();
      // Only match if in patient/ID context, not device context
      if (
        /\b(?:mrn|patient|id|record|chart)\b/i.test(ctx) ||
        !/\b(?:model|scope|device|system|platform)\b/i.test(ctx)
      ) {
        spans.push({
          start: match.index,
          end: match.index + idGroup.length,
          label: "ID",
          score: 0.8,
          source: "regex_alphanumeric_id",
        });
      }
    }
  }

  // 5m) "pt [Name]" patterns in informal notes: "pt Juan C R long term trach"
  for (const match of text.matchAll(PT_LOWERCASE_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.toLowerCase().indexOf(nameGroup.toLowerCase());
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        if (
          !isFollowedByCredentials(text, nameEnd) &&
          !isPrecededByProviderContext(text, match.index)
        ) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.9,
            source: "regex_pt_lowercase",
          });
        }
      }
    }
  }

  // 5n) Lowercase names followed by date: "oscar godsey 5/15/19"
  for (const match of text.matchAll(LOWERCASE_NAME_DATE_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, nameEnd) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.85,
          source: "regex_lowercase_date",
        });
      }
    }
  }

  // 5o) Lowercase names followed by age/gender: "barbara kim 69 female"
  for (const match of text.matchAll(LOWERCASE_NAME_AGE_GENDER_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, nameEnd) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.85,
          source: "regex_lowercase_age_gender",
        });
      }
    }
  }

  // 5p) Lowercase names followed by "here to": "gilbert barkley here to get his stents out"
  for (const match of text.matchAll(INFORMAL_NAME_HERE_TO_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, nameEnd) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.85,
          source: "regex_informal_here_to",
        });
      }
    }
  }

  // 5q) Lowercase names followed by "note": "michael foster note hard to read"
  for (const match of text.matchAll(LOWERCASE_NAME_NOTE_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      if (
        !isFollowedByCredentials(text, nameEnd) &&
        !isPrecededByProviderContext(text, match.index)
      ) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.85,
          source: "regex_lowercase_note",
        });
      }
    }
  }

  // 6) DOB header dates
  for (const match of text.matchAll(DOB_HEADER_RE)) {
    const dateGroup = match[1];
    const fullMatch = match[0];
    if (dateGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(dateGroup);
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + dateGroup.length,
          label: "DATE",
          score: 1.0,
          source: "regex_dob",
        });
      }
    }
  }

  // 7) Space-separated dates: "13 Feb 2028"
  for (const match of text.matchAll(DATE_DDMMMYYYY_SPACED_RE)) {
    const dateGroup = match[1];
    if (dateGroup && match.index != null) {
      spans.push({
        start: match.index,
        end: match.index + dateGroup.length,
        label: "DATE",
        score: 0.96,
        source: "regex_date_ddmmm_spaced",
      });
    }
  }

  // 8-10) Other date patterns
  for (const match of text.matchAll(DATE_DDMMMYYYY_RE)) {
    const dateGroup = match[1];
    if (dateGroup && match.index != null) {
      spans.push({
        start: match.index,
        end: match.index + dateGroup.length,
        label: "DATE",
        score: 0.95,
        source: "regex_date_ddmmm",
      });
    }
  }

  for (const match of text.matchAll(DATE_SLASH_RE)) {
    const dateGroup = match[1];
    if (dateGroup && match.index != null) {
      spans.push({
        start: match.index,
        end: match.index + dateGroup.length,
        label: "DATE",
        score: 0.9,
        source: "regex_date_slash",
      });
    }
  }

  for (const match of text.matchAll(DATE_ISO_RE)) {
    const dateGroup = match[1];
    if (dateGroup && match.index != null) {
      spans.push({
        start: match.index,
        end: match.index + dateGroup.length,
        label: "DATE",
        score: 0.95,
        source: "regex_date_iso",
      });
    }
  }

  // Facility/institution names
  for (const re of [FACILITY_NAME_RE, FACILITY_CAMEL_HEALTH_RE, FACILITY_ENDING_HEALTH_RE, STATE_MEDICINE_RE]) {
    re.lastIndex = 0;
    for (const match of text.matchAll(re)) {
      if (match.index == null) continue;
      const raw = match[0];
      if (!raw || raw.length < 4) continue;
      if (/\d/.test(raw)) continue;
      spans.push({
        start: match.index,
        end: match.index + raw.length,
        label: "GEO",
        score: 0.95,
        source: "regex_facility",
      });
    }
  }

  return spans;
}

// =============================================================================
// Merge overlapping spans
// =============================================================================

function mergeOverlapsBestOf(spans) {
  if (!spans.length) return [];

  const sorted = [...spans].sort((a, b) => a.start - b.start || b.end - a.end);
  const merged = [];

  for (const span of sorted) {
    if (!merged.length) {
      merged.push(span);
      continue;
    }

    const last = merged[merged.length - 1];
    if (span.start < last.end) {
      // Overlap - prefer regex over NER, then higher score
      const lastIsRegex = (last.source || "").startsWith("regex");
      const spanIsRegex = (span.source || "").startsWith("regex");

      if (spanIsRegex && !lastIsRegex) {
        merged[merged.length - 1] = span;
      } else if (!spanIsRegex && lastIsRegex) {
        // Keep last
      } else if ((span.score || 0) > (last.score || 0)) {
        merged[merged.length - 1] = span;
      }
      // Extend end if needed
      merged[merged.length - 1].end = Math.max(last.end, span.end);
    } else {
      merged.push(span);
    }
  }

  return merged;
}

// =============================================================================
// Word boundary expansion
// =============================================================================

function expandToWordBoundaries(spans, fullText) {
  return spans.map((span) => {
    let { start, end } = span;

    // Expand left
    while (start > 0 && /[a-zA-Z0-9'_-]/.test(fullText[start - 1])) {
      start--;
    }

    // Expand right
    while (end < fullText.length && /[a-zA-Z0-9'_-]/.test(fullText[end])) {
      end++;
    }

    if (start !== span.start || end !== span.end) {
      return { ...span, start, end, text: fullText.slice(start, end) };
    }
    return span;
  });
}

// =============================================================================
// Header name extension
// =============================================================================

function extendPatientSpansForTrailingNameToken(spans, fullText) {
  return spans.map((span) => {
    const label = String(span.label || "").toUpperCase().replace(/^[BI]-/, "");
    if (label !== "PATIENT") return span;

    const lineStart = fullText.lastIndexOf("\n", Math.max(0, span.start - 1));
    const start = lineStart === -1 ? 0 : lineStart + 1;
    const lineEnd = fullText.indexOf("\n", span.end);
    const end = lineEnd === -1 ? fullText.length : lineEnd;
    const lineText = fullText.slice(start, end);

    const relStart = span.start - start;
    const relEnd = span.end - start;
    const before = lineText.slice(0, relStart);

    HEADER_NAME_LABEL_RE.lastIndex = 0;
    if (!HEADER_NAME_LABEL_RE.test(before)) return span;

    const after = lineText.slice(relEnd);
    const termIdx = after.search(HEADER_NAME_TERMINATOR_RE);
    const candidate = termIdx === -1 ? after : after.slice(0, termIdx);
    const match = candidate.match(/^\s+([A-Z][A-Za-z'.-]{1,})/);
    if (!match) return span;

    const token = match[1];
    if (HEADER_NAME_STOPWORDS.has(token.toLowerCase())) return span;

    const newEnd = span.end + match[0].length;
    return {
      ...span,
      end: newEnd,
      text: fullText.slice(span.start, newEnd),
    };
  });
}

// =============================================================================
// GEO span extension
// =============================================================================

const CITY_PREFIXES = new Set([
  "san", "los", "las", "new", "fort", "saint", "st", "santa", "el", "la",
  "port", "mount", "mt", "north", "south", "east", "west", "upper", "lower",
  "lake", "palm", "long", "grand", "great", "little", "old", "big",
]);

function extendGeoSpans(spans, fullText) {
  return spans.map((span) => {
    const label = String(span.label || "").toUpperCase().replace(/^[BI]-/, "");
    if (label !== "GEO") return span;

    let { start } = span;
    const beforeWindow = fullText.slice(Math.max(0, start - 20), start);
    const prefixMatch = beforeWindow.match(/\b([A-Za-z]+)\s+$/);

    if (prefixMatch) {
      const prefix = prefixMatch[1].toLowerCase();
      if (CITY_PREFIXES.has(prefix)) {
        const prefixStart = start - prefixMatch[0].length;
        return {
          ...span,
          start: prefixStart,
          end: span.end,
          text: fullText.slice(prefixStart, span.end),
        };
      }
    }

    return span;
  });
}

// =============================================================================
// Apply redactions to text
// =============================================================================

function applyRedactions(text, spans) {
  const sorted = [...spans].sort((a, b) => b.start - a.start);
  let result = text;

  for (const span of sorted) {
    const label = String(span.label || "UNKNOWN").toUpperCase().replace(/^[BI]-/, "");
    const placeholder = `[REDACTED_${label}]`;
    result = result.slice(0, span.start) + placeholder + result.slice(span.end);
  }

  return result;
}

// =============================================================================
// Load golden notes
// =============================================================================

function loadGoldenNotes(goldenDir, limit = null) {
  const notes = [];
  const files = fs.readdirSync(goldenDir).filter((f) => f.match(/^golden_\d+\.json$/));

  for (const file of files) {
    try {
      const filepath = path.join(goldenDir, file);
      const content = fs.readFileSync(filepath, "utf-8");
      const data = JSON.parse(content);

      const entries = Array.isArray(data) ? data : [data];
      for (let i = 0; i < entries.length; i++) {
        const entry = entries[i];
        if (entry?.note_text && entry.note_text.trim().length > 50) {
          notes.push({
            text: entry.note_text,
            source: file,
            index: i,
          });
        }
      }
    } catch (err) {
      console.error(`Warning: Could not load ${file}: ${err.message}`);
    }
  }

  return notes;
}

// =============================================================================
// Main
// =============================================================================

async function main() {
  const args = process.argv.slice(2);
  let count = 10;
  let seed = null;
  let outputFile = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--count" || args[i] === "-n") {
      count = parseInt(args[++i], 10);
    } else if (args[i] === "--seed") {
      seed = parseInt(args[++i], 10);
    } else if (args[i] === "--output" || args[i] === "-o") {
      outputFile = args[++i];
    }
  }

  console.error("Loading protected terms...");
  const protectedTerms = JSON.parse(fs.readFileSync(PROTECTED_TERMS_PATH, "utf-8"));

  console.error("Loading DistilBERT NER model...");
  let nerPipeline;
  try {
    nerPipeline = await pipeline("token-classification", MODEL_PATH, {
      local_files_only: true,
      quantized: false,
    });
    console.error("Model loaded successfully.");
  } catch (err) {
    console.error(`Warning: Could not load NER model: ${err.message}`);
    console.error("Running in regex-only mode.");
    nerPipeline = null;
  }

  console.error("Loading golden notes...");
  const allNotes = loadGoldenNotes(GOLDEN_DIR);
  console.error(`Found ${allNotes.length} notes.`);

  // Sample random notes
  if (seed !== null) {
    // Simple seeded random
    let s = seed;
    const random = () => {
      s = (s * 1103515245 + 12345) & 0x7fffffff;
      return s / 0x7fffffff;
    };
    allNotes.sort(() => random() - 0.5);
  } else {
    allNotes.sort(() => Math.random() - 0.5);
  }

  const sampleNotes = allNotes.slice(0, Math.min(count, allNotes.length));
  console.error(`Selected ${sampleNotes.length} random notes.\n`);

  const outputLines = [];
  outputLines.push("=".repeat(80));
  outputLines.push("PHI REDACTION TEST REPORT (DistilBERT ONNX + Veto Layer)");
  outputLines.push(`Sample size: ${sampleNotes.length} notes`);
  outputLines.push(`Model: ${nerPipeline ? "DistilBERT ONNX" : "Regex-only (model not loaded)"}`);
  outputLines.push("=".repeat(80));
  outputLines.push("");

  let totalRedactions = 0;

  for (let i = 0; i < sampleNotes.length; i++) {
    const note = sampleNotes[i];
    console.error(`Processing note ${i + 1}/${sampleNotes.length}...`);

    try {
      let allSpans = [];

      // 1) NER inference (if model loaded)
      if (nerPipeline) {
        const nerResults = await nerPipeline(note.text);
        for (const entity of nerResults) {
          if (entity.entity && entity.start != null && entity.end != null) {
            allSpans.push({
              start: entity.start,
              end: entity.end,
              label: entity.entity,
              score: entity.score || 0.5,
              source: "ner",
            });
          }
        }
      }

      // 2) Regex detection
      const regexSpans = runRegexDetectors(note.text);
      allSpans.push(...regexSpans);

      // 3) Merge overlaps
      let merged = mergeOverlapsBestOf(allSpans);

      // 4) Expand to word boundaries
      merged = expandToWordBoundaries(merged, note.text);

      // 5) Extend PATIENT spans for trailing header name tokens
      merged = extendPatientSpansForTrailingNameToken(merged, note.text);

      // 6) Extend GEO spans
      merged = extendGeoSpans(merged, note.text);

      // 7) Re-merge
      merged = mergeOverlapsBestOf(merged);

      // 8) Apply veto
      const final = applyVeto(merged, note.text, protectedTerms, { debug: false });

      // 9) Apply redactions
      const redactedText = applyRedactions(note.text, final);

      totalRedactions += final.length;

      // Format output
      outputLines.push("=".repeat(80));
      outputLines.push(`NOTE ${i + 1}: ${note.source} [entry ${note.index}]`);
      outputLines.push("=".repeat(80));
      outputLines.push("");
      outputLines.push("ORIGINAL TEXT:");
      outputLines.push("-".repeat(80));
      outputLines.push(note.text.trim());
      outputLines.push("");
      outputLines.push("-".repeat(80));
      outputLines.push("REDACTED TEXT:");
      outputLines.push("-".repeat(80));
      outputLines.push(redactedText.trim());
      outputLines.push("");
      outputLines.push("-".repeat(80));
      outputLines.push(`REDACTION SUMMARY: ${final.length} items redacted`);
      outputLines.push("-".repeat(80));

      if (final.length > 0) {
        outputLines.push("Detected PHI:");
        for (const span of final) {
          const label = String(span.label || "UNKNOWN").toUpperCase().replace(/^[BI]-/, "");
          const text = note.text.slice(span.start, span.end).slice(0, 50);
          const score = span.score?.toFixed(2) || "N/A";
          const source = span.source || "unknown";
          outputLines.push(`  - [${label}] "${text}" (score=${score}, source=${source})`);
        }
      } else {
        outputLines.push("No PHI detected.");
      }

      outputLines.push("");
    } catch (err) {
      outputLines.push(`ERROR processing note ${i + 1}: ${err.message}`);
      outputLines.push("");
    }
  }

  // Summary
  outputLines.push("=".repeat(80));
  outputLines.push("SUMMARY");
  outputLines.push("=".repeat(80));
  outputLines.push(`Notes processed: ${sampleNotes.length}`);
  outputLines.push(`Total redactions: ${totalRedactions}`);
  outputLines.push(`Average redactions per note: ${(totalRedactions / sampleNotes.length).toFixed(1)}`);
  outputLines.push("");

  const output = outputLines.join("\n");

  if (outputFile) {
    fs.writeFileSync(outputFile, output);
    console.error(`Results saved to ${outputFile}`);
  } else {
    console.log(output);
  }

  console.error("Done!");
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
