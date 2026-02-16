/**
 * redactor.worker.js — “Best-of” Hybrid PHI Detector (ML + Regex)
 *
 * Combines:
 *  - Version B robustness: quantized→unquantized fallback, offset recovery, cancel, debug hooks
 *  - Version A fix: expand spans to word boundaries to prevent partial-word redactions
 *  - Hybrid regex injection BEFORE veto (cold-start header guarantees)
 *  - Smarter merge rules: prefer regex spans over overlapping ML spans to avoid double-highlights
 */

const BASE_URL = new URL("./", self.location).toString();
const CACHE_BUST = Date.now();
importScripts(`${BASE_URL}transformers.legacy.js?v=${CACHE_BUST}`);
importScripts(`${BASE_URL}protectedVeto.legacy.js?v=${CACHE_BUST}`);

const { pipeline, env } = self.transformers || {};
// applyVeto is already in global scope from importScripts(protectedVeto.legacy.js)

const MODEL_BASE_URL = new URL("./vendor/", self.location).toString();
const MODEL_CANDIDATES = [
  { id: "phi_distilbert_ner_quant", path: "./vendor/phi_distilbert_ner_quant/" },
  { id: "phi_distilbert_ner", path: "./vendor/phi_distilbert_ner/" },
];
const TASK = "token-classification";

// Character windowing (simple + robust)
const WINDOW = 2500;
const OVERLAP = 250;
const STEP = WINDOW - OVERLAP;

// =============================================================================
// MERGE MODE CONFIGURATION
// =============================================================================

/**
 * MERGE_MODE controls how overlapping spans from different sources are handled.
 *
 * "best_of" (legacy): Prefers regex over ML on overlap, runs merge BEFORE veto.
 *   - PROBLEM: If regex span is vetoed, overlapping ML span is already lost.
 *
 * "union" (recommended): Keeps all candidates until AFTER veto.
 *   - Removes only exact duplicates before veto
 *   - Resolves overlaps AFTER veto has approved survivors
 *   - Safer: veto can't cause span loss from merge happening too early
 */
const MERGE_MODE_DEFAULT = "union";

/**
 * Get the merge mode from config, with fallback to default.
 * Main thread can pass mergeMode via config (from query param or localStorage).
 */
function getMergeMode(config) {
  const mode = config?.mergeMode;
  if (mode === "union" || mode === "best_of") return mode;
  return MERGE_MODE_DEFAULT;
}

// =============================================================================
// Phase 4: PER-LABEL AI THRESHOLDS
// =============================================================================

/**
 * Default thresholds by label type.
 * Quantized models often have worse calibration; ID should be stricter.
 *
 * - PATIENT: 0.50 (moderate - names are important to catch)
 * - ID: 0.70 (stricter - short numbers often false positives)
 * - DATE: 0.45 (relaxed - dates are lower risk)
 * - GEO: 0.55 (moderate)
 * - CONTACT: 0.55 (moderate)
 */
const DEFAULT_THRESHOLDS_BY_LABEL = {
  PATIENT: 0.50,
  ID: 0.70,
  DATE: 0.45,
  GEO: 0.55,
  CONTACT: 0.55,
};

// Quantized model threshold bump (added to base thresholds)
const QUANTIZED_THRESHOLD_BUMP = 0.05;

/**
 * Get the effective threshold for a given label.
 * Supports both legacy single-threshold and per-label threshold configs.
 *
 * @param {Object} config - Worker config from main thread
 * @param {string} label - Entity label (PATIENT, ID, DATE, GEO, CONTACT)
 * @param {boolean} isQuantized - Whether using quantized model
 * @returns {number} Threshold value
 */
function getThresholdForLabel(config, label, isQuantized = false) {
  const normLabel = String(label || "").toUpperCase().replace(/^[BI]-/, "");

  // Check for per-label thresholds in config
  if (config?.aiThresholdsByLabel && typeof config.aiThresholdsByLabel === "object") {
    const labelThreshold = config.aiThresholdsByLabel[normLabel];
    if (typeof labelThreshold === "number") {
      return isQuantized ? labelThreshold + QUANTIZED_THRESHOLD_BUMP : labelThreshold;
    }
  }

  // Fall back to single threshold if provided
  if (typeof config?.aiThreshold === "number") {
    // For ID label, always bump up slightly even with single threshold
    if (normLabel === "ID") {
      return config.aiThreshold + 0.15;
    }
    return config.aiThreshold;
  }

  // Use default per-label thresholds
  const baseThreshold = DEFAULT_THRESHOLDS_BY_LABEL[normLabel] ?? 0.45;
  return isQuantized ? baseThreshold + QUANTIZED_THRESHOLD_BUMP : baseThreshold;
}

// =============================================================================
// HYBRID REGEX DETECTION (guarantees headers/IDs)
// =============================================================================

// Matches: "Patient: Smith, John" or "Pt Name: John Smith" or "Patient Name: Carey , Cloyd D" (footer format)
// Also matches "Pt: C. Rodriguez" (Initial. Lastname format) and "Pt: White, E." (Last, Initial format)
// IMPORTANT: Requires colon/dash delimiter to avoid matching "patient went into" as a name
const PATIENT_HEADER_RE =
  /(?:Patient(?:\s+Name)?|Pt\.?(?:\s+Name)?|Pat\.?|Name|Subject)\s*[:\-]\s*([A-Z][a-z]+\s*,\s*[A-Z]\.?|[A-Z]\.?\s+[A-Z][a-z]+|[A-Z][a-z]+\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?|[A-Z][a-z]+\s+[A-Z]'?[A-Za-z]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)/gim;

// Matches ALL-CAPS patient names after header labels: "PATIENT NAME: CHARLES D HOLLINGER"
// NER often fails on all-uppercase names, so we need a dedicated regex
// Captures 2-4 uppercase words (with optional middle initial) after patient/name labels
const PATIENT_HEADER_ALLCAPS_RE =
  /(?:PATIENT(?:\s+NAME)?|PT\.?(?:\s+NAME)?|NAME|SUBJECT)\s*[:\-]\s*([A-Z]{2,}(?:\s+[A-Z]\.?)?\s+[A-Z]{2,}(?:\s+[A-Z]{2,})?)/g;

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

// =============================================================================
// HEADER ZONE + CLINICAL CONTEXT GATING (Phase 1 - False Positive Reduction)
// =============================================================================

// Maximum characters from document start to consider "header zone" for name patterns
// Patient demographics typically appear in the first ~1200 chars of procedure notes
const HEADER_ZONE_MAX_CHARS = 1200;

// Clinical terms that should NEVER be treated as patient names
// Used to gate "Last, First" patterns (e.g., "Elastography, First" is NOT a name)
const NAME_REGEX_CLINICAL_STOPLIST = new Set([
  // Pathology/lab report headers
  "elastography", "cytology", "pathology", "histology", "microbiology",
  "cultures", "specimen", "immunohistochemistry", "flow",
  // Cell types (from path reports)
  "lymphocytes", "macrophages", "histiocytes", "neutrophils", "eosinophils",
  "epithelial", "squamous", "columnar", "ciliated",
  // Lymph node wording (often appears as headings like "Lymph Nodes Evaluated:")
  "lymph", "node", "nodes",
  // Demographic nouns that should never be interpreted as a name
  "patient",
  // Anatomical segments/regions
  "apical", "basal", "anterior", "posterior", "lateral", "medial",
  "superior", "inferior", "proximal", "distal", "segmental", "subsegmental",
  // Findings/descriptors
  "first", "second", "third", "target", "additional", "primary", "secondary",
  // Common clinical nouns that start capitalized
  "lesion", "nodule", "mass", "tumor", "stenosis", "obstruction",
  "serial", "irrigation", "dilation", "aspiration", "suction",
  // Procedure / modality terms that can look like names
  "radial", "ebus", "ct", "us", "mri", "pet", "cbct",
  // Specialties/roles often mis-detected as "Last, First"
  "pulmonology", "critical", "medicine", "anesthesia", "radiology", "pathology",
  // Common EHR/vitals labels that can be mis-detected as names
  "range", "ending", "pressure", "cuff", "calc", "injection", "once", "vitals", "family",
  // Directions/locations
  "left", "right", "bilateral", "central", "peripheral",
  // Anatomy terms
  "adrenal", "bronchus", "carina", "trachea", "hilum", "mediastinum",
  "lobe", "segment", "mainstem", "lingula", "station"
]);

// Clinical single-word terms that should not match FIRST_NAME_CLINICAL_RE
// These appear as "Air came", "Still is" etc in clinical prose
const SINGLE_NAME_CLINICAL_STOPLIST = new Set([
  // Common clinical words mistaken for first names
  "air", "still", "serial", "flow", "pain", "mass", "clear", "free",
  "deep", "mild", "moderate", "severe", "acute", "chronic",
  "good", "fair", "poor", "stable", "normal", "adequate",
  "sterile", "clean", "patent", "open", "closed",
  // Anatomical/positional
  "left", "right", "upper", "lower", "middle", "lateral", "medial",
  "apical", "basal", "anterior", "posterior", "superior", "inferior",
  // Clinical procedures/actions
  "suction", "lavage", "dilation", "ablation", "biopsy", "aspiration",
  // Procedure / modality terms that can look like names
  "radial", "ebus",
  // Equipment/device terms
  "scope", "probe", "needle", "catheter", "balloon", "stent",
  // Common sentence starters
  "there", "then", "here", "both", "each", "some", "all", "most",
  // Vitals/labels
  "range", "ending", "pressure", "cuff", "calc", "injection", "once", "vitals", "family", "no",
  // Common clinical adjectives that appear after "for"
  "progressive", "inflammation", "newly", "acquired",
  // Specialties (avoid session name amplification)
  "pulmonology", "critical", "medicine"
]);

// Helper: check if a word is in the clinical stoplist (case-insensitive)
function isInClinicalStoplist(word) {
  if (!word) return false;
  return NAME_REGEX_CLINICAL_STOPLIST.has(word.toLowerCase());
}

// Helper: check if span is within header zone (first N chars of document)
function isInHeaderZone(matchIndex, headerZoneChars = HEADER_ZONE_MAX_CHARS) {
  return matchIndex < headerZoneChars;
}

// Helper: check if text has patient demographic context nearby
function hasPatientDemographicContext(text, matchIndex, windowSize = 80) {
  const start = Math.max(0, matchIndex - windowSize);
  const end = Math.min(text.length, matchIndex + windowSize);
  const context = text.slice(start, end).toLowerCase();
  return /\b(?:patient|name|mrn|dob|age|year[\s-]*old|pt\b|subject)\b/i.test(context);
}

// Matches: "MRN: 12345" or "ID: 55-22-11" or "DOD NUMBER: 194174412" or "DOD#: 12345678"
// IMPORTANT: Must contain at least one digit to avoid matching medical acronyms like "rEBUS"
const MRN_RE =
  /\b(?:MRN|MR|Medical\s*Record|Patient\s*ID|ID|EDIPI|DOD\s*(?:ID|NUMBER|NUM|#))\s*[:\#]?\s*([A-Z0-9\-]*\d[A-Z0-9\-]*)\b/gi;

// Matches: MRN with spaces like "A92 555" or "AB 123 456" (2-3 groups of alphanumerics)
// IMPORTANT: Each segment MUST contain at least one digit to avoid matching "Li in the" as MRN
// Removed plain "ID" prefix as too generic (matches "ID Li in the ICU")
const MRN_SPACED_RE =
  /\b(?:MRN|MR|Medical\s*Record|Patient\s*ID)\s*[:\#]?\s*([A-Z0-9]*\d[A-Z0-9]*\s+[A-Z0-9]*\d[A-Z0-9]*(?:\s+[A-Z0-9]*\d[A-Z0-9]*)?)\b/gi;

// Matches inline narrative patient names followed by age: "Emma Jones, a 64-year-old male..."
// Also supports "Last, First" format: "Belardes, Lisa is a 64-year-old..."
// Captures: "Emma Jones" or "Belardes, Lisa" when followed by age descriptor
// IMPORTANT: Excludes "The patient", "A patient", etc. via negative lookahead
const INLINE_PATIENT_NAME_RE =
  /\b(?!(?:The|A|An)\s+(?:patient|subject|candidate|individual|person)\b)([A-Z][a-z]+(?:(?:\s+|,\s*)[A-Z]\.?)?(?:\s+|,\s*)[A-Z][a-z]+),?\s+(?:(?:is|was)\s+)?(?:a\s+)?(?:\d{1,3}\s*-?\s*(?:year|yr|y\/?o|yo)[\s-]*old|aged?\s+\d{1,3})\b/gi;

// Matches names after procedural verbs: "performed on Robert Chen", "procedure for Jane Doe"
// Captures patient name when following "performed on/for", "procedure on/for", etc.
// IMPORTANT: Case-sensitive (no 'i' flag) to avoid matching lowercase words like "the core"
const PROCEDURAL_NAME_RE =
  /\b(?:performed|completed|done|scheduled|procedure)\s+(?:on|for)\s+([A-Z][a-z]+(?:'[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:'[A-Z][a-z]+)?)\b/g;

// Matches: "pt Name mrn 1234" pattern common in IP notes
// Captures the name between "pt" and "mrn"
const PT_NAME_MRN_RE =
  /\bpt\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+mrn\s+\d+/gi;

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

// Matches "did an EBUS on [Name]", "did a bronchoscopy on [Name]"
// Fixes: "We did an EBUS on Gregory Martinez today" - verb "did" with intervening procedure name
const DID_PROCEDURE_NAME_RE =
  /\bdid\s+(?:an?\s+)?(?:EBUS|bronch(?:oscopy)?|procedure|biopsy|tbna|bal|navigation|bronch)\s+on\s+([A-Z][a-z]+(?:'[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:'[A-Z][a-z]+)?)\b/gi;

// Matches "EBUS for [Name]", "procedure for [Name]" followed by sentence boundary or common words
// Fixes: "EBUS for Arthur Curry. We looked at all the nodes."
const PROCEDURE_FOR_NAME_RE =
  /\b(?:EBUS|bronch(?:oscopy)?|procedure|biopsy|tbna|bal|navigation)\s+for\s+([A-Z][a-z]+(?:'[A-Z][a-z]+)?\s+[A-Z][a-z]+(?:'[A-Z][a-z]+)?)(?=\s*[\.!\?,;:]|\s+(?:we|he|she|they|who|with|has|had|is|was|were|today|yesterday|this|that|the|a|an|and|but|or|so|then|now|here|there)\b)/gi;

// Matches de-identification placeholders/artifacts: "<PERSON>", "[Name]", "[REDACTED]", "***NAME***"
// These appear in pre-processed or dirty data and should be redacted to prevent leakage
const PLACEHOLDER_NAME_RE =
  /(?:Patient(?:\s+Name)?|Pt(?:\s+Name)?|Name|Subject)\s*[:\-]?\s*(<[A-Z]+>|\[[A-Za-z_]+\]|\*{2,}[A-Za-z_]+\*{2,})/gi;

// REMOVED: PATIENT_SHORTHAND_RE - Age/gender demographics (e.g., "68 female") are NOT PHI
// Pattern was causing false positives by redacting age/gender info after "Patient"
// const PATIENT_SHORTHAND_RE = /\bPatient\s+(\d{1,3}\s*[MF](?:emale)?)\b/gi;

// Matches case/accession/specimen IDs: "case c-847", "specimen A-12345", "pathology P-9876"
// These are identifiers that can be PHI and should be captured
const CASE_ID_RE =
  /\b(?:case|accession|specimen|path(?:ology)?)\s*[:\#]?\s*([A-Za-z]-?\d{3,6})\b/gi;

// Matches names at sentence start followed by clinical verbs: "Robert has a LLL nodule..."
// Must be at sentence start (after period/newline) and followed by clinical context
// Requires both first and last name to reduce false positives
const SENTENCE_START_NAME_RE =
  /(?:^|[.!?]\s+|\n\s*)([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:has|had|is|was|presents|presented|underwent|undergoing|needs|needed|required|denies|denied|reports|reported|complained|complains|notes|noted|states|stated|describes|described|exhibits|exhibited|demonstrates|demonstrated|developed|shows|showed|appears|appeared)\b/gm;

// Matches names at very start of line/document followed by period: "Kimberly Garcia. Ion Bronchoscopy."
// For notes that begin directly with patient name without a header label
const LINE_START_NAME_RE =
  /^([A-Z][a-z]+\s+[A-Z][a-z]+)\s*[.,]/gm;

// Matches names at line start followed by clinical abbreviations/terms
// Fixes: "Daniel Rivera LLL nodule small 14mm." - name at absolute start
// Fixes: "Ryan Williams procedure note" - name at start of procedure note
const LINE_START_CLINICAL_NAME_RE =
  /^([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:LLL|RLL|RUL|LUL|RML|RB\d|LB\d|nodule|mass|lesion|lung|lobe|procedure|bronch|ebus|ion|underwent|scheduled|post|status|transplant|bilateral|unilateral)\b/gm;

// Matches informal/lowercase names followed by "here for": "jason phillips here for right lung lavage"
// Case-insensitive to catch dictation notes where names aren't capitalized
// The "here for" phrase is a strong indicator of patient context in informal notes
// Uses negative lookahead to exclude pronouns and common words that aren't names
const INFORMAL_NAME_HERE_RE =
  /^(?!(?:it|he|she|we|they|you|this|that|here|there|i|me|us|pt|who|what)\s+)([a-z]+\s+[a-z]+)\s+here\s+for\b/gim;

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

// Matches parenthetical IDs after patient context: "Patient: Smith, John (22352321)"
// Requires 6-15 digits to avoid matching list markers like (1) or (2)
const PAREN_ID_RE =
  /\((\d{6,15})\)/g;

// Matches "pt [Name]" or "patient [Name]" when followed by identifier keywords
// Requires: "patient angela davis mrn" or "pt john doe dob" etc.
// SAFE: Won't match "patient severe pneumonia" because "pneumonia" is not a keyword
const PT_LOWERCASE_NAME_RE =
  /\b(?:pt|patient)\s+([a-z]+\s+[a-z]+)(?=\s+(?:mrn|id|dob|age|here|for)\b)/gi;

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

// Matches names at absolute start followed by clinical context (no punctuation required)
// E.g., "Brenda Lewis transplant patient with stenosis" or "John Smith status post lobectomy"
// Requires clinical follow-word to reduce false positives
const NAME_START_CLINICAL_CONTEXT_RE =
  /^([A-Z][a-z]+\s+[A-Z][a-z]+)\s+(?:transplant|patient|pt|status|post|s\/p|here|presented|presenting|hx|history|scheduled|referred|admitted|seen|evaluated|is|was|with|who|underwent)\b/gim;

// Matches lowercase names after "for" in narrative text (not just at line start)
// E.g., "diag bronch for charlene king she has hilar adenopathy"
// Requires pronoun or clinical word after name to confirm patient context
const LOWERCASE_FOR_NAME_RE =
  /\bfor\s+([a-z]+\s+[a-z]+)\s+(?:she|he|they|who|patient|pt|with|has|had|is|was)\b/gi;

// Matches "Last, First M" or "Last , First M" format with trailing initial/suffix
// Common in footers, headers, and patient identifiers: "Carey , Cloyd D", "Smith, John Jr"
// Captures full name including trailing initial (D, M, etc.) or suffix (Jr, Sr, III)
const LAST_FIRST_INITIAL_RE =
  /\b([A-Z][a-z]+\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?|\s+(?:Jr|Sr|II|III|IV)\.?)?)\b/g;

// Provider/staff role lines: "Staff: Miller", "Fellow: Derek Booth"
const PROVIDER_ROLE_LINE_RE =
  /^(?:Staff|Fellow|Attending|Proceduralist|Surgeon|Operator|Anesthesiologist|Physician|Provider)\s*:\s*([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){0,3}|[A-Z]{2,}(?:\s+[A-Z]{2,}){0,5})\b/gim;

// Credentialed names: "Derek Booth, DO", "Jane Doe, MD"
const CREDENTIAL_NAME_RE =
  /\b([A-Z][A-Za-z'’.-]+(?:\s+[A-Z][A-Za-z'’.-]+){1,3})\s*,\s*(?:MD|DO|RN|RT|PA|NP|CRNA|PhD|FCCP|DAABIP)\b/g;

// All-caps "LAST, FIRST ..." formats often used in signatures/headers
const ALLCAPS_LAST_FIRST_RE =
  /\b([A-Z]{2,}(?:\s+[A-Z]{2,}){0,4}\s*,\s*[A-Z]{2,}(?:\s+[A-Z]{2,}){0,4})\b/g;

// Signature blocks: underscore separator followed by an ALL-CAPS name line
const SIGNATURE_BLOCK_ALLCAPS_RE =
  /^_{5,}\s*\n([A-Z]{2,}(?:\s+[A-Z]{2,}){1,5})\b/gm;

// Facility acronyms + city: "NMRTC San Diego"
const FACILITY_ACRONYM_CITY_RE =
  /\b(?:NMRTC|NMCSD|NMCP|NMCL|NMC)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b/g;

// Facility / institution patterns (treated as PHI → GEO)
// Goal: prevent partial redactions like "[REDACTED] Ridge Medical Center" by capturing the full facility span.
// Note: explicitly avoids "Becker's Hospital Review" by blocking "Hospital" matches followed by "Review".
const FACILITY_NAME_RE =
  /\b(?:The\s+)?(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*|&|of|the|and|for|at|de|la|st\.?|st|saint|mount|mt)){0,12}\s+(?:Medical\s+(?:Center|Centre|Pavilion)|Hospital\s+Center|Hospital|Hospitals|Clinic|Clinics|Health\s+(?:System|Center)|Cancer\s+Center|Institute|Clinical\s+Center)\b(?!\s+Review\b)(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)){0,2})?/g;

// CamelCase health-system brands: "AdventHealth Orlando", "OhioHealth Riverside Methodist Hospital"
const FACILITY_CAMEL_HEALTH_RE =
  /\b[A-Z][A-Za-z]+Health\b(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)){0,6}\b/g;

// Multi-token health-system names ending with Health/Healthcare: "Indiana University Health", "Lakeland Regional Health"
// Requires >= 2 tokens before the Health word to avoid matching specialties like "Mental Health".
const FACILITY_ENDING_HEALTH_RE =
  /\b(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*)(?:\s+(?:[A-Z]{2,}|(?:[A-Z]\.){2,}|[A-Z][A-Za-z'’.-]*|&|of|the|and)){1,10}\s+(?:Health|Healthcare)\b/g;

// State-name medicine institutions: "Michigan Medicine"
const STATE_MEDICINE_RE =
  /\b(?:Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|Mississippi|Missouri|Montana|Nebraska|Nevada|New\s+Hampshire|New\s+Jersey|New\s+Mexico|New\s+York|North\s+Carolina|North\s+Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode\s+Island|South\s+Carolina|South\s+Dakota|Tennessee|Texas|Utah|Vermont|Virginia|Washington|West\s+Virginia|Wisconsin|Wyoming)\s+Medicine\b/g;

// Date patterns - various formats commonly found in medical notes
// Matches: "18Apr2022", "18-Apr-2022", "18 Apr 2022" (DDMMMYYYY variants)
const DATE_DDMMMYYYY_RE =
  /\b(\d{1,2}[-\s]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s]?\d{2,4})\b/gi;

// Matches: "13 Feb 2028" (DD Mon YYYY with required spaces)
const DATE_DDMMMYYYY_SPACED_RE =
  /\b(\d{1,2}\s+[A-Za-z]{3}\s+\d{4})\b/gi;

// Matches: "6/3/2016", "06/03/2016", "6-3-2016" (M/D/YYYY or MM/DD/YYYY)
const DATE_SLASH_RE =
  /\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b/g;

// Matches: "2024-01-15" (YYYY-MM-DD ISO format)
const DATE_ISO_RE =
  /\b(\d{4}[-\/]\d{1,2}[-\/]\d{1,2})\b/g;

// Matches: "January 15, 1960", "Jan 15, 1960", and (conservatively) "Jan 15" (year optional).
// Used to prevent ZK bundle date-leak rejections for month-name date strings.
const DATE_MONTH_NAME_RE =
  /\b((?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?(?:\s*,?\s*(?:19|20)\d{2})?)\b/gi;

// Matches: "DOB: 01/15/1960" or "Date of Birth: January 15, 1960"
const DOB_HEADER_RE =
  /\b(?:DOB|Date\s+of\s+Birth|Birth\s*Date|Birthdate)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{1,2}[-\s]?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[-\s,]?\s*\d{1,2}[-,\s]+\d{2,4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}[-,\s]+\d{2,4})\b/gi;

// Matches timestamps: "10:00:00 AM", "14:30", "2:15 PM", "08:45:30"
// Used to capture time components when they appear near procedure dates
const TIME_RE =
  /\b(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?)\b/g;

// =============================================================================
// Transformers.js env
// =============================================================================

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = MODEL_BASE_URL;

// Disable browser cache temporarily while iterating (you can re-enable later)
env.useBrowserCache = false;

if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.proxy = false;
  env.backends.onnx.wasm.numThreads = 1;
}

// =============================================================================
// Worker state
// =============================================================================

let classifier = null;
let classifierQuantized = null;
let classifierUnquantized = null;

let modelPromiseQuantized = null;
let modelPromiseUnquantized = null;
let activeModel = null;
let activeModelPromise = null;

let protectedTerms = null;
let termsPromise = null;

const DEFAULT_PROTECTED_TERMS = {
  anatomy_terms: [],
  device_manufacturers: [],
  protected_device_names: [],
  ln_station_regex: "^\\\\d{1,2}[LRlr](?:[is])?$",
  segment_regex: "^[LRlr][Bb]\\\\d{1,2}(?:\\\\+\\\\d{1,2})?$",
  address_markers: [],
  code_markers: [],
  station_markers: [],
};

let cancelled = false;
let debug = false;
let didTokenDebug = false;
let didLogitsDebug = false;

function log(...args) {
  if (debug) console.log(...args);
}

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function toFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function toFiniteInt(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return null;
  return Number.isInteger(value) ? value : Math.trunc(value);
}

function normalizeLabel(entity) {
  if (!entity) return "PHI";
  const raw = String(entity).toUpperCase();
  return raw.replace(/^B-/, "").replace(/^I-/, "");
}

function normalizeRawOutput(raw) {
  if (Array.isArray(raw)) return raw;
  if (!raw || typeof raw !== "object") return [];
  if (Array.isArray(raw.data)) return raw.data;
  if (Array.isArray(raw.entities)) return raw.entities;
  return [];
}

// =============================================================================
// Protected terms (veto list) loader
// =============================================================================

async function loadProtectedTerms() {
  if (protectedTerms) return protectedTerms;
  if (termsPromise) return termsPromise;

  termsPromise = (async () => {
    const model = await resolveActiveModel();
    if (!model) {
      protectedTerms = { ...DEFAULT_PROTECTED_TERMS };
      return protectedTerms;
    }

    const termsUrl = new URL(`${model.path}protected_terms.json`, self.location);
    try {
      const res = await fetch(termsUrl);
      if (res.ok) {
        protectedTerms = await res.json();
        return protectedTerms;
      }
    } catch {
      // Fall back to defaults when terms are unavailable in local dev.
    }

    protectedTerms = { ...DEFAULT_PROTECTED_TERMS };
    return protectedTerms;
  })();

  return termsPromise;
}

async function resolveActiveModel() {
  if (activeModel) return activeModel;
  if (activeModelPromise) return activeModelPromise;

  activeModelPromise = (async () => {
    for (const candidate of MODEL_CANDIDATES) {
      const configUrl = new URL(`${candidate.path}config.json`, self.location);
      try {
        const res = await fetch(configUrl);
        if (res.ok) return candidate;
      } catch {
        // Continue probing candidates.
      }
    }
    return null;
  })();

  try {
    activeModel = await activeModelPromise;
    return activeModel;
  } finally {
    activeModelPromise = null;
  }
}

// =============================================================================
// Model loading (quantized -> fallback unquantized)
// =============================================================================

async function loadQuantizedModel() {
  if (classifierQuantized) return classifierQuantized;
  if (modelPromiseQuantized) return modelPromiseQuantized;
  const model = await resolveActiveModel();
  if (!model) throw new Error("No local PHI model bundle found under ./vendor/");

  modelPromiseQuantized = pipeline(TASK, model.id, { device: "wasm", quantized: true })
    .then((c) => {
      classifierQuantized = c;
      return c;
    })
    .catch((err) => {
      modelPromiseQuantized = null;
      classifierQuantized = null;
      throw err;
    });

  return modelPromiseQuantized;
}

async function loadUnquantizedModel() {
  if (classifierUnquantized) return classifierUnquantized;
  if (modelPromiseUnquantized) return modelPromiseUnquantized;
  const model = await resolveActiveModel();
  if (!model) throw new Error("No local PHI model bundle found under ./vendor/");

  modelPromiseUnquantized = pipeline(TASK, model.id, { device: "wasm", quantized: false })
    .then((c) => {
      classifierUnquantized = c;
      return c;
    })
    .catch((err) => {
      modelPromiseUnquantized = null;
      classifierUnquantized = null;
      throw err;
    });

  return modelPromiseUnquantized;
}

async function loadModel(config = {}) {
  const resolvedModel = await resolveActiveModel();
  if (!resolvedModel) {
    throw new Error(
      "Local PHI model not found. Expected ui/static/phi_redactor/vendor/phi_distilbert_ner or ui/static/phi_redactor/vendor/phi_distilbert_ner_quant."
    );
  }

  const forceUnquantized = Boolean(config.forceUnquantized);

  if (forceUnquantized) {
    post("progress", { stage: "Loading local PHI model (unquantized; forced)…" });
    classifier = await loadUnquantizedModel();
    post("progress", { stage: "AI model ready" });
    return classifier;
  }

  post("progress", { stage: "Loading local PHI model (quantized)…" });
  try {
    classifier = await loadQuantizedModel();
    post("progress", { stage: "AI model ready" });
    return classifier;
  } catch (err) {
    classifier = null;
    log("[PHI Worker] Quantized load failed; falling back to unquantized", err);
  }

  post("progress", { stage: "Loading local PHI model (unquantized)…" });
  classifier = await loadUnquantizedModel();
  post("progress", { stage: "AI model ready" });
  return classifier;
}

// =============================================================================
// Regex injection (deterministic)
// =============================================================================

function runRegexDetectors(text) {
  const spans = [];

  // Reset global regex state
  PATIENT_HEADER_RE.lastIndex = 0;
  PATIENT_HEADER_ALLCAPS_RE.lastIndex = 0;
  HEADER_NAME_LABEL_RE.lastIndex = 0;
  FACILITY_NAME_RE.lastIndex = 0;
  FACILITY_CAMEL_HEALTH_RE.lastIndex = 0;
  FACILITY_ENDING_HEALTH_RE.lastIndex = 0;
  STATE_MEDICINE_RE.lastIndex = 0;
  MRN_RE.lastIndex = 0;
  MRN_SPACED_RE.lastIndex = 0;
  INLINE_PATIENT_NAME_RE.lastIndex = 0;
  PROCEDURAL_NAME_RE.lastIndex = 0;
  PT_NAME_MRN_RE.lastIndex = 0;
  PT_STANDALONE_RE.lastIndex = 0;
  TITLE_NAME_RE.lastIndex = 0;
  NARRATIVE_FOR_NAME_RE.lastIndex = 0;
  DID_PROCEDURE_NAME_RE.lastIndex = 0;
  PROCEDURE_FOR_NAME_RE.lastIndex = 0;
  PLACEHOLDER_NAME_RE.lastIndex = 0;
  CASE_ID_RE.lastIndex = 0;
  // REMOVED: PATIENT_SHORTHAND_RE.lastIndex = 0; (pattern deleted)
  SENTENCE_START_NAME_RE.lastIndex = 0;
  LINE_START_NAME_RE.lastIndex = 0;
  LINE_START_CLINICAL_NAME_RE.lastIndex = 0;
  INFORMAL_NAME_HERE_RE.lastIndex = 0;
  UNDERSCORE_NAME_RE.lastIndex = 0;
  UNDERSCORE_ID_RE.lastIndex = 0;
  UNDERSCORE_DATE_RE.lastIndex = 0;
  TITLE_NAME_LOWERCASE_RE.lastIndex = 0;
  FIRST_NAME_CLINICAL_RE.lastIndex = 0;
  PROCEDURE_NOTE_NAME_RE.lastIndex = 0;
  STANDALONE_ALPHANUMERIC_ID_RE.lastIndex = 0;
  PAREN_ID_RE.lastIndex = 0;
  PT_LOWERCASE_NAME_RE.lastIndex = 0;
  LOWERCASE_NAME_DATE_RE.lastIndex = 0;
  LOWERCASE_NAME_AGE_GENDER_RE.lastIndex = 0;
  INFORMAL_NAME_HERE_TO_RE.lastIndex = 0;
  LOWERCASE_NAME_NOTE_RE.lastIndex = 0;
  NAME_START_CLINICAL_CONTEXT_RE.lastIndex = 0;
  LOWERCASE_FOR_NAME_RE.lastIndex = 0;
  LAST_FIRST_INITIAL_RE.lastIndex = 0;
  PROVIDER_ROLE_LINE_RE.lastIndex = 0;
  CREDENTIAL_NAME_RE.lastIndex = 0;
  ALLCAPS_LAST_FIRST_RE.lastIndex = 0;
  SIGNATURE_BLOCK_ALLCAPS_RE.lastIndex = 0;
  FACILITY_ACRONYM_CITY_RE.lastIndex = 0;
  DATE_DDMMMYYYY_RE.lastIndex = 0;
  DATE_DDMMMYYYY_SPACED_RE.lastIndex = 0;
  DATE_SLASH_RE.lastIndex = 0;
  DATE_ISO_RE.lastIndex = 0;
  DOB_HEADER_RE.lastIndex = 0;
  TIME_RE.lastIndex = 0;

  // Helper: check if followed by provider credentials (to exclude provider names)
  function isFollowedByCredentials(matchEnd) {
    const after = text.slice(matchEnd, Math.min(text.length, matchEnd + 40));
    return /^,?\s*(?:MD|DO|RN|RT|PA|NP|CRNA|PhD|FCCP|DAABIP)\b/i.test(after);
  }

  // Helper: check if preceded by provider context
  function isPrecededByProviderContext(matchStart) {
    const before = text.slice(Math.max(0, matchStart - 60), matchStart).toLowerCase();
    return /(?:dr\.?|attending|proceduralist|assistant|fellow|resident|surgeon|operator|anesthesiologist|physician)\s*[:\-]?\s*$/i.test(before);
  }

  // 1) Patient header names
  for (const match of text.matchAll(PATIENT_HEADER_RE)) {
    const fullMatch = match[0];
    const nameGroup = match[1];
    const groupOffset = fullMatch.indexOf(nameGroup);
    if (groupOffset !== -1 && match.index != null) {
      spans.push({
        start: match.index + groupOffset,
        end: match.index + groupOffset + nameGroup.length,
        label: "PATIENT",
        score: 1.0,
        source: "regex_header",
      });
    }
  }

  // 1-allcaps) ALL-CAPS patient names after headers: "PATIENT NAME: CHARLES D HOLLINGER"
  // NER often fails on all-uppercase names, so we need dedicated regex
  for (const match of text.matchAll(PATIENT_HEADER_ALLCAPS_RE)) {
    const fullMatch = match[0];
    const nameGroup = match[1];
    const groupOffset = fullMatch.indexOf(nameGroup);
    if (groupOffset !== -1 && match.index != null) {
      // Skip if the name is a known medical term (e.g., "PREOPERATIVE DIAGNOSIS")
      const nameNorm = nameGroup.toLowerCase().replace(/[^a-z\s]/g, "").trim();
      const isMedicalTerm = /^(preoperative|postoperative|intraoperative|surgical|medical|clinical|diagnostic|therapeutic)\s+(diagnosis|procedure|findings|impression|history)$/i.test(nameGroup);
      if (!isMedicalTerm) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + nameGroup.length,
          label: "PATIENT",
          score: 1.0,
          source: "regex_header_allcaps",
        });
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

  // REMOVED: 1b) Patient shorthand - Age/gender demographics are NOT PHI
  // Pattern PATIENT_SHORTHAND_RE was deleted to prevent "68 female" false positives

  // 2) MRN / IDs
  for (const match of text.matchAll(MRN_RE)) {
    const fullMatch = match[0];
    const idGroup = match[1];
    const groupOffset = fullMatch.indexOf(idGroup);
    if (groupOffset !== -1 && match.index != null) {
      spans.push({
        start: match.index + groupOffset,
        end: match.index + groupOffset + idGroup.length,
        label: "ID",
        score: 1.0,
        source: "regex_mrn",
      });
    }
  }

  // 2a) MRN with spaces: "A92 555" or "AB 123 456"
  for (const match of text.matchAll(MRN_SPACED_RE)) {
    const fullMatch = match[0];
    const idGroup = match[1];
    const groupOffset = fullMatch.indexOf(idGroup);
    if (groupOffset !== -1 && match.index != null) {
      spans.push({
        start: match.index + groupOffset,
        end: match.index + groupOffset + idGroup.length,
        label: "ID",
        score: 1.0,
        source: "regex_mrn_spaced",
      });
    }
  }

  // 2b) Case/accession/specimen IDs: "case c-847", "specimen A-12345"
  for (const match of text.matchAll(CASE_ID_RE)) {
    const idGroup = match[1];
    if (idGroup && match.index != null) {
      const fullMatch = match[0];
      const groupOffset = fullMatch.indexOf(idGroup);
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + idGroup.length,
          label: "ID",
          score: 0.95,
          source: "regex_case_id",
        });
      }
    }
  }

  // 2c) Parenthetical IDs: "(22352321)" - numeric IDs in parentheses after patient context
  for (const match of text.matchAll(PAREN_ID_RE)) {
    const idGroup = match[1];
    if (idGroup && match.index != null) {
      // Capture the ID inside the parentheses (not the parentheses themselves)
      spans.push({
        start: match.index + 1, // Skip opening paren
        end: match.index + 1 + idGroup.length,
        label: "ID",
        score: 0.95,
        source: "regex_paren_id",
      });
    }
  }

  // 3) Inline narrative names: "Emma Jones, a 64-year-old..."
  for (const match of text.matchAll(INLINE_PATIENT_NAME_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const matchEnd = match.index + nameGroup.length;
      // Skip if followed by credentials (likely provider, not patient)
      if (!isFollowedByCredentials(matchEnd) && !isPrecededByProviderContext(match.index)) {
        spans.push({
          start: match.index,
          end: matchEnd,
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
        // Skip if followed by credentials (likely provider)
        if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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

  // 4) "pt Name mrn 1234" pattern
  for (const match of text.matchAll(PT_NAME_MRN_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      // Find where the name starts within the match
      const fullMatch = match[0];
      const groupOffset = fullMatch.toLowerCase().indexOf(nameGroup.toLowerCase());
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + nameGroup.length,
          label: "PATIENT",
          score: 0.95,
          source: "regex_pt_mrn",
        });
      }
    }
  }

  // 5) Title + Name: "Mr. Smith", "Mrs. Johnson"
  for (const match of text.matchAll(TITLE_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const matchEnd = match.index + fullMatch.length;
      // Skip if followed by credentials (likely provider)
      if (!isFollowedByCredentials(matchEnd) && !isPrecededByProviderContext(match.index)) {
        spans.push({
          start: match.index,
          end: matchEnd,
          label: "PATIENT",
          score: 0.9,
          source: "regex_title_name",
        });
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
        // Skip if followed by credentials (likely provider)
        if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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
        // Skip if followed by credentials (likely provider)
        if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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

  // 5b2) "did an EBUS on [Name]" pattern: "We did an EBUS on Gregory Martinez today"
  for (const match of text.matchAll(DID_PROCEDURE_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        // Skip if followed by credentials (likely provider)
        if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.9,
            source: "regex_did_procedure",
          });
        }
      }
    }
  }

  // 5b3) "EBUS for [Name]" pattern: "EBUS for Arthur Curry. We looked at all the nodes."
  for (const match of text.matchAll(PROCEDURE_FOR_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      // Phase 1 gating: Skip obvious clinical phrases (e.g., "peripheral lesion")
      const nameParts = nameGroup.trim().split(/\s+/);
      const firstName = nameParts[0] || "";
      const lastName = nameParts[1] || "";
      if (SINGLE_NAME_CLINICAL_STOPLIST.has(firstName.toLowerCase()) ||
          SINGLE_NAME_CLINICAL_STOPLIST.has(lastName.toLowerCase()) ||
          isInClinicalStoplist(firstName) ||
          isInClinicalStoplist(lastName)) {
        continue;
      }

      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        // Skip if followed by credentials (likely provider)
        if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.9,
            source: "regex_procedure_for",
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
        // Skip if followed by credentials (likely provider)
        if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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
      // Skip if followed by credentials (likely provider)
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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

  // 5d2) Line-start name with clinical context: "Daniel Rivera LLL nodule small 14mm."
  for (const match of text.matchAll(LINE_START_CLINICAL_NAME_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      // Skip if followed by credentials (likely provider)
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.9,
          source: "regex_line_start_clinical",
        });
      }
    }
  }

  // 5e) Informal lowercase names: "jason phillips here for right lung lavage"
  for (const match of text.matchAll(INFORMAL_NAME_HERE_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;
      // Skip if followed by credentials (unlikely for lowercase but check anyway)
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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
      // Redact the entire match including underscores
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
      // Skip if followed by credentials (likely provider)
      if (!isFollowedByCredentials(matchEnd) && !isPrecededByProviderContext(match.index)) {
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
  // GATED: Skip single-word clinical terms like "Air", "Still", "Flow"
  for (const match of text.matchAll(FIRST_NAME_CLINICAL_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      // Phase 1 gating: Skip if name is a single-word clinical term
      if (SINGLE_NAME_CLINICAL_STOPLIST.has(nameGroup.toLowerCase())) {
        continue;
      }

      const nameEnd = match.index + nameGroup.length;
      // Skip if followed by credentials or preceded by provider context
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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
        // Skip if followed by credentials (likely provider)
        if (!isFollowedByCredentials(nameEnd)) {
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
      // Check context to confirm this looks like an identifier (not a device model)
      const ctx = text.slice(Math.max(0, match.index - 30), Math.min(text.length, match.index + idGroup.length + 30)).toLowerCase();
      // Only match if in patient/ID context, not device context
      if (/\b(?:mrn|patient|id|record|chart)\b/i.test(ctx) || !(/\b(?:model|scope|device|system|platform)\b/i.test(ctx))) {
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
        if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
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

  // 5m) Name at start with clinical context: "Brenda Lewis transplant patient with stenosis"
  // GATED: Skip if first or second word is a clinical term (e.g., "Serial irrigation")
  for (const match of text.matchAll(NAME_START_CLINICAL_CONTEXT_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      // Phase 1 gating: Parse name parts and check clinical stoplist
      const nameParts = nameGroup.trim().split(/\s+/);
      const firstName = nameParts[0] || "";
      const lastName = nameParts[1] || "";

      // Skip if either name part is a clinical term
      if (SINGLE_NAME_CLINICAL_STOPLIST.has(firstName.toLowerCase()) ||
          SINGLE_NAME_CLINICAL_STOPLIST.has(lastName.toLowerCase()) ||
          isInClinicalStoplist(firstName) ||
          isInClinicalStoplist(lastName)) {
        continue;
      }

      const nameEnd = match.index + nameGroup.length;
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.85,
          source: "regex_name_start_clinical",
        });
      }
    }
  }

  // 5n) Lowercase name after "for": "diag bronch for charlene king she has hilar adenopathy"
  // GATED: Skip if either word is a clinical term (e.g., "dilation to")
  for (const match of text.matchAll(LOWERCASE_FOR_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      // Phase 1 gating: Parse name parts and check clinical stoplist
      const nameParts = nameGroup.trim().split(/\s+/);
      const firstName = nameParts[0] || "";
      const secondName = nameParts[1] || "";

      // Skip if either word is a clinical term
      if (SINGLE_NAME_CLINICAL_STOPLIST.has(firstName.toLowerCase()) ||
          SINGLE_NAME_CLINICAL_STOPLIST.has(secondName.toLowerCase()) ||
          isInClinicalStoplist(firstName) ||
          isInClinicalStoplist(secondName)) {
        continue;
      }

      // Find where the name starts within the match (after "for ")
      const groupOffset = fullMatch.toLowerCase().indexOf(nameGroup.toLowerCase());
      if (groupOffset !== -1) {
        const nameStart = match.index + groupOffset;
        const nameEnd = nameStart + nameGroup.length;
        if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
          spans.push({
            start: nameStart,
            end: nameEnd,
            label: "PATIENT",
            score: 0.8,
            source: "regex_lowercase_for",
          });
        }
      }
    }
  }

  // 5o) "Last, First M" format with trailing initial: "Carey , Cloyd D", "Smith, John Jr"
  // GATED: Skip if either part matches clinical stoplist (e.g., "Elastography, First")
  for (const match of text.matchAll(LAST_FIRST_INITIAL_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const nameEnd = match.index + nameGroup.length;

      // Phase 1 gating: Parse "Last, First" and check clinical stoplist
      const commaParts = nameGroup.split(/\s*,\s*/);
      const lastName = commaParts[0]?.trim() || "";
      const firstNamePart = commaParts[1]?.trim().split(/\s+/)[0] || "";

      // Skip if either name part is a clinical term
      if (isInClinicalStoplist(lastName) || isInClinicalStoplist(firstNamePart)) {
        continue;
      }

      // Skip if followed by credentials (likely provider) or preceded by provider context
      if (!isFollowedByCredentials(nameEnd) && !isPrecededByProviderContext(match.index)) {
        spans.push({
          start: match.index,
          end: nameEnd,
          label: "PATIENT",
          score: 0.9,
          source: "regex_last_first_initial",
        });
      }
    }
  }

  // 5p) Provider/staff role lines: "Staff: Miller", "Fellow: Derek Booth"
  for (const match of text.matchAll(PROVIDER_ROLE_LINE_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + nameGroup.length,
          label: "PATIENT",
          score: 0.98,
          source: "regex_provider_role",
        });
      }
    }
  }

  // 5q) Credentialed provider names: "Derek Booth, DO"
  for (const match of text.matchAll(CREDENTIAL_NAME_RE)) {
    const nameGroup = match[1];
    const fullMatch = match[0];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + nameGroup.length,
          label: "PATIENT",
          score: 0.98,
          source: "regex_provider_credential",
        });
      }
    }
  }

  // 5r) All-caps "LAST, FIRST" signature formats: "BOOTH, DEREK ALLEN"
  for (const match of text.matchAll(ALLCAPS_LAST_FIRST_RE)) {
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const words = nameGroup
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, " ")
        .trim()
        .split(/\s+/)
        .filter(Boolean);
      if (words.some((w) => SINGLE_NAME_CLINICAL_STOPLIST.has(w) || NAME_REGEX_CLINICAL_STOPLIST.has(w))) {
        continue;
      }
      spans.push({
        start: match.index,
        end: match.index + nameGroup.length,
        label: "PATIENT",
        score: 0.97,
        source: "regex_provider_allcaps_last_first",
      });
    }
  }

  // 5s) Signature blocks with underscore separators + ALL-CAPS name line
  for (const match of text.matchAll(SIGNATURE_BLOCK_ALLCAPS_RE)) {
    const fullMatch = match[0];
    const nameGroup = match[1];
    if (nameGroup && match.index != null) {
      const groupOffset = fullMatch.indexOf(nameGroup);
      if (groupOffset !== -1) {
        spans.push({
          start: match.index + groupOffset,
          end: match.index + groupOffset + nameGroup.length,
          label: "PATIENT",
          score: 0.98,
          source: "regex_provider_signature_block",
        });
      }
    }
  }

  // 6) DOB header dates: "DOB: 01/15/1960"
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

  // 8) Date formats: "18Apr2022", "18-Apr-2022"
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

  // 9) Slash/dash dates: "6/3/2016", "06-03-2016"
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

  // 10) ISO dates: "2024-01-15" (YYYY-MM-DD)
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

  // 10b) Month-name dates: "January 15, 1960" / "Jan 15" (year optional)
  for (const match of text.matchAll(DATE_MONTH_NAME_RE)) {
    const dateGroup = match[1];
    if (dateGroup && match.index != null) {
      spans.push({
        start: match.index,
        end: match.index + dateGroup.length,
        label: "DATE",
        score: 0.93,
        source: "regex_date_month_name",
      });
    }
  }

  // 11) Timestamps: "10:00:00 AM", "14:30", "2:15 PM"
  // Only match if preceded by date context (to avoid matching times in other contexts like "station 4:30")
  for (const match of text.matchAll(TIME_RE)) {
    const timeGroup = match[1];
    if (timeGroup && match.index != null) {
      // Check if preceded by date-related context to reduce false positives
      const before = text.slice(Math.max(0, match.index - 80), match.index).toLowerCase();
      // Allow various delimiters between date and time: space, "/", ",", "@", "at"
      const hasDateContext =
        // "date/time of procedure:", "scheduled for:", etc.
        /(?:date|time|procedure|scheduled|at|on)\s*[:\-]?\s*(?:\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})?[\s\/,@]*$/i.test(before) ||
        // Date immediately before (with optional delimiter): "2/18/2018/ " or "2/18/2018 "
        /\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}[\/\s,@]*$/.test(before) ||
        // Time header without date: "TIME:" or "TIME OF PROCEDURE:"
        /\btime\s*(?:of\s+procedure)?[:\-]\s*$/i.test(before);
      if (hasDateContext) {
        spans.push({
          start: match.index,
          end: match.index + timeGroup.length,
          label: "DATE",
          score: 0.85,
          source: "regex_time",
        });
      }
    }
  }

  // 12) Facility/institution names
  // Ensures facility names are redacted as PHI (GEO) and avoids partial-token redactions.
  for (const re of [FACILITY_NAME_RE, FACILITY_CAMEL_HEALTH_RE, FACILITY_ENDING_HEALTH_RE, STATE_MEDICINE_RE, FACILITY_ACRONYM_CITY_RE]) {
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
// NER: robust offsets
// =============================================================================

function getOffsetsMappingFromTokenizerEncoding(encoding) {
  const mapping = encoding?.offset_mapping ?? encoding?.offsets ?? encoding?.offsetMapping;
  if (!Array.isArray(mapping)) return null;

  // Some tokenizers return a batch: [ [ [s,e], ... ] ]
  const candidate = Array.isArray(mapping[0]) && Array.isArray(mapping[0][0]) ? mapping[0] : mapping;
  if (!Array.isArray(candidate) || candidate.length === 0) return null;
  if (!Array.isArray(candidate[0]) || candidate[0].length < 2) return null;
  if (typeof candidate[0][0] !== "number" || typeof candidate[0][1] !== "number") return null;
  return candidate;
}

function getOffsetPair(offsets, index) {
  if (!Array.isArray(offsets) || typeof index !== "number" || !Number.isFinite(index)) return null;
  const idx = toFiniteInt(index);
  if (idx === null) return null;
  const pair = offsets[idx] ?? (idx > 0 ? offsets[idx - 1] : null);
  if (!Array.isArray(pair) || pair.length < 2) return null;
  const start = pair[0];
  const end = pair[1];
  if (typeof start !== "number" || typeof end !== "number") return null;
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) return null;
  return [start, end];
}

function getEntityText(ent) {
  const text =
    typeof ent?.word === "string"
      ? ent.word
      : typeof ent?.token === "string"
      ? ent.token
      : typeof ent?.text === "string"
      ? ent.text
      : typeof ent?.value === "string"
      ? ent.value
      : null;
  if (!text) return null;
  return String(text).replace(/^##/, "");
}

async function runNER(chunk, config = {}, isQuantized = false) {
  if (!classifier) return [];
  const raw = await classifier(chunk, {
    aggregation_strategy: "simple",
    ignore_labels: ["O"],
  });

  const rawList = normalizeRawOutput(raw);
  log("[PHI] raw spans (simple) count:", rawList.length);
  if (debug) log("[PHI] using per-label thresholds, isQuantized:", isQuantized);

  const spans = [];
  let offsets = null;
  let offsetsTried = false;
  let searchCursor = 0;

  for (const ent of rawList) {
    let start =
      toFiniteNumber(ent?.start) ??
      toFiniteNumber(ent?.start_offset) ??
      toFiniteNumber(ent?.begin);

    let end =
      toFiniteNumber(ent?.end) ??
      toFiniteNumber(ent?.end_offset) ??
      toFiniteNumber(ent?.finish);

    const score =
      toFiniteNumber(ent?.score) ??
      toFiniteNumber(ent?.confidence) ??
      toFiniteNumber(ent?.probability) ??
      0.0;

    // If offsets are missing/bad, try to recover.
    if (typeof start !== "number" || typeof end !== "number" || end <= start) {
      const tokenIndex =
        toFiniteInt(ent?.index) ??
        toFiniteInt(ent?.token) ??
        toFiniteInt(ent?.position) ??
        toFiniteInt(ent?.token_index) ??
        toFiniteInt(ent?.tokenIndex) ??
        null;

      const startTokenIndex =
        toFiniteInt(ent?.start_token) ??
        toFiniteInt(ent?.startToken) ??
        toFiniteInt(ent?.start_index) ??
        toFiniteInt(ent?.startIndex) ??
        null;

      const endTokenIndex =
        toFiniteInt(ent?.end_token) ??
        toFiniteInt(ent?.endToken) ??
        toFiniteInt(ent?.end_index) ??
        toFiniteInt(ent?.endIndex) ??
        null;

      const needsOffsets =
        tokenIndex !== null ||
        startTokenIndex !== null ||
        endTokenIndex !== null ||
        Boolean(getEntityText(ent));

      if (needsOffsets && !offsetsTried) {
        offsetsTried = true;
        try {
          const enc = await classifier.tokenizer(chunk, { return_offsets_mapping: true });
          offsets = getOffsetsMappingFromTokenizerEncoding(enc);
          log("[PHI] tokenizer offsets mapping count:", offsets ? offsets.length : null);
        } catch (err) {
          log("[PHI] tokenizer return_offsets_mapping failed:", err);
        }
      }

      if (offsets) {
        if (startTokenIndex !== null || endTokenIndex !== null) {
          const sPair =
            getOffsetPair(offsets, startTokenIndex) ??
            (startTokenIndex !== null ? getOffsetPair(offsets, startTokenIndex + 1) : null);
          const ePair =
            getOffsetPair(offsets, endTokenIndex) ??
            (endTokenIndex !== null ? getOffsetPair(offsets, endTokenIndex + 1) : null);
          if (sPair && ePair) {
            start = sPair[0];
            end = ePair[1];
          }
        } else if (tokenIndex !== null) {
          const pair = getOffsetPair(offsets, tokenIndex) ?? getOffsetPair(offsets, tokenIndex + 1);
          if (pair) {
            start = pair[0];
            end = pair[1];
          }
        }
      }

      // Last-resort: find token text in the chunk (case-insensitive) with cursor.
      if (typeof start !== "number" || typeof end !== "number" || end <= start) {
        const tokenText = getEntityText(ent);
        if (tokenText) {
          const candidates = tokenText.trim() !== tokenText ? [tokenText, tokenText.trim()] : [tokenText];
          const chunkLower = chunk.toLowerCase();

          let found = -1;
          let foundLen = 0;

          for (const t of candidates) {
            const tLower = t.toLowerCase();
            const idx = chunkLower.indexOf(tLower, searchCursor);
            if (idx !== -1) {
              found = idx;
              foundLen = t.length;
              break;
            }
          }

          if (found === -1 && searchCursor > 0) {
            for (const t of candidates) {
              const tLower = t.toLowerCase();
              const idx = chunkLower.indexOf(tLower);
              if (idx !== -1) {
                found = idx;
                foundLen = t.length;
                break;
              }
            }
          }

          if (found !== -1 && foundLen > 0) {
            start = found;
            end = found + foundLen;
          }
        }
      }
    }

    if (typeof start !== "number" || typeof end !== "number" || end <= start) continue;
    if (end - start < 1) continue;

    // Get label for threshold lookup
    const entLabel = normalizeLabel(ent?.entity_group || ent?.entity || ent?.label);

    // Phase 4: Apply per-label thresholds
    const labelThreshold = getThresholdForLabel(config, entLabel, isQuantized);
    if (typeof score === "number" && score < labelThreshold) {
      if (debug) log("[PHI] skipping span below threshold:", entLabel, score, "<", labelThreshold);
      continue;
    }

    searchCursor = Math.max(searchCursor, end);

    spans.push({
      start,
      end,
      label: entLabel,
      score: typeof score === "number" ? score : 0.0,
      source: "ner",
    });
  }

  // If model returns nothing, optionally dump token debug once per run.
  if (spans.length === 0 && debug && !didTokenDebug) {
    didTokenDebug = true;
    await debugTokenPredictions(chunk);
  }

  return spans;
}

// =============================================================================
// Span utilities: dedupe, merge, word-boundary expansion
// =============================================================================

function dedupeSpans(spans) {
  const seen = new Set();
  const out = [];
  for (const s of spans) {
    const k = `${s.start}:${s.end}:${s.label}:${s.source || ""}`;
    if (!seen.has(k)) {
      seen.add(k);
      out.push(s);
    }
  }
  return out;
}

/**
 * Deduplicate EXACT duplicates only (same start, end, label).
 * Does NOT drop spans due to overlap with different source/label.
 * Used in union mode before veto to preserve all candidates.
 *
 * Key difference from dedupeSpans: ignores source in the key, so two spans
 * at the same position with the same label are treated as duplicates even
 * if one is from regex and one from ML.
 *
 * @param {Array<{start: number, end: number, label: string, source?: string, score?: number}>} spans
 * @returns {Array} Deduplicated spans (only exact matches removed)
 */
function dedupeExactSpansOnly(spans) {
  const seen = new Map(); // key -> span with highest score

  for (const s of spans) {
    // Key includes start, end, and label (but NOT source)
    // Two spans at same position with same label are duplicates regardless of source
    const key = `${s.start}:${s.end}:${s.label}`;

    const existing = seen.get(key);
    if (!existing) {
      seen.set(key, s);
    } else {
      // Keep the one with higher score (prefer regex > ML on tie)
      const existingScore = existing.score ?? 0;
      const newScore = s.score ?? 0;
      const existingIsRegex = isRegexSpan(existing);
      const newIsRegex = isRegexSpan(s);

      if (newScore > existingScore || (newScore === existingScore && newIsRegex && !existingIsRegex)) {
        seen.set(key, s);
      }
    }
  }

  return Array.from(seen.values());
}

function isRegexSpan(s) {
  return typeof s?.source === "string" && s.source.startsWith("regex_");
}

function overlapsOrAdjacent(aStart, aEnd, bStart, bEnd) {
  // include adjacency (aEnd === bStart)
  return aStart <= bEnd && bStart <= aEnd;
}

function mergeOverlapsBestOf(spans) {
  const sorted = [...spans].sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    // Prefer regex then higher score
    const aR = isRegexSpan(a) ? 1 : 0;
    const bR = isRegexSpan(b) ? 1 : 0;
    if (aR !== bR) return bR - aR;
    return (b.score ?? 0) - (a.score ?? 0);
  });

  const out = [];
  for (const s of sorted) {
    const last = out[out.length - 1];
    if (!last || !overlapsOrAdjacent(last.start, last.end, s.start, s.end)) {
      out.push({ ...s });
      continue;
    }

    // Overlapping or adjacent
    const overlapLen = Math.max(0, Math.min(last.end, s.end) - Math.max(last.start, s.start));
    const lastIsRegex = isRegexSpan(last);
    const sIsRegex = isRegexSpan(s);

    // If same label and either is regex, UNION the spans (take min start, max end)
    // This ensures trailing initials like "D" in "Carey , Cloyd D" are captured
    if (overlapLen > 0 && last.label === s.label && (lastIsRegex || sIsRegex)) {
      out[out.length - 1] = {
        ...(lastIsRegex ? last : s), // Keep regex span's metadata
        start: Math.min(last.start, s.start),
        end: Math.max(last.end, s.end),
        score: Math.max(last.score ?? 0, s.score ?? 0),
      };
      continue;
    }

    // If different labels and either is regex, prefer the regex span
    if (overlapLen > 0 && (lastIsRegex || sIsRegex)) {
      const keep = lastIsRegex ? last : s;
      out[out.length - 1] = { ...keep };
      continue;
    }

    // If same label, union them (also merges adjacent token pieces nicely)
    if (last.label === s.label) {
      out[out.length - 1] = {
        ...last,
        start: Math.min(last.start, s.start),
        end: Math.max(last.end, s.end),
        score: Math.max(last.score ?? 0, s.score ?? 0),
        source: last.source || s.source,
      };
      continue;
    }

    // Different labels: only replace if overlap is huge; otherwise keep both.
    const lastLen = Math.max(1, last.end - last.start);
    const sLen = Math.max(1, s.end - s.start);
    const overlapRatio = overlapLen / Math.min(lastLen, sLen);

    if (overlapRatio >= 0.8) {
      if ((s.score ?? 0) > (last.score ?? 0)) out[out.length - 1] = { ...s };
    } else {
      out.push({ ...s });
    }
  }

  return out;
}

/**
 * Expand spans to full word boundaries to prevent partial-word redactions.
 * - Fixes cases like "id[REDACTED]" when the model only tagged part of a token.
 */
function expandToWordBoundaries(spans, fullText) {
  function isWordCharAt(i) {
    if (i < 0 || i >= fullText.length) return false;
    const ch = fullText[i];
    if (/[A-Za-z0-9]/.test(ch)) return true;

    // Treat apostrophe/hyphen as word-char only when adjacent to alnum
    if (ch === "'" || ch === "’" || ch === "-") {
      const left = i > 0 ? fullText[i - 1] : "";
      const right = i + 1 < fullText.length ? fullText[i + 1] : "";
      return /[A-Za-z0-9]/.test(left) || /[A-Za-z0-9]/.test(right);
    }
    return false;
  }

  return spans.map((span) => {
    let { start, end } = span;

    while (start > 0 && isWordCharAt(start - 1)) start--;
    while (end < fullText.length && isWordCharAt(end)) end++;

    if (start !== span.start || end !== span.end) {
      return { ...span, start, end, text: fullText.slice(start, end) };
    }
    return span;
  });
}

/**
 * Escape special regex characters in a string.
 */
function escapeRegex(str) {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function attachStableIds(spans, fullText) {
  const seen = new Map();
  const input = Array.isArray(spans) ? spans : [];
  return input.map((s) => {
    if (s && typeof s.id === "string" && s.id) return s;
    const label = String(s?.label ?? "OTHER");
    const source = String(s?.source ?? "unknown");
    const start = Number.isFinite(s?.start) ? s.start : -1;
    const end = Number.isFinite(s?.end) ? s.end : -1;
    const base = `${label}:${source}:${start}:${end}`;
    const n = (seen.get(base) || 0) + 1;
    seen.set(base, n);
    const id = n === 1 ? base : `${base}:${n}`;
    const text =
      typeof s?.text === "string"
        ? s.text
        : typeof fullText === "string" && start >= 0 && end > start
        ? fullText.slice(start, end)
        : undefined;
    return { ...s, id, text };
  });
}

/**
 * Session-based name tracking for document consistency.
 *
 * Collects high-confidence PATIENT names from existing detections,
 * then scans for any undetected occurrences of those names elsewhere
 * in the document. This ensures that if "John Smith" is detected once
 * with high confidence, all other mentions of "John Smith" are also caught.
 *
 * @param {Array} spans - Current span array
 * @param {string} text - Full document text
 * @param {Object} options - { debug: boolean }
 * @returns {Array} Updated spans with additional session name matches
 */
function addSessionNameMatches(spans, text, options = {}) {
  const { debug } = options;
  const log = debug ? console.log.bind(console) : () => {};

  if (!spans || spans.length === 0 || !text) return spans;

  // Helper: check if a phrase contains clinical terms (should not be session-tracked)
  function containsClinicalTerms(phrase) {
    const words = phrase
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, " ")
      .trim()
      .split(/\s+/)
      .filter(Boolean);
    for (const word of words) {
      if (SINGLE_NAME_CLINICAL_STOPLIST.has(word) || NAME_REGEX_CLINICAL_STOPLIST.has(word)) {
        return true;
      }
    }
    // Also check for laterality + anatomy patterns
    if (/\b(left|right|bilateral)\s+(adrenal|lobe|segment|bronchus|carina|hilum|mainstem)/i.test(phrase)) {
      return true;
    }
    return false;
  }

  // Collect confirmed high-confidence PATIENT names
  const confirmedNames = new Set();
  for (const span of spans) {
    const labelNorm = String(span.label || "").toUpperCase().replace(/^[BI]-/, "");
    if (labelNorm === "PATIENT" && (span.score ?? 0) >= 0.85) {
      const nameText = text.slice(span.start, span.end).trim();
      // Only track names with at least 4 characters (avoid initials)
      // Phase 1 gating: Skip clinical phrases like "left adrenal", "apical segment"
      if (nameText.length >= 4 && !containsClinicalTerms(nameText)) {
        confirmedNames.add(nameText);
      }
    }
  }

  if (confirmedNames.size === 0) {
    if (debug) log("[PHI] sessionNames: no high-confidence names found");
    return spans;
  }

  if (debug) log("[PHI] sessionNames: tracking", confirmedNames.size, "confirmed names");

  // Build a set of already-covered ranges for efficient lookup
  const coveredRanges = spans.map(s => ({ start: s.start, end: s.end }));

  function isCovered(start, end) {
    for (const r of coveredRanges) {
      // Fully contained within existing span
      if (start >= r.start && end <= r.end) return true;
    }
    return false;
  }

  const newSpans = [...spans];
  let addedCount = 0;

  // Scan for undetected occurrences of confirmed names
  for (const name of confirmedNames) {
    const nameRe = new RegExp(escapeRegex(name), 'gi');
    let match;
    while ((match = nameRe.exec(text)) !== null) {
      const start = match.index;
      const end = start + match[0].length;

      // Skip if already covered by an existing span
      if (isCovered(start, end)) continue;

      // Add as a new PATIENT span with session source
      newSpans.push({
        start,
        end,
        label: "PATIENT",
        score: 0.95,
        source: "regex_session_name",
        text: match[0],
      });
      addedCount++;
    }
  }

  if (debug && addedCount > 0) {
    log("[PHI] sessionNames: added", addedCount, "new matches");
  }

  return newSpans;
}

/**
 * Final overlap resolution AFTER veto has approved all survivors.
 * Produces non-overlapping spans sorted by start position.
 *
 * Selection rules for overlaps:
 * 1. Same label → union (min start, max end)
 * 2. Different labels:
 *    a) Prefer larger coverage (more characters)
 *    b) On tie: risk priority (ID > PATIENT > CONTACT > GEO > DATE)
 *    c) On tie: higher score
 *
 * IMPORTANT: Never creates spans bigger than the chosen candidate (no gap-bridging).
 *
 * @param {Array<{start: number, end: number, label: string, score?: number, source?: string}>} spans
 * @returns {Array} Non-overlapping spans sorted by start
 */
function finalResolveOverlaps(spans) {
  if (!spans || spans.length === 0) return [];

  // Risk priority: higher number = more critical to redact
  const RISK_PRIORITY = {
    ID: 5, // MRNs, SSNs, etc. - highest risk
    PATIENT: 4, // Patient names
    CONTACT: 3, // Phone, email, fax
    GEO: 2, // Addresses, locations
    DATE: 1, // Dates (often lower risk)
  };

  function getRiskPriority(label) {
    const normalized = String(label || "").toUpperCase().replace(/^[BI]-/, "");
    return RISK_PRIORITY[normalized] ?? 0;
  }

  function spanLength(span) {
    return span.end - span.start;
  }

  // Sort by start, then by end descending (larger spans first on same start)
  const sorted = [...spans].sort((a, b) => {
    if (a.start !== b.start) return a.start - b.start;
    return b.end - a.end; // Larger span first
  });

  const result = [];

  for (const span of sorted) {
    if (result.length === 0) {
      result.push({ ...span });
      continue;
    }

    const last = result[result.length - 1];

    // Check for overlap (not just adjacency)
    if (span.start < last.end) {
      // Overlapping spans
      const lastLabel = String(last.label || "").toUpperCase().replace(/^[BI]-/, "");
      const spanLabel = String(span.label || "").toUpperCase().replace(/^[BI]-/, "");

      // Rule 1: Same label → union
      if (lastLabel === spanLabel) {
        // Union: extend last to cover both
        result[result.length - 1] = {
          ...last,
          start: Math.min(last.start, span.start),
          end: Math.max(last.end, span.end),
          score: Math.max(last.score ?? 0, span.score ?? 0),
        };
        continue;
      }

      // Rule 2: Different labels → selection based on priority
      const lastLen = spanLength(last);
      const spanLen = spanLength(span);
      const lastPriority = getRiskPriority(lastLabel);
      const spanPriority = getRiskPriority(spanLabel);
      const lastScore = last.score ?? 0;
      const spanScore = span.score ?? 0;

      // Calculate overlap ratio to decide if we should keep both
      const overlapStart = Math.max(last.start, span.start);
      const overlapEnd = Math.min(last.end, span.end);
      const overlapLen = Math.max(0, overlapEnd - overlapStart);
      const overlapRatio = overlapLen / Math.min(lastLen, spanLen);

      // If overlap is < 50%, keep both (they cover different regions)
      if (overlapRatio < 0.5) {
        result.push({ ...span });
        continue;
      }

      // High overlap - pick winner based on: coverage > risk priority > score
      let keepLast = true;

      if (spanLen > lastLen) {
        keepLast = false;
      } else if (spanLen === lastLen) {
        if (spanPriority > lastPriority) {
          keepLast = false;
        } else if (spanPriority === lastPriority && spanScore > lastScore) {
          keepLast = false;
        }
      }

      if (!keepLast) {
        result[result.length - 1] = { ...span };
      }
      // If keepLast, we simply don't add span - last stays as winner
    } else {
      // No overlap - add span
      result.push({ ...span });
    }
  }

  return result;
}

/**
 * Extend GEO spans to include common multi-word city prefixes.
 * Fixes partial redactions like "San [REDACTED]" → "[REDACTED]" for "San Francisco".
 */
const CITY_PREFIXES = new Set([
  "san", "los", "las", "new", "fort", "saint", "st", "santa", "el", "la",
  "port", "mount", "mt", "north", "south", "east", "west", "upper", "lower",
  "lake", "palm", "long", "grand", "great", "little", "old", "big"
]);

// Facility suffix words that should be included in GEO spans
// Fixes: "Horizon University Medical Center" being split, leaving "Center" as separate
const FACILITY_SUFFIXES = new Set([
  "center", "hospital", "clinic", "institute", "university", "foundation",
  "medical", "health", "healthcare", "memorial", "regional", "general",
  "community", "children", "childrens", "pediatric", "veterans", "va"
]);

function extendGeoSpans(spans, fullText) {
  return spans.map((span) => {
    // Only extend GEO-labeled spans
    const label = String(span.label || "").toUpperCase().replace(/^[BI]-/, "");
    if (label !== "GEO") return span;

    let { start, end } = span;
    let newStart = start;
    let newEnd = end;

    // Look for city prefix word before the span
    const beforeWindow = fullText.slice(Math.max(0, start - 20), start);
    const prefixMatch = beforeWindow.match(/\b([A-Za-z]+)\s+$/);

    if (prefixMatch) {
      const prefix = prefixMatch[1].toLowerCase();
      if (CITY_PREFIXES.has(prefix)) {
        // Extend start to include the prefix
        newStart = start - prefixMatch[0].length;
      }
    }

    // Look for facility suffix words AFTER the span
    // Fixes: "Horizon University Medical Center" where "Center" was split off
    // Fixes: "Center," being split due to trailing punctuation
    const afterWindow = fullText.slice(end, Math.min(fullText.length, end + 40));
    // Match optional comma/space + one or more facility suffix words + optional trailing punctuation
    const suffixMatch = afterWindow.match(/^(\s*,?\s*(?:(?:Medical|Health|Healthcare)\s+)?(?:Center|Hospital|Clinic|Institute|University|Foundation|Memorial|Regional|General|Community|Children(?:'?s)?|Pediatric|Veterans|VA)(?:\s+(?:Medical\s+)?(?:Center|Hospital|Clinic))?[,;.]?)/i);

    if (suffixMatch) {
      newEnd = end + suffixMatch[1].length;
    }

    // Return extended span if any extension occurred
    if (newStart !== start || newEnd !== end) {
      return {
        ...span,
        start: newStart,
        end: newEnd,
        text: fullText.slice(newStart, newEnd)
      };
    }

    return span;
  });
}

/**
 * Extend PATIENT spans to include trailing initials (D, M, Jr, Sr, II, III, IV)
 * Fixes cases like "Carey , Cloyd D" where "D" is left as a dangling initial
 */
function extendPatientSpansForTrailingInitials(spans, fullText) {
  return spans.map((span) => {
    // Only extend PATIENT-labeled spans
    const label = String(span.label || "").toUpperCase().replace(/^[BI]-/, "");
    if (label !== "PATIENT") return span;

    const { end } = span;

    // Look for trailing initial or suffix after the span
    const afterWindow = fullText.slice(end, Math.min(fullText.length, end + 10));

    // Match: space + single capital letter (optional period) OR suffix like Jr, Sr, II, III, IV
    const trailingMatch = afterWindow.match(/^(\s+[A-Z]\.?|\s+(?:Jr|Sr|II|III|IV)\.?)(?:\s|$|,|;)/i);

    if (trailingMatch) {
      const newEnd = end + trailingMatch[1].length;
      return {
        ...span,
        end: newEnd,
        text: fullText.slice(span.start, newEnd)
      };
    }

    return span;
  });
}

/**
 * Extend PATIENT spans by one trailing name token when in a Name:/Patient: header line.
 * Fixes: "Patient: DIEDRICH, THERESA MARIE" where "MARIE" is left behind.
 */
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
      text: fullText.slice(span.start, newEnd)
    };
  });
}

// =============================================================================
// Debug helpers (optional)
// =============================================================================

function formatTokenPreview(token) {
  const word =
    typeof token?.word === "string"
      ? token.word
      : typeof token?.token === "string"
      ? token.token
      : typeof token?.text === "string"
      ? token.text
      : typeof token?.index === "number"
      ? String(token.index)
      : "(tok)";
  const label = normalizeLabel(token?.entity || token?.entity_group || token?.label);
  const score = typeof token?.score === "number" ? token.score : 0;
  return `${word} -> ${label} (${score.toFixed(3)})`;
}

async function debugTokenPredictions(chunk) {
  if (!debug || !classifier) return;
  try {
    let tokenRaw;
    try {
      tokenRaw = await classifier(chunk, {
        aggregation_strategy: "none",
        ignore_labels: [],
        return_offsets_mapping: true,
        topk: 1,
      });
    } catch (err) {
      log("[PHI] token preds debug (with offsets) failed; retrying without offsets:", err);
      tokenRaw = await classifier(chunk, {
        aggregation_strategy: "none",
        ignore_labels: [],
        topk: 1,
      });
    }

    const tokenList = normalizeRawOutput(tokenRaw);
    log("[PHI] token preds count:", tokenList.length);
    if (tokenList.length > 0) {
      log("[PHI] token preds preview:", tokenList.slice(0, 10).map(formatTokenPreview));
      return;
    }
  } catch (err) {
    log("[PHI] token preds debug failed:", err);
  }

  if (didLogitsDebug) return;
  didLogitsDebug = true;

  try {
    const inputs = await classifier.tokenizer(chunk, { return_tensors: "np" });
    const out = await classifier.model(inputs);
    const logits = out?.logits;
    log("[PHI] logits dims:", logits?.dims);
    const data = logits?.data;
    log("[PHI] logits sample:", data ? Array.from(data.slice(0, 20)) : null);
  } catch (err) {
    log("[PHI] logits debug failed:", err);
  }
}

// =============================================================================
// Worker message loop
// =============================================================================

self.onmessage = async (e) => {
  const msg = e.data;
  if (!msg || typeof msg.type !== "string") return;

  if (msg.type === "cancel") {
    cancelled = true;
    return;
  }

  if (msg.type === "init") {
    cancelled = false;
    didTokenDebug = false;
    didLogitsDebug = false;

    const config = msg.config && typeof msg.config === "object" ? msg.config : {};
    debug = Boolean(msg.debug ?? config.debug);

    try {
      await loadProtectedTerms();
      await loadModel(config);
      post("ready");
    } catch (err) {
      post("error", { message: String(err?.message || err) });
    }
    return;
  }

  if (msg.type === "start") {
    cancelled = false;
    didTokenDebug = false;
    didLogitsDebug = false;

    try {
      await loadProtectedTerms();

      const text = String(msg.text || "");
      const config = msg.config && typeof msg.config === "object" ? msg.config : {};
      debug = Boolean(config.debug);

      await loadModel(config);

      // Phase 4: Determine if using quantized model for threshold adjustments
      const isQuantized = !config.forceUnquantized && classifier === classifierQuantized;

      const allSpans = [];
      const windowCount = Math.max(1, Math.ceil(Math.max(0, text.length - OVERLAP) / STEP));
      let windowIndex = 0;

      post("progress", { stage: "Running detection (local model)…", windowIndex, windowCount });

      for (let start = 0; start < text.length; start += STEP) {
        const end = Math.min(text.length, start + WINDOW);
        const chunk = text.slice(start, end);

        // Avoid an extra tiny tail window (often low signal / higher false positives)
        if (start > 0 && chunk.length < 50) break;

        windowIndex += 1;

        // 1) ML spans (robust offsets) - now with per-label thresholds
        const nerSpans = await runNER(chunk, config, isQuantized);

        // 2) Regex injection spans (header guarantees)
        const regexSpans = runRegexDetectors(chunk);

        // 3) Combine (still chunk-relative)
        const combined = dedupeSpans([...nerSpans, ...regexSpans]);

        // 4) Convert to absolute offsets
        for (const s of combined) {
          allSpans.push({
            ...s,
            start: s.start + start,
            end: s.end + start,
          });
        }

        post("progress", { windowIndex, windowCount });
        if (cancelled) break;
      }

      // Determine merge mode from config
      const mergeMode = getMergeMode(config);

      if (debug) {
        log("[PHI] mergeMode:", mergeMode);
        log("[PHI] allSpans (all windows):", allSpans.length);
        // Count ML vs regex spans
        const mlCount = allSpans.filter((s) => !isRegexSpan(s)).length;
        const regexCount = allSpans.filter((s) => isRegexSpan(s)).length;
        log("[PHI] mlSpans:", mlCount, "regexSpans:", regexCount);
      }

      let merged;

      if (mergeMode === "union") {
        // ========== UNION MODE PIPELINE ==========
        // Keeps all candidates until AFTER veto, then resolves overlaps.
        // This prevents valid ML spans from being dropped when overlapping
        // regex spans are later vetoed as false positives.

        // 5) Remove only exact duplicates (keeps all overlap candidates)
        merged = dedupeExactSpansOnly(allSpans);
        if (debug) log("[PHI] afterExactDedupe:", merged.length);

        // 6) Expand to word boundaries (fixes partial-word redactions)
        merged = expandToWordBoundaries(merged, text);

        // 7) Extend PATIENT spans for trailing initials
        merged = extendPatientSpansForTrailingInitials(merged, text);
        merged = extendPatientSpansForTrailingNameToken(merged, text);

        // 8) Extend GEO spans to include city prefixes
        merged = extendGeoSpans(merged, text);
        if (debug) log("[PHI] afterExpand:", merged.length);

        // 9) Apply veto BEFORE final overlap resolution
        const beforeVetoCount = merged.length;
        const vetoOpts = { debug };
        if (typeof config.protectProviders === "boolean") vetoOpts.protectProviders = config.protectProviders;
        merged = applyVeto(merged, text, protectedTerms, vetoOpts);
        if (debug) {
          log("[PHI] vetoedCount:", beforeVetoCount - merged.length);
          log("[PHI] afterVeto:", merged.length);
        }

        // 9b) Session-based name tracking for document consistency
        merged = addSessionNameMatches(merged, text, { debug });
        if (debug) log("[PHI] afterSessionNames:", merged.length);

        // 10) Final overlap resolution AFTER veto has approved survivors
        merged = finalResolveOverlaps(merged);
        if (debug) log("[PHI] afterFinalResolve:", merged.length);

      } else {
        // ========== LEGACY BEST_OF MODE PIPELINE ==========
        // (Original behavior - may drop valid ML spans if regex span is later vetoed)

        // 5) Merge/dedupe across windows (may drop ML spans on overlap)
        merged = mergeOverlapsBestOf(allSpans);
        if (debug) log("[PHI] afterMergeBestOf:", merged.length);

        // 6) Expand to word boundaries (fixes partial-word redactions)
        merged = expandToWordBoundaries(merged, text);

        // 7) Extend PATIENT spans for trailing initials
        merged = extendPatientSpansForTrailingInitials(merged, text);
        merged = extendPatientSpansForTrailingNameToken(merged, text);

        // 8) Extend GEO spans to include city prefixes
        merged = extendGeoSpans(merged, text);
        if (debug) log("[PHI] afterExpand:", merged.length);

        // 9) Re-merge after expansion
        merged = mergeOverlapsBestOf(merged);
        if (debug) log("[PHI] afterReMerge:", merged.length);

        // 10) Apply veto
        const beforeVetoCount = merged.length;
        const vetoOpts = { debug };
        if (typeof config.protectProviders === "boolean") vetoOpts.protectProviders = config.protectProviders;
        merged = applyVeto(merged, text, protectedTerms, vetoOpts);
        if (debug) {
          log("[PHI] vetoedCount:", beforeVetoCount - merged.length);
          log("[PHI] afterVeto:", merged.length);
        }

        // 10b) Session-based name tracking for document consistency
        merged = addSessionNameMatches(merged, text, { debug });
        if (debug) log("[PHI] afterSessionNames:", merged.length);
      }

      merged = attachStableIds(merged, text);
      post("done", { detections: merged });
    } catch (err) {
      post("error", { message: String(err?.message || err) });
    }
  }
};
