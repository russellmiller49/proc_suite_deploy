const NUMBER_WORDS = {
  one: "1",
  two: "2",
  three: "3",
  four: "4",
  five: "5",
  six: "6",
  seven: "7",
  eight: "8",
  nine: "9",
  ten: "10",
  eleven: "11",
  twelve: "12",
};

const STATION_NUMBER_WORDS_WITH_SIDE = {
  ...NUMBER_WORDS,
  won: "1",
  to: "2",
  too: "2",
  tree: "3",
  for: "4",
  fore: "4",
  sicks: "6",
  sex: "6",
  ate: "8",
  "one one": "11",
  "1 1": "11",
  "one two": "12",
  "1 2": "12",
};

// Plain station numbers intentionally omit common homophones such as "to", "for",
// and "ate" because phrases like "station for the procedure" should not become
// "station 4 the procedure". Those homophones are only accepted when a side follows.
const STATION_NUMBER_WORDS_PLAIN = {
  ...NUMBER_WORDS,
  "one one": "11",
  "1 1": "11",
  "one two": "12",
  "1 2": "12",
};

const SIDE_WORDS = {
  r: "R",
  ar: "R",
  are: "R",
  our: "R",
  right: "R",
  l: "L",
  el: "L",
  ell: "L",
  left: "L",
};

const GAUGE_WORDS = {
  eighteen: "18",
  nineteen: "19",
  twenty: "20",
  "twenty one": "21",
  "twenty-one": "21",
  "twenty two": "22",
  "twenty-two": "22",
  "twenty five": "25",
  "twenty-five": "25",
};

const PERCENT_WORDS = {
  half: "0.5",
  one: "1",
  two: "2",
};

const STATION_PREFIX_WITH_SIDE_PATTERN = "(?:lymph\\s+node\\s+station|node\\s+station|station|level)";
const STATION_PREFIX_PLAIN_PATTERN = "(?:lymph\\s+node\\s+station|node\\s+station|station)";
const STATION_SIDE_PATTERN = "(?:right|left|are|our|ell|el|ar|r|l)";
const STATION_NUMBER_WITH_SIDE_PATTERN = [
  "one\\s+one",
  "1\\s+1",
  "one\\s+two",
  "1\\s+2",
  "eleven",
  "twelve",
  "three",
  "seven",
  "eight",
  "four",
  "five",
  "nine",
  "sicks",
  "six",
  "tree",
  "fore",
  "for",
  "sex",
  "ate",
  "ten",
  "too",
  "two",
  "to",
  "won",
  "one",
  "\\d{1,2}",
].join("|");
const STATION_NUMBER_PLAIN_PATTERN = [
  "one\\s+one",
  "1\\s+1",
  "one\\s+two",
  "1\\s+2",
  "eleven",
  "twelve",
  "three",
  "seven",
  "eight",
  "four",
  "five",
  "nine",
  "six",
  "ten",
  "two",
  "one",
  "\\d{1,2}",
].join("|");

const LOBE_SUFFIX = "(?:lobe|lobes|low|load|love|lob|lope)";
const LETTER_R = "(?:r|are|our)";
const LETTER_L = "(?:l|el|ell)";
const LETTER_U = "(?:u|you)";
const LETTER_M = "(?:m|em)";
const LETTER_SEPARATOR = "(?:\\s|[-.])+";

function normalizeWhitespace(text) {
  return String(text || "")
    .replace(/\r\n/g, "\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/[ \t]{2,}/g, " ")
    .replace(/[ \t]+([,.;:])/g, "$1")
    .replace(/([([{])\s+/g, "$1")
    .replace(/\s+([)\]}])/g, "$1")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}

function getReplacementOffset(args) {
  const maybeGroups = args[args.length - 1];
  const offset = typeof maybeGroups === "object" ? args[args.length - 3] : args[args.length - 2];
  return Number.isInteger(offset) ? offset : undefined;
}

function applyReplacement(current, regex, replacer, replacements, label) {
  return current.replace(regex, (...args) => {
    const match = String(args[0] || "");
    const next = typeof replacer === "function" ? replacer(...args) : String(replacer);
    if (!next || next === match) return match;
    replacements.push({
      label,
      from: match,
      to: next,
      index: getReplacementOffset(args),
    });
    return next;
  });
}

function applyRules(current, rules, replacements) {
  return rules.reduce(
    (updated, rule) => applyReplacement(updated, rule.regex, rule.replacement, replacements, rule.label),
    current,
  );
}

function cleanToken(raw) {
  return String(raw || "")
    .toLowerCase()
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function normalizeStationPrefix(rawPrefix) {
  const prefix = cleanToken(rawPrefix);
  if (prefix === "level") return "station";
  return prefix;
}

function normalizeStationNumber(rawNumber, allowSideHomophones = false) {
  const key = cleanToken(rawNumber);
  const map = allowSideHomophones ? STATION_NUMBER_WORDS_WITH_SIDE : STATION_NUMBER_WORDS_PLAIN;
  const compactDigits = key.replace(/\s+/g, "");
  const number = map[key] || (/^\d{1,2}$/.test(compactDigits) ? compactDigits : "");
  if (!number) return "";
  const numeric = Number(number);
  if (!Number.isInteger(numeric) || numeric < 1 || numeric > 12) return "";
  return String(numeric);
}

function normalizeSide(rawSide = "") {
  return SIDE_WORDS[cleanToken(rawSide)] || "";
}

function normalizeStationToken(rawNumber, rawSide = "") {
  const side = normalizeSide(rawSide);
  const number = normalizeStationNumber(rawNumber, Boolean(side));
  if (!number) return "";
  return `${number}${side}`;
}

function normalizeGauge(rawGauge) {
  const key = cleanToken(rawGauge);
  return GAUGE_WORDS[key] || key.replace(/[^0-9]/g, "");
}

function normalizePercent(rawPercent) {
  const key = cleanToken(rawPercent);
  return PERCENT_WORDS[key] || key;
}

const PROCEDURE_AND_DEVICE_RULES = [
  {
    label: "proc_ebus_guided_staging",
    regex: /\b(?:hebus|ebus|evis|evus|eviss?)\s+(?:skydits|guid(?:ed|ance)|guided)\s+staging\b/gi,
    replacement: "EBUS guided staging",
  },
  { label: "proc_ebus_staging", regex: /\b(?:even|evan|ibus|ebis|e[\s.-]*bus|e[\s.-]*best)\s+staging\b/gi, replacement: "EBUS staging" },
  { label: "proc_ebus_tbna", regex: /\b(?:hebus|ebus|evis|evus|eviss?)\s+tbna\b/gi, replacement: "EBUS-TBNA" },
  { label: "proc_ebus", regex: /\b(?:e[\s.-]*bus|e[\s.-]*bis|ibus|ebis|e[\s.-]*best)\b/gi, replacement: "EBUS" },
  { label: "proc_staging_ebus", regex: /\bstaging\s+(?:hebus|ebus|evis|evus|eviss?)\b/gi, replacement: "staging EBUS" },
  { label: "proc_endobronchial_ultrasound", regex: /\bendo\s+bronchial\s+ultra\s*sound\b/gi, replacement: "endobronchial ultrasound" },
  { label: "proc_endobronchial", regex: /\bendo\s+bronchial\b/gi, replacement: "endobronchial" },
  { label: "proc_transbronchial", regex: /\btrans\s+bronchial\b/gi, replacement: "transbronchial" },
  { label: "proc_peribronchial", regex: /\bperi\s+bronchial\b/gi, replacement: "peribronchial" },
  { label: "proc_flexible_bronchoscopy", regex: /\bflex(?:ible)?\s+(?:will|well|full|fill)?\s*bronch\b/gi, replacement: "Flexible bronchoscopy" },

  { label: "proc_bal_full", regex: /\b(?:broncho|bronco|bronchial|bronch)\s+alveolar\s+lavage\b/gi, replacement: "bronchoalveolar lavage" },
  { label: "proc_bal_joined", regex: /\bbroncho\s*alveolar\b/gi, replacement: "bronchoalveolar" },
  { label: "proc_bal", regex: /\b(?:(?:b|bee|be)[\s.-]*(?:a|ay)[\s.-]*(?:l|el|ell)|bals?)\b/gi, replacement: "BAL" },

  { label: "proc_tbna", regex: /\b(?:(?:t|tee)[\s.-]*(?:b|bee)[\s.-]*(?:n|en)[\s.-]*(?:a|ay)|tvna|tpna)\b/gi, replacement: "TBNA" },
  { label: "proc_tbna_full", regex: /\btransbronchial\s+needle\s+aspiration\b/gi, replacement: "transbronchial needle aspiration" },
  { label: "proc_tbbx", regex: /\b(?:t|tee)[\s.-]*(?:b|bee)[\s.-]*(?:b|bee)[\s.-]*(?:x|ex)\b/gi, replacement: "TBBx" },
  { label: "proc_ebbx", regex: /\b(?:e|ee)[\s.-]*(?:b|bee)[\s.-]*(?:b|bee)[\s.-]*(?:x|ex)\b/gi, replacement: "EBBx" },

  { label: "term_bronchoscope", regex: /\bbroncho\s+scope\b/gi, replacement: "bronchoscope" },
  { label: "term_bronchoscopic", regex: /\bbroncho\s+scopic\b/gi, replacement: "bronchoscopic" },
  {
    label: "term_bronchoscopy",
    regex: /\b(?:broncho\s+scopy|bronco\s+scopy|bronchosby|bronchoski|broncoscopy|run\s+cosby)\b/gi,
    replacement: "bronchoscopy",
  },

  { label: "proc_cryobiopsies", regex: /\bcryo(?:\s*|-)?(?:biopsies|biospies|bopsies|opsies|oopsies|volopsies)\b/gi, replacement: "cryobiopsies" },
  { label: "proc_cryobiopsy", regex: /\bcryo(?:\s*|-)?(?:biopsy|biospy|bopsy|opsy|oopsy|volopsy)\b/gi, replacement: "cryobiopsy" },

  { label: "proc_forceps_biopsy", regex: /\b(?:four\s+steps|forcepts?|forcep)\s+biops(?:y|ies)\b/gi, replacement: "forceps biopsy" },
  { label: "term_forceps", regex: /\b(?:forcepts?|forcep)\b/gi, replacement: "forceps" },
  { label: "term_catheter", regex: /\b(?:cathart|cathater|catheder)\b/gi, replacement: "catheter" },
  { label: "term_guide_sheath", regex: /\bguide\s+(?:sheet|sheeth|she)\b/gi, replacement: "guide sheath" },
  { label: "term_radial_probe", regex: /\bradial\s+(?:pro|prope)\b/gi, replacement: "radial probe" },
  { label: "term_fiducial", regex: /\b(?:fidu\s*seal|fid\s+u\s+cial|feducial)\b/gi, replacement: "fiducial" },

  { label: "term_fluoroscopy", regex: /\b(?:fluoro\s+scopy|floro\s+scopy|flow\s+row\s+scopy|fluoroscoopy|fluoresce\s+b)\b/gi, replacement: "fluoroscopy" },
  { label: "term_fluoro", regex: /\bfloro\b/gi, replacement: "fluoro" },
  { label: "term_c_arm", regex: /\b(?:c|see)\s*[-\s]?arm\b/gi, replacement: "C-arm" },
  { label: "term_cone_beam_ct", regex: /\b(?:cone|comb|combo|combi|combat(?:ing)?)\s*(?:beam\s*)?[-\s]*ct\b/gi, replacement: "cone beam CT" },
  { label: "term_cone_beam", regex: /\b(?:comb|combo|combi)\s+beam\b/gi, replacement: "cone beam" },
  { label: "term_conebeam", regex: /\bcone\s+beam\b/gi, replacement: "cone beam" },
  { label: "term_3d_reconstruction", regex: /\b3\s*d\s+reconstruction\b/gi, replacement: "3D reconstruction" },
  { label: "term_ground_glass_opacity", regex: /\bground\s+glass\s+capacity\b/gi, replacement: "ground glass opacity" },

  {
    label: "robotic_ion_phrase",
    regex: /\b(?:hi,\s*)?i(?:'m|\s+am)\s+robotic\s+bronchoscopy\b/gi,
    replacement: "Ion robotic bronchoscopy",
  },
  { label: "robotic_ion", regex: /\bion(?=\s+(?:robotic|platform|system|bronchoscopy|bronchoscope))\b/gi, replacement: "Ion" },
  { label: "robotic_monarch", regex: /\bmonarch(?=\s+(?:robotic|platform|system|bronchoscopy|bronchoscope))\b/gi, replacement: "Monarch" },
  { label: "robotic_superdimension", regex: /\bsuper\s+dimension\b/gi, replacement: "SuperDimension" },

  { label: "proc_apc", regex: /\b(?:a|ay)[\s.-]*(?:p|pee)[\s.-]*(?:c|see)\b/gi, replacement: "APC" },
  { label: "proc_argon_plasma", regex: /\bargon\s+plasma\s+(?:coagulation|coag)\b/gi, replacement: "argon plasma coagulation" },
  { label: "proc_balloon_dilation", regex: /\bballoon\s+dilatation\b/gi, replacement: "balloon dilation" },
];

const LOBE_RULES = [
  { label: "lobe_rul_phrase", regex: new RegExp(`\\b(?:right|write)\\s+upper\\s+${LOBE_SUFFIX}\\b`, "gi"), replacement: "right upper lobe" },
  { label: "lobe_rml_phrase", regex: new RegExp(`\\b(?:right|write)\\s+(?:middle|metal|medial)\\s+${LOBE_SUFFIX}\\b`, "gi"), replacement: "right middle lobe" },
  { label: "lobe_rll_phrase", regex: new RegExp(`\\b(?:right|write)\\s+lower\\s+${LOBE_SUFFIX}\\b`, "gi"), replacement: "right lower lobe" },
  { label: "lobe_lul_phrase", regex: new RegExp(`\\b(?:left\\s+upper|leftover|left\\s+over)\\s+${LOBE_SUFFIX}\\b`, "gi"), replacement: "left upper lobe" },
  { label: "lobe_lll_phrase", regex: new RegExp(`\\bleft\\s+lower\\s+${LOBE_SUFFIX}\\b`, "gi"), replacement: "left lower lobe" },
  {
    label: "lesion_rul_phrase",
    regex: /\bright\s+up\s+to\s+low[-\s]*veluation\b/gi,
    replacement: "right upper lobe lesion",
  },
  { label: "lobe_lingula", regex: /\b(?:lingular|lingula)\s+(?:low|load|love|lob|lope)\b/gi, replacement: "lingula" },
  { label: "lobe_rul", regex: new RegExp(`\\b${LETTER_R}${LETTER_SEPARATOR}${LETTER_U}${LETTER_SEPARATOR}${LETTER_L}\\b`, "gi"), replacement: "RUL" },
  { label: "lobe_rml", regex: new RegExp(`\\b${LETTER_R}${LETTER_SEPARATOR}${LETTER_M}${LETTER_SEPARATOR}${LETTER_L}\\b`, "gi"), replacement: "RML" },
  { label: "lobe_rll", regex: new RegExp(`\\b${LETTER_R}${LETTER_SEPARATOR}${LETTER_L}${LETTER_SEPARATOR}${LETTER_L}\\b`, "gi"), replacement: "RLL" },
  { label: "lobe_lul", regex: new RegExp(`\\b${LETTER_L}${LETTER_SEPARATOR}${LETTER_U}${LETTER_SEPARATOR}${LETTER_L}\\b`, "gi"), replacement: "LUL" },
  { label: "lobe_lll", regex: new RegExp(`\\b${LETTER_L}${LETTER_SEPARATOR}${LETTER_L}${LETTER_SEPARATOR}${LETTER_L}\\b`, "gi"), replacement: "LLL" },
];

const PATH_AND_SPECIMEN_RULES = [
  { label: "term_brushings_possessive", regex: /\bbrushings['’]s\b/gi, replacement: "brushings" },
  { label: "term_biopsies_possessive", regex: /\bbiopsies['’]s\b/gi, replacement: "biopsies" },
  { label: "term_washings_possessive", regex: /\bwashings['’]s\b/gi, replacement: "washings" },
  { label: "proc_rose_with", regex: /\bwith\s+(?:rows|rose|rohse)\b/gi, replacement: "with ROSE" },
  { label: "proc_rose_result", regex: /\b(?:rows|rose|rohse)(?=\s*[,.:;-]?\s+(?:positive|negative|adequate|inadequate|present|available|showed|shows|demonstrated|preliminary|cytology|confirmed|suggested|malignant)\b)/gi, replacement: "ROSE" },
  { label: "proc_rose_showed", regex: /\brose\s+road(?:,?\s*ship)?\b/gi, replacement: "ROSE showed" },
  { label: "proc_rose_full", regex: /\brapid\s+on[-\s]?site\s+evaluation\b/gi, replacement: "rapid on-site evaluation" },
  { label: "path_atypical_cells", regex: /\ba\s+(?:double|typical)\s+cells\b/gi, replacement: "atypical cells" },
  { label: "path_granulomas", regex: /\b(?:granny|granby|grandy)\s+(?:aloma(?:s| is)?|lumbar)\b/gi, replacement: "granulomas" },
  { label: "path_granulomas_commas", regex: /\bgranny,\s*lumbar\b/gi, replacement: "granulomas" },
  { label: "path_sarcoidosis", regex: /\b(?:sacriosis|sarcoid\s*osis)\b/gi, replacement: "sarcoidosis" },
  { label: "term_mediastinal_lymphadenopathy", regex: /\bmeters?\s+panel(?:\s+and)?\s+fed\s+an\s+op\b/gi, replacement: "mediastinal lymphadenopathy" },
  { label: "path_cytology", regex: /\bcyto\s+logy\b/gi, replacement: "cytology" },
  { label: "path_cell_block", regex: /\bcell\s+bloc\b/gi, replacement: "cell block" },
  { label: "specimen_formalin", regex: /\bformal\s+in\b/gi, replacement: "formalin" },
  { label: "specimen_cytolyt", regex: /\b(?:cyto\s*light|cyto\s*lit|sight\s*a\s*light|cyto\s*lyt)\b/gi, replacement: "CytoLyt" },
  { label: "specimen_rpmi", regex: /\b(?:r|are)[\s.-]*(?:p|pee)[\s.-]*(?:m|em)[\s.-]*(?:i|eye)\b/gi, replacement: "RPMI" },
];

const MEDICATION_AND_UNIT_RULES = [
  { label: "med_lidocaine", regex: /\b(?:lido\s*cane|lido\s*kane|lidocane|xylocaine|xylo\s*cane)\b/gi, replacement: "lidocaine" },
  { label: "med_epinephrine", regex: /\bepi\s+nephrine\b/gi, replacement: "epinephrine" },
  {
    label: "unit_lidocaine_percent",
    regex: /\b(half|one|two|0\.5|1|2)\s*percent\s+(lidocaine)\b/gi,
    replacement: (_match, percentToken, medication) => `${normalizePercent(percentToken)}% ${String(medication).toLowerCase()}`,
  },
  {
    label: "unit_ml",
    regex: /\b(\d+(?:\.\d+)?)\s*(?:cc|m\s*l|milliliters?|millilitres?|mils?)\b/gi,
    replacement: (_match, volume) => `${volume} mL`,
  },
  {
    label: "unit_cm",
    regex: /\b(\d+(?:\.\d+)?)\s*[-\s]*(?:seminer|centimeters?|centimetres?)\b/gi,
    replacement: (_match, value) => `${value} cm`,
  },
  {
    label: "unit_gauge_with_device",
    regex: /\b(twenty[-\s]?one|twenty[-\s]?two|twenty[-\s]?five|nineteen|eighteen|21|22|25|19|18)\s*g\s+(needle|catheter|forceps)\b/gi,
    replacement: (_match, gaugeToken, device) => `${normalizeGauge(gaugeToken)}-gauge ${String(device).toLowerCase()}`,
  },
  {
    label: "unit_gauge",
    regex: /\b(twenty[-\s]?one|twenty[-\s]?two|twenty[-\s]?five|nineteen|eighteen|21|22|25|19|18)\s*gauge\b/gi,
    replacement: (_match, gaugeToken) => `${normalizeGauge(gaugeToken)}-gauge`,
  },
];

const PLEURAL_AND_COMPLICATION_RULES = [
  { label: "proc_thoracentesis", regex: /\b(?:thora\s+centesis|thor\s+a\s+centesis|thoracentisis|thoracenteses)\b/gi, replacement: "thoracentesis" },
  { label: "proc_thoracoscopy", regex: /\bthoraco\s+scopy\b/gi, replacement: "thoracoscopy" },
  { label: "term_pleural_context", regex: /\bplural(?=\s+(?:effusion|fluid|space|catheter|biopsy|pressure|manometry|thickening|plaque|nodule|disease|drainage))\b/gi, replacement: "pleural" },
  { label: "term_pleurx", regex: /\b(?:pleur\s*x|pleurex|plurex)\b/gi, replacement: "PleurX" },
  { label: "term_chest_tube", regex: /\bchess\s+tube\b/gi, replacement: "chest tube" },
  { label: "complication_pneumothorax", regex: /\b(?:pneumo|new\s+mo|newmo|neo\s*morpho)\s+thorax\b/gi, replacement: "pneumothorax" },
  { label: "complication_pneumothorax_direct", regex: /\bneomorphoax\b/gi, replacement: "pneumothorax" },
  { label: "complication_no_complications", regex: /\b(?:new|no\s+new)\s+complications\b/gi, replacement: "no complications" },
  { label: "term_post_procedure_cxr", regex: /\bpost[-\s]?(?:resver|procedure|procedural?)\s+(?:church|chest)\s+s?\s*x[-\s]?ray\b/gi, replacement: "post-procedure chest x-ray" },
  { label: "term_hemostasis", regex: /\b(?:hema\s+stasis|he\s+mostasis)\b/gi, replacement: "hemostasis" },
  { label: "count_times_six", regex: /\btime\s+six\b/gi, replacement: "x 6" },
];

const DICTATED_PUNCTUATION_RULES = [
  { label: "dictation_new_paragraph", regex: /\s*\bnew\s+paragraph\b\s*/gi, replacement: "\n\n" },
  { label: "dictation_new_line", regex: /\s*\bnew\s+line\b\s*/gi, replacement: "\n" },
  { label: "dictation_period", regex: /\s+\b(?:period|full\s+stop)\b(?=\s+|$)/gi, replacement: ". " },
  { label: "dictation_comma", regex: /\s+\bcomma\b\s+/gi, replacement: ", " },
  { label: "dictation_colon", regex: /\s+\bcolon\b\s+/gi, replacement: ": " },
];

const DEFAULT_RULES = [
  ...PROCEDURE_AND_DEVICE_RULES,
  ...LOBE_RULES,
  ...PATH_AND_SPECIMEN_RULES,
  ...MEDICATION_AND_UNIT_RULES,
  ...PLEURAL_AND_COMPLICATION_RULES,
];

function applyStationRepairs(current, replacements) {
  let repaired = current;

  repaired = applyReplacement(
    repaired,
    new RegExp(
      `\\b(${STATION_PREFIX_WITH_SIDE_PATTERN})\\s+(${STATION_NUMBER_WITH_SIDE_PATTERN})\\s*(?:[-/]?\\s*)(${STATION_SIDE_PATTERN})\\b`,
      "gi",
    ),
    (_match, prefixToken, numberToken, sideToken) => {
      const station = normalizeStationToken(numberToken, sideToken);
      if (!station) return _match;
      return `${normalizeStationPrefix(prefixToken)} ${station}`;
    },
    replacements,
    "station_with_side",
  );

  repaired = applyReplacement(
    repaired,
    new RegExp(`\\b(${STATION_PREFIX_PLAIN_PATTERN})\\s+(${STATION_NUMBER_PLAIN_PATTERN})\\b`, "gi"),
    (_match, prefixToken, numberToken) => {
      const station = normalizeStationToken(numberToken);
      if (!station) return _match;
      return `${normalizeStationPrefix(prefixToken)} ${station}`;
    },
    replacements,
    "station_plain",
  );

  return repaired;
}

function applyNodeLabelMeasurementRepairs(current, replacements) {
  let repaired = current;

  repaired = applyReplacement(
    repaired,
    /\bfor\s+our(?=\s+\d+(?:\.\d+)?\s+millimeters?\b)/gi,
    "4R",
    replacements,
    "node_4r_measurement",
  );

  repaired = applyReplacement(
    repaired,
    /\beleven\s+out(?=\s+\d+(?:\.\d+)?\s+millimeters?\b)/gi,
    "11L",
    replacements,
    "node_11l_measurement",
  );

  repaired = applyReplacement(
    repaired,
    /\b11\s+out(?=\s+\d+(?:\.\d+)?\s+millimeters?\b)/gi,
    "11L",
    replacements,
    "node_11l_measurement_numeric",
  );

  repaired = applyReplacement(
    repaired,
    /\b([1-9]|1[0-2])([rRlL])(?=\s*,?\s*\d+(?:\.\d+)?\s+millimeters?\b)/g,
    (_match, numberToken, sideToken) => `${numberToken}${String(sideToken).toUpperCase()}`,
    replacements,
    "node_label_uppercase_measurement",
  );

  repaired = applyReplacement(
    repaired,
    /\bstation\s+7(?=\d{2}\s+millimeters?\b)/gi,
    "station 7 ",
    replacements,
    "station_measurement_spacing",
  );

  return repaired;
}

function normalizeCustomRule(rule) {
  if (!rule || !(rule.regex instanceof RegExp)) return null;
  if (typeof rule.replacement !== "string" && typeof rule.replacement !== "function") return null;
  return {
    label: rule.label || "custom_rule",
    regex: rule.regex,
    replacement: rule.replacement,
  };
}

export function repairSpeechTranscript(text, options = {}) {
  const replacements = [];
  const opts = {
    normalizeDictatedPunctuation: false,
    customRules: [],
    ...options,
  };

  let repaired = String(text || "");
  repaired = applyRules(repaired, DEFAULT_RULES, replacements);
  repaired = applyStationRepairs(repaired, replacements);
  repaired = applyNodeLabelMeasurementRepairs(repaired, replacements);

  const customRules = Array.isArray(opts.customRules)
    ? opts.customRules.map(normalizeCustomRule).filter(Boolean)
    : [];
  if (customRules.length) repaired = applyRules(repaired, customRules, replacements);

  if (opts.normalizeDictatedPunctuation) {
    repaired = applyRules(repaired, DICTATED_PUNCTUATION_RULES, replacements);
  }

  return {
    text: normalizeWhitespace(repaired),
    replacements,
  };
}

export default repairSpeechTranscript;
