export const RedactionType = Object.freeze({
  EMAIL: "EMAIL",
  PHONE: "PHONE",
  SSN: "SSN",
  MRN: "MRN",
  ACCOUNT: "ACCOUNT",
  DATE: "DATE",
  URL: "URL",
  IP: "IP",
});

/**
 * @typedef {keyof typeof RedactionType} RedactionTypeKey
 */

/**
 * @typedef {Object} RedactionSpan
 * @property {number} start
 * @property {number} end
 * @property {string} type
 * @property {number} confidence
 * @property {'regex'|'ner'|'manual'} source
 */
