import { pipeline, env } from '@huggingface/transformers';

// Configure for browser usage
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = new URL('../../api/static/phi_redactor/vendor/', import.meta.url).toString();
env.useBrowserCache = true;

// =============================================================================
// CONFIGURATION
// =============================================================================

export const DEFAULT_CONFIG = {
    redactPatientNames: true,
    redactMRN: true,
    redactDOB: true,
    redactProcedureDates: true,
    redactFacilities: true,
    redactAgesOver89: true,
    protectPhysicianNames: true,
    protectDeviceNames: true,
    protectAnatomicalTerms: true,
    aiThreshold: 0.85
};

const MODEL_ID = 'phi_distilbert_ner_quant';

// =============================================================================
// PROTECTION LISTS (Terms that should NEVER be redacted)
// =============================================================================

// Robotic/Navigation platforms (can look like names)
const ROBOTIC_PLATFORMS = new Set([
    "ion", "monarch", "galaxy", "superdimension", "illumisite", "lungvision",
    "veran", "spin", "archimedes", "zephyr", "body vision", "inavision"
]);

// Valve/stent products
const AIRWAY_DEVICES = new Set([
    "spiration", "zephyr", "chartis", "pleurx", "aspira", "dumon", 
    "ultraflex", "alair", "emprint", "neuwave", "aero", "atmos"
]);

// Clinical terms/procedures
const CLINICAL_TERMS = new Set([
    "rose", "ebus", "tbna", "bal", "tbbx", "tblb", "fna", "eus",
    "bronchoscopy", "thoracentesis", "pleurodesis", "cryobiopsy",
    "adenocarcinoma", "squamous", "nsclc", "sclc", "carcinoid",
    "lymphocytes", "malignant", "benign", "atypical", "necrosis",
    "fibrosis", "granuloma", "inflammation", "mucosa", "submucosa"
]);

// Anatomical terms (lobes, segments, stations)
const ANATOMICAL_TERMS = new Set([
    // Lobes
    "rul", "rml", "rll", "lul", "lll", "lingula",
    "right upper lobe", "right middle lobe", "right lower lobe",
    "left upper lobe", "left lower lobe",
    // Structures
    "carina", "trachea", "bronchus", "bronchi", "pleura", "mediastinum",
    "mainstem", "main stem", "intermedius", "interlobar",
    // Segments (B1-B10)
    "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9", "b10",
    "rb1", "rb2", "rb3", "rb4", "rb5", "rb6", "rb7", "rb8", "rb9", "rb10",
    "lb1", "lb2", "lb3", "lb4", "lb5", "lb6", "lb7", "lb8", "lb9", "lb10",
    // Lymph node stations
    "station 1", "station 2", "station 2l", "station 2r",
    "station 3", "station 4", "station 4l", "station 4r",
    "station 5", "station 6", "station 7",
    "station 8", "station 9", "station 10", "station 10l", "station 10r",
    "station 11", "station 11l", "station 11r", "station 12",
    "2l", "2r", "4l", "4r", "7", "10l", "10r", "11l", "11r"
]);

// Device manufacturers that look like names
const DEVICE_BRAND_NAMES = [
    "Noah", "Wang", "Cook", "Mark", "Baker", "Young", "King", "Edwards",
    "Olympus", "Boston", "Stryker", "Intuitive", "Auris", "Fujifilm", 
    "Pentax", "Karl Storz", "Medtronic", "Merit"
];

// Device context keywords
const DEVICE_CONTEXT_WORDS = [
    "Medical", "Needle", "Catheter", "System", "Biopsy", "Scope",
    "Bronchoscope", "Endoscope", "Ultrasound", "EchoTip", "Expect",
    "Vizishot", "Platform", "Navigation", "Robotic"
];

// =============================================================================
// DETECTION PATTERNS (PHI to redact)
// =============================================================================

const PHI_PATTERNS = [
    // Patient header - captures name from structured header
    {
        type: 'PATIENT_NAME',
        regex: /(?:^|\n)\s*(?:Patient(?:\s+Name)?|Pt|Name|Subject)\s*[:\-]?\s*([A-Z][a-z]+,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?|[A-Z][a-z]+\s+[A-Z]'?[A-Za-z]+|[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+|[A-Z][a-z]+\s+[A-Z]\.|[A-Z][a-z]+)/gim,
        confidence: 0.95
    },
    // MRN / Medical Record Number
    {
        type: 'MRN',
        regex: /\b(?:MRN|MR|Medical\s*Record|Patient\s*ID|ID|EDIPI|DOD\s*ID)\s*[:#]?\s*([A-Z0-9\-]{4,15})\b/gi,
        confidence: 0.95
    },
    // DOB with label
    {
        type: 'DOB',
        regex: /\b(?:DOB|Date\s*of\s*Birth|Birth\s*Date|Born)\s*[:\-]?\s*(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})/gi,
        confidence: 0.95
    },
    // Generic dates
    {
        type: 'DATE',
        regex: /\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|\d{4}[\/\-]\d{1,2}[\/\-]\d{1,2})\b/g,
        confidence: 0.85
    },
    // Ages over 89 (HIPAA)
    {
        type: 'AGE_OVER_89',
        regex: /\b(9\d|1[0-4]\d)\s*[-]?\s*(?:year[s]?\s*[-]?\s*old|y\.?o\.?|yo)\b/gi,
        confidence: 0.95
    },
    {
        type: 'AGE_OVER_89',
        regex: /\bAge\s*[:\-]?\s*(9\d|1[0-4]\d)\b/gi,
        confidence: 0.95
    },
    // Phone numbers
    {
        type: 'PHONE',
        regex: /\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b/g,
        confidence: 0.90
    },
    // SSN
    {
        type: 'SSN',
        regex: /\b\d{3}[-]\d{2}[-]\d{4}\b/g,
        confidence: 0.95
    },
    // Email
    {
        type: 'EMAIL',
        regex: /\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b/g,
        confidence: 0.95
    },
    // Facility with City, State
    {
        type: 'FACILITY',
        regex: /\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Medical\s+Center|Hospital|Clinic|Health\s+System|Healthcare)),?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),?\s*(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)(?:\s+\d{5}(?:-\d{4})?)?\b/g,
        confidence: 0.90
    }
];

// =============================================================================
// PROTECTION PATTERNS
// =============================================================================

const PROTECTION_PATTERNS = [
    // Physician headers
    {
        name: 'physician_header',
        regex: /(?:Attending|Fellow|Resident|Physician|Anesthesiologist|Cytopathologist|Pathologist|Radiologist|Surgeon|Provider)\s*(?:Physician)?[:\s]+(?:Dr\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?/gi
    },
    // "Dr. Name" pattern
    {
        name: 'dr_title',
        regex: /\b(?:Dr\.?|Doctor)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?/gi
    },
    // "Name, MD/DO" credential pattern
    {
        name: 'credential',
        regex: /\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*(?:MD|DO|RN|NP|PA|MBBS|PhD|PharmD)(?:\s*,\s*(?:FCCP|FACP|FACS|PhD|MPH|MS|MBA))?\b/gi
    },
    // Electronic signatures
    {
        name: 'signature',
        regex: /(?:Electronically\s+(?:signed|attested)|Signed)\s+by[:\s]+(?:Dr\.?\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?/gi
    },
    // ROSE context
    {
        name: 'rose_context',
        regex: /ROSE\s+(?:cytopathology|cytopathologist|pathologist|technician|present|interpretation|review|evaluation|analysis)/gi
    },
    // Device brand + context
    {
        name: 'device_context',
        regex: new RegExp(
            `\\b(${DEVICE_BRAND_NAMES.join('|')})\\s+(${DEVICE_CONTEXT_WORDS.join('|')})`,
            'gi'
        )
    },
    // Robotic platform context
    {
        name: 'robotic_platform',
        regex: /\b(?:Ion|Monarch|Galaxy|Veran|SuperDimension|Body\s*Vision)\s+(?:system|platform|navigation|robotic|bronchoscopy|procedure)/gi
    }
];

// =============================================================================
// REDACTION SERVICE CLASS
// =============================================================================

export class RedactionService {
    constructor(config = {}) {
        this.model = null;
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.learnedPatientNames = new Set();
    }

    async init() {
        if (this.model) return;
        
        this.model = await pipeline(
            'token-classification',
            MODEL_ID,
            { quantized: true }
        );
    }

    setConfig(config) {
        this.config = { ...this.config, ...config };
    }

    resetLearnedNames() {
        this.learnedPatientNames = new Set();
    }

    /**
     * Get all protected ranges
     */
    getProtectedRanges(text) {
        const ranges = [];
        const lowerText = text.toLowerCase();

        // 1. Protect clinical/anatomical terms
        if (this.config.protectAnatomicalTerms) {
            const allTerms = new Set([
                ...ROBOTIC_PLATFORMS,
                ...AIRWAY_DEVICES,
                ...CLINICAL_TERMS,
                ...ANATOMICAL_TERMS
            ]);

            allTerms.forEach(term => {
                let idx = lowerText.indexOf(term);
                while (idx !== -1) {
                    const charBefore = idx > 0 ? lowerText[idx - 1] : ' ';
                    const charAfter = idx + term.length < lowerText.length 
                        ? lowerText[idx + term.length] : ' ';
                    
                    if (!/[a-z0-9]/.test(charBefore) && !/[a-z0-9]/.test(charAfter)) {
                        ranges.push({ 
                            start: idx, 
                            end: idx + term.length,
                            reason: 'clinical_term'
                        });
                    }
                    idx = lowerText.indexOf(term, idx + 1);
                }
            });
        }

        // 2. Protect physician names and device contexts
        if (this.config.protectPhysicianNames || this.config.protectDeviceNames) {
            PROTECTION_PATTERNS.forEach(({ name, regex }) => {
                if (!this.config.protectDeviceNames && 
                    (name === 'device_context' || name === 'robotic_platform')) {
                    return;
                }
                if (!this.config.protectPhysicianNames && 
                    ['physician_header', 'dr_title', 'credential', 'signature'].includes(name)) {
                    return;
                }

                regex.lastIndex = 0;
                let match;
                while ((match = regex.exec(text)) !== null) {
                    ranges.push({
                        start: match.index,
                        end: match.index + match[0].length,
                        reason: name
                    });
                }
            });
        }

        return ranges;
    }

    /**
     * Learn patient name from header
     */
    learnPatientName(text) {
        const headerPattern = /(?:^|\n)\s*(?:Patient(?:\s+Name)?|Pt)\s*[:\-]?\s*([A-Z][a-z]+(?:[,\s]+[A-Z][a-z']+)+)/im;
        const match = headerPattern.exec(text);
        
        if (match && match[1]) {
            const fullName = match[1].trim();
            this.learnedPatientNames.add(fullName.toLowerCase());
            
            const parts = fullName.split(/[\s,]+/).filter(p => p.length > 1);
            parts.forEach(part => {
                if (part.length > 2 && !/^[A-Z]\.?$/.test(part)) {
                    this.learnedPatientNames.add(part.toLowerCase());
                }
            });
            
            return fullName;
        }
        return null;
    }

    /**
     * Find subsequent mentions of learned names
     */
    findNameMentions(text) {
        const detections = [];
        const lowerText = text.toLowerCase();

        this.learnedPatientNames.forEach(name => {
            if (name.length < 3) return;
            
            let idx = lowerText.indexOf(name);
            while (idx !== -1) {
                const charBefore = idx > 0 ? lowerText[idx - 1] : ' ';
                const charAfter = idx + name.length < lowerText.length 
                    ? lowerText[idx + name.length] : ' ';
                
                if (!/[a-z]/.test(charBefore) && !/[a-z]/.test(charAfter)) {
                    detections.push({
                        start: idx,
                        end: idx + name.length,
                        text: text.substring(idx, idx + name.length),
                        label: 'PATIENT_NAME',
                        confidence: 0.90,
                        source: 'regex'
                    });
                }
                idx = lowerText.indexOf(name, idx + 1);
            }
        });

        return detections;
    }

    /**
     * Apply regex patterns
     */
    applyRegexPatterns(text) {
        const detections = [];

        PHI_PATTERNS.forEach(({ type, regex, confidence }) => {
            if (type === 'PATIENT_NAME' && !this.config.redactPatientNames) return;
            if (type === 'MRN' && !this.config.redactMRN) return;
            if (type === 'DOB' && !this.config.redactDOB) return;
            if (type === 'DATE' && !this.config.redactProcedureDates) return;
            if (type === 'FACILITY' && !this.config.redactFacilities) return;
            if (type === 'AGE_OVER_89' && !this.config.redactAgesOver89) return;

            regex.lastIndex = 0;
            let match;
            while ((match = regex.exec(text)) !== null) {
                detections.push({
                    start: match.index,
                    end: match.index + match[0].length,
                    text: match[0],
                    label: type,
                    confidence,
                    source: 'regex'
                });
            }
        });

        return detections;
    }

    /**
     * Apply AI model
     */
    async applyAIModel(text) {
        if (!this.model) return [];

        const detections = [];
        const labelMap = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'DATE': 'DATE',
            'PHONE': 'PHONE',
            'EMAIL': 'EMAIL',
            'SSN': 'SSN',
            'ID': 'ID'
        };

        try {
            const results = await this.model(text, {
                ignore_labels: ['O'],
                aggregation_strategy: 'simple'
            });

            for (const entity of results) {
                if (entity.score >= this.config.aiThreshold) {
                    const label = labelMap[entity.entity_group?.toUpperCase()] || 
                                  entity.entity_group?.toUpperCase() || 
                                  'PHI';
                    
                    detections.push({
                        start: entity.start,
                        end: entity.end,
                        text: text.substring(entity.start, entity.end),
                        label,
                        confidence: entity.score,
                        source: 'ai'
                    });
                }
            }
        } catch (error) {
            console.warn('AI model detection failed:', error);
        }

        return detections;
    }

    /**
     * Check if detection overlaps protected range
     */
    isProtected(detection, protectedRanges) {
        return protectedRanges.some(p =>
            (detection.start >= p.start && detection.start < p.end) ||
            (detection.end > p.start && detection.end <= p.end) ||
            (detection.start <= p.start && detection.end >= p.end)
        );
    }

    /**
     * Resolve overlapping detections
     */
    resolveOverlaps(detections) {
        if (detections.length === 0) return [];

        const sorted = [...detections].sort((a, b) => {
            if (a.start !== b.start) return a.start - b.start;
            return b.confidence - a.confidence;
        });

        const resolved = [];
        let lastEnd = -1;

        for (const detection of sorted) {
            if (detection.start >= lastEnd) {
                resolved.push(detection);
                lastEnd = detection.end;
            } else if (detection.confidence > (resolved[resolved.length - 1]?.confidence || 0)) {
                resolved[resolved.length - 1] = detection;
                lastEnd = detection.end;
            }
        }

        return resolved;
    }

    /**
     * Main redaction method
     */
    async redact(text) {
        if (!this.model) await this.init();

        // Step 1: Learn patient name
        this.learnPatientName(text);

        // Step 2: Get protected ranges
        const protectedRanges = this.getProtectedRanges(text);

        // Step 3: Apply regex patterns
        let detections = this.applyRegexPatterns(text);

        // Step 4: Find learned name mentions
        const nameMentions = this.findNameMentions(text);
        detections.push(...nameMentions);

        // Step 5: Apply AI model
        const aiDetections = await this.applyAIModel(text);
        detections.push(...aiDetections);

        // Step 6: Filter out protected ranges
        detections = detections.filter(d => !this.isProtected(d, protectedRanges));

        // Step 7: Resolve overlaps
        detections = this.resolveOverlaps(detections);

        // Step 8: Apply redactions (back to front)
        detections.sort((a, b) => b.start - a.start);

        let redactedText = text;
        for (const detection of detections) {
            const mask = `[REDACTED_${detection.label}]`;
            redactedText = 
                redactedText.substring(0, detection.start) +
                mask +
                redactedText.substring(detection.end);
        }

        detections.sort((a, b) => a.start - b.start);

        return {
            redactedText,
            detections,
            protectedRanges
        };
    }

    /**
     * Batch redaction
     */
    async redactBatch(texts) {
        if (!this.model) await this.init();
        
        const results = [];
        for (const text of texts) {
            this.resetLearnedNames();
            results.push(await this.redact(text));
        }
        return results;
    }
}

// Convenience factory
export async function createRedactionService(config) {
    const service = new RedactionService(config);
    await service.init();
    return service;
}
