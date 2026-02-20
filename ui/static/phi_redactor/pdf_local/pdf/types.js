/**
 * @typedef {'native' | 'ocr'} PageSource
 */

/**
 * @typedef {'native' | 'ocr' | 'hybrid'} SourceDecision
 */

/**
 * @typedef {{x:number,y:number,width:number,height:number}} Region
 */

/**
 * @typedef {{text:string,conf:number|null,bbox:Region}} OcrWord
 */

/**
 * @typedef {{text:string,confidence:number|null,bbox:Region,words:OcrWord[],pageIndex:number}} OcrLine
 */

/**
 * @typedef {{lines:OcrLine[],fullText:string,pageIndex:number}} OcrPage
 */

/**
 * @typedef {{start:number,end:number,kind?:string}} Span
 */

/**
 * @typedef {Object} LayoutBlock
 * @property {string} id
 * @property {Region} bbox
 * @property {number} lineCount
 * @property {string} textPreview
 */

/**
 * @typedef {Object} ExtractionStats
 * @property {number} charCount
 * @property {number} itemCount
 * @property {number} nonPrintableRatio
 * @property {number} singleCharItemRatio
 * @property {number} imageOpCount
 * @property {number} textShowOpCount
 * @property {number} layoutBlockCount
 * @property {number} overlapRatio
 * @property {number} contaminationScore
 * @property {number} completenessConfidence
 */

/**
 * @typedef {Object} PageClassification
 * @property {boolean} needsOcr
 * @property {string} reason
 * @property {number} confidence
 * @property {number} completenessConfidence
 * @property {string[]} qualityFlags
 */

/**
 * @typedef {'force_native' | 'force_ocr' | undefined} PageOverride
 */

/**
 * @typedef {Object} PdfPageModel
 * @property {number} pageIndex
 * @property {string} text
 * @property {string} rawText
 * @property {ExtractionStats} stats
 * @property {PageSource} source
 * @property {SourceDecision} sourceDecision
 * @property {PageClassification} classification
 * @property {PageOverride} [userOverride]
 * @property {LayoutBlock[]} layoutBlocks
 * @property {Region[]} imageRegions
 * @property {Region[]} textRegions
 * @property {Span[]} contaminatedSpans
 * @property {string[]} qualityFlags
 * @property {string} sourceReason
 * @property {string} [blockedReason]
 * @property {string} [ocrText]
 * @property {{
 * confidence?:number,
 * source?:string,
 * width?:number,
 * height?:number,
 * scale?:number,
 * durationMs?:number,
 * masking?:{
 * applied?:boolean,
 * mode?:string,
 * reason?:string,
 * coverageRatio?:number,
 * candidateCount?:number,
 * maskedCount?:number
 * },
 * crop?:{
 * applied?:boolean,
 * mode?:string,
 * reason?:string,
 * box?:number[]
 * },
 * header?:{
 * psmUsed?:string,
 * scaleUsed?:number,
 * retries?:number,
 * retryNeeded?:boolean
 * },
 * filterMode?:{
 * disableFigureOverlap?:boolean,
 * reason?:string
 * },
 * metrics?:{
 * preMask?:object,
 * preFilter?:object,
 * postOcr?:object,
 * postFilter?:object
 * },
 * lines?:OcrLine[],
 * droppedLineSummary?:Record<string,number>,
 * droppedLines?:Array<{reason:string,text:string,confidence?:number,overlapRatio?:number}>,
 * figureRegions?:Region[]
 * }} [ocrMeta]
 * @property {{
 * charCount:number,
 * alphaRatio:number,
 * meanLineConf:number|null,
 * lowConfLineFrac:number|null,
 * numLines:number,
 * medianTokenLen:number,
 * footerBoilerplateHits:number,
 * lowConfidence:boolean,
 * stages?:object
 * }} [qualityMetrics]
 */

/**
 * @typedef {Object} PdfExtractionGate
 * @property {'pass'|'blocked'} status
 * @property {boolean} blocked
 * @property {string} [reason]
 * @property {boolean} ocrAvailable
 * @property {boolean} requiresOcr
 * @property {{minCompletenessConfidence:number,maxContaminationScore:number,hardBlockWhenUnsafeWithoutOcr:boolean}} thresholds
 */

/**
 * @typedef {Object} PdfDocumentModel
 * @property {string} fileName
 * @property {PdfPageModel[]} pages
 * @property {string} fullText
 * @property {number[]} pageStartOffsets
 * @property {boolean} requiresOcr
 * @property {boolean} blocked
 * @property {string} [blockReason]
 * @property {{lowConfidencePages:number,contaminatedPages:number,pageMetrics?:string[]}} qualitySummary
 * @property {PdfExtractionGate} gate
 */

/**
 * @typedef {Object} ExtractionProgressEvent
 * @property {'progress'} kind
 * @property {number} completedPages
 * @property {number} totalPages
 */

/**
 * @typedef {Object} ExtractionStageEvent
 * @property {'stage'} kind
 * @property {'layout_analysis'|'contamination_detection'|'adaptive_assembly'|'native_extraction_done'|'ocr_prepare'|'ocr_loading_assets'|'ocr_rendering'|'ocr_recognizing'|'ocr_failed'} stage
 * @property {number} pageIndex
 * @property {number} totalPages
 */

/**
 * @typedef {Object} OcrProgressEvent
 * @property {'ocr_progress'} kind
 * @property {number} completedPages
 * @property {number} totalPages
 */

/**
 * @typedef {Object} OcrStatusEvent
 * @property {'ocr_status'} kind
 * @property {string} status
 * @property {number} progress
 */

/**
 * @typedef {Object} OcrErrorEvent
 * @property {'ocr_error'} kind
 * @property {string} error
 */

/**
 * @typedef {Object} OcrPageEvent
 * @property {'ocr_page'} kind
 * @property {{pageIndex:number,text:string,meta?:object}} page
 */

/**
 * @typedef {Object} PageResultEvent
 * @property {'page'} kind
 * @property {PdfPageModel} page
 */

/**
 * @typedef {Object} ExtractionDoneEvent
 * @property {'done'} kind
 * @property {PdfPageModel[]} pages
 * @property {PdfDocumentModel} [document]
 * @property {PdfExtractionGate} [gate]
 */
