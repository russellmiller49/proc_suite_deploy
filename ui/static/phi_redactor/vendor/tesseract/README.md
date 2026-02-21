Self-hosted Tesseract assets for local-only OCR.

Contents:
- tesseract.esm.min.js (tesseract.js runtime)
- worker.min.js (tesseract.js worker runtime)
- tesseract-core-simd.wasm.js (WASM core bundle)
- tessdata/eng.traineddata (English language model)

All OCR requests in this app must resolve these assets from same-origin paths.
