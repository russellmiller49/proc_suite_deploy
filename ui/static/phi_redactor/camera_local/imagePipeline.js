let cameraJobCounter = 0;

function hasFunction(value) {
  return typeof value === "function";
}

function clamp01(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return Math.max(0, Math.min(1, n));
}

function normalizeCropBox(crop) {
  if (!crop || typeof crop !== "object") return null;
  const x0 = Number(crop.x0);
  const y0 = Number(crop.y0);
  const x1 = Number(crop.x1);
  const y1 = Number(crop.y1);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;

  const left = clamp01(Math.min(x0, x1));
  const right = clamp01(Math.max(x0, x1));
  const top = clamp01(Math.min(y0, y1));
  const bottom = clamp01(Math.max(y0, y1));
  if (right - left < 0.05 || bottom - top < 0.05) return null;

  return { x0: left, y0: top, x1: right, y1: bottom };
}

function nextCameraJobId() {
  cameraJobCounter += 1;
  return `camera_ocr_${Date.now()}_${cameraJobCounter}`;
}

function normalizePages(inputPages) {
  return (Array.isArray(inputPages) ? inputPages : [])
    .filter((page) => page && page.bitmap)
    .map((page, idx) => ({
      pageIndex: Number.isFinite(page.pageIndex) ? Number(page.pageIndex) : idx,
      bitmap: page.bitmap,
      width: Number.isFinite(page.width) ? Number(page.width) : 0,
      height: Number.isFinite(page.height) ? Number(page.height) : 0,
      crop: normalizeCropBox(page.crop),
    }))
    .sort((a, b) => a.pageIndex - b.pageIndex);
}

async function cloneBitmap(bitmap) {
  if (!bitmap) return null;
  if (!hasFunction(globalThis.createImageBitmap)) return bitmap;
  try {
    return await globalThis.createImageBitmap(bitmap);
  } catch {
    return bitmap;
  }
}

async function clonePagesForWorker(pages) {
  const out = [];
  for (const page of pages) {
    const clonedBitmap = await cloneBitmap(page.bitmap);
    if (!clonedBitmap) continue;
    out.push({
      pageIndex: page.pageIndex,
      bitmap: clonedBitmap,
      width: page.width,
      height: page.height,
      crop: page.crop
        ? {
            x0: Number(page.crop.x0),
            y0: Number(page.crop.y0),
            x1: Number(page.crop.x1),
            y1: Number(page.crop.y1),
          }
        : null,
    });
  }
  return out;
}

export function makeCameraWorkerUrl() {
  return new URL("./workers/image_ocr.worker.js", import.meta.url);
}

export function cancelCameraOcrJob(worker, jobId) {
  if (!worker || !jobId || !hasFunction(worker.postMessage)) return;
  try {
    worker.postMessage({ type: "camera_ocr_cancel", jobId: String(jobId) });
  } catch {
    // ignore
  }
}

export async function runCameraOcrJob(worker, pages, options = {}, handlers = {}) {
  if (!worker || !hasFunction(worker.postMessage)) {
    throw new Error("camera OCR worker is required");
  }

  const preparedPages = normalizePages(pages);
  if (!preparedPages.length) {
    throw new Error("No captured pages to OCR");
  }

  const transferablePages = await clonePagesForWorker(preparedPages);
  if (!transferablePages.length) {
    throw new Error("Unable to prepare captured pages for OCR");
  }

  const jobId = String(options.jobId || nextCameraJobId());
  const payload = {
    type: "camera_ocr_run",
    jobId,
    pages: transferablePages,
    options: {
      lang: options.lang === "eng" ? "eng" : "eng",
      mode: options.mode === "high_accuracy" ? "high_accuracy" : "fast",
      sceneHint: options.sceneHint === "monitor"
        ? "monitor"
        : options.sceneHint === "document"
          ? "document"
          : "auto",
      warningProfile: options.warningProfile === "ios_safari" ? "ios_safari" : "default",
      preprocess: {
        mode: options.preprocess?.mode === "bw_high_contrast"
          ? "bw_high_contrast"
          : options.preprocess?.mode === "grayscale"
            ? "grayscale"
            : options.preprocess?.mode === "off"
              ? "off"
              : "auto",
      },
    },
  };

  const transferList = transferablePages
    .map((page) => page.bitmap)
    .filter((bitmap) => bitmap && typeof bitmap === "object");

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      if (hasFunction(worker.removeEventListener)) {
        worker.removeEventListener("message", onMessage);
        worker.removeEventListener("error", onError);
      }
    };

    const fail = (error) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(error instanceof Error ? error : new Error(String(error)));
    };

    const done = (result) => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve(result);
    };

    const onError = (event) => {
      const message = event?.message || "camera OCR worker failed";
      fail(new Error(message));
    };

    const onMessage = (event) => {
      const data = event?.data || {};
      if (data.jobId && String(data.jobId) !== jobId) return;

      if (data.type === "camera_ocr_progress") {
        handlers.onProgress?.(data);
        return;
      }

      if (data.type === "camera_ocr_page") {
        handlers.onPage?.(data.page || null, data);
        return;
      }

      if (data.type === "camera_ocr_done") {
        done({
          jobId,
          pages: Array.isArray(data.pages) ? data.pages : [],
        });
        return;
      }

      if (data.type === "camera_ocr_cancelled") {
        const error = new Error("Camera OCR cancelled");
        error.name = "AbortError";
        fail(error);
        return;
      }

      if (data.type === "camera_ocr_error") {
        const message = String(data.error || "camera OCR failed");
        fail(new Error(message));
      }
    };

    if (hasFunction(worker.addEventListener)) {
      worker.addEventListener("message", onMessage);
      worker.addEventListener("error", onError);
    } else {
      const originalOnMessage = worker.onmessage;
      worker.onmessage = (event) => {
        onMessage(event);
        if (hasFunction(originalOnMessage)) originalOnMessage(event);
      };
    }

    try {
      worker.postMessage(payload, transferList);
    } catch (error) {
      fail(error);
    }
  });
}

export function buildCameraOcrDocumentText(pages) {
  const source = (Array.isArray(pages) ? pages : [])
    .filter((page) => Number.isFinite(page?.pageIndex))
    .sort((a, b) => Number(a.pageIndex) - Number(b.pageIndex));

  if (!source.length) return "";

  const chunks = [];
  for (const page of source) {
    const pageLabel = Number(page.pageIndex) + 1;
    const text = String(page.text || "").trim();
    chunks.push(`===== PAGE ${pageLabel} (CAMERA_OCR) =====\n${text}`);
  }
  return `${chunks.join("\n\n")}\n`;
}
