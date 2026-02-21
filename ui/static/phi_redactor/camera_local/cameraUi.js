function hasFunction(value) {
  return typeof value === "function";
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function normalizeCropBox(input) {
  if (!input || typeof input !== "object") return null;
  const x0 = Number(input.x0);
  const y0 = Number(input.y0);
  const x1 = Number(input.x1);
  const y1 = Number(input.y1);
  if (![x0, y0, x1, y1].every(Number.isFinite)) return null;

  const left = clamp(Math.min(x0, x1), 0, 1);
  const right = clamp(Math.max(x0, x1), 0, 1);
  const top = clamp(Math.min(y0, y1), 0, 1);
  const bottom = clamp(Math.max(y0, y1), 0, 1);
  const minSpan = 0.05;
  if (right - left < minSpan || bottom - top < minSpan) return null;

  return { x0: left, y0: top, x1: right, y1: bottom };
}

function defaultUrlApi() {
  const runtimeUrl = globalThis.URL;
  if (runtimeUrl && hasFunction(runtimeUrl.createObjectURL) && hasFunction(runtimeUrl.revokeObjectURL)) {
    return runtimeUrl;
  }
  return null;
}

export function releaseCapturedPage(page, urlApi = defaultUrlApi()) {
  if (!page || typeof page !== "object") return;

  if (page.bitmap && hasFunction(page.bitmap.close)) {
    try {
      page.bitmap.close();
    } catch {
      // ignore
    }
  }

  if (page.previewUrl && urlApi && hasFunction(urlApi.revokeObjectURL)) {
    try {
      urlApi.revokeObjectURL(page.previewUrl);
    } catch {
      // ignore
    }
  }

  page.bitmap = null;
  page.previewUrl = "";
}

export function createCameraCaptureQueue(opts = {}) {
  const urlApi = opts.urlApi || defaultUrlApi();
  const pages = [];

  const getPages = () => pages;

  const addPage = (input = {}) => {
    if (!input.bitmap) {
      throw new Error("bitmap is required");
    }

    let previewUrl = "";
    if (input.blob && urlApi && hasFunction(urlApi.createObjectURL)) {
      try {
        previewUrl = urlApi.createObjectURL(input.blob);
      } catch {
        previewUrl = "";
      }
    }

    const page = {
      pageIndex: Number.isFinite(input.pageIndex) ? Number(input.pageIndex) : pages.length,
      bitmap: input.bitmap,
      previewUrl,
      width: Number.isFinite(input.width) ? Number(input.width) : 0,
      height: Number.isFinite(input.height) ? Number(input.height) : 0,
      capturedAt: Number.isFinite(input.capturedAt) ? Number(input.capturedAt) : Date.now(),
      warnings: Array.isArray(input.warnings) ? [...input.warnings] : [],
      crop: normalizeCropBox(input.crop),
    };

    pages.push(page);
    return page;
  };

  const retakeLast = () => {
    if (!pages.length) return null;
    const last = pages.pop();
    releaseCapturedPage(last, urlApi);
    return last;
  };

  const clearAll = () => {
    let cleared = 0;
    while (pages.length) {
      const page = pages.pop();
      releaseCapturedPage(page, urlApi);
      cleared += 1;
    }
    return cleared;
  };

  const setPageCrop = (pageIndex, crop) => {
    const idx = Number(pageIndex);
    if (!Number.isFinite(idx)) return null;
    if (idx < 0 || idx >= pages.length) return null;
    const normalized = normalizeCropBox(crop);
    pages[idx].crop = normalized;
    return pages[idx];
  };

  const clearAllCrops = () => {
    let changed = 0;
    for (const page of pages) {
      if (page?.crop) {
        page.crop = null;
        changed += 1;
      }
    }
    return changed;
  };

  const exportForOcr = () => pages.map((page, idx) => ({
    pageIndex: idx,
    bitmap: page.bitmap,
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
  }));

  return {
    get pages() {
      return getPages();
    },
    addPage,
    retakeLast,
    clearAll,
    setPageCrop,
    clearAllCrops,
    exportForOcr,
  };
}
