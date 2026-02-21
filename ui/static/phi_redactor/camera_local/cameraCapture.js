function hasFunction(value) {
  return typeof value === "function";
}

function resolveEnv(env) {
  if (env && typeof env === "object") return env;
  return globalThis;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function canUseCameraScan(env) {
  const runtime = resolveEnv(env);
  if (!runtime.isSecureContext) {
    return {
      ok: false,
      reason: "Camera scan requires HTTPS and a secure context.",
      code: "secure_context_required",
    };
  }

  const mediaDevices = runtime.navigator?.mediaDevices;
  if (!mediaDevices || !hasFunction(mediaDevices.getUserMedia)) {
    return {
      ok: false,
      reason: "Camera scan requires browser support for live camera capture.",
      code: "getusermedia_unavailable",
    };
  }

  if (!hasFunction(runtime.Worker)) {
    return {
      ok: false,
      reason: "Camera scan requires Web Worker support.",
      code: "worker_unavailable",
    };
  }

  if (!hasFunction(runtime.OffscreenCanvas)) {
    return {
      ok: false,
      reason: "Camera scan requires OffscreenCanvas support.",
      code: "offscreen_canvas_unavailable",
    };
  }

  if (!hasFunction(runtime.createImageBitmap)) {
    return {
      ok: false,
      reason: "Camera scan requires ImageBitmap support.",
      code: "image_bitmap_unavailable",
    };
  }

  return { ok: true, reason: "ok", code: "ok" };
}

function buildConstraints(options = {}) {
  const facingMode = String(options.facingMode || "environment").trim() || "environment";
  const preferredWidth = Number.isFinite(options.preferredWidth)
    ? clamp(Number(options.preferredWidth), 640, 3840)
    : 1280;
  const preferredHeight = Number.isFinite(options.preferredHeight)
    ? clamp(Number(options.preferredHeight), 480, 2160)
    : undefined;

  const video = {
    facingMode: { ideal: facingMode },
    width: { ideal: preferredWidth },
  };
  if (preferredHeight) {
    video.height = { ideal: preferredHeight };
  }

  return {
    video,
    audio: false,
  };
}

export async function startCamera(videoEl, options = {}, env) {
  const runtime = resolveEnv(env);
  const mediaDevices = runtime.navigator?.mediaDevices;
  if (!videoEl) throw new Error("camera preview element is required");
  if (!mediaDevices || !hasFunction(mediaDevices.getUserMedia)) {
    throw new Error("getUserMedia is unavailable");
  }

  const constraints = buildConstraints(options);
  let stream = null;
  try {
    stream = await mediaDevices.getUserMedia(constraints);
  } catch (error) {
    const requestedFacing = String(options.facingMode || "environment").toLowerCase();
    if (requestedFacing !== "environment") throw error;
    // Fallback for devices that refuse rear camera constraints.
    stream = await mediaDevices.getUserMedia({
      video: { width: constraints.video.width },
      audio: false,
    });
  }

  videoEl.setAttribute("playsinline", "true");
  videoEl.muted = true;
  videoEl.srcObject = stream;
  if (hasFunction(videoEl.play)) {
    try {
      await videoEl.play();
    } catch {
      // iOS can block autoplay until explicit gesture; caller starts via button click.
    }
  }

  return stream;
}

export function stopCamera(target) {
  let stream = null;
  if (target && hasFunction(target.getTracks)) {
    stream = target;
  } else if (target && target.srcObject && hasFunction(target.srcObject.getTracks)) {
    stream = target.srcObject;
  }

  if (!stream) return 0;

  const tracks = stream.getTracks();
  for (const track of tracks) {
    try {
      track.stop();
    } catch {
      // ignore per-track stop failures
    }
  }

  if (target && Object.prototype.hasOwnProperty.call(target, "srcObject")) {
    try {
      target.srcObject = null;
    } catch {
      // ignore
    }
  }

  return tracks.length;
}

function createCanvas(width, height, runtime) {
  if (hasFunction(runtime.OffscreenCanvas)) {
    return new runtime.OffscreenCanvas(width, height);
  }

  const doc = runtime.document;
  if (!doc || !hasFunction(doc.createElement)) {
    throw new Error("No canvas API available for camera capture");
  }

  const canvas = doc.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

function canvasToBlob(canvas, quality) {
  if (hasFunction(canvas.convertToBlob)) {
    return canvas.convertToBlob({ type: "image/jpeg", quality });
  }

  if (!hasFunction(canvas.toBlob)) {
    return Promise.resolve(null);
  }

  return new Promise((resolve) => {
    canvas.toBlob(
      (blob) => resolve(blob || null),
      "image/jpeg",
      quality,
    );
  });
}

function resolveCaptureDimensions(videoEl, options = {}) {
  const sourceWidth = Math.max(1, Number(videoEl.videoWidth) || 0);
  const sourceHeight = Math.max(1, Number(videoEl.videoHeight) || 0);
  if (!sourceWidth || !sourceHeight) {
    throw new Error("Camera has not produced a frame yet");
  }

  const maxDim = Number.isFinite(options.maxDim)
    ? clamp(Number(options.maxDim), 320, 4096)
    : 2000;
  const scale = Math.min(1, maxDim / Math.max(sourceWidth, sourceHeight));
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  return { sourceWidth, sourceHeight, width, height };
}

function drawVideoFrameToCanvas(context, videoEl, dims) {
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, dims.width, dims.height);
  context.drawImage(
    videoEl,
    0,
    0,
    dims.sourceWidth,
    dims.sourceHeight,
    0,
    0,
    dims.width,
    dims.height,
  );
}

function imageDataFromBuffer(context, buffer, width, height) {
  if (!context || !buffer) return null;
  try {
    if (typeof ImageData === "function") {
      return new ImageData(buffer, width, height);
    }
  } catch {
    // fall through
  }
  if (hasFunction(context.createImageData)) {
    const imageData = context.createImageData(width, height);
    imageData.data.set(buffer);
    return imageData;
  }
  return null;
}

function computeFrameSharpnessScore(imageData) {
  const data = imageData?.data;
  const width = Math.max(1, Number(imageData?.width) || 1);
  const height = Math.max(1, Number(imageData?.height) || 1);
  if (!data || !data.length) return 0;

  const sampleStride = Math.max(1, Math.floor(Math.max(width, height) / 900));
  let sum = 0;
  let sqSum = 0;
  let colorDeltaSum = 0;
  let count = 0;

  for (let y = 0; y < height; y += sampleStride) {
    for (let x = 0; x < width; x += sampleStride) {
      const i = (y * width + x) * 4;
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const delta = Math.max(Math.abs(r - g), Math.abs(g - b), Math.abs(r - b));
      sum += luma;
      sqSum += luma * luma;
      colorDeltaSum += delta;
      count += 1;
    }
  }

  if (!count) return 0;
  const mean = sum / count;
  const variance = Math.max(0, sqSum / count - mean * mean);
  const stdDev = Math.sqrt(variance);
  const meanColorDelta = colorDeltaSum / count;

  // Prefer high contrast while penalizing severe chroma fringing (monitor moire).
  return stdDev - meanColorDelta * 0.34;
}

function sleep(ms) {
  const delay = Math.max(0, Number(ms) || 0);
  return new Promise((resolve) => setTimeout(resolve, delay));
}

async function finalizeCanvasCapture(canvas, runtime, dims, options = {}) {
  const bitmap = await runtime.createImageBitmap(canvas);
  const quality = Number.isFinite(options.jpegQuality)
    ? clamp(Number(options.jpegQuality), 0.5, 0.98)
    : 0.92;
  const blob = await canvasToBlob(canvas, quality);
  return {
    bitmap,
    width: dims.width,
    height: dims.height,
    blob,
    sourceWidth: dims.sourceWidth,
    sourceHeight: dims.sourceHeight,
  };
}

export async function captureFrame(videoEl, options = {}, env) {
  const runtime = resolveEnv(env);
  if (!videoEl) throw new Error("camera preview element is required");
  const dims = resolveCaptureDimensions(videoEl, options);

  const canvas = createCanvas(dims.width, dims.height, runtime);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!context) throw new Error("Unable to acquire 2D context for capture");

  drawVideoFrameToCanvas(context, videoEl, dims);
  return finalizeCanvasCapture(canvas, runtime, dims, options);
}

export async function captureBestFrame(videoEl, options = {}, env) {
  const runtime = resolveEnv(env);
  if (!videoEl) throw new Error("camera preview element is required");

  const sampleCount = Number.isFinite(options.framesToSample)
    ? clamp(Math.round(Number(options.framesToSample)), 2, 9)
    : 5;
  if (sampleCount <= 1) {
    return captureFrame(videoEl, options, runtime);
  }

  const delayMs = Number.isFinite(options.delayMs)
    ? clamp(Number(options.delayMs), 40, 400)
    : 120;
  const dims = resolveCaptureDimensions(videoEl, options);
  const canvas = createCanvas(dims.width, dims.height, runtime);
  const context = canvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!context) throw new Error("Unable to acquire 2D context for burst capture");

  let bestScore = Number.NEGATIVE_INFINITY;
  let bestBuffer = null;
  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    drawVideoFrameToCanvas(context, videoEl, dims);
    const imageData = context.getImageData(0, 0, dims.width, dims.height);
    const score = computeFrameSharpnessScore(imageData);
    if (score > bestScore || !bestBuffer) {
      bestScore = score;
      bestBuffer = new Uint8ClampedArray(imageData.data);
    }

    if (hasFunction(options.onProgress)) {
      try {
        options.onProgress({
          sampleIndex: sampleIndex + 1,
          sampleCount,
          score,
          bestScore,
        });
      } catch {
        // ignore callback errors
      }
    }

    if (sampleIndex < sampleCount - 1) {
      await sleep(delayMs);
    }
  }

  if (bestBuffer) {
    const bestImageData = imageDataFromBuffer(context, bestBuffer, dims.width, dims.height);
    if (bestImageData) {
      context.putImageData(bestImageData, 0, 0);
    }
  }

  return finalizeCanvasCapture(canvas, runtime, dims, options);
}
