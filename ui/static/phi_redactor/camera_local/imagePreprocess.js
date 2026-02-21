function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function safeNumber(value, fallback = 0) {
  return Number.isFinite(value) ? Number(value) : fallback;
}

function buildHistogram(grayValues) {
  const histogram = new Uint32Array(256);
  for (let i = 0; i < grayValues.length; i += 1) {
    histogram[grayValues[i]] += 1;
  }
  return histogram;
}

function computeOtsuThreshold(histogram, totalCount) {
  const total = Math.max(1, safeNumber(totalCount, 1));
  let sum = 0;
  for (let i = 0; i < 256; i += 1) {
    sum += i * histogram[i];
  }

  let sumBackground = 0;
  let weightBackground = 0;
  let bestThreshold = 127;
  let bestVariance = Number.NEGATIVE_INFINITY;

  for (let i = 0; i < 256; i += 1) {
    weightBackground += histogram[i];
    if (!weightBackground) continue;

    const weightForeground = total - weightBackground;
    if (!weightForeground) break;

    sumBackground += i * histogram[i];
    const meanBackground = sumBackground / weightBackground;
    const meanForeground = (sum - sumBackground) / weightForeground;
    const diff = meanBackground - meanForeground;
    const variance = weightBackground * weightForeground * diff * diff;
    if (variance > bestVariance) {
      bestVariance = variance;
      bestThreshold = i;
    }
  }

  return bestThreshold;
}

function percentileFromHistogram(histogram, totalCount, percentile) {
  const total = Math.max(1, Math.floor(safeNumber(totalCount, 1)));
  const pct = clamp(safeNumber(percentile, 0.5), 0, 1);
  const target = Math.max(0, Math.min(total - 1, Math.floor(total * pct)));
  let seen = 0;
  for (let i = 0; i < 256; i += 1) {
    seen += histogram[i];
    if (seen > target) return i;
  }
  return 255;
}

export function computeGrayStats(grayValues) {
  const gray = grayValues instanceof Uint8ClampedArray
    ? grayValues
    : new Uint8ClampedArray(Array.isArray(grayValues) ? grayValues : []);
  if (!gray.length) {
    return {
      p05: 0,
      p10: 0,
      p50: 0,
      p90: 0,
      p95: 0,
      dynamicRange: 0,
      histogram: new Uint32Array(256),
      pixelCount: 0,
    };
  }

  const histogram = buildHistogram(gray);
  const pixelCount = gray.length;
  const p05 = percentileFromHistogram(histogram, pixelCount, 0.05);
  const p10 = percentileFromHistogram(histogram, pixelCount, 0.1);
  const p50 = percentileFromHistogram(histogram, pixelCount, 0.5);
  const p90 = percentileFromHistogram(histogram, pixelCount, 0.9);
  const p95 = percentileFromHistogram(histogram, pixelCount, 0.95);

  return {
    p05,
    p10,
    p50,
    p90,
    p95,
    dynamicRange: Math.max(0, p90 - p10),
    histogram,
    pixelCount,
  };
}

function applyContrastStretch(grayValues, stats, opts = {}) {
  const gray = grayValues instanceof Uint8ClampedArray
    ? grayValues
    : new Uint8ClampedArray(Array.isArray(grayValues) ? grayValues : []);
  if (!gray.length) return gray;

  const lowerPct = clamp(safeNumber(opts.lowerPercentile, 0.06), 0, 0.45);
  const upperPct = clamp(safeNumber(opts.upperPercentile, 0.94), 0.55, 1);
  const lower = percentileFromHistogram(stats.histogram, stats.pixelCount, lowerPct);
  const upper = percentileFromHistogram(stats.histogram, stats.pixelCount, upperPct);
  const span = Math.max(12, upper - lower);

  const out = new Uint8ClampedArray(gray.length);
  for (let i = 0; i < gray.length; i += 1) {
    const normalized = ((gray[i] - lower) * 255) / span;
    out[i] = Math.max(0, Math.min(255, Math.round(normalized)));
  }
  return out;
}

function computeColorDeltaStats(data, width, height) {
  const safeWidth = Math.max(1, Math.floor(safeNumber(width, 1)));
  const safeHeight = Math.max(1, Math.floor(safeNumber(height, 1)));
  const pixelCount = Math.max(1, Math.floor(data.length / 4));
  const sampleStride = Math.max(1, Math.floor(Math.max(safeWidth, safeHeight) / 900));
  const samplePixelStep = Math.max(1, sampleStride * sampleStride);

  let sampled = 0;
  let highDelta = 0;
  let sumDelta = 0;
  for (let p = 0; p < pixelCount; p += samplePixelStep) {
    const i = p * 4;
    const r = data[i];
    const g = data[i + 1];
    const b = data[i + 2];
    const delta = Math.max(
      Math.abs(r - g),
      Math.abs(g - b),
      Math.abs(r - b),
    );
    if (delta >= 24) highDelta += 1;
    sumDelta += delta;
    sampled += 1;
  }

  return {
    meanColorDelta: sampled ? sumDelta / sampled : 0,
    highColorDeltaFrac: sampled ? highDelta / sampled : 0,
    sampled,
  };
}

function shouldApplyMonitorMoireReduction(data, width, height, captureMetrics, plan = {}) {
  const hint = String(plan?.sceneHint || "auto");
  if (hint === "monitor") return true;
  if (hint === "document") return false;

  const colorStats = computeColorDeltaStats(data, width, height);
  const dynamicRange = safeNumber(captureMetrics?.dynamicRange, 0);
  const blurVariance = safeNumber(captureMetrics?.blurVariance, 0);
  const overexposureRatio = safeNumber(captureMetrics?.overexposureRatio, 0);

  // Keep auto-detection conservative to avoid over-smoothing paper captures.
  const strongSignal =
    colorStats.highColorDeltaFrac > 0.28 &&
    colorStats.meanColorDelta > 14 &&
    dynamicRange < 110 &&
    blurVariance > 210;
  const extremeSignal =
    colorStats.highColorDeltaFrac > 0.34 &&
    colorStats.meanColorDelta > 16 &&
    overexposureRatio < 0.5;

  return strongSignal || extremeSignal;
}

function buildIntegralImage(grayValues, width, height) {
  const safeWidth = Math.max(1, Math.floor(safeNumber(width, 1)));
  const safeHeight = Math.max(1, Math.floor(safeNumber(height, 1)));
  const integral = new Uint32Array((safeWidth + 1) * (safeHeight + 1));

  for (let y = 1; y <= safeHeight; y += 1) {
    let rowSum = 0;
    for (let x = 1; x <= safeWidth; x += 1) {
      rowSum += grayValues[(y - 1) * safeWidth + (x - 1)];
      integral[y * (safeWidth + 1) + x] = integral[(y - 1) * (safeWidth + 1) + x] + rowSum;
    }
  }
  return integral;
}

function sumRectIntegral(integral, width, x0, y0, x1, y1) {
  const stride = width + 1;
  const a = integral[y0 * stride + x0];
  const b = integral[y0 * stride + x1];
  const c = integral[y1 * stride + x0];
  const d = integral[y1 * stride + x1];
  return d - b - c + a;
}

function boxBlurGray(grayValues, width, height, radius = 1) {
  const safeWidth = Math.max(1, Math.floor(safeNumber(width, 1)));
  const safeHeight = Math.max(1, Math.floor(safeNumber(height, 1)));
  const safeRadius = clamp(Math.floor(safeNumber(radius, 1)), 1, 6);
  const out = new Uint8ClampedArray(grayValues.length);
  const integral = buildIntegralImage(grayValues, safeWidth, safeHeight);

  for (let y = 0; y < safeHeight; y += 1) {
    const y0 = Math.max(0, y - safeRadius);
    const y1 = Math.min(safeHeight - 1, y + safeRadius);
    for (let x = 0; x < safeWidth; x += 1) {
      const x0 = Math.max(0, x - safeRadius);
      const x1 = Math.min(safeWidth - 1, x + safeRadius);
      const sum = sumRectIntegral(integral, safeWidth, x0, y0, x1 + 1, y1 + 1);
      const count = Math.max(1, (x1 - x0 + 1) * (y1 - y0 + 1));
      out[y * safeWidth + x] = Math.round(sum / count);
    }
  }
  return out;
}

function resampleGrayBilinear(grayValues, sourceWidth, sourceHeight, targetWidth, targetHeight) {
  const src = grayValues instanceof Uint8ClampedArray
    ? grayValues
    : new Uint8ClampedArray(Array.isArray(grayValues) ? grayValues : []);
  const srcWidth = Math.max(1, Math.floor(safeNumber(sourceWidth, 1)));
  const srcHeight = Math.max(1, Math.floor(safeNumber(sourceHeight, 1)));
  const dstWidth = Math.max(1, Math.floor(safeNumber(targetWidth, srcWidth)));
  const dstHeight = Math.max(1, Math.floor(safeNumber(targetHeight, srcHeight)));
  if (!src.length || (srcWidth === dstWidth && srcHeight === dstHeight)) {
    return new Uint8ClampedArray(src);
  }

  const out = new Uint8ClampedArray(dstWidth * dstHeight);
  const scaleX = srcWidth / dstWidth;
  const scaleY = srcHeight / dstHeight;

  for (let y = 0; y < dstHeight; y += 1) {
    const srcY = (y + 0.5) * scaleY - 0.5;
    const y0 = Math.max(0, Math.min(srcHeight - 1, Math.floor(srcY)));
    const y1 = Math.max(0, Math.min(srcHeight - 1, y0 + 1));
    const wy = srcY - y0;
    for (let x = 0; x < dstWidth; x += 1) {
      const srcX = (x + 0.5) * scaleX - 0.5;
      const x0 = Math.max(0, Math.min(srcWidth - 1, Math.floor(srcX)));
      const x1 = Math.max(0, Math.min(srcWidth - 1, x0 + 1));
      const wx = srcX - x0;

      const p00 = src[y0 * srcWidth + x0];
      const p10 = src[y0 * srcWidth + x1];
      const p01 = src[y1 * srcWidth + x0];
      const p11 = src[y1 * srcWidth + x1];

      const top = p00 + (p10 - p00) * wx;
      const bottom = p01 + (p11 - p01) * wx;
      out[y * dstWidth + x] = Math.round(top + (bottom - top) * wy);
    }
  }

  return out;
}

function downscaleUpscaleGray(grayValues, width, height, opts = {}) {
  const srcWidth = Math.max(1, Math.floor(safeNumber(width, 1)));
  const srcHeight = Math.max(1, Math.floor(safeNumber(height, 1)));
  const src = grayValues instanceof Uint8ClampedArray
    ? grayValues
    : new Uint8ClampedArray(Array.isArray(grayValues) ? grayValues : []);
  if (!src.length || srcWidth < 180 || srcHeight < 180) return src;

  const scale = clamp(safeNumber(opts.scale, 0.62), 0.45, 0.9);
  const downWidth = Math.max(24, Math.round(srcWidth * scale));
  const downHeight = Math.max(24, Math.round(srcHeight * scale));
  if (downWidth >= srcWidth || downHeight >= srcHeight) return src;

  const downsampled = resampleGrayBilinear(src, srcWidth, srcHeight, downWidth, downHeight);
  return resampleGrayBilinear(downsampled, downWidth, downHeight, srcWidth, srcHeight);
}

function applyUnsharpMask(grayValues, width, height, opts = {}) {
  const gray = grayValues instanceof Uint8ClampedArray
    ? grayValues
    : new Uint8ClampedArray(Array.isArray(grayValues) ? grayValues : []);
  if (!gray.length) return gray;

  const radius = clamp(Math.floor(safeNumber(opts.radius, 1)), 1, 5);
  const amount = clamp(safeNumber(opts.amount, 0.8), 0, 2.4);
  if (amount <= 0) return gray;

  const blurred = boxBlurGray(gray, width, height, radius);
  const out = new Uint8ClampedArray(gray.length);
  for (let i = 0; i < gray.length; i += 1) {
    const sharpened = gray[i] + amount * (gray[i] - blurred[i]);
    out[i] = clamp(Math.round(sharpened), 0, 255);
  }
  return out;
}

function applyAdaptiveThreshold(grayValues, width, height, opts = {}) {
  const gray = grayValues instanceof Uint8ClampedArray
    ? grayValues
    : new Uint8ClampedArray(Array.isArray(grayValues) ? grayValues : []);
  if (!gray.length) return gray;

  const safeWidth = Math.max(1, Math.floor(safeNumber(width, 1)));
  const safeHeight = Math.max(1, Math.floor(safeNumber(height, 1)));
  const minDim = Math.max(1, Math.min(safeWidth, safeHeight));
  const windowFraction = clamp(safeNumber(opts.windowFraction, 0.09), 0.04, 0.22);
  const thresholdRatio = clamp(safeNumber(opts.thresholdRatio, 0.14), 0.06, 0.26);
  const radius = Math.max(2, Math.floor((minDim * windowFraction) / 2));
  const integral = buildIntegralImage(gray, safeWidth, safeHeight);
  const out = new Uint8ClampedArray(gray.length);

  for (let y = 0; y < safeHeight; y += 1) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(safeHeight - 1, y + radius);
    for (let x = 0; x < safeWidth; x += 1) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(safeWidth - 1, x + radius);
      const sum = sumRectIntegral(integral, safeWidth, x0, y0, x1 + 1, y1 + 1);
      const area = Math.max(1, (x1 - x0 + 1) * (y1 - y0 + 1));
      const localMean = sum / area;
      const threshold = localMean * (1 - thresholdRatio);
      out[y * safeWidth + x] = gray[y * safeWidth + x] <= threshold ? 0 : 255;
    }
  }
  return out;
}

export function resolveAutoPreprocessMode(grayStats, qualityMetrics) {
  const dynamicRange = safeNumber(grayStats?.dynamicRange, 0);
  const overexposureRatio = safeNumber(qualityMetrics?.overexposureRatio, 0);
  const underexposureRatio = safeNumber(qualityMetrics?.underexposureRatio, 0);

  if (overexposureRatio > 0.34) return "bw_high_contrast";
  if (underexposureRatio > 0.46) return "bw_high_contrast";
  if (dynamicRange < 54) return "bw_high_contrast";
  return "grayscale";
}

export function computeScaledDimensions(width, height, maxDim = 2000) {
  const srcWidth = Math.max(1, Math.floor(safeNumber(width, 1)));
  const srcHeight = Math.max(1, Math.floor(safeNumber(height, 1)));
  const safeMaxDim = clamp(Math.floor(safeNumber(maxDim, 2000)), 320, 4096);
  const scale = Math.min(1, safeMaxDim / Math.max(srcWidth, srcHeight));
  return {
    width: Math.max(1, Math.round(srcWidth * scale)),
    height: Math.max(1, Math.round(srcHeight * scale)),
    scale,
  };
}

export function buildPreprocessPlan(input = {}) {
  const mode = input.mode === "bw_high_contrast"
    ? "bw_high_contrast"
    : input.mode === "grayscale"
      ? "grayscale"
      : input.mode === "auto"
        ? "auto"
        : "off";

  const dims = computeScaledDimensions(input.width, input.height, input.maxDim);
  const sceneHint = input.sceneHint === "monitor"
    ? "monitor"
    : input.sceneHint === "document"
      ? "document"
      : "auto";
  return {
    mode,
    sceneHint,
    resolvedMode: mode,
    targetWidth: dims.width,
    targetHeight: dims.height,
    scale: dims.scale,
    applyGrayscale: mode === "grayscale" || mode === "bw_high_contrast" || mode === "auto",
    applyThreshold: mode === "bw_high_contrast",
    applyAdaptiveThreshold: mode === "bw_high_contrast" || mode === "auto",
    autoTuning: mode === "auto",
  };
}

export function computeGrayFromImageData(data, opts = {}) {
  const pixelCount = Math.floor(data.length / 4);
  const gray = new Uint8ClampedArray(pixelCount);
  const channel = opts?.channel === "green" ? "green" : "luma";
  let p = 0;
  for (let i = 0; i < data.length; i += 4) {
    gray[p] = channel === "green"
      ? data[i + 1]
      : Math.round(0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]);
    p += 1;
  }
  return gray;
}

export function applyPreprocessToImageData(imageData, plan) {
  const data = imageData?.data;
  if (!data) {
    return {
      changed: false,
      gray: new Uint8ClampedArray(),
      sourceGray: new Uint8ClampedArray(),
      threshold: null,
      resolvedMode: "off",
      grayStats: computeGrayStats([]),
    };
  }

  const srcWidth = Math.max(1, Math.floor(safeNumber(imageData?.width, 1)));
  const srcHeight = Math.max(1, Math.floor(safeNumber(imageData?.height, 1)));
  const neutralGray = computeGrayFromImageData(data);
  const baselineMetrics = computeCaptureQualityMetrics(neutralGray, srcWidth, srcHeight);
  const monitorMoireReduction = shouldApplyMonitorMoireReduction(
    data,
    srcWidth,
    srcHeight,
    baselineMetrics,
    plan,
  );
  const sourceGray = monitorMoireReduction
    ? computeGrayFromImageData(data, { channel: "green" })
    : neutralGray;
  const sourceStats = computeGrayStats(sourceGray);
  const captureMetrics = computeCaptureQualityMetrics(sourceGray, srcWidth, srcHeight);
  const forcedMonitorHint = String(plan?.sceneHint || "auto") === "monitor";

  let resolvedMode = String(plan?.mode || "off");
  if (resolvedMode === "auto") {
    resolvedMode = resolveAutoPreprocessMode(sourceStats, captureMetrics);
    // Monitor captures often OCR better with grayscale than hard binarization.
    if (
      monitorMoireReduction &&
      resolvedMode === "bw_high_contrast" &&
      safeNumber(captureMetrics.dynamicRange, 0) >= 40
    ) {
      resolvedMode = "grayscale";
    }
  }

  if (resolvedMode !== "grayscale" && resolvedMode !== "bw_high_contrast") {
    return {
      changed: false,
      gray: sourceGray,
      sourceGray,
      threshold: null,
      resolvedMode: "off",
      grayStats: sourceStats,
      monitorMoireReduction,
    };
  }

  let threshold = null;
  const glareHeavy = safeNumber(captureMetrics.overexposureRatio, 0) > 0.26;
  const lowContrastScene = safeNumber(captureMetrics.dynamicRange, 999) < 54;
  let output = sourceGray;
  if (monitorMoireReduction) {
    output = downscaleUpscaleGray(output, srcWidth, srcHeight, {
      scale: forcedMonitorHint ? 0.76 : 0.82,
    });
    if (!forcedMonitorHint) {
      output = boxBlurGray(output, srcWidth, srcHeight, 1);
    }
  }

  output = applyUnsharpMask(output, srcWidth, srcHeight, {
    radius: resolvedMode === "bw_high_contrast" ? 2 : 1,
    amount: resolvedMode === "bw_high_contrast"
      ? monitorMoireReduction
        ? glareHeavy
          ? 0.78
          : 0.88
        : glareHeavy
          ? 0.95
          : 1.08
      : 0.76,
  });
  const sharpenedStats = computeGrayStats(output);
  output = applyContrastStretch(output, sharpenedStats, {
    lowerPercentile: resolvedMode === "bw_high_contrast"
      ? monitorMoireReduction
        ? glareHeavy
          ? 0.078
          : 0.068
        : glareHeavy
          ? 0.065
          : 0.055
      : 0.065,
    upperPercentile: resolvedMode === "bw_high_contrast"
      ? monitorMoireReduction
        ? glareHeavy
          ? 0.916
          : 0.932
        : glareHeavy
          ? 0.935
          : 0.945
      : 0.935,
  });

  if (resolvedMode === "bw_high_contrast") {
    let binary = applyAdaptiveThreshold(output, srcWidth, srcHeight, {
      windowFraction: monitorMoireReduction
        ? lowContrastScene
          ? 0.102
          : 0.092
        : lowContrastScene
          ? 0.094
          : 0.082,
      thresholdRatio: monitorMoireReduction
        ? glareHeavy
          ? 0.106
          : 0.122
        : glareHeavy
          ? 0.112
          : 0.132,
    });

    // Guardrail: fallback to Otsu if adaptive output collapses to near-solid mask.
    let darkCount = 0;
    for (let i = 0; i < binary.length; i += 1) {
      if (binary[i] === 0) darkCount += 1;
    }
    const darkFrac = darkCount / Math.max(1, binary.length);
    if (darkFrac < 0.01 || darkFrac > 0.99) {
      const histogram = buildHistogram(output);
      threshold = computeOtsuThreshold(histogram, output.length);
      binary = new Uint8ClampedArray(output.length);
      for (let i = 0; i < output.length; i += 1) {
        binary[i] = output[i] <= threshold ? 0 : 255;
      }
    } else {
      threshold = null;
    }
    output = binary;
  }

  let p = 0;
  for (let i = 0; i < data.length; i += 4) {
    const v = output[p];
    data[i] = v;
    data[i + 1] = v;
    data[i + 2] = v;
    data[i + 3] = 255;
    p += 1;
  }

  return {
    changed: true,
    gray: output,
    sourceGray,
    threshold,
    resolvedMode,
    grayStats: computeGrayStats(output),
    monitorMoireReduction,
  };
}

function computeLaplacianVariance(gray, width, height) {
  if (!gray.length || width < 3 || height < 3) return 0;

  let sum = 0;
  let sumSquares = 0;
  let count = 0;
  const stride = Math.max(1, Math.floor(Math.max(width, height) / 600));

  for (let y = 1; y < height - 1; y += stride) {
    for (let x = 1; x < width - 1; x += stride) {
      const idx = y * width + x;
      const lap =
        gray[idx - width] +
        gray[idx - 1] +
        gray[idx + 1] +
        gray[idx + width] -
        4 * gray[idx];

      sum += lap;
      sumSquares += lap * lap;
      count += 1;
    }
  }

  if (!count) return 0;
  const mean = sum / count;
  return Math.max(0, sumSquares / count - mean * mean);
}

export function computeCaptureQualityMetrics(gray, width, height, opts = {}) {
  const safeWidth = Math.max(1, Math.floor(safeNumber(width, 1)));
  const safeHeight = Math.max(1, Math.floor(safeNumber(height, 1)));
  const whiteThreshold = clamp(Math.floor(safeNumber(opts.whiteThreshold, 245)), 180, 254);
  const darkThreshold = clamp(Math.floor(safeNumber(opts.darkThreshold, 32)), 1, 120);

  let whiteCount = 0;
  let darkCount = 0;
  for (let i = 0; i < gray.length; i += 1) {
    if (gray[i] >= whiteThreshold) whiteCount += 1;
    if (gray[i] <= darkThreshold) darkCount += 1;
  }

  const pixelCount = Math.max(1, gray.length);
  const histogram = buildHistogram(gray);
  const p10 = percentileFromHistogram(histogram, pixelCount, 0.1);
  const p90 = percentileFromHistogram(histogram, pixelCount, 0.9);
  const overexposureRatio = whiteCount / pixelCount;
  const underexposureRatio = darkCount / pixelCount;
  const blurVariance = computeLaplacianVariance(gray, safeWidth, safeHeight);

  return {
    overexposureRatio,
    underexposureRatio,
    dynamicRange: Math.max(0, p90 - p10),
    blurVariance,
    pixelCount,
  };
}

function resolveWarningProfile(value) {
  const profile = String(value || "default").trim().toLowerCase();
  return profile === "ios_safari" ? "ios_safari" : "default";
}

export function resolveCaptureWarningThresholds(metrics, opts = {}) {
  const profile = resolveWarningProfile(opts.warningProfile || opts.profile);
  const defaults = profile === "ios_safari"
    ? {
        blurMinVariance: 92,
        maxOverexposureRatio: 0.62,
        maxUnderexposureRatio: 0.66,
        minDynamicRange: 44,
      }
    : {
        blurMinVariance: 110,
        maxOverexposureRatio: 0.55,
        maxUnderexposureRatio: 0.58,
        minDynamicRange: 50,
      };

  let blurMinVariance = safeNumber(opts.blurMinVariance, defaults.blurMinVariance);
  let maxOverexposureRatio = safeNumber(opts.maxOverexposureRatio, defaults.maxOverexposureRatio);
  let maxUnderexposureRatio = safeNumber(opts.maxUnderexposureRatio, defaults.maxUnderexposureRatio);
  let minDynamicRange = safeNumber(opts.minDynamicRange, defaults.minDynamicRange);

  const dynamicRange = safeNumber(metrics?.dynamicRange, 0);
  const overexposureRatio = safeNumber(metrics?.overexposureRatio, 0);
  const underexposureRatio = safeNumber(metrics?.underexposureRatio, 0);
  const pixelCount = Math.max(1, safeNumber(metrics?.pixelCount, 0));

  if (dynamicRange >= 72) {
    blurMinVariance -= 14;
  } else if (dynamicRange >= 60) {
    blurMinVariance -= 8;
  } else if (dynamicRange < 38) {
    blurMinVariance += 8;
  }
  if (overexposureRatio > maxOverexposureRatio * 0.7 || underexposureRatio > maxUnderexposureRatio * 0.7) {
    blurMinVariance -= 10;
  }
  if (pixelCount >= 1_000_000) {
    blurMinVariance -= 6;
  }
  if (Math.max(overexposureRatio, underexposureRatio) > 0.34) {
    minDynamicRange -= 6;
  }

  blurMinVariance = clamp(Math.round(blurMinVariance), 68, 160);
  maxOverexposureRatio = clamp(maxOverexposureRatio, 0.35, 0.85);
  maxUnderexposureRatio = clamp(maxUnderexposureRatio, 0.35, 0.85);
  minDynamicRange = clamp(Math.round(minDynamicRange), 36, 85);

  return {
    profile,
    blurMinVariance,
    maxOverexposureRatio,
    maxUnderexposureRatio,
    minDynamicRange,
  };
}

export function buildCaptureWarnings(metrics, opts = {}) {
  const thresholds = resolveCaptureWarningThresholds(metrics, opts);
  const blurVariance = safeNumber(metrics?.blurVariance, 0);
  const overexposureRatio = safeNumber(metrics?.overexposureRatio, 0);
  const underexposureRatio = safeNumber(metrics?.underexposureRatio, 0);
  const severeExposure =
    overexposureRatio > thresholds.maxOverexposureRatio + 0.16 ||
    underexposureRatio > thresholds.maxUnderexposureRatio + 0.16;
  const warnings = [];

  if (
    blurVariance < thresholds.blurMinVariance &&
    (!severeExposure || blurVariance < thresholds.blurMinVariance * 0.62)
  ) {
    warnings.push("Image may be blurry; consider retaking.");
  }
  if (overexposureRatio > thresholds.maxOverexposureRatio) {
    warnings.push("Image may be overexposed; reduce glare and retake.");
  }
  if (underexposureRatio > thresholds.maxUnderexposureRatio) {
    warnings.push("Image may be underexposed; increase lighting and retake.");
  }
  if (safeNumber(metrics.dynamicRange, 999) < thresholds.minDynamicRange) {
    warnings.push("Image has low contrast; use flatter lighting or high-contrast enhance mode.");
  }

  return warnings;
}

function createOffscreenCanvas(width, height) {
  return new OffscreenCanvas(width, height);
}

export function preprocessCanvasForOcr(canvas, options = {}) {
  const sourceWidth = Math.max(1, Math.floor(safeNumber(canvas?.width, 1)));
  const sourceHeight = Math.max(1, Math.floor(safeNumber(canvas?.height, 1)));
  const plan = buildPreprocessPlan({
    width: sourceWidth,
    height: sourceHeight,
    maxDim: options.maxDim,
    mode: options.mode,
    sceneHint: options.sceneHint,
  });

  const targetCanvas = createOffscreenCanvas(plan.targetWidth, plan.targetHeight);
  const ctx = targetCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
  if (!ctx) {
    throw new Error("Unable to acquire 2D context for preprocess");
  }

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, plan.targetWidth, plan.targetHeight);
  ctx.drawImage(canvas, 0, 0, sourceWidth, sourceHeight, 0, 0, plan.targetWidth, plan.targetHeight);

  const imageData = ctx.getImageData(0, 0, plan.targetWidth, plan.targetHeight);
  const processed = applyPreprocessToImageData(imageData, plan);
  if (processed.changed) {
    ctx.putImageData(imageData, 0, 0);
  }

  const qualityGray = processed.sourceGray.length
    ? processed.sourceGray
    : computeGrayFromImageData(imageData.data);
  const metrics = computeCaptureQualityMetrics(qualityGray, plan.targetWidth, plan.targetHeight, options);
  const warnings = buildCaptureWarnings(metrics, options);
  if (processed.monitorMoireReduction) {
    warnings.push("Applied anti-moire monitor smoothing before OCR.");
  }

  return {
    canvas: targetCanvas,
    plan: {
      ...plan,
      resolvedMode: processed.resolvedMode || plan.mode,
    },
    threshold: processed.threshold,
    grayStats: processed.grayStats,
    metrics,
    warnings,
  };
}
