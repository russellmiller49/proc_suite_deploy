import { pipeline, env } from "./transformers.min.js";

import { repairSpeechTranscript } from "./speechTranscriptRepair.js";

const TASK = "automatic-speech-recognition";
const SPEECH_MODELS = Object.freeze({
  base: Object.freeze({
    key: "base",
    label: "Base",
    bundleId: "speech_whisper_base_en",
  }),
  tiny: Object.freeze({
    key: "tiny",
    label: "Tiny",
    bundleId: "speech_whisper_tiny_en",
  }),
});
const DEFAULT_MODEL_KEY = "tiny";
const MODEL_BASE_URL = new URL("./vendor/", import.meta.url).toString();
const ONNX_WASM_BASE_URL = new URL("./vendor/transformers/", import.meta.url).toString();
const ONNX_WASM_REQUIRED_FILES = [
  "ort-wasm.wasm",
  "ort-wasm-threaded.wasm",
  "ort-wasm-simd.wasm",
  "ort-wasm-simd-threaded.wasm",
];

env.allowRemoteModels = false;
env.localModelPath = MODEL_BASE_URL;
if (env.backends?.onnx?.wasm) {
  env.backends.onnx.wasm.proxy = false;
  env.backends.onnx.wasm.numThreads = 1;
  env.backends.onnx.wasm.wasmPaths = ONNX_WASM_BASE_URL;
}

let modelPromise = null;
let assetsVerified = false;
let assetsVerifiedBundleId = "";
let activeModel = SPEECH_MODELS[DEFAULT_MODEL_KEY];
let lastStatusMessage = "";

function emitStatus(message) {
  const text = String(message || "").trim();
  if (!text || text === lastStatusMessage) return;
  lastStatusMessage = text;
  self.postMessage({
    type: "status",
    modelKey: activeModel.key,
    model: activeModel.bundleId,
    modelLabel: activeModel.label,
    message: text,
  });
}

function resolveSpeechModel(modelKey) {
  return SPEECH_MODELS[String(modelKey || "").trim()] || SPEECH_MODELS[DEFAULT_MODEL_KEY];
}

function setActiveSpeechModel(modelKey) {
  const nextModel = resolveSpeechModel(modelKey);
  if (nextModel.bundleId !== activeModel.bundleId) {
    modelPromise = null;
    assetsVerified = false;
    assetsVerifiedBundleId = "";
    lastStatusMessage = "";
  }
  activeModel = nextModel;
  return nextModel;
}

function buildModelConfigUrl(model) {
  return new URL(`./vendor/${model.bundleId}/config.json`, import.meta.url).toString();
}

async function verifyAssets(model = activeModel) {
  if (assetsVerified && assetsVerifiedBundleId === model.bundleId) return true;
  emitStatus(`Checking local ${model.label} speech assets…`);
  const modelResponse = await fetch(buildModelConfigUrl(model), { cache: "no-store" });
  if (!modelResponse.ok) {
    throw new Error(
      `Local ${model.label} speech model assets are not available. Re-run the speech vendor bootstrap or switch models.`,
    );
  }
  for (const filename of ONNX_WASM_REQUIRED_FILES) {
    const runtimeResponse = await fetch(new URL(`./vendor/transformers/${filename}`, import.meta.url), {
      cache: "no-store",
    });
    if (!runtimeResponse.ok) {
      throw new Error("Local speech runtime assets are not available");
    }
  }
  assetsVerified = true;
  assetsVerifiedBundleId = model.bundleId;
  return true;
}

function buildProgressMessage(progress, model) {
  const status = String(progress?.status || "").trim().toLowerCase();
  if (!status) return "";
  if (status === "initiate") return `Loading local ${model.label} model files…`;
  if (status === "download") return `Downloading local ${model.label} model files…`;
  if (status === "progress") {
    const percent = Number(progress?.progress);
    if (Number.isFinite(percent) && percent > 0) {
      return `Loading local ${model.label} model… ${Math.round(percent)}%`;
    }
    return `Loading local ${model.label} model…`;
  }
  if (status === "done") return `Finishing local ${model.label} model load…`;
  return "";
}

async function getSpeechPipeline(model = activeModel, { warmup = false } = {}) {
  await verifyAssets(model);
  if (!modelPromise) {
    emitStatus(
      warmup
        ? `Preparing local ${model.label} speech model…`
        : `Loading local ${model.label} speech model…`,
    );
    modelPromise = pipeline(TASK, model.bundleId, {
      device: "wasm",
      local_files_only: true,
      progress_callback: (progress) => {
        const message = buildProgressMessage(progress, model);
        if (message) emitStatus(message);
      },
    });
  }
  return modelPromise;
}

function buildRepairWarnings(replacements) {
  if (!Array.isArray(replacements) || !replacements.length) return [];
  return [`REPORTER_SPEECH_LOCAL_REPAIR: applied_${replacements.length}_deterministic_fixes`];
}

self.onmessage = async (event) => {
  const msg = event?.data || {};
  if (!msg || typeof msg.type !== "string") return;

  if (msg.type === "init") {
    const model = setActiveSpeechModel(msg.modelKey);
    try {
      await getSpeechPipeline(model, { warmup: true });
      self.postMessage({
        type: "ready",
        modelKey: model.key,
        model: model.bundleId,
        modelLabel: model.label,
      });
    } catch (error) {
      self.postMessage({
        type: "unavailable",
        modelKey: model.key,
        model: model.bundleId,
        modelLabel: model.label,
        message: error?.message || "Local speech model assets are unavailable",
      });
    }
    return;
  }

  if (msg.type !== "transcribe") return;

  const requestId = String(msg.requestId || "");
  try {
    emitStatus(`Transcribing locally with ${activeModel.label}…`);
    const speechPipeline = await getSpeechPipeline(activeModel);
    const audio = new Float32Array(msg.audio || []);
    const result = await speechPipeline(audio, {
      chunk_length_s: 30,
      stride_length_s: 5,
      return_timestamps: false,
    });
    const rawTranscript = String(result?.text || "").trim();
    emitStatus(`Applying local ${activeModel.label} transcript repairs…`);
    const repaired = repairSpeechTranscript(rawTranscript);
    self.postMessage({
      type: "transcription_result",
      requestId,
      modelKey: activeModel.key,
      model: activeModel.bundleId,
      modelLabel: activeModel.label,
      transcript: repaired.text,
      warnings: buildRepairWarnings(repaired.replacements),
    });
  } catch (error) {
    self.postMessage({
      type: "transcription_error",
      requestId,
      modelKey: activeModel.key,
      model: activeModel.bundleId,
      modelLabel: activeModel.label,
      message: error?.message || "Local speech transcription failed",
    });
  }
};
