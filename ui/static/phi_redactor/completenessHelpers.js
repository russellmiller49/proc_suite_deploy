const COMPLETENESS_STATION_TOKEN_RE = /\b(?:1[0-2]|[1-9])(?:[LR](?:[SI])?)?\b/i;

export function normalizeCompletenessStationToken(value) {
  const raw = String(value || "").trim().toUpperCase();
  if (!raw) return "";
  const compact = raw.replace(/[\s_-]+/g, "");
  const loose = compact.match(/(?:1[0-2]|[1-9])(?:[LR](?:[SI])?)?/i);
  return loose ? String(loose[0] || "").toUpperCase() : "";
}

export function extractCompletenessStationFromPrompt(prompt) {
  const candidates = [
    String(prompt?.label || ""),
    String(prompt?.message || ""),
  ];
  for (const text of candidates) {
    const match = String(text || "").match(COMPLETENESS_STATION_TOKEN_RE);
    if (!match) continue;
    const normalized = normalizeCompletenessStationToken(match[0]);
    if (normalized) return normalized;
  }
  return "";
}

export function buildCompletenessEbusStationHintsByIndex(prompts) {
  const map = new Map();
  const list = Array.isArray(prompts) ? prompts : [];
  list.forEach((prompt) => {
    const targetPath = String(prompt?.target_path || prompt?.path || "").trim();
    const match = targetPath.match(/^granular_data\.linear_ebus_stations_detail\[(\d+)\]\.[^.]+$/);
    if (!match) return;
    const idx = Number.parseInt(match[1], 10);
    if (!Number.isFinite(idx) || idx < 0) return;
    if (map.has(idx)) return;
    const station = extractCompletenessStationFromPrompt(prompt);
    if (station) map.set(idx, station);
  });
  return map;
}

export function getCompletenessEbusStationHintForPath(
  prompt,
  effectivePath,
  hintsByIndex,
  getRowStationByIndex = null
) {
  const path = String(effectivePath || "").trim();
  const match = path.match(/^granular_data\.linear_ebus_stations_detail\[(\d+)\]\.[^.]+$/);
  if (!match) return "";
  const idx = Number.parseInt(match[1], 10);
  if (!Number.isFinite(idx) || idx < 0) return "";

  const mapped = hintsByIndex?.get(idx);
  if (mapped) return mapped;

  const fromPrompt = extractCompletenessStationFromPrompt(prompt);
  if (fromPrompt) {
    hintsByIndex?.set(idx, fromPrompt);
    return fromPrompt;
  }

  if (typeof getRowStationByIndex === "function") {
    const rowStation = normalizeCompletenessStationToken(getRowStationByIndex(idx));
    return rowStation || "";
  }

  return "";
}

export function isCompletenessRawValueEmpty(rawValue) {
  return (
    rawValue === null ||
    (typeof rawValue === "string" && rawValue.trim() === "") ||
    (Array.isArray(rawValue) && rawValue.length === 0)
  );
}

export function collectStagedCompletenessEntries({
  prompts,
  registry,
  rawValueByPath,
  resolvePromptPath,
  getStoredWildcardEffectivePathsForPrompt,
  getInputSpec,
  coerceValue,
  getStationHintForPath,
}) {
  const list = Array.isArray(prompts) ? prompts : [];
  const out = [];
  let invalidCount = 0;

  list.forEach((prompt) => {
    const promptPath = String(prompt?.target_path || prompt?.path || "").trim();
    const sourcePath = String(prompt?.path || "").trim();
    if (!promptPath) return;

    const resolved = resolvePromptPath(registry, promptPath);
    if (resolved.hasWildcard && resolved.wildcardCount === 0) return;

    const spec = getInputSpec(promptPath);

    if (spec.type === "ecog") {
      const effectivePath = resolved.effectivePath;
      const rawValue = rawValueByPath.get(effectivePath);
      if (rawValue === undefined) return;
      const raw = String(rawValue || "").trim();
      out.push({
        kind: "ecog",
        promptPath,
        sourcePath,
        effectivePath,
        raw,
      });
      return;
    }

    const effectivePaths = resolved.hasWildcard
      ? getStoredWildcardEffectivePathsForPrompt(promptPath)
      : [resolved.effectivePath];

    effectivePaths.forEach((effectivePath) => {
      const rawValue = rawValueByPath.get(effectivePath);
      if (rawValue === undefined) return;

      const rawEmpty = isCompletenessRawValueEmpty(rawValue);
      const stationHint =
        typeof getStationHintForPath === "function"
          ? String(getStationHintForPath(prompt, effectivePath) || "")
          : "";

      if (rawEmpty) {
        out.push({
          kind: "field",
          promptPath,
          sourcePath,
          effectivePath,
          rawEmpty: true,
          stationHint,
          invalid: false,
          coerced: null,
        });
        return;
      }

      const coerced = coerceValue(spec, rawValue);
      const invalid = coerced === null;
      if (invalid) invalidCount += 1;

      out.push({
        kind: "field",
        promptPath,
        sourcePath,
        effectivePath,
        rawEmpty: false,
        stationHint,
        invalid,
        coerced,
      });
    });
  });

  return { entries: out, invalidCount };
}

function _extractIndices(path) {
  const out = [];
  const matches = String(path || "").matchAll(/\[(\d+)\]/g);
  for (const m of matches) out.push(String(m[1]));
  return out;
}

function _applyWildcardIndices(path, indices) {
  const list = Array.isArray(indices) ? [...indices] : [];
  return String(path || "").replaceAll("[*]", () => `[${list.length ? list.shift() : "0"}]`);
}

export function buildCompletenessCandidatePaths(entry) {
  const effectivePath = String(entry?.effectivePath || "").trim();
  const promptPath = String(entry?.promptPath || "").trim();
  const sourcePath = String(entry?.sourcePath || "").trim();

  const candidates = [];
  const add = (value) => {
    const v = String(value || "").trim();
    if (!v) return;
    if (candidates.includes(v)) return;
    candidates.push(v);
  };

  add(effectivePath);

  if (sourcePath && sourcePath !== promptPath) {
    const indices = _extractIndices(effectivePath);
    if (sourcePath.includes("[*]")) add(_applyWildcardIndices(sourcePath, indices));
    else add(sourcePath);
  }

  return candidates;
}
