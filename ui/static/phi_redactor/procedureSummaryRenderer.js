function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function asTrimmedString(value) {
  return String(value ?? "").trim();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function normalizeStationToken(raw) {
  return asTrimmedString(raw).replace(/[^0-9A-Za-z]/g, "").toUpperCase();
}

export function normalizeTargetId(rawLabel) {
  const raw = asTrimmedString(rawLabel);
  if (!raw) return "";
  const station = normalizeStationToken(raw);
  if (/^(?:[1-9]|1[0-4])(?:[LR])?$/.test(station)) {
    return `ln:${station}`;
  }

  const slug = raw
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return slug ? `lesion:${slug}` : "";
}

export function buildCanonicalTargets(registryData) {
  const registry = isRecord(registryData) ? registryData : {};
  const targets = [];
  const targetIndex = new Map();

  const addTarget = (row) => {
    if (!row || !row.id) return;
    if (targetIndex.has(row.id)) return;
    targetIndex.set(row.id, row);
    targets.push(row);
  };

  const mediastinal = Array.isArray(registry?.targets?.mediastinal_targets) ? registry.targets.mediastinal_targets : [];
  mediastinal.forEach((t) => {
    if (!isRecord(t)) return;
    const station = asTrimmedString(t.station);
    const id = normalizeTargetId(station);
    if (!id) return;
    addTarget({
      id,
      kind: "mediastinal_target",
      label: station || id,
      station: station || null,
      location_text: asTrimmedString(t.location_text) || null,
    });
  });

  const nodeEvents = Array.isArray(registry?.procedures_performed?.linear_ebus?.node_events)
    ? registry.procedures_performed.linear_ebus.node_events
    : [];
  nodeEvents.forEach((ev) => {
    if (!isRecord(ev)) return;
    const station = asTrimmedString(ev.station);
    const id = normalizeTargetId(station);
    if (!id) return;
    addTarget({
      id,
      kind: "ebus_node_event",
      label: station || id,
      station: station || null,
    });
  });

  const peripheral = Array.isArray(registry?.targets?.peripheral_targets) ? registry.targets.peripheral_targets : [];
  peripheral.forEach((t) => {
    if (!isRecord(t)) return;
    const targetKey = asTrimmedString(t.target_key);
    const labelParts = [asTrimmedString(t.laterality), asTrimmedString(t.lobe), asTrimmedString(t.segment)].filter(Boolean);
    const fallbackLabel = labelParts.join(" ");
    const id = normalizeTargetId(targetKey || fallbackLabel);
    if (!id) return;
    addTarget({
      id,
      kind: "peripheral_target",
      label: targetKey || fallbackLabel || id,
      target_key: targetKey || null,
      laterality: asTrimmedString(t.laterality) || null,
      lobe: asTrimmedString(t.lobe) || null,
      segment: asTrimmedString(t.segment) || null,
    });
  });

  return { targets, targetIndex };
}

function normalizeEventInput(row) {
  const ev = isRecord(row) ? row : {};
  const extracted = isRecord(ev.extracted_json) ? ev.extracted_json : isRecord(ev.extracted) ? ev.extracted : null;
  return {
    event_type: asTrimmedString(ev.event_type || ev.type),
    extracted,
    structured: isRecord(ev.structured_data) ? ev.structured_data : null,
    relative_day_offset: Number.isFinite(Number(ev.relative_day_offset)) ? Number(ev.relative_day_offset) : 0,
    created_at: asTrimmedString(ev.created_at),
    event_title: asTrimmedString(ev.event_title),
    source_modality: asTrimmedString(ev.source_modality),
    event_subtype: asTrimmedString(ev.event_subtype),
    id: asTrimmedString(ev.id),
  };
}

function chooseLatest(existing, incoming) {
  if (!incoming) return existing;
  if (!existing) return incoming;
  const a = Date.parse(existing.created_at || "");
  const b = Date.parse(incoming.created_at || "");
  if (Number.isFinite(a) && Number.isFinite(b)) return b >= a ? incoming : existing;
  return incoming;
}

export function buildJoinedTargetSummary(registryData, eventPayloads) {
  const { targets, targetIndex } = buildCanonicalTargets(registryData);
  const join = new Map();
  targets.forEach((t) => {
    join.set(t.id, { target: t, pathology: null, imaging: null });
  });

  const rows = Array.isArray(eventPayloads) ? eventPayloads : [];
  rows.forEach((row) => {
    const ev = normalizeEventInput(row);
    const extracted = ev.extracted;
    if (!extracted) return;
    const normalizedType = String(ev.event_type || "").toLowerCase();

    if (normalizedType === "pathology") {
      const nodeUpdates = Array.isArray(extracted?.node_updates) ? extracted.node_updates : [];
      nodeUpdates.forEach((nu) => {
        if (!isRecord(nu)) return;
        const station = asTrimmedString(nu.station);
        const id = normalizeTargetId(station);
        if (!id) return;
        const base = join.get(id) || { target: targetIndex.get(id) || { id, label: station || id }, pathology: null, imaging: null };
        base.pathology = chooseLatest(base.pathology, {
          created_at: ev.created_at,
          relative_day_offset: ev.relative_day_offset,
          source_event_id: ev.id || null,
          result: asTrimmedString(nu.path_result) || null,
          diagnosis: asTrimmedString(nu.path_diagnosis_text) || null,
        });
        join.set(id, base);
      });

      const peripheralUpdates = Array.isArray(extracted?.peripheral_updates) ? extracted.peripheral_updates : [];
      peripheralUpdates.forEach((pu) => {
        if (!isRecord(pu)) return;
        const key = asTrimmedString(pu.target_key) || [pu.laterality, pu.lobe, pu.segment].map(asTrimmedString).filter(Boolean).join(" ");
        const id = normalizeTargetId(key);
        if (!id) return;
        const base = join.get(id) || { target: targetIndex.get(id) || { id, label: key || id }, pathology: null, imaging: null };
        base.pathology = chooseLatest(base.pathology, {
          created_at: ev.created_at,
          relative_day_offset: ev.relative_day_offset,
          source_event_id: ev.id || null,
          result: asTrimmedString(pu.path_result) || null,
          diagnosis: asTrimmedString(pu.path_diagnosis_text) || null,
        });
        join.set(id, base);
      });
    }

    if (normalizedType === "imaging") {
      const targetsUpdate = isRecord(extracted?.targets_update) ? extracted.targets_update : {};
      const mediastinalTargets = Array.isArray(targetsUpdate?.mediastinal_targets) ? targetsUpdate.mediastinal_targets : [];
      mediastinalTargets.forEach((mt) => {
        if (!isRecord(mt)) return;
        const station = asTrimmedString(mt.station);
        const id = normalizeTargetId(station);
        if (!id) return;
        const base = join.get(id) || { target: targetIndex.get(id) || { id, label: station || id }, pathology: null, imaging: null };
        base.imaging = chooseLatest(base.imaging, {
          created_at: ev.created_at,
          relative_day_offset: ev.relative_day_offset,
          source_event_id: ev.id || null,
          short_axis_mm: Number.isFinite(Number(mt.short_axis_mm)) ? Number(mt.short_axis_mm) : null,
          pet_avid: typeof mt.pet_avid === "boolean" ? mt.pet_avid : null,
          suvmax: Number.isFinite(Number(mt.pet_suvmax)) ? Number(mt.pet_suvmax) : null,
          change: asTrimmedString(mt.comparative_change) || null,
        });
        join.set(id, base);
      });

      const peripheralTargets = Array.isArray(targetsUpdate?.peripheral_targets) ? targetsUpdate.peripheral_targets : [];
      peripheralTargets.forEach((pt) => {
        if (!isRecord(pt)) return;
        const key = asTrimmedString(pt.target_key) || [pt.laterality, pt.lobe, pt.segment].map(asTrimmedString).filter(Boolean).join(" ");
        const id = normalizeTargetId(key);
        if (!id) return;
        const base = join.get(id) || { target: targetIndex.get(id) || { id, label: key || id }, pathology: null, imaging: null };
        base.imaging = chooseLatest(base.imaging, {
          created_at: ev.created_at,
          relative_day_offset: ev.relative_day_offset,
          source_event_id: ev.id || null,
          size_mm_long: Number.isFinite(Number(pt.size_mm_long)) ? Number(pt.size_mm_long) : null,
          size_mm_short: Number.isFinite(Number(pt.size_mm_short)) ? Number(pt.size_mm_short) : null,
          suvmax: Number.isFinite(Number(pt.pet_suvmax)) ? Number(pt.pet_suvmax) : null,
          change: asTrimmedString(pt.comparative_change) || null,
        });
        join.set(id, base);
      });
    }
  });

  const mediastinalRows = [];
  const peripheralRows = [];
  join.forEach((entry) => {
    const kind = String(entry?.target?.kind || "");
    if (kind.includes("mediastinal") || kind.includes("ebus")) mediastinalRows.push(entry);
    else peripheralRows.push(entry);
  });

  const sortByLabel = (a, b) => asTrimmedString(a?.target?.label).localeCompare(asTrimmedString(b?.target?.label));
  mediastinalRows.sort(sortByLabel);
  peripheralRows.sort(sortByLabel);

  return { mediastinal: mediastinalRows, peripheral: peripheralRows };
}

function formatTOffset(offset) {
  const n = Number(offset || 0);
  if (!Number.isFinite(n)) return "T?";
  if (n === 0) return "T0";
  return n > 0 ? `T+${Math.trunc(n)}` : `T${Math.trunc(n)}`;
}

function renderTimeline(eventList) {
  const events = Array.isArray(eventList) ? eventList.map(normalizeEventInput) : [];
  const sorted = events
    .slice()
    .sort((a, b) => (a.relative_day_offset || 0) - (b.relative_day_offset || 0) || String(a.created_at).localeCompare(String(b.created_at)));
  if (!sorted.length) return '<div class="dash-empty" style="padding: 8px 10px;">No events available.</div>';
  const items = sorted
    .map((ev) => {
      const label = `${formatTOffset(ev.relative_day_offset)} · ${ev.event_type || "event"}${ev.event_title ? ` · ${ev.event_title}` : ""}`;
      return `<li>${escapeHtml(label)}</li>`;
    })
    .join("");
  return `<ul style="margin:0; padding-left: 18px; color: rgba(15,23,42,0.78); font-size: 12px; line-height: 1.4;">${items}</ul>`;
}

function renderTargetTable(rows, kind) {
  const list = Array.isArray(rows) ? rows : [];
  if (!list.length) {
    return '<div class="dash-empty" style="padding: 8px 10px;">No targets.</div>';
  }

  const head = kind === "mediastinal"
    ? ["Target", "Imaging", "Pathology"]
    : ["Target", "Imaging", "Pathology"];

  const body = list
    .map((row) => {
      const t = row.target || {};
      const imaging = row.imaging || {};
      const pathology = row.pathology || {};

      const imagingBits = [];
      if (Number.isFinite(imaging.short_axis_mm)) imagingBits.push(`SA ${imaging.short_axis_mm} mm`);
      if (Number.isFinite(imaging.size_mm_long) || Number.isFinite(imaging.size_mm_short)) {
        const long = Number.isFinite(imaging.size_mm_long) ? imaging.size_mm_long : "—";
        const short = Number.isFinite(imaging.size_mm_short) ? imaging.size_mm_short : "—";
        imagingBits.push(`${long}×${short} mm`);
      }
      if (typeof imaging.pet_avid === "boolean") imagingBits.push(imaging.pet_avid ? "PET avid" : "PET not avid");
      if (Number.isFinite(imaging.suvmax)) imagingBits.push(`SUV ${imaging.suvmax}`);
      if (imaging.change) imagingBits.push(imaging.change);

      const pathBits = [];
      if (pathology.result) pathBits.push(pathology.result);
      if (pathology.diagnosis) pathBits.push(pathology.diagnosis);

      return `<tr>
        <td style="font-weight:600; color:#475569;">${escapeHtml(t.label || t.station || t.target_key || t.id || "—")}</td>
        <td>${escapeHtml(imagingBits.join(" · ") || "—")}</td>
        <td>${escapeHtml(pathBits.join(" · ") || "—")}</td>
      </tr>`;
    })
    .join("");

  const ths = head.map((h) => `<th>${escapeHtml(h)}</th>`).join("");
  return `<table class="dash-table striped" style="margin-top:6px;">
    <thead><tr>${ths}</tr></thead>
    <tbody>${body}</tbody>
  </table>`;
}

export function renderProcedureSummaryCard(registryData, joinedTables, eventList) {
  const registry = isRecord(registryData) ? registryData : {};
  const hasTargets =
    Array.isArray(registry?.targets?.mediastinal_targets) ||
    Array.isArray(registry?.targets?.peripheral_targets) ||
    Array.isArray(registry?.procedures_performed?.linear_ebus?.node_events);

  const joined = isRecord(joinedTables) ? joinedTables : { mediastinal: [], peripheral: [] };
  const mediastinal = Array.isArray(joined.mediastinal) ? joined.mediastinal : [];
  const peripheral = Array.isArray(joined.peripheral) ? joined.peripheral : [];

  const timelineHtml = renderTimeline(eventList);

  if (!hasTargets && (!Array.isArray(eventList) || eventList.length === 0)) {
    return '<div class="dash-empty" style="padding: 10px 12px;">No case targets or events available.</div>';
  }

  return `
    <div style="display:flex; flex-direction:column; gap: 10px;">
      <div>
        <div style="font-weight:700; color:#0f172a; margin-bottom: 4px;">Timeline</div>
        ${timelineHtml}
      </div>

      <div>
        <div style="font-weight:700; color:#0f172a; margin-bottom: 4px;">Mediastinal / LN Targets</div>
        ${renderTargetTable(mediastinal, "mediastinal")}
      </div>

      <div>
        <div style="font-weight:700; color:#0f172a; margin-bottom: 4px;">Peripheral / Lesion Targets</div>
        ${renderTargetTable(peripheral, "peripheral")}
      </div>
    </div>
  `;
}

