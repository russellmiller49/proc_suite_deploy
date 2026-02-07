// Minimal PHI demo client (synthetic data only). No logging of raw PHI.
//
// IMPORTANT: This page is intended to be served by the FastAPI app at /ui/phi_demo.html.
// If someone opens the HTML file directly (file://...), relative fetch() calls will fail.
// We defensively fall back to http://localhost:8000 in that case.

const API_BASE =
    (typeof window !== "undefined" &&
        window.location &&
        window.location.origin &&
        window.location.origin !== "null")
        ? window.location.origin
        : "http://localhost:8000";

async function fetchJson(url, options) {
    const resp = await fetch(url, options);
    if (resp.ok) return resp.json();
    const err = await resp.json().catch(() => ({}));
    const detail = err.detail || err.message || `${resp.status} ${resp.statusText}`.trim();
    throw new Error(detail || "Request failed");
}

const api = {
    async preview(text, document_type = null, specialty = null) {
        return fetchJson(`${API_BASE}/v1/phi/scrub/preview`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text, document_type, specialty }),
        });
    },
    async submit(text, submitted_by = "demo_physician", document_type = null, specialty = null, confirmed_entities = null) {
        const payload = { text, submitted_by, document_type, specialty };
        if (confirmed_entities) {
            payload.confirmed_entities = confirmed_entities;
        }
        return fetchJson(`${API_BASE}/v1/phi/submit`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    },
    async status(procedure_id) {
        return fetchJson(`${API_BASE}/v1/phi/status/${procedure_id}`);
    },
    async procedure(procedure_id) {
        return fetchJson(`${API_BASE}/v1/phi/procedure/${procedure_id}`);
    },
    async feedback(procedure_id, payload) {
        return fetchJson(`${API_BASE}/v1/phi/procedure/${procedure_id}/feedback`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    },
    async extract(scrubbed_text, include_financials = true) {
        return fetchJson(`${API_BASE}/api/v1/process`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                note: scrubbed_text,
                already_scrubbed: true,
                include_financials,
                explain: true,
            }),
        });
    },
    async reidentify(procedure_id) {
        return fetchJson(`${API_BASE}/v1/phi/reidentify`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ procedure_id, user_id: "phi_demo_user" }),
        });
    },
    async listCases() {
        return fetchJson(`${API_BASE}/api/v1/phi-demo/cases`);
    },
    async createCase(payload) {
        return fetchJson(`${API_BASE}/api/v1/phi-demo/cases`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
    },
    async attachProcedure(case_id, procedure_id) {
        return fetchJson(`${API_BASE}/api/v1/phi-demo/cases/${case_id}/procedure`, {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ procedure_id }),
        });
    },
};

const state = {
    preview: null,
    procedureId: null,
    scrubbedText: null,
    entities: [],
};

function $(id) {
    return document.getElementById(id);
}

function setStatus(text, cls = "bg-secondary") {
    const el = $("phi-status");
    el.textContent = text;
    el.className = `badge ${cls}`;
}

let entityEditModal;

function renderPreview(result) {
    const preview = $("scrubbed-preview");
    preview.textContent = result.scrubbed_text;
    preview.classList.remove("text-muted");
    const list = $("entity-list");
    list.innerHTML = "";
    (result.entities || []).forEach((ent, idx) => {
        const badge = document.createElement("span");
        badge.className = "badge bg-info text-dark entity-badge d-inline-flex align-items-center";
        
        const span = document.createElement("span");
        span.textContent = `${ent.entity_type} → ${ent.placeholder}`;
        badge.appendChild(span);

        // Add edit button (pencil)
        const editBtn = document.createElement("i");
        editBtn.className = "bi bi-pencil-square ms-2";
        editBtn.style.cursor = "pointer";
        editBtn.onclick = () => openEditModal(idx);
        badge.appendChild(editBtn);

        // Add close button (x)
        const closeBtn = document.createElement("i");
        closeBtn.className = "bi bi-x ms-2";
        closeBtn.style.cursor = "pointer";
        closeBtn.onclick = () => removeEntity(idx);
        badge.appendChild(closeBtn);

        list.appendChild(badge);
    });
    $("preview-meta").textContent = `${(result.entities || []).length} entities`;
}

function openEditModal(index) {
    const ent = state.entities[index];
    if (!ent) return;
    
    $("edit-entity-index").value = index;
    $("edit-entity-type").value = ent.entity_type;
    $("edit-entity-placeholder").value = ent.placeholder;
    
    if (!entityEditModal) {
        entityEditModal = new bootstrap.Modal($("entityEditModal"));
    }
    entityEditModal.show();
}

function saveEntity() {
    const index = parseInt($("edit-entity-index").value);
    const type = $("edit-entity-type").value;
    const placeholder = $("edit-entity-placeholder").value;
    
    if (state.entities[index]) {
        state.entities[index].entity_type = type;
        state.entities[index].placeholder = placeholder;
        
        // Re-generate preview text locally
        const rawText = $("note-input").value || "";
        const updatedScrubbedText = generateScrubbedText(rawText, state.entities);
        state.scrubbedText = updatedScrubbedText;
        
        // Re-render
        renderPreview({
            scrubbed_text: updatedScrubbedText,
            entities: state.entities
        });
    }
    
    if (entityEditModal) {
        entityEditModal.hide();
    }
}

function removeEntity(index) {
    // Remove entity at index
    state.entities.splice(index, 1);
    
    // Re-generate preview text locally
    const rawText = $("note-input").value || "";
    const updatedScrubbedText = generateScrubbedText(rawText, state.entities);
    
    state.scrubbedText = updatedScrubbedText;
    
    // Re-render
    renderPreview({
        scrubbed_text: updatedScrubbedText,
        entities: state.entities
    });
}

function generateScrubbedText(text, entities) {
    // Client-side implementation of scrub_with_manual_entities
    // 1. Sort entities by start desc
    const sorted = [...entities].sort((a, b) => b.original_start - a.original_start);
    
    let chars = text.split("");
    
    sorted.forEach(ent => {
        const start = ent.original_start;
        const end = ent.original_end;
        if (start < 0 || end > chars.length) return;
        
        const placeholder = ent.placeholder || `[${ent.entity_type}]`;
        // Replace
        chars.splice(start, end - start, ...placeholder.split(""));
    });
    
    return chars.join("");
}

function renderStatus(status, procedureId) {
    $("procedure-id").textContent = procedureId || "-";
    $("procedure-status").textContent = status || "-";

    // Update prompt based on status
    const prompt = $("extraction-prompt");
    if (status === "PHI_REVIEWED") {
        prompt.textContent = "PHI reviewed. Running extraction...";
        prompt.className = "small text-info";
    } else if (status === "PENDING_REVIEW") {
        prompt.textContent = "Click 'Approve & Extract' to run extraction.";
        prompt.className = "small text-warning";
    } else {
        prompt.textContent = "Submit and approve PHI review to run extraction.";
        prompt.className = "small text-muted";
    }
}

function renderExtractionResults(result) {
    const card = $("extraction-results-card");
    card.style.display = "block";

    // Processing time
    $("extraction-time").textContent = `${result.processing_time_ms}ms`;

    // CPT codes with descriptions
    const cptList = $("cpt-codes-list");
    cptList.innerHTML = "";
    (result.suggestions || []).forEach(s => {
        const badge = document.createElement("span");
        badge.className = "badge bg-primary";
        badge.title = s.description || "";
        badge.innerHTML = `${s.code}`;
        if (s.confidence) {
            const conf = document.createElement("span");
            conf.className = "ms-1 badge bg-light text-dark";
            conf.textContent = `${Math.round(s.confidence * 100)}%`;
            badge.appendChild(conf);
        }
        cptList.appendChild(badge);
    });

    if (result.cpt_codes && result.cpt_codes.length === 0) {
        cptList.innerHTML = '<span class="text-muted">No CPT codes derived</span>';
    }

    // RVU/payment
    const rvuBadge = $("cpt-rvu");
    if (result.total_work_rvu !== null && result.total_work_rvu !== undefined) {
        let rvuText = `${result.total_work_rvu} wRVU`;
        if (result.estimated_payment !== null && result.estimated_payment !== undefined) {
            rvuText += ` / $${result.estimated_payment.toFixed(2)}`;
        }
        rvuBadge.textContent = rvuText;
        rvuBadge.style.display = "inline";
    } else {
        rvuBadge.style.display = "none";
    }

    // Registry fields (formatted as key-value pairs)
    const registryDiv = $("registry-fields");
    registryDiv.innerHTML = "";
    if (result.registry && Object.keys(result.registry).length > 0) {
        const pre = document.createElement("pre");
        pre.className = "mb-0";
        pre.style.fontSize = "0.75rem";
        pre.textContent = JSON.stringify(result.registry, null, 2);
        registryDiv.appendChild(pre);
    } else {
        registryDiv.innerHTML = '<span class="text-muted">No registry fields extracted</span>';
    }

    // Audit warnings
    const warningsSection = $("audit-warnings-section");
    const warningsList = $("audit-warnings-list");
    warningsList.innerHTML = "";
    if (result.audit_warnings && result.audit_warnings.length > 0) {
        warningsSection.style.display = "block";
        result.audit_warnings.forEach(w => {
            const li = document.createElement("li");
            li.textContent = w;
            warningsList.appendChild(li);
        });
    } else {
        warningsSection.style.display = "none";
    }

    // Update prompt
    const prompt = $("extraction-prompt");
    if (result.needs_manual_review) {
        prompt.textContent = "Manual review recommended.";
        prompt.className = "small text-warning";
    } else {
        prompt.textContent = "Extraction complete.";
        prompt.className = "small text-success";
    }
}

function renderCases(cases) {
    const list = $("case-list");
    list.innerHTML = "";
    cases.forEach((c) => {
        const li = document.createElement("li");
        li.className = "list-group-item d-flex justify-content-between align-items-center";
        li.innerHTML = `<div>
            <div class="fw-semibold">${c.scenario_label || "Demo case"}</div>
            <div class="small text-muted">${c.synthetic_patient_label || ""} • ${c.procedure_date || ""} • ${c.operator_name || ""}</div>
            <div class="small">procedure_id: ${c.procedure_id || "-"}</div>
        </div>
        <button class="btn btn-sm btn-outline-primary">Load</button>`;
        li.querySelector("button").onclick = async () => {
            if (c.procedure_id) {
                state.procedureId = c.procedure_id;
                renderStatus("loading...", state.procedureId);
                try {
                    const proc = await api.procedure(c.procedure_id);
                    state.scrubbedText = proc.scrubbed_text;
                    state.entities = proc.entities;
                    renderPreview(proc);
                    renderStatus(proc.status, c.procedure_id);
                    $("btn-refresh-status").disabled = false;
                    $("btn-mark-reviewed").disabled = false;
                    $("btn-reidentify").disabled = false;
                } catch (err) {
                    setStatus("Case load failed", "bg-danger");
                }
            }
        };
        list.appendChild(li);
    });
}

async function handlePreview() {
    const text = $("note-input").value || "";
    if (!text.trim()) {
        setStatus("Enter synthetic note text", "bg-warning text-dark");
        return;
    }
    setStatus("Previewing...", "bg-info text-dark");
    try {
        const res = await api.preview(text);
        state.preview = res;
        state.scrubbedText = res.scrubbed_text;
        state.entities = res.entities || [];
        renderPreview(res);
        $("btn-submit").disabled = false;
        setStatus("Preview complete", "bg-success");
    } catch (err) {
        console.error("PHI preview failed:", err);
        setStatus(`Preview failed`, "bg-danger");
    }
}

async function handleSubmit() {
    const text = $("note-input").value || "";
    if (!state.preview) {
        setStatus("Run preview first", "bg-warning text-dark");
        return;
    }
    setStatus("Submitting...", "bg-info text-dark");
    try {
        // Pass current state.entities as confirmed_entities to support manual overrides
        const res = await api.submit(text, "demo_physician", null, null, state.entities);
        state.procedureId = res.procedure_id;
        renderStatus(res.status, res.procedure_id);
        $("btn-refresh-status").disabled = false;
        $("btn-mark-reviewed").disabled = false;
        $("btn-reidentify").disabled = true; // only after review
        // Attach procedure_id to latest created case if any
        const cases = await api.listCases();
        if (cases.length > 0) {
            const latest = cases[cases.length - 1];
            if (!latest.procedure_id) {
                await api.attachProcedure(latest.id, res.procedure_id);
                loadCases();
            }
        }
        setStatus("Submitted", "bg-success");
    } catch (err) {
        setStatus("Submit failed", "bg-danger");
    }
}

async function handleStatus() {
    if (!state.procedureId) return;
    try {
        const res = await api.status(state.procedureId);
        renderStatus(res.status, res.procedure_id || state.procedureId);
        setStatus("Status refreshed", "bg-secondary");
    } catch (err) {
        setStatus("Status error", "bg-danger");
    }
}

async function handleReview() {
    if (!state.procedureId || !state.scrubbedText) return;
    setStatus("Approving PHI review...", "bg-info text-dark");
    try {
        // Step 1: Mark as reviewed
        const res = await api.feedback(state.procedureId, {
            scrubbed_text: state.scrubbedText,
            entities: state.entities || [],
            reviewer_id: "reviewer_demo",
            reviewer_email: "reviewer@example.com",
            reviewer_role: "physician",
            comment: "Auto-confirmed in demo",
        });
        renderStatus(res.status, state.procedureId);
        $("btn-reidentify").disabled = false;
        setStatus("PHI Approved - Extracting...", "bg-info text-dark");

        // Step 2: Run extraction
        try {
            const extraction = await api.extract(state.scrubbedText, true);
            renderExtractionResults(extraction);
            setStatus("Extraction Complete", "bg-success");
        } catch (extractErr) {
            console.error("Extraction failed:", extractErr);
            setStatus("Reviewed (extraction failed)", "bg-warning text-dark");
            $("extraction-prompt").textContent = `Extraction failed: ${extractErr.message}`;
            $("extraction-prompt").className = "small text-danger";
        }
    } catch (err) {
        setStatus("Review failed", "bg-danger");
    }
}

async function handleReidentify() {
    if (!state.procedureId) return;
    setStatus("Reidentifying...", "bg-warning text-dark");
    try {
        const res = await api.reidentify(state.procedureId);
        $("reid-text").value = res.raw_text || "";
        setStatus("Reidentified", "bg-success");
    } catch (err) {
        setStatus("Reidentify failed", "bg-danger");
    }
}

async function loadCases() {
    try {
        const cases = await api.listCases();
        renderCases(cases);
    } catch (err) {
        setStatus("Case list failed", "bg-danger");
    }
}

async function createCase() {
    const scenarios = [
        "EBUS for RUL nodule",
        "Pleural effusion thoracentesis",
        "Bronchoscopy BAL follow-up",
    ];
    const scenario = scenarios[Math.floor(Math.random() * scenarios.length)];
    try {
        await api.createCase({
            synthetic_patient_label: "Patient X",
            procedure_date: dayjs().format("YYYY-MM-DD"),
            operator_name: "Dr. Jane Test",
            scenario_label: scenario,
        });
        loadCases();
        setStatus("Case created", "bg-success");
    } catch (err) {
        setStatus("Case create failed", "bg-danger");
    }
}

function init() {
    $("btn-preview").onclick = handlePreview;
    $("btn-submit").onclick = handleSubmit;
    $("btn-refresh-status").onclick = handleStatus;
    $("btn-mark-reviewed").onclick = handleReview;
    $("btn-reidentify").onclick = handleReidentify;
    $("btn-create-case").onclick = createCase;
    $("btn-save-entity").onclick = saveEntity;
    loadCases();
}

document.addEventListener("DOMContentLoaded", init);
