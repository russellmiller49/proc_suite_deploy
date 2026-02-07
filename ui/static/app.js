// Global state
let currentMode = 'unified';
let lastResult = null;

// PHI preview state (for unified mode two-step workflow)
const phiState = {
    rawText: null,           // Original text before scrubbing
    scrubbedText: null,      // Scrubbed text after preview
    entities: [],            // Entity array from preview (editable)
    previewDone: false,      // Whether preview step completed
};

const reporterBuilderState = {
    bundle: null,
    questions: [],
    strict: false,
};

/**
 * Convert snake_case to Title Case for display
 */
function toTitleCase(str) {
    return str.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

/**
 * Format a boolean value as a badge
 */
function formatBool(val) {
    if (val === null || val === undefined) return '';
    return val ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>';
}

/**
 * Extract non-null fields from an object and format as readable list
 */
function formatObjectFields(obj, fieldLabels = {}) {
    if (!obj || typeof obj !== 'object') return '<span class="text-muted">—</span>';

    const parts = [];
    for (const [key, value] of Object.entries(obj)) {
        if (value === null || value === undefined) continue;

        const label = fieldLabels[key] || toTitleCase(key);

        if (typeof value === 'boolean') {
            if (value) parts.push(`<strong>${label}</strong>`);
        } else if (Array.isArray(value)) {
            if (value.length > 0) {
                parts.push(`<strong>${label}:</strong> ${value.join(', ')}`);
            }
        } else if (typeof value === 'object') {
            // Nested object - recursively format
            const nested = formatObjectFields(value);
            if (nested !== '<span class="text-muted">—</span>') {
                parts.push(`<strong>${label}:</strong> ${nested}`);
            }
        } else {
            parts.push(`<strong>${label}:</strong> ${value}`);
        }
    }

    if (parts.length === 0) return '<span class="text-muted">—</span>';
    return parts.join(' · ');
}

/**
 * Format providers object for human-readable display
 */
function formatProviders(providers) {
    if (!providers) return '<span class="text-muted">—</span>';

    const parts = [];
    if (providers.attending_name) {
        let attending = `<strong>Attending:</strong> ${providers.attending_name}`;
        if (providers.attending_npi) attending += ` (NPI: ${providers.attending_npi})`;
        parts.push(attending);
    }
    if (providers.fellow_name) {
        let fellow = `<strong>Fellow:</strong> ${providers.fellow_name}`;
        if (providers.fellow_pgy_level) fellow += ` (PGY-${providers.fellow_pgy_level})`;
        parts.push(fellow);
    }
    if (providers.assistant_name) {
        let assistant = `<strong>Assistant:</strong> ${providers.assistant_name}`;
        if (providers.assistant_role) assistant += ` (${providers.assistant_role})`;
        parts.push(assistant);
    }
    if (providers.trainee_present) parts.push('Trainee present');
    if (providers.rose_present) parts.push('ROSE present');

    if (parts.length === 0) return '<span class="text-muted">—</span>';
    return parts.join('<br>');
}

/**
 * Format clinical_context object for human-readable display
 */
function formatClinicalContext(ctx) {
    if (!ctx) return '<span class="text-muted">—</span>';

    const parts = [];
    if (ctx.asa_class) parts.push(`<strong>ASA:</strong> ${ctx.asa_class}`);
    if (ctx.primary_indication) parts.push(`<strong>Indication:</strong> ${ctx.primary_indication}`);
    if (ctx.indication_category) parts.push(`<strong>Category:</strong> ${ctx.indication_category}`);
    if (ctx.lesion_location) parts.push(`<strong>Location:</strong> ${ctx.lesion_location}`);
    if (ctx.lesion_size_mm) parts.push(`<strong>Size:</strong> ${ctx.lesion_size_mm} mm`);
    if (ctx.radiographic_findings) parts.push(`<strong>Imaging:</strong> ${ctx.radiographic_findings}`);
    if (ctx.pet_avidity) parts.push(`<strong>PET:</strong> ${ctx.pet_avidity}`);
    if (ctx.suv_max) parts.push(`<strong>SUV max:</strong> ${ctx.suv_max}`);
    if (ctx.bronchus_sign !== null && ctx.bronchus_sign !== undefined) {
        parts.push(`<strong>Bronchus sign:</strong> ${ctx.bronchus_sign ? 'Yes' : 'No'}`);
    }

    if (parts.length === 0) return '<span class="text-muted">—</span>';
    return parts.join(' · ');
}

/**
 * Format sedation object for human-readable display
 */
function formatSedation(sed) {
    if (!sed) return '<span class="text-muted">—</span>';

    const parts = [];
    if (sed.type) parts.push(`<strong>Type:</strong> ${sed.type}`);
    if (sed.anesthesia_provider) parts.push(`<strong>Provider:</strong> ${sed.anesthesia_provider}`);
    if (sed.agents_used && sed.agents_used.length) parts.push(`<strong>Agents:</strong> ${sed.agents_used.join(', ')}`);
    if (sed.paralytic_used) parts.push('Paralytic used');
    if (sed.reversal_given) {
        let rev = 'Reversal given';
        if (sed.reversal_agent) rev += ` (${sed.reversal_agent})`;
        parts.push(rev);
    }
    if (sed.intraservice_minutes) parts.push(`<strong>Duration:</strong> ${sed.intraservice_minutes} min`);

    if (parts.length === 0) return '<span class="text-muted">—</span>';
    return parts.join(' · ');
}

/**
 * Format equipment object for human-readable display
 */
function formatEquipment(eq) {
    if (!eq) return '<span class="text-muted">—</span>';

    const parts = [];
    if (eq.bronchoscope_type) parts.push(`<strong>Scope:</strong> ${eq.bronchoscope_type}`);
    if (eq.bronchoscope_model) parts.push(`(${eq.bronchoscope_model})`);
    if (eq.bronchoscope_outer_diameter_mm) parts.push(`${eq.bronchoscope_outer_diameter_mm} mm OD`);
    if (eq.fluoroscopy_used) {
        let fluoro = 'Fluoroscopy';
        if (eq.fluoroscopy_time_seconds) fluoro += ` (${eq.fluoroscopy_time_seconds}s)`;
        parts.push(fluoro);
    }
    if (eq.navigation_platform) parts.push(`<strong>Navigation:</strong> ${eq.navigation_platform}`);
    if (eq.cbct_used) parts.push('CBCT');
    if (eq.augmented_fluoroscopy) parts.push('Augmented fluoroscopy');

    if (parts.length === 0) return '<span class="text-muted">—</span>';
    return parts.join(' · ');
}

/**
 * Format a single procedure sub-object (e.g., airway_stent, transbronchial_biopsy)
 */
function formatProcedureDetail(proc, procName) {
    if (!proc || typeof proc !== 'object') return null;
    if (proc.performed === false) return null;

    const parts = [];

    // Common fields
    if (proc.locations && proc.locations.length) parts.push(`locations: ${proc.locations.join(', ')}`);
    if (proc.location) parts.push(`location: ${proc.location}`);
    if (proc.number_of_samples) parts.push(`${proc.number_of_samples} samples`);
    if (proc.stent_type) parts.push(`type: ${proc.stent_type}`);
    if (proc.stent_brand) parts.push(`brand: ${proc.stent_brand}`);
    if (proc.diameter_mm) parts.push(`${proc.diameter_mm} mm`);
    if (proc.length_mm) parts.push(`${proc.length_mm} mm length`);
    if (proc.forceps_type) parts.push(`forceps: ${proc.forceps_type}`);
    if (proc.cryoprobe_size_mm) parts.push(`cryo: ${proc.cryoprobe_size_mm} mm`);
    if (proc.action) parts.push(`action: ${proc.action}`);
    if (proc.indication) parts.push(`indication: ${proc.indication}`);
    if (proc.deployment_successful !== null && proc.deployment_successful !== undefined) {
        parts.push(proc.deployment_successful ? 'successful' : 'unsuccessful');
    }

    const detail = parts.length > 0 ? ` (${parts.join(', ')})` : '';
    return `<strong>${toTitleCase(procName)}</strong>${detail}`;
}

/**
 * Format procedures_performed object for human-readable display
 */
function formatProceduresPerformed(procs) {
    if (!procs) return '<span class="text-muted">—</span>';

    const items = [];

    // Simple boolean procedures
    const simpleProcs = [
        'diagnostic_bronchoscopy', 'bal', 'bronchial_wash', 'brushings',
        'endobronchial_biopsy', 'tbna_conventional', 'peripheral_tbna', 'linear_ebus', 'radial_ebus',
        'navigational_bronchoscopy', 'therapeutic_aspiration', 'foreign_body_removal',
        'cryotherapy', 'photodynamic_therapy', 'brachytherapy_catheter',
        'bronchial_thermoplasty', 'whole_lung_lavage', 'rigid_bronchoscopy'
    ];

    for (const proc of simpleProcs) {
        if (procs[proc] === true) {
            items.push(`<strong>${toTitleCase(proc)}</strong>`);
        }
    }

    // Complex procedures with sub-objects
    const complexProcs = [
        'transbronchial_biopsy', 'transbronchial_cryobiopsy', 'airway_dilation',
        'airway_stent', 'thermal_ablation', 'mechanical_debulking', 'blvr', 'peripheral_ablation'
    ];

    for (const proc of complexProcs) {
        if (procs[proc] && typeof procs[proc] === 'object') {
            const formatted = formatProcedureDetail(procs[proc], proc);
            if (formatted) items.push(formatted);
        }
    }

    if (items.length === 0) return '<span class="text-muted">—</span>';
    return '<ul class="list-unstyled mb-0" style="font-size: 0.9em;">' +
           items.map(i => `<li>${i}</li>`).join('') + '</ul>';
}

/**
 * Format complications object for human-readable display
 */
function formatComplications(comp) {
    if (!comp) return '<span class="text-muted">—</span>';

    if (comp.any_complication === false) {
        return '<span class="badge bg-success">None</span>';
    }

    const parts = [];

    if (comp.complication_list && comp.complication_list.length) {
        parts.push(`<strong>Complications:</strong> ${comp.complication_list.join(', ')}`);
    }

    if (comp.bleeding) {
        let bleed = 'Bleeding';
        if (typeof comp.bleeding === 'object') {
            if (comp.bleeding.severity) bleed += ` (${comp.bleeding.severity})`;
        }
        parts.push(bleed);
    }

    if (comp.pneumothorax) {
        let ptx = 'Pneumothorax';
        if (typeof comp.pneumothorax === 'object') {
            if (comp.pneumothorax.chest_tube_required) ptx += ' (chest tube)';
        }
        parts.push(ptx);
    }

    if (comp.respiratory) {
        const resp = comp.respiratory;
        const respParts = [];
        if (resp.hypoxia_occurred) {
            let hyp = 'Hypoxia';
            if (resp.lowest_spo2) hyp += ` (SpO2 ${resp.lowest_spo2}%)`;
            respParts.push(hyp);
        }
        if (resp.intubation_required) respParts.push('Intubation required');
        if (resp.respiratory_failure) respParts.push('Respiratory failure');
        if (respParts.length) parts.push(respParts.join(', '));
    }

    if (comp.other_complication_details) {
        parts.push(`Other: ${comp.other_complication_details}`);
    }

    if (parts.length === 0 && comp.any_complication) {
        return '<span class="badge bg-warning text-dark">Yes (details unknown)</span>';
    }

    if (parts.length === 0) return '<span class="text-muted">—</span>';
    return parts.join('<br>');
}

/**
 * Format outcomes object for human-readable display
 */
function formatOutcomes(out) {
    if (!out) return '<span class="text-muted">—</span>';

    const parts = [];

    if (out.procedure_completed !== null && out.procedure_completed !== undefined) {
        parts.push(out.procedure_completed ?
            '<span class="badge bg-success">Completed</span>' :
            '<span class="badge bg-warning text-dark">Incomplete</span>');
    }
    if (out.procedure_aborted_reason) parts.push(`Aborted: ${out.procedure_aborted_reason}`);
    if (out.preliminary_diagnosis) parts.push(`<strong>Dx:</strong> ${out.preliminary_diagnosis}`);
    if (out.preliminary_staging) parts.push(`<strong>Stage:</strong> ${out.preliminary_staging}`);
    if (out.disposition) parts.push(`<strong>Disposition:</strong> ${out.disposition}`);
    if (out.follow_up_plan_text) parts.push(`<strong>Follow-up:</strong> ${out.follow_up_plan_text}`);
    if (out.follow_up_actions && out.follow_up_actions.length) {
        parts.push(`<strong>Actions:</strong> ${out.follow_up_actions.join(', ')}`);
    }

    if (parts.length === 0) return '<span class="text-muted">—</span>';
    return parts.join(' · ');
}

/**
 * Format registry values for display, handling complex types like arrays and objects.
 * Special handling for EBUS station details and other structured fields.
 */
function formatRegistryValue(key, value) {
    // Handle null/undefined
    if (value === null || value === undefined) {
        return '<span class="text-muted">—</span>';
    }

    // Handle arrays
    if (Array.isArray(value)) {
        // Empty array
        if (value.length === 0) {
            return '<span class="text-muted">—</span>';
        }

        // Special handling for ebus_stations_detail - MUST come first before primitive check
        if (key === 'ebus_stations_detail') {
            return formatEbusStationDetails(value);
        }

        // Check if array contains only primitives (strings, numbers, booleans, null)
        const allPrimitives = value.every(item => {
            const t = typeof item;
            return item === null || t === 'string' || t === 'number' || t === 'boolean';
        });

        if (allPrimitives) {
            return value.join(', ');
        }

        // Array of objects - format as expandable list
        return '<ul class="list-unstyled mb-0 small">' +
               value.map(v => `<li>${formatObjectFields(v)}</li>`).join('') + '</ul>';
    }

    // Handle objects with specialized formatters based on key
    if (typeof value === 'object' && value !== null) {
        switch (key) {
            case 'providers':
                return formatProviders(value);
            case 'clinical_context':
                return formatClinicalContext(value);
            case 'sedation':
                return formatSedation(value);
            case 'equipment':
                return formatEquipment(value);
            case 'procedures_performed':
                return formatProceduresPerformed(value);
            case 'complications':
                return formatComplications(value);
            case 'outcomes':
                return formatOutcomes(value);
            case 'granular_data':
                return formatGranularData(value);
            case 'patient_demographics':
            case 'procedure_setting':
            case 'pleural_procedures':
            case 'specimens':
            case 'pathology_results':
            case 'billing':
            case 'metadata':
                return formatObjectFields(value);
            default:
                // Unknown object - use generic formatter
                return formatObjectFields(value);
        }
    }

    // Handle booleans
    if (typeof value === 'boolean') {
        return value ? '<span class="badge bg-success">Yes</span>' : '<span class="badge bg-secondary">No</span>';
    }

    // Default: return as-is (strings, numbers)
    return String(value);
}

/**
 * Format EBUS station details array into a readable list format
 * Example output:
 *   - 11L: size 5.4 mm, ROSE: Nondiagnostic
 *   - 4R: size 5.5 mm, ROSE: Benign
 */
function formatEbusStationDetails(stations) {
    if (!stations || stations.length === 0) {
        return '<span class="text-muted">[]</span>';
    }

    let html = '<ul class="list-unstyled mb-0" style="font-size: 0.9em;">';

    stations.forEach(s => {
        const station = s.station || '?';
        const parts = [];

        // Size
        if (s.size_mm !== null && s.size_mm !== undefined) {
            parts.push(`size ${s.size_mm} mm`);
        }

        // Passes next for quick per-station review
        if (s.passes !== null && s.passes !== undefined) {
            parts.push(`${s.passes} passes`);
        }

        // ROSE result
        if (s.rose_result) {
            parts.push(`ROSE: ${s.rose_result}`);
        }

        // Morphology in consistent order
        if (s.shape) parts.push(`shape: ${s.shape}`);
        if (s.margin) parts.push(`margin: ${s.margin}`);
        if (s.echogenicity) parts.push(`echo: ${s.echogenicity}`);
        if (s.chs_present !== null && s.chs_present !== undefined) {
            parts.push(`CHS: ${s.chs_present ? "present" : "absent"}`);
        }

        const details = parts.length > 0 ? parts.join('; ') : 'no details';
        html += `<li><strong>${station}</strong>: ${details}</li>`;
    });

    html += '</ul>';
    return html;
}

/**
 * Derive a summary ROSE result from per-station details
 * Returns "Mixed (11L: Nondiagnostic, 4R: Benign)" if results differ
 */
function deriveRoseSummary(stations, globalRose) {
    if (!stations || stations.length === 0) {
        return globalRose || '<span class="text-muted">null</span>';
    }

    const stationRose = stations
        .filter(s => s.rose_result)
        .map(s => ({ station: s.station, rose: s.rose_result }));

    if (stationRose.length === 0) {
        return globalRose || '<span class="text-muted">null</span>';
    }

    // Check if all ROSE results are the same
    const uniqueRose = [...new Set(stationRose.map(s => s.rose))];

    if (uniqueRose.length === 1) {
        // All the same
        return uniqueRose[0];
    }

    // Mixed results - show each station's result
    const summary = stationRose.map(s => `${s.station}: ${s.rose}`).join(', ');
    return `<span class="badge bg-warning text-dark">Mixed</span> (${summary})`;
}

// ============================================================================
// GRANULAR DATA FORMATTERS
// ============================================================================

/**
 * Format the granular_data container with all per-site arrays
 */
function formatGranularData(data) {
    if (!data || typeof data !== 'object') {
        return '<span class="text-muted">—</span>';
    }

    const sections = [];

    // EBUS Stations Detail
    if (data.linear_ebus_stations_detail && data.linear_ebus_stations_detail.length > 0) {
        sections.push({
            title: '🔬 EBUS Stations',
            icon: 'bi-bullseye',
            color: 'primary',
            content: formatGranularEbusStations(data.linear_ebus_stations_detail)
        });
    }

    // Navigation Targets
    if (data.navigation_targets && data.navigation_targets.length > 0) {
        sections.push({
            title: '🎯 Navigation Targets',
            icon: 'bi-geo-alt',
            color: 'success',
            content: formatGranularNavTargets(data.navigation_targets)
        });
    }

    // CAO Interventions
    if (data.cao_interventions_detail && data.cao_interventions_detail.length > 0) {
        sections.push({
            title: '⚡ CAO Interventions',
            icon: 'bi-lightning',
            color: 'danger',
            content: formatGranularCAO(data.cao_interventions_detail)
        });
    }

    // BLVR Valves
    if (data.blvr_valve_placements && data.blvr_valve_placements.length > 0) {
        sections.push({
            title: '🫁 BLVR Valves',
            icon: 'bi-plug',
            color: 'info',
            content: formatGranularBLVRValves(data.blvr_valve_placements)
        });
    }

    // Chartis Measurements
    if (data.blvr_chartis_measurements && data.blvr_chartis_measurements.length > 0) {
        sections.push({
            title: '📊 Chartis Measurements',
            icon: 'bi-graph-up',
            color: 'info',
            content: formatGranularChartis(data.blvr_chartis_measurements)
        });
    }

    // Cryobiopsy Sites
    if (data.cryobiopsy_sites && data.cryobiopsy_sites.length > 0) {
        sections.push({
            title: '❄️ Cryobiopsy Sites',
            icon: 'bi-snow',
            color: 'primary',
            content: formatGranularCryobiopsy(data.cryobiopsy_sites)
        });
    }

    // Thoracoscopy Findings
    if (data.thoracoscopy_findings_detail && data.thoracoscopy_findings_detail.length > 0) {
        sections.push({
            title: '👁️ Thoracoscopy Findings',
            icon: 'bi-eye',
            color: 'warning',
            content: formatGranularThoracoscopy(data.thoracoscopy_findings_detail)
        });
    }

    // Specimens Collected
    if (data.specimens_collected && data.specimens_collected.length > 0) {
        sections.push({
            title: '🧪 Specimens Collected',
            icon: 'bi-cup',
            color: 'secondary',
            content: formatGranularSpecimens(data.specimens_collected)
        });
    }

    if (sections.length === 0) {
        return '<span class="text-muted">No granular data</span>';
    }

    // Build accordion-style display
    let html = '<div class="granular-data-container">';
    sections.forEach((section, idx) => {
        html += `
            <div class="card mb-2 border-${section.color}">
                <div class="card-header py-1 px-2 bg-${section.color} bg-opacity-10">
                    <strong class="text-${section.color}">${section.title}</strong>
                </div>
                <div class="card-body py-2 px-2" style="font-size: 0.85em;">
                    ${section.content}
                </div>
            </div>
        `;
    });
    html += '</div>';

    return html;
}

/**
 * Format granular EBUS station details
 */
function formatGranularEbusStations(stations) {
    if (!stations || stations.length === 0) return '<span class="text-muted">—</span>';

    let html = '<div class="table-responsive"><table class="table table-sm table-bordered mb-0" style="font-size: 0.9em;">';
    html += `<thead class="table-light">
        <tr>
            <th>Station</th>
            <th>Size</th>
            <th>Morphology</th>
            <th>Impression</th>
            <th>Passes</th>
            <th>ROSE</th>
        </tr>
    </thead><tbody>`;

    stations.forEach(s => {
        const station = s.station || '?';
        const size = s.short_axis_mm ? `${s.short_axis_mm} mm` : '—';

        // Build morphology summary
        const morphParts = [];
        if (s.shape) morphParts.push(s.shape);
        if (s.echogenicity) morphParts.push(s.echogenicity);
        if (s.chs_present !== null && s.chs_present !== undefined) {
            morphParts.push(s.chs_present ? 'CHS+' : 'CHS−');
        }
        if (s.necrosis_present) morphParts.push('necrosis');
        const morphology = morphParts.length > 0 ? morphParts.join(', ') : '—';

        // Impression badge
        let impressionBadge = '—';
        if (s.morphologic_impression) {
            const colors = {
                'benign': 'success',
                'suspicious': 'warning',
                'malignant': 'danger',
                'indeterminate': 'secondary'
            };
            const color = colors[s.morphologic_impression] || 'secondary';
            impressionBadge = `<span class="badge bg-${color}">${s.morphologic_impression}</span>`;
        }

        const passes = s.number_of_passes || '—';

        // ROSE result
        let roseBadge = '—';
        if (s.rose_result) {
            const roseColors = {
                'Malignant': 'danger',
                'Suspicious for malignancy': 'warning',
                'Adequate lymphocytes': 'success',
                'Granuloma': 'info',
                'Nondiagnostic': 'secondary'
            };
            const rColor = roseColors[s.rose_result] || 'light';
            roseBadge = `<span class="badge bg-${rColor}">${s.rose_result}</span>`;
        }

        html += `<tr>
            <td><strong>${station}</strong>${s.sampled === false ? ' <small class="text-muted">(not sampled)</small>' : ''}</td>
            <td>${size}</td>
            <td>${morphology}</td>
            <td>${impressionBadge}</td>
            <td>${passes}</td>
            <td>${roseBadge}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    return html;
}

/**
 * Format granular navigation targets
 */
function formatGranularNavTargets(targets) {
    if (!targets || targets.length === 0) return '<span class="text-muted">—</span>';

    let html = '<div class="table-responsive"><table class="table table-sm table-bordered mb-0" style="font-size: 0.9em;">';
    html += `<thead class="table-light">
        <tr>
            <th>#</th>
            <th>Location</th>
            <th>Size</th>
            <th>rEBUS</th>
            <th>TIL</th>
            <th>Samples</th>
            <th>ROSE</th>
        </tr>
    </thead><tbody>`;

    targets.forEach(t => {
        const num = t.target_number || '?';
        const loc = t.target_location_text || t.target_lobe || '—';
        const size = t.lesion_size_mm ? `${t.lesion_size_mm} mm` : '—';

        // rEBUS view
        let rebusView = '—';
        if (t.rebus_used && t.rebus_view) {
            const viewColors = {
                'Concentric': 'success',
                'Eccentric': 'warning',
                'Adjacent': 'info',
                'Not visualized': 'secondary'
            };
            rebusView = `<span class="badge bg-${viewColors[t.rebus_view] || 'light'}">${t.rebus_view}</span>`;
        } else if (t.rebus_used === false) {
            rebusView = '<span class="text-muted">not used</span>';
        }

        // TIL confirmation
        let tilBadge = '—';
        if (t.tool_in_lesion_confirmed !== null && t.tool_in_lesion_confirmed !== undefined) {
            if (t.tool_in_lesion_confirmed) {
                const method = t.confirmation_method ? ` (${t.confirmation_method})` : '';
                tilBadge = `<span class="badge bg-success">✓ TIL${method}</span>`;
            } else {
                tilBadge = `<span class="badge bg-warning text-dark">✗ No TIL</span>`;
            }
        }

        // Sampling tools
        const samples = [];
        if (t.number_of_forceps_biopsies) samples.push(`${t.number_of_forceps_biopsies} forceps`);
        if (t.number_of_needle_passes) samples.push(`${t.number_of_needle_passes} needle`);
        if (t.number_of_cryo_biopsies) samples.push(`${t.number_of_cryo_biopsies} cryo`);
        const sampleStr = samples.length > 0 ? samples.join(', ') : '—';

        // ROSE
        const rose = t.rose_result || '—';

        html += `<tr>
            <td><strong>${num}</strong></td>
            <td>${loc}</td>
            <td>${size}</td>
            <td>${rebusView}</td>
            <td>${tilBadge}</td>
            <td>${sampleStr}</td>
            <td>${rose}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    return html;
}

/**
 * Format granular CAO interventions
 */
function formatGranularCAO(interventions) {
    if (!interventions || interventions.length === 0) return '<span class="text-muted">—</span>';

    let html = '<ul class="list-unstyled mb-0">';

    interventions.forEach(i => {
        const location = i.location || '?';
        const parts = [];

        if (i.obstruction_type) parts.push(i.obstruction_type);
        if (i.etiology) parts.push(i.etiology);

        // Obstruction change
        if (i.pre_obstruction_pct !== null && i.post_obstruction_pct !== null) {
            const improvement = i.pre_obstruction_pct - i.post_obstruction_pct;
            parts.push(`<span class="badge bg-success">${i.pre_obstruction_pct}% → ${i.post_obstruction_pct}%</span> (−${improvement}%)`);
        }

        // Modalities
        if (i.modalities_applied && i.modalities_applied.length > 0) {
            const mods = i.modalities_applied.map(m => {
                let modStr = m.modality;
                if (m.power_setting_watts) modStr += ` ${m.power_setting_watts}W`;
                if (m.number_of_applications) modStr += ` ×${m.number_of_applications}`;
                return modStr;
            }).join(', ');
            parts.push(`<strong>Modalities:</strong> ${mods}`);
        }

        // Hemostasis
        if (i.hemostasis_required) {
            const methods = i.hemostasis_methods ? i.hemostasis_methods.join(', ') : 'methods unknown';
            parts.push(`<span class="badge bg-warning text-dark">Hemostasis: ${methods}</span>`);
        }

        // Stent
        if (i.stent_placed_at_site) {
            parts.push('<span class="badge bg-info">Stent placed</span>');
        }

        const details = parts.length > 0 ? '<br>' + parts.join('<br>') : '';
        html += `<li class="mb-2"><strong class="text-danger">${location}</strong>${details}</li>`;
    });

    html += '</ul>';
    return html;
}

/**
 * Format granular BLVR valve placements
 */
function formatGranularBLVRValves(valves) {
    if (!valves || valves.length === 0) return '<span class="text-muted">—</span>';

    let html = '<div class="table-responsive"><table class="table table-sm table-bordered mb-0" style="font-size: 0.9em;">';
    html += `<thead class="table-light">
        <tr>
            <th>#</th>
            <th>Lobe</th>
            <th>Segment</th>
            <th>Size</th>
            <th>Type</th>
            <th>Status</th>
        </tr>
    </thead><tbody>`;

    valves.forEach(v => {
        const num = v.valve_number || '?';
        const lobe = v.target_lobe || '—';
        const segment = v.segment || '—';
        const size = v.valve_size || '—';
        const type = v.valve_type ? v.valve_type.replace(/\s*\([^)]*\)/, '') : '—'; // Remove brand in parens

        let status = '—';
        if (v.deployment_successful === true) {
            status = '<span class="badge bg-success">✓ Deployed</span>';
            if (v.seal_confirmed) status += ' <small class="text-success">sealed</small>';
            if (v.repositioned) status += ' <small class="text-warning">(repositioned)</small>';
        } else if (v.deployment_successful === false) {
            status = '<span class="badge bg-danger">✗ Failed</span>';
        }

        html += `<tr>
            <td>${num}</td>
            <td><strong>${lobe}</strong></td>
            <td>${segment}</td>
            <td>${size}</td>
            <td>${type}</td>
            <td>${status}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    return html;
}

/**
 * Format granular Chartis measurements
 */
function formatGranularChartis(measurements) {
    if (!measurements || measurements.length === 0) return '<span class="text-muted">—</span>';

    let html = '<ul class="list-unstyled mb-0">';

    measurements.forEach(m => {
        const lobe = m.lobe_assessed || '?';
        const segment = m.segment_assessed ? ` (${m.segment_assessed})` : '';

        let resultBadge = '—';
        if (m.cv_result) {
            const colors = {
                'CV Negative': 'success',
                'CV Positive': 'danger',
                'Indeterminate': 'warning',
                'Low flow': 'warning',
                'No seal': 'secondary',
                'Aborted': 'secondary'
            };
            resultBadge = `<span class="badge bg-${colors[m.cv_result] || 'light'}">${m.cv_result}</span>`;
        }

        const duration = m.measurement_duration_seconds ? ` · ${m.measurement_duration_seconds}s` : '';
        const seal = m.adequate_seal === true ? ' · seal ✓' : (m.adequate_seal === false ? ' · no seal' : '');

        html += `<li><strong>${lobe}</strong>${segment}: ${resultBadge}${duration}${seal}</li>`;
    });

    html += '</ul>';
    return html;
}

/**
 * Format granular cryobiopsy sites
 */
function formatGranularCryobiopsy(sites) {
    if (!sites || sites.length === 0) return '<span class="text-muted">—</span>';

    let html = '<div class="table-responsive"><table class="table table-sm table-bordered mb-0" style="font-size: 0.9em;">';
    html += `<thead class="table-light">
        <tr>
            <th>#</th>
            <th>Location</th>
            <th>Probe</th>
            <th>Freeze</th>
            <th>Biopsies</th>
            <th>Bleeding</th>
        </tr>
    </thead><tbody>`;

    sites.forEach(s => {
        const num = s.site_number || '?';
        const loc = s.lobe + (s.segment ? ` ${s.segment}` : '');
        const probe = s.probe_size_mm ? `${s.probe_size_mm}mm` : '—';
        const freeze = s.freeze_time_seconds ? `${s.freeze_time_seconds}s` : '—';
        const biopsies = s.number_of_biopsies || '—';

        let bleeding = '—';
        if (s.bleeding_severity) {
            const colors = {
                'None/Scant': 'success',
                'Mild': 'info',
                'Moderate': 'warning',
                'Severe': 'danger'
            };
            bleeding = `<span class="badge bg-${colors[s.bleeding_severity] || 'light'}">${s.bleeding_severity}</span>`;
        }

        html += `<tr>
            <td>${num}</td>
            <td><strong>${loc}</strong></td>
            <td>${probe}</td>
            <td>${freeze}</td>
            <td>${biopsies}</td>
            <td>${bleeding}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    return html;
}

/**
 * Format granular thoracoscopy findings
 */
function formatGranularThoracoscopy(findings) {
    if (!findings || findings.length === 0) return '<span class="text-muted">—</span>';

    let html = '<ul class="list-unstyled mb-0">';

    findings.forEach(f => {
        const location = f.location || '?';
        const finding = f.finding_type || '—';
        const extent = f.extent ? ` (${f.extent})` : '';

        let impressionBadge = '';
        if (f.impression) {
            const colors = {
                'Benign appearing': 'success',
                'Malignant appearing': 'danger',
                'Infectious appearing': 'warning',
                'Indeterminate': 'secondary'
            };
            impressionBadge = ` <span class="badge bg-${colors[f.impression] || 'light'}">${f.impression}</span>`;
        }

        let biopsy = '';
        if (f.biopsied) {
            const count = f.number_of_biopsies ? `${f.number_of_biopsies}×` : '';
            const tool = f.biopsy_tool ? ` ${f.biopsy_tool}` : '';
            biopsy = ` · <small class="text-info">${count}biopsied${tool}</small>`;
        }

        html += `<li><strong>${location}</strong>: ${finding}${extent}${impressionBadge}${biopsy}</li>`;
    });

    html += '</ul>';
    return html;
}

/**
 * Format granular specimens collected
 */
function formatGranularSpecimens(specimens) {
    if (!specimens || specimens.length === 0) return '<span class="text-muted">—</span>';

    let html = '<div class="table-responsive"><table class="table table-sm table-bordered mb-0" style="font-size: 0.9em;">';
    html += `<thead class="table-light">
        <tr>
            <th>#</th>
            <th>Source</th>
            <th>Location</th>
            <th>Tool</th>
            <th>Destinations</th>
            <th>ROSE</th>
        </tr>
    </thead><tbody>`;

    specimens.forEach(s => {
        const num = s.specimen_number || '?';
        const source = s.source_procedure || '—';
        const location = s.source_location || '—';
        const tool = s.collection_tool || '—';

        let destinations = '—';
        if (s.destinations && s.destinations.length > 0) {
            destinations = s.destinations.map(d => {
                // Abbreviate long names
                const abbrev = {
                    'Histology/Surgical pathology': 'Histo',
                    'Molecular/NGS': 'NGS',
                    'Flow cytometry': 'Flow',
                    'Cell block': 'CB',
                    'Bacterial culture': 'Bact',
                    'AFB culture': 'AFB',
                    'Fungal culture': 'Fungal'
                };
                return abbrev[d] || d;
            }).join(', ');
        }

        let rose = '—';
        if (s.rose_result) {
            rose = s.rose_result;
        }

        html += `<tr>
            <td>${num}</td>
            <td>${source}</td>
            <td><strong>${location}</strong></td>
            <td>${tool}</td>
            <td><small>${destinations}</small></td>
            <td>${rose}</td>
        </tr>`;
    });

    html += '</tbody></table></div>';
    return html;
}

// ============================================================================
// END GRANULAR DATA FORMATTERS
// ============================================================================

function getPreferredFieldValue(field) {
    if (
        field &&
        typeof field === "object" &&
        !Array.isArray(field) &&
        ("clean" in field || "raw" in field)
    ) {
        return field.clean ?? field.raw;
    }
    return field;
}

function buildScopeSummary(equipment) {
    const equip = getPreferredFieldValue(equipment);
    if (!equip || typeof equip !== "object") return null;

    const parts = [];

    const formatMillimeters = (value) => {
        const preferred = getPreferredFieldValue(value);
        if (preferred === null || preferred === undefined) {
            return null;
        }
        if (typeof preferred === "number" && !Number.isNaN(preferred)) {
            return `${preferred} mm`;
        }
        if (typeof preferred === "string") {
            const trimmed = preferred.trim();
            if (!trimmed) return null;
            if (/\bmm\b/i.test(trimmed)) {
                return trimmed.replace(/\s+/g, " ").trim();
            }
            return `${trimmed} mm`;
        }
        return `${preferred} mm`;
    };

    const buildRigidLabel = (diameter) => {
        const od = formatMillimeters(diameter);
        if (!od) return "Rigid bronchoscope";
        const lower = od.toLowerCase();
        if (lower.includes("od")) {
            return `Rigid bronchoscope – ${od}`;
        }
        return `Rigid bronchoscope – ${od} OD`;
    };

    const buildFlexibleLabel = (channel, diameter) => {
        const chan = formatMillimeters(channel);
        if (chan) {
            const lowerChan = chan.toLowerCase();
            if (lowerChan.includes("channel")) {
                return `Flexible bronchoscope – ${chan}`;
            }
            return `Flexible bronchoscope – ${chan} working channel`;
        }
        const od = formatMillimeters(diameter);
        if (!od) return "Flexible bronchoscope";
        const lower = od.toLowerCase();
        if (lower.includes("od")) {
            return `Flexible bronchoscope – ${od}`;
        }
        return `Flexible bronchoscope – ${od} OD`;
    };

    const pushRigidScopes = (scopes) => {
        if (!Array.isArray(scopes)) return;
        scopes.forEach(scope => {
            const entry = getPreferredFieldValue(scope) || scope || {};
            parts.push(
                buildRigidLabel(
                    entry.outer_diameter_mm ??
                    entry.bronchoscope_outer_diameter_mm ??
                    entry.od_mm
                )
            );
        });
    };

    const pushFlexibleScopes = (scopes) => {
        if (!Array.isArray(scopes)) return;
        scopes.forEach(scope => {
            const entry = getPreferredFieldValue(scope) || scope || {};
            parts.push(
                buildFlexibleLabel(
                    entry.working_channel_mm ??
                    entry.bronchoscope_working_channel_mm ??
                    entry.working_channel_size_mm,
                    entry.outer_diameter_mm ??
                    entry.bronchoscope_outer_diameter_mm ??
                    entry.od_mm
                )
            );
        });
    };

    pushRigidScopes(getPreferredFieldValue(equip.rigid_bronchoscopes));
    pushFlexibleScopes(getPreferredFieldValue(equip.flexible_bronchoscopes));

    if (!parts.length) {
        const typeValue = getPreferredFieldValue(equip.bronchoscope_type);
        const typeString = typeof typeValue === "string" ? typeValue.toLowerCase() : "";
        const workingChannelValue =
            equip.bronchoscope_working_channel_mm ??
            equip.working_channel_mm ??
            equip.scope_working_channel_mm;
        const odValue = equip.bronchoscope_outer_diameter_mm ??
            equip.scope_outer_diameter_mm ??
            equip.outer_diameter_mm;

        const hasChannel = !!formatMillimeters(workingChannelValue);
        const hasOd = !!formatMillimeters(odValue);
        const typeIndicatesRigid = !!typeString && typeString.includes("rigid");
        const flexibleKeywords = ["flex", "diagnostic", "therapeutic", "ultra", "ebus", "single", "robotic"];
        const typeIndicatesFlexible = !!typeString && flexibleKeywords.some(keyword => typeString.includes(keyword));

        if (typeIndicatesRigid || (!typeString && hasOd && !hasChannel)) {
            parts.push(buildRigidLabel(odValue));
        }

        const shouldAddFlexible = typeIndicatesFlexible || hasChannel || (!parts.length && typeString && !typeIndicatesRigid);
        if (shouldAddFlexible) {
            parts.push(buildFlexibleLabel(workingChannelValue, odValue));
        }
    }

    if (!parts.length) return null;
    return parts.join("; ");
}

function buildRegistryDisplayRows(payload) {
    const rawRecord = payload?.record || payload?.raw_record || payload || {};
    const cleanRecord = payload?.clean_record || payload?.reviewed_record || null;
    const rec = cleanRecord || rawRecord || {};

    const rows = [];

    const LABEL_OVERRIDES = {
        asa_class: "ASA Class",
        primary_indication: "Primary Indication",
        radiographic_findings: "Radiographic Findings",
        lesion_size_mm: "Lesion Size (mm)",
        lesion_location: "Lesion Location",
        sedation: "Sedation",
        equipment: "Equipment",
        navigation_platform: "Navigation Platform",
        procedures_performed: "Procedures",
        pleural_procedures: "Pleural Procedures",
        tbna_conventional: "TBNA",
        peripheral_tbna: "Peripheral TBNA",
        radial_ebus: "Radial EBUS",
        navigational_bronchoscopy: "Navigational Bronchoscopy",
        transbronchial_biopsy: "Transbronchial Biopsy",
        transbronchial_cryobiopsy: "Transbronchial Cryobiopsy",
        therapeutic_aspiration: "Therapeutic Aspiration",
        passes_per_station: "Passes per Station",
        stations_sampled: "Stations Sampled",
        probe_position: "View",
        tool_in_lesion_confirmed: "Tool-in-Lesion Confirmed",
        confirmation_method: "Tool-in-Lesion Confirmation Method",
        sampling_tools_used: "Sampling Tools Used",
        freeze_time_seconds: "Freeze Time (sec)",
        number_of_samples: "Sample Count",
        number_of_biopsies: "Biopsy Count",
        number_of_cryo_biopsies: "Cryobiopsy Count",
        number_of_needle_passes: "Needle Pass Count",
        material: "Material",
        location: "Location",
        disposition: "Disposition",
        procedure_completed: "Procedure Completed",
        procedure_families: "Procedure Families",
        version: "Registry Version",
        granular_validation_warnings: "Validation Warnings",
        navigation_targets: "Navigation Target",
        linear_ebus_stations_detail: "EBUS Station",
        cryobiopsy_sites: "Cryobiopsy Site",
        specimens_collected: "Specimen",
        probe_size_mm: "Probe Size (mm)",
        distance_from_pleura: "Distance from Pleura",
        blocker_used: "Blocker Used",
        bleeding_severity: "Bleeding Severity",
        target_lobe: "Target Lobe",
        target_segment: "Target Segment",
        target_location_text: "Target Location",
        target_number: "Navigation Target Number",
        bronchus_sign: "Bronchus Sign",
        cpt_codes_simple: "CPT Codes",
        cpt_codes: "CPT Codes",
    };

    const ACRONYM_MAP = {
        tbna: "TBNA",
        ebus: "EBUS",
        cbct: "CBCT",
        suv: "SUV",
    };

    const SKIP_KEYS = new Set(["evidence"]);
    const COLLAPSE_KEYS = new Set([
        "procedures_performed",
        "granular_data",
        "clinical_context",
        "equipment",
        "outcomes",
        "billing",
        "providers",
        "patient_demographics",
        "procedure_setting",
        "metadata",
        "complications",
        "coding_support",
        "pleural_procedures",
        "specimens",
        "pathology_results",
    ]);

    const prettifySegment = (segment) => {
        if (LABEL_OVERRIDES[segment]) return LABEL_OVERRIDES[segment];
        // Preserve already formatted labels (e.g., with index)
        if (segment.match(/^[A-Z].*\d+$/)) return segment;
        return segment
            .split("_")
            .map(part => {
                const lower = part.toLowerCase();
                if (ACRONYM_MAP[lower]) return ACRONYM_MAP[lower];
                return lower.charAt(0).toUpperCase() + lower.slice(1);
            })
            .join(" ");
    };

    const singularize = (word) => {
        if (word.endsWith("s")) return word.slice(0, -1);
        return word;
    };

    const formatPrimitive = (value) => {
        if (typeof value === "boolean") return value ? "Yes" : "No";
        if (typeof value === "number") return String(value);
        if (typeof value === "string") return value.trim();
        return JSON.stringify(value);
    };

    const labelFromPath = (parts) => {
        const filtered = parts.filter(p => !COLLAPSE_KEYS.has(p));
        const effective = filtered.length ? filtered : parts;
        return effective.map(prettifySegment).join(" ");
    };

    const flatten = (value, pathParts) => {
        const preferred = getPreferredFieldValue(value);
        if (preferred === null || preferred === undefined) return;

        // Handle arrays
        if (Array.isArray(preferred)) {
            const filtered = preferred.filter(item => item !== null && item !== undefined);
            if (!filtered.length) return;

            const allPrimitives = filtered.every(item => {
                return (
                    item === null ||
                    ["string", "number", "boolean"].includes(typeof item)
                );
            });

            if (allPrimitives) {
                rows.push({
                    field: labelFromPath(pathParts),
                    value: filtered.map(formatPrimitive).join("; "),
                });
                return;
            }

            // Array of objects -> recurse with index
            filtered.forEach((item, idx) => {
                const base = singularize(pathParts[pathParts.length - 1] || "Item");
                flatten(item, [...pathParts.slice(0, -1), `${prettifySegment(base)} ${idx + 1}`]);
            });
            return;
        }

        // Handle objects
        if (typeof preferred === "object") {
            const entries = Object.entries(preferred).filter(([, v]) => v !== null && v !== undefined);
            if (!entries.length) return;
            entries.forEach(([key, val]) => {
                if (SKIP_KEYS.has(key)) return;
                flatten(val, [...pathParts, key]);
            });
            return;
        }

        // Primitive
        const label = labelFromPath(pathParts);
        const formatted = formatPrimitive(preferred);
        if (formatted === "") return;
        rows.push({ field: label, value: formatted });
    };

    Object.entries(rec).forEach(([key, val]) => {
        if (SKIP_KEYS.has(key)) return;
        flatten(val, [key]);
    });

    return rows;
}

function ensureReporterTemplates() {
    const sel = document.getElementById('reporter-template');
    if (!sel) return;
    const desired = [
        { value: 'knowledge', label: 'Comprehensive (knowledge)' },
        { value: 'comprehensive', label: 'Comprehensive (alias)' },
        { value: 'comprehensive_ip', label: 'Comprehensive IP (alias)' },
    ];
    desired.forEach(({ value, label }) => {
        if (!Array.from(sel.options).some(opt => opt.value === value)) {
            const opt = document.createElement('option');
            opt.value = value;
            opt.textContent = label;
            sel.appendChild(opt);
        }
    });
}

function setMode(mode) {
    currentMode = mode;

    // Reset PHI state when switching modes
    resetPHIState();
    if (mode !== 'reporter') {
        reporterBuilderState.bundle = null;
        reporterBuilderState.questions = [];
    }

    // Update Tab UI
    document.querySelectorAll('#mode-tabs .nav-link').forEach(el => {
        el.classList.remove('active');
        if (el.dataset.mode === mode) el.classList.add('active');
    });

    // Update Options UI
    document.querySelectorAll('.mode-opt').forEach(el => el.style.display = 'none');
    document.getElementById(`opt-${mode}`).style.display = 'block';

    // Toggle button visibility for unified vs other modes
    const unifiedButtons = document.getElementById('unified-buttons');
    const otherButtons = document.getElementById('other-buttons');
    const phiPreviewArea = document.getElementById('phi-preview-area');

    if (unifiedButtons) unifiedButtons.style.display = mode === 'unified' ? 'grid' : 'none';
    if (otherButtons) otherButtons.style.display = mode === 'unified' ? 'none' : 'grid';
    if (phiPreviewArea) phiPreviewArea.style.display = 'none';
}

function resetPHIState() {
    phiState.rawText = null;
    phiState.scrubbedText = null;
    phiState.entities = [];
    phiState.previewDone = false;

    const btnPreview = document.getElementById('btn-phi-preview');
    const btnExtract = document.getElementById('btn-extract');

    if (btnPreview) {
        btnPreview.classList.remove('btn-success');
        btnPreview.classList.add('btn-warning');
        btnPreview.innerHTML = '<i class="bi bi-shield-check"></i> 1. Redact PHI';
    }
    if (btnExtract) btnExtract.disabled = true;
}

// ============================================================================
// PHI PREVIEW FUNCTIONS (Two-step workflow for unified mode)
// ============================================================================

/**
 * Handle PHI Preview button click (Step 1 of unified workflow)
 */
async function handlePHIPreview() {
    const text = document.getElementById('input-text').value;
    if (!text.trim()) {
        alert("Please enter a procedure note.");
        return;
    }

    showLoading(true);
    try {
        // Call preview endpoint (no persistence, no database writes)
        const resp = await fetch('/v1/phi/scrub/preview', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, document_type: 'procedure_note' })
        });
        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || 'PHI preview failed');
        }
        const result = await resp.json();

        // Store state
        phiState.rawText = text;
        phiState.scrubbedText = result.scrubbed_text;
        phiState.entities = result.entities || [];
        phiState.previewDone = true;

        // Show preview area and render
        renderPHIPreview();

        // Enable extract button and update preview button
        document.getElementById('btn-extract').disabled = false;
        const btnPreview = document.getElementById('btn-phi-preview');
        btnPreview.classList.remove('btn-warning');
        btnPreview.classList.add('btn-success');
        btnPreview.innerHTML = '<i class="bi bi-check-circle"></i> PHI Redacted';

    } catch (error) {
        alert(`PHI Preview Error: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * Render the PHI preview area with scrubbed text and entity badges
 */
function renderPHIPreview() {
    const previewArea = document.getElementById('phi-preview-area');
    previewArea.style.display = 'block';

    const previewDiv = document.getElementById('scrubbed-preview');
    previewDiv.textContent = phiState.scrubbedText;
    previewDiv.classList.remove('text-muted');

    document.getElementById('phi-entity-count').textContent =
        `${phiState.entities.length} entities`;

    renderEntityBadges();
}

/**
 * Render entity badges with edit and remove buttons
 */
function renderEntityBadges() {
    const list = document.getElementById('entity-list');
    list.innerHTML = '';

    phiState.entities.forEach((ent, idx) => {
        const badge = document.createElement('span');
        badge.className = 'badge bg-info text-dark entity-badge d-inline-flex align-items-center';

        const span = document.createElement('span');
        span.textContent = `${ent.entity_type} → ${ent.placeholder}`;
        badge.appendChild(span);

        // Edit button (pencil)
        const editBtn = document.createElement('i');
        editBtn.className = 'bi bi-pencil-square ms-2';
        editBtn.style.cursor = 'pointer';
        editBtn.onclick = () => openEditModal(idx);
        badge.appendChild(editBtn);

        // Remove button (x)
        const closeBtn = document.createElement('i');
        closeBtn.className = 'bi bi-x ms-2';
        closeBtn.style.cursor = 'pointer';
        closeBtn.onclick = () => removeEntity(idx);
        badge.appendChild(closeBtn);

        list.appendChild(badge);
    });
}

/**
 * Remove an entity from the list and regenerate scrubbed text
 */
function removeEntity(index) {
    phiState.entities.splice(index, 1);
    phiState.scrubbedText = generateScrubbedText(phiState.rawText, phiState.entities);
    renderPHIPreview();
}

/**
 * Open the entity edit modal
 */
function openEditModal(index) {
    const ent = phiState.entities[index];
    if (!ent) return;

    document.getElementById('edit-entity-index').value = index;
    document.getElementById('edit-entity-type').value = ent.entity_type;
    document.getElementById('edit-entity-placeholder').value = ent.placeholder;

    const modal = new bootstrap.Modal(document.getElementById('entityEditModal'));
    modal.show();
}

/**
 * Save entity changes from the modal
 */
function saveEntity() {
    const idx = parseInt(document.getElementById('edit-entity-index').value);
    const type = document.getElementById('edit-entity-type').value;
    const placeholder = document.getElementById('edit-entity-placeholder').value;

    if (phiState.entities[idx]) {
        phiState.entities[idx].entity_type = type;
        phiState.entities[idx].placeholder = placeholder;
        phiState.scrubbedText = generateScrubbedText(phiState.rawText, phiState.entities);
        renderPHIPreview();
    }

    const modalEl = document.getElementById('entityEditModal');
    const modal = bootstrap.Modal.getInstance(modalEl);
    if (modal) modal.hide();
}

/**
 * Generate scrubbed text by replacing entities with their placeholders.
 * Processes entities from end to beginning to avoid index shifting.
 */
function generateScrubbedText(text, entities) {
    if (!text || !entities || entities.length === 0) return text;

    // Sort entities by original_start descending to avoid index shift issues
    const sorted = [...entities].sort((a, b) => b.original_start - a.original_start);

    let chars = text.split('');

    sorted.forEach(ent => {
        const start = ent.original_start;
        const end = ent.original_end;
        if (start < 0 || end > chars.length) return;

        const placeholder = ent.placeholder || `[${ent.entity_type}]`;
        // Replace the character range with placeholder characters
        chars.splice(start, end - start, ...placeholder.split(''));
    });

    return chars.join('');
}

// ============================================================================
// END PHI PREVIEW FUNCTIONS
// ============================================================================

function showLoading(show) {
    const overlay = document.getElementById('loading-overlay');
    overlay.style.display = show ? 'flex' : 'none';
    document.querySelector('button').disabled = show;
}

async function postJSON(url, payload) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    if (!response.ok) {
        const text = await response.text();
        throw new Error(`API Error: ${response.status} ${response.statusText} - ${text}`);
    }
    return response.json();
}

function escapeHtml(value) {
    return String(value ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function decodePointerToken(token) {
    return token.replace(/~1/g, '/').replace(/~0/g, '~');
}

function pointerExists(documentValue, pointer) {
    if (!pointer || pointer === '/') return true;
    const tokens = pointer.split('/').slice(1).map(decodePointerToken);
    let current = documentValue;

    for (const token of tokens) {
        if (Array.isArray(current)) {
            if (!/^\d+$/.test(token)) return false;
            const index = Number(token);
            if (!Number.isInteger(index) || index < 0 || index >= current.length) return false;
            current = current[index];
            continue;
        }
        if (current && typeof current === 'object') {
            if (!(token in current)) return false;
            current = current[token];
            continue;
        }
        return false;
    }

    return true;
}

function parseReporterQuestionValue(question, index) {
    const input = document.getElementById(`reporter-question-input-${index}`);
    if (!input) return { hasValue: false, value: null };

    const inputType = question.input_type || 'string';
    if (inputType === 'multiselect') {
        const selected = Array.from(input.selectedOptions || []).map(opt => opt.value).filter(v => v !== '');
        if (!selected.length) return { hasValue: false, value: null };
        return { hasValue: true, value: selected };
    }

    if (inputType === 'boolean') {
        const raw = (input.value || '').trim().toLowerCase();
        if (raw === '') return { hasValue: false, value: null };
        if (raw === 'true') return { hasValue: true, value: true };
        if (raw === 'false') return { hasValue: true, value: false };
        throw new Error(`Invalid boolean value for "${question.label}".`);
    }

    const raw = typeof input.value === 'string' ? input.value.trim() : input.value;
    if (raw === '' || raw === null || raw === undefined) return { hasValue: false, value: null };

    if (inputType === 'integer') {
        const parsed = Number.parseInt(raw, 10);
        if (!Number.isFinite(parsed)) {
            throw new Error(`"${question.label}" must be an integer.`);
        }
        return { hasValue: true, value: parsed };
    }

    if (inputType === 'number') {
        const parsed = Number.parseFloat(raw);
        if (!Number.isFinite(parsed)) {
            throw new Error(`"${question.label}" must be a number.`);
        }
        return { hasValue: true, value: parsed };
    }

    return { hasValue: true, value: raw };
}

function renderReporterQuestionsForm(questions) {
    if (!questions || !questions.length) {
        return '<div class="alert alert-success mb-3">No required follow-up questions. The report appears complete.</div>';
    }

    const groups = new Map();
    questions.forEach((question, index) => {
        const group = question.group || 'Additional Details';
        if (!groups.has(group)) groups.set(group, []);
        groups.get(group).push({ question, index });
    });

    let html = '<div class="card border-info mb-3">';
    html += '<div class="card-header bg-info-subtle"><strong>Interactive Reporter Builder</strong></div>';
    html += '<div class="card-body">';
    html += '<p class="text-muted small mb-3">Answer one or more questions, then apply patch and re-render.</p>';

    groups.forEach((items, group) => {
        html += `<div class="mb-3"><h6 class="mb-2">${escapeHtml(group)}</h6>`;
        items.forEach(({ question, index }) => {
            const requiredBadge = question.required ? '<span class="text-danger">*</span>' : '';
            const helpText = question.help ? `<div class="form-text">${escapeHtml(question.help)}</div>` : '';
            const inputId = `reporter-question-input-${index}`;
            const inputType = question.input_type || 'string';
            const options = Array.isArray(question.options) ? question.options : [];
            let control = '';

            if (inputType === 'boolean') {
                control = `
                    <select class="form-select form-select-sm" id="${inputId}">
                        <option value="">-- Select --</option>
                        <option value="true">Yes</option>
                        <option value="false">No</option>
                    </select>
                `;
            } else if ((inputType === 'enum' || inputType === 'multiselect') && options.length) {
                if (inputType === 'multiselect') {
                    const size = Math.min(Math.max(options.length, 3), 6);
                    control = `
                        <select class="form-select form-select-sm" id="${inputId}" multiple size="${size}">
                            ${options.map(opt => `<option value="${escapeHtml(opt)}">${escapeHtml(opt)}</option>`).join('')}
                        </select>
                    `;
                } else {
                    control = `
                        <select class="form-select form-select-sm" id="${inputId}">
                            <option value="">-- Select --</option>
                            ${options.map(opt => `<option value="${escapeHtml(opt)}">${escapeHtml(opt)}</option>`).join('')}
                        </select>
                    `;
                }
            } else if (inputType === 'textarea') {
                control = `<textarea class="form-control form-control-sm" id="${inputId}" rows="2" placeholder="Enter details"></textarea>`;
            } else {
                const htmlInputType = inputType === 'integer' || inputType === 'number' ? 'number' : 'text';
                const step = inputType === 'integer' ? ' step="1"' : '';
                control = `<input type="${htmlInputType}" class="form-control form-control-sm" id="${inputId}" placeholder="Enter value"${step}>`;
            }

            html += `
                <div class="mb-2">
                    <label class="form-label form-label-sm mb-1" for="${inputId}">
                        ${escapeHtml(question.label)} ${requiredBadge}
                        <span class="text-muted small">(${escapeHtml(question.pointer)})</span>
                    </label>
                    ${control}
                    ${helpText}
                </div>
            `;
        });
        html += '</div>';
    });

    html += `
        <div class="d-flex gap-2">
            <button type="button" class="btn btn-primary btn-sm" onclick="applyReporterQuestionPatch()">Apply Answers & Re-render</button>
            <button type="button" class="btn btn-outline-secondary btn-sm" onclick="refreshReporterQuestions()">Refresh Questions</button>
        </div>
    `;
    html += '</div></div>';
    return html;
}

async function runReporterFlow(noteText) {
    // Step 0: PHI scrubbing
    const { scrubbedText } = await submitAndApprovePHI(noteText);

    // Step 1: registry extraction with scrubbed text (for UI transparency)
    const extraction = await postJSON('/v1/registry/run', {
        note: scrubbedText,
        explain: document.getElementById('registry-explain')?.checked || false,
        mode: document.getElementById('registry-disable-llm')?.checked ? 'engine_only' : null,
    });

    // Step 2: seed bundle + questions + draft markdown
    const strict = false;
    const seed = await postJSON('/report/seed_from_text', {
        text: scrubbedText,
        strict: false,
    });

    reporterBuilderState.bundle = seed.bundle || null;
    reporterBuilderState.questions = seed.questions || [];
    reporterBuilderState.strict = strict;

    return {
        extraction,
        seed,
        verify: {
            bundle: seed.bundle,
            issues: seed.issues || [],
            warnings: seed.warnings || [],
            inference_notes: seed.inference_notes || [],
            suggestions: seed.suggestions || [],
        },
        render: {
            bundle: seed.bundle,
            markdown: seed.markdown || '',
            issues: seed.issues || [],
            warnings: seed.warnings || [],
            inference_notes: seed.inference_notes || [],
            suggestions: seed.suggestions || [],
        },
        questions: seed.questions || [],
    };
}

async function refreshReporterQuestions() {
    if (!reporterBuilderState.bundle) {
        alert('Run Reporter mode first to initialize the bundle.');
        return;
    }

    showLoading(true);
    try {
        const questionsResp = await postJSON('/report/questions', {
            bundle: reporterBuilderState.bundle,
            strict: reporterBuilderState.strict,
        });

        reporterBuilderState.bundle = questionsResp.bundle;
        reporterBuilderState.questions = questionsResp.questions || [];

        if (!lastResult) lastResult = {};
        lastResult.verify = {
            bundle: questionsResp.bundle,
            issues: questionsResp.issues || [],
            warnings: questionsResp.warnings || [],
            inference_notes: questionsResp.inference_notes || [],
            suggestions: questionsResp.suggestions || [],
        };
        lastResult.questions = questionsResp.questions || [];
        renderResult();
    } catch (error) {
        alert(`Failed to refresh questions: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

async function applyReporterQuestionPatch() {
    if (!reporterBuilderState.bundle) {
        alert('Run Reporter mode first to initialize the bundle.');
        return;
    }

    const questions = reporterBuilderState.questions || [];
    if (!questions.length) {
        alert('No outstanding questions to patch.');
        return;
    }

    const patchOps = [];
    try {
        questions.forEach((question, index) => {
            const parsed = parseReporterQuestionValue(question, index);
            if (!parsed.hasValue) return;
            patchOps.push({
                op: pointerExists(reporterBuilderState.bundle, question.pointer) ? 'replace' : 'add',
                path: question.pointer,
                value: parsed.value,
            });
        });
    } catch (error) {
        alert(error.message);
        return;
    }

    if (!patchOps.length) {
        alert('Enter at least one answer before applying.');
        return;
    }

    showLoading(true);
    try {
        const render = await postJSON('/report/render', {
            bundle: reporterBuilderState.bundle,
            patch: patchOps,
            embed_metadata: false,
            strict: reporterBuilderState.strict,
        });

        const questionsResp = await postJSON('/report/questions', {
            bundle: render.bundle,
            strict: reporterBuilderState.strict,
        });

        reporterBuilderState.bundle = questionsResp.bundle;
        reporterBuilderState.questions = questionsResp.questions || [];

        if (!lastResult) lastResult = {};
        lastResult.render = render;
        lastResult.verify = {
            bundle: questionsResp.bundle,
            issues: questionsResp.issues || [],
            warnings: questionsResp.warnings || [],
            inference_notes: questionsResp.inference_notes || [],
            suggestions: questionsResp.suggestions || [],
        };
        lastResult.questions = questionsResp.questions || [];

        renderResult();
    } catch (error) {
        alert(`Reporter patch failed: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * Submit text to PHI vault and auto-approve.
 * Returns the procedure_id and scrubbed_text.
 */
async function submitAndApprovePHI(text) {
    // Step 1: Submit to PHI vault for scrubbing
    const submitResp = await fetch('/v1/phi/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            submitted_by: 'workbench_user',
            document_type: 'procedure_note',
        })
    });
    if (!submitResp.ok) {
        const err = await submitResp.json().catch(() => ({}));
        throw new Error(`PHI submit failed: ${err.detail || submitResp.statusText}`);
    }
    const submitResult = await submitResp.json();
    const procedureId = submitResult.procedure_id;
    const scrubbedText = submitResult.scrubbed_text;

    // Step 2: Auto-approve (mark as reviewed)
    const feedbackResp = await fetch(`/v1/phi/procedure/${procedureId}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            scrubbed_text: scrubbedText,
            entities: submitResult.entities || [],
            reviewer_id: 'workbench_auto',
            reviewer_email: 'workbench@local',
            reviewer_role: 'auto',
            comment: 'Auto-approved via workbench'
        })
    });
    if (!feedbackResp.ok) {
        const err = await feedbackResp.json().catch(() => ({}));
        throw new Error(`PHI review failed: ${err.detail || feedbackResp.statusText}`);
    }

    return { procedureId, scrubbedText };
}

/**
 * Run coder through PHI workflow:
 * 1. Submit and approve PHI
 * 2. Call coder endpoint with scrubbed text
 */
async function runCoderViaPHI(text, options) {
    const { scrubbedText } = await submitAndApprovePHI(text);

    // Call coder with scrubbed text
    const coderResp = await fetch('/v1/coder/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            note: scrubbedText,
            explain: options.explain,
            allow_weak_sedation_docs: options.allow_weak_sedation_docs,
            locality: options.locality,
            setting: options.setting,
            use_ml_first: options.use_ml_first,
            mode: options.mode || null
        })
    });
    if (!coderResp.ok) {
        const err = await coderResp.json().catch(() => ({}));
        throw new Error(`Coder failed: ${err.detail || coderResp.statusText}`);
    }

    return coderResp.json();
}

/**
 * Run registry through PHI workflow:
 * 1. Submit and approve PHI
 * 2. Call registry endpoint with scrubbed text
 */
async function runRegistryViaPHI(text, options) {
    const { scrubbedText } = await submitAndApprovePHI(text);

    // Call registry with scrubbed text
    const registryResp = await fetch('/v1/registry/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            note: scrubbedText,
            explain: options.explain,
            mode: options.mode || null
        })
    });
    if (!registryResp.ok) {
        const err = await registryResp.json().catch(() => ({}));
        const hint = registryResp.status >= 500 ? " Retry extraction or switch to manual mode." : "";
        throw new Error(`Registry failed: ${err.detail || registryResp.statusText}.${hint}`);
    }

    return registryResp.json();
}

/**
 * Run unified extraction through PHI workflow:
 * 1. Submit note to PHI vault (scrub)
 * 2. Auto-approve the scrubbed text
 * 3. Call PHI-gated extraction endpoint
 *
 * If phiState.previewDone is true, uses pre-edited entities (two-step workflow).
 * Otherwise, uses auto-scrubbing (single-step fallback).
 */
async function runUnifiedViaPHI(text, options) {
    // If preview was completed, use confirmed entities (no re-scrubbing)
    if (phiState.previewDone) {
        return await runUnifiedWithConfirmedEntities(options);
    }

    // Fallback to original single-step behavior (auto-scrub + auto-approve)
    const { scrubbedText } = await submitAndApprovePHI(text);

    // Run unified extraction on scrubbed text
    const extractResp = await fetch(`/api/v1/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            note: scrubbedText,
            already_scrubbed: true,
            include_financials: options.include_financials,
            explain: options.explain,
        })
    });
    if (!extractResp.ok) {
        const err = await extractResp.json().catch(() => ({}));
        const hint = extractResp.status >= 500 ? " Retry extraction or switch to manual mode." : "";
        throw new Error(`Extraction failed: ${err.detail || extractResp.statusText}.${hint}`);
    }

    return extractResp.json();
}

/**
 * Run unified extraction with confirmed (pre-edited) entities.
 * This is the two-step workflow:
 * 1. Submit with confirmed_entities (bypass Presidio re-scrubbing)
 * 2. Auto-approve the scrubbed text
 * 3. Run extraction (no Presidio at this point - text already clean)
 */
async function runUnifiedWithConfirmedEntities(options) {
    // Step 1: Submit with confirmed_entities (uses pre-edited entities, no re-scrubbing)
    const submitResp = await fetch('/v1/phi/submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: phiState.rawText,
            submitted_by: 'workbench_user',
            document_type: 'procedure_note',
            confirmed_entities: phiState.entities  // Pass edited entities
        })
    });
    if (!submitResp.ok) {
        const err = await submitResp.json().catch(() => ({}));
        throw new Error(`PHI submit failed: ${err.detail || submitResp.statusText}`);
    }
    const { procedure_id, scrubbed_text } = await submitResp.json();

    // Step 2: Mark as reviewed (auto-approve with confirmed entities)
    const feedbackResp = await fetch(`/v1/phi/procedure/${procedure_id}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            scrubbed_text: scrubbed_text,
            entities: phiState.entities,
            reviewer_id: 'workbench_auto',
            reviewer_email: 'workbench@local',
            reviewer_role: 'auto',
            comment: 'Approved via workbench two-step workflow (user reviewed PHI)'
        })
    });
    if (!feedbackResp.ok) {
        const err = await feedbackResp.json().catch(() => ({}));
        throw new Error(`PHI review failed: ${err.detail || feedbackResp.statusText}`);
    }

    // Step 3: Run extraction (no Presidio at this point - text already scrubbed)
    const extractResp = await fetch(`/api/v1/process`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            note: scrubbed_text,
            already_scrubbed: true,
            include_financials: options.include_financials,
            explain: options.explain,
        })
    });
    if (!extractResp.ok) {
        const err = await extractResp.json().catch(() => ({}));
        const hint = extractResp.status >= 500 ? " Retry extraction or switch to manual mode." : "";
        throw new Error(`Extraction failed: ${err.detail || extractResp.statusText}.${hint}`);
    }

    // Reset PHI state after successful extraction
    resetPHIState();

    return extractResp.json();
}

/**
 * Main Orchestrator: Renders the clean clinical dashboard
 */
function renderDashboard(data) {
    renderStatCards(data);
    renderUnifiedTable(data);
    renderRegistrySummary(data);
    renderDebugLogs(data);
}

/**
 * 1. Renders the Executive Summary (Stat Cards)
 */
function renderStatCards(data) {
    const container = document.getElementById('statCards');
    if (!container) return;

    // Determine review status
    let statusText = "Ready";
    let statusClass = "";
    if (data.needs_manual_review || (data.audit_warnings && data.audit_warnings.length > 0)) {
        statusText = "⚠️ Review Required";
        statusClass = "warning";
    }

    // Format currency and RVU
    const payment = data.estimated_payment
        ? new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(data.estimated_payment)
        : '$0.00';
    const rvu = data.total_work_rvu ? data.total_work_rvu.toFixed(2) : '0.00';

    container.innerHTML = `
        <div class="stat-card">
            <span class="stat-label">Review Status</span>
            <div class="stat-value ${statusClass}">${statusText}</div>
        </div>
        <div class="stat-card">
            <span class="stat-label">Total wRVU</span>
            <div class="stat-value">${rvu}</div>
        </div>
        <div class="stat-card">
            <span class="stat-label">Est. Payment</span>
            <div class="stat-value currency">${payment}</div>
        </div>
        <div class="stat-card">
            <span class="stat-label">CPT Count</span>
            <div class="stat-value">${(data.per_code_billing || []).length}</div>
        </div>
    `;
}

/**
 * Transforms API data into a unified "Golden Record"
 * FIX: Now prioritizes backend rationale over generic placeholders
 */
function transformToUnifiedTable(rawData) {
    const unifiedMap = new Map();

    // Helper: Get explanation from specific coding_support backend map
    const getBackendRationale = (code) => {
        if (rawData.registry?.coding_support?.code_rationales?.[code]) {
            return rawData.registry.coding_support.code_rationales[code];
        }
        // Fallback to "evidence" array if available
        const billingEntry = rawData.registry?.billing?.cpt_codes?.find(c => c.code === code);
        if (billingEntry?.evidence?.length > 0) {
            return billingEntry.evidence.map(e => e.text).join('; ');
        }
        return null;
    };

    // 1. Process Header Codes (Raw)
    (rawData.header_codes || []).forEach(item => {
        unifiedMap.set(item.code, {
            code: item.code,
            desc: item.description || "Unknown Procedure",
            inHeader: true,
            inBody: false,
            status: 'pending',
            rationale: "Found in header scan", // Default start
            rvu: 0.00,
            payment: 0.00
        });
    });

    // 2. Process Derived Codes (Body)
    (rawData.derived_codes || []).forEach(item => {
        const existing = unifiedMap.get(item.code) || {
            code: item.code,
            inHeader: false,
            rvu: 0.00,
            payment: 0.00
        };

        existing.desc = item.description || existing.desc;
        existing.inBody = true;

        // FIX: Grab specific backend rationale if available
        const backendReason = getBackendRationale(item.code);
        if (backendReason) {
            existing.rationale = backendReason;
        } else {
            existing.rationale = "Derived from procedure actions";
        }

        unifiedMap.set(item.code, existing);
    });

    // 3. Process Final Selection (The "Truth")
    (rawData.per_code_billing || []).forEach(item => {
        const existing = unifiedMap.get(item.cpt_code) || {
            code: item.cpt_code,
            inHeader: false,
            inBody: true,
            rationale: "Selected"
        };

        existing.code = item.cpt_code; // Ensure code is set
        existing.desc = item.description || existing.desc;
        existing.status = item.status || 'selected';
        existing.rvu = item.work_rvu;
        existing.payment = item.facility_payment;

        // FIX: Ensure suppression/bundling logic is visible
        if (item.work_rvu === 0) {
            existing.status = 'Bundled/Suppressed';
            // If we have a specific bundling warning, append it
            const warning = (rawData.audit_warnings || []).find(w => w.includes(item.cpt_code));
            if (warning) existing.rationale = warning;
        } else {
            // Refresh rationale from backend to ensure it's not "Derived..."
            const backendReason = getBackendRationale(item.cpt_code);
            if (backendReason) existing.rationale = backendReason;
        }

        unifiedMap.set(item.cpt_code, existing);
    });

    // Sort: High Value -> Suppressed -> Header Only
    return Array.from(unifiedMap.values()).sort((a, b) => {
        if (a.rvu > 0 && b.rvu === 0) return -1;
        if (b.rvu > 0 && a.rvu === 0) return 1;
        return a.code.localeCompare(b.code);
    });
}

/**
 * 2. Renders the Unified Billing Reconciliation Table
 * Merges Header, Derived, and Final codes into one view.
 */
function renderUnifiedTable(data) {
    const tbody = document.getElementById('unifiedTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const sortedRows = transformToUnifiedTable(data);

    // Render Rows
    sortedRows.forEach(row => {
        const tr = document.createElement('tr');

        // Logic Badges
        let sourceBadge = '';
        if (row.inHeader && row.inBody) sourceBadge = `<span class="badge badge-both">Match</span>`;
        else if (row.inHeader) sourceBadge = `<span class="badge badge-header">Header Only</span>`;
        else sourceBadge = `<span class="badge badge-body">Derived</span>`;

        // Status Badge
        let statusBadge = `<span class="badge badge-primary">Primary</span>`;
        if (row.rvu === 0 || row.status === 'Bundled/Suppressed') {
            statusBadge = `<span class="badge badge-bundled">Bundled</span>`;
            tr.classList.add('row-suppressed');
        }

        // Rationale cleaning
        const rationale = row.rationale || (row.inBody ? 'Derived from procedure actions' : 'Found in header scan');
        const rvuDisplay = Number.isFinite(row.rvu) ? row.rvu.toFixed(2) : '0.00';
        const paymentDisplay = Number.isFinite(row.payment) ? row.payment.toFixed(2) : '0.00';

        tr.innerHTML = `
            <td><span class="code-cell">${row.code}</span></td>
            <td>
                <span class="desc-text">${row.desc || 'Unknown Procedure'}</span>
                ${sourceBadge}
            </td>
            <td>${statusBadge}</td>
            <td><span class="rationale-text">${rationale}</span></td>
            <td><strong>${rvuDisplay}</strong></td>
            <td>$${paymentDisplay}</td>
        `;
        tbody.appendChild(tr);
    });
}

/**
 * Renders the Clinical/Registry Data Table (Restored)
 * Flattens nested registry objects into a clean key-value view.
 */
function renderRegistrySummary(data) {
    const tbody = document.getElementById('registryTableBody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const registry = data.registry || {};

    // 1. Clinical Context (Top Priority)
    if (registry.clinical_context) {
        addRegistryRow(tbody, "Indication", registry.clinical_context.primary_indication);
        if (registry.clinical_context.indication_category) {
            addRegistryRow(tbody, "Category", registry.clinical_context.indication_category);
        }
    }

    // 2. Anesthesia/Sedation
    if (registry.sedation) {
        const sedationStr = `${registry.sedation.type || 'Not specified'} (${registry.sedation.anesthesia_provider || 'Provider unknown'})`;
        addRegistryRow(tbody, "Sedation", sedationStr);
    }

    // 3. EBUS Details (Granular)
    if (registry.procedures_performed?.linear_ebus?.performed) {
        const ebus = registry.procedures_performed.linear_ebus;
        const stations = Array.isArray(ebus.stations_sampled) ? ebus.stations_sampled.join(", ") : "None";
        const needle = ebus.needle_gauge || "Not specified";
        addRegistryRow(tbody, "Linear EBUS", `<strong>Stations:</strong> ${stations} <br> <span style="font-size:11px; color:#64748b;">Gauge: ${needle} | Elastography: ${ebus.elastography_used ? 'Yes' : 'No'}</span>`);
    }

    // 4. Other Procedures (Iterate generic performed flags)
    const procs = registry.procedures_performed || {};
    Object.keys(procs).forEach(key => {
        if (key === 'linear_ebus') return; // Handled above
        const p = procs[key];
        if (p === true || (p && p.performed)) {
            // Convert snake_case to Title Case (e.g., radial_ebus -> Radial Ebus)
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

            // Extract useful details if they exist (e.g., "lobes", "sites")
            let details = "Performed";
            if (p?.sites) details = `Sites: ${Array.isArray(p.sites) ? p.sites.join(', ') : p.sites}`;
            else if (p?.target_lobes) details = `Lobes: ${p.target_lobes.join(', ')}`;
            else if (p?.action) details = p.action;

            addRegistryRow(tbody, label, details);
        }
    });
}

// Helper to append rows
function addRegistryRow(tbody, label, content) {
    if (!content) return;
    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td style="font-weight:600; color:#475569;">${label}</td>
        <td>${content}</td>
    `;
    tbody.appendChild(tr);
}

/**
 * 3. Renders Technical Logs (Collapsed by default)
 */
function renderDebugLogs(data) {
    const logBox = document.getElementById('systemLogs');
    if (!logBox) return;

    let logs = [];

    // Collect all warnings and logs
    if (data.audit_warnings) logs.push(...data.audit_warnings.map(w => `[AUDIT] ${w}`));
    if (data.warnings) logs.push(...data.warnings.map(w => `[WARN] ${w}`));
    if (data.self_correction) {
        data.self_correction.forEach(sc => {
            logs.push(`[SELF-CORRECT] Applied patch for ${sc.trigger.target_cpt}: ${sc.trigger.reason}`);
        });
    }

    if (logs.length === 0) {
        logBox.textContent = "No system warnings or overrides.";
    } else {
        logBox.textContent = logs.join('\n');
    }
}

async function run() {
    const text = document.getElementById('input-text').value;
    if (!text.trim()) {
        alert("Please enter a procedure note.");
        return;
    }

    showLoading(true);

    try {
        let url, payload;

        if (currentMode === 'unified') {
            // Route through PHI workflow for unified mode
            lastResult = await runUnifiedViaPHI(text, {
                include_financials: document.getElementById('unified-financials').checked,
                explain: document.getElementById('unified-explain').checked,
                mode: document.getElementById('unified-disable-llm').checked ? 'engine_only' : null
            });
            console.log('API Response (via PHI):', lastResult);
            renderDashboard(lastResult);
            renderResult();
            return;
        } else if (currentMode === 'coder') {
            // Route coder through PHI workflow too
            lastResult = await runCoderViaPHI(text, {
                explain: document.getElementById('coder-explain').checked,
                allow_weak_sedation_docs: document.getElementById('coder-weak-sedation').checked,
                locality: document.getElementById('coder-locality').value || '00',
                setting: document.getElementById('coder-setting').value || 'facility',
                use_ml_first: document.getElementById('coder-ml-first').checked,
                mode: document.getElementById('coder-disable-llm').checked ? 'rules_only' : null
            });
            console.log('API Response (via PHI):', lastResult);
            renderResult();
            return;
        } else if (currentMode === 'registry') {
            // Route registry through PHI workflow too
            lastResult = await runRegistryViaPHI(text, {
                explain: document.getElementById('registry-explain').checked,
                mode: document.getElementById('registry-disable-llm').checked ? 'engine_only' : null
            });
            console.log('API Response (via PHI):', lastResult);
            renderResult();
            return;
        } else if (currentMode === 'reporter') {
            lastResult = await runReporterFlow(text);
            renderResult();
            return;
        }

        lastResult = await postJSON(url, payload);
        console.log('API Response:', lastResult);
        console.log('Has financials:', !!lastResult.financials);
        if (lastResult.financials) {
            console.log('Financials data:', lastResult.financials);
        }
        if (currentMode === 'unified') {
            renderDashboard(lastResult);
        }
        renderResult();

    } catch (error) {
        const legacy = document.getElementById('legacyResults');
        const dashboard = document.getElementById('resultsContainer');
        const rawJson = document.getElementById('rawJson');

        if (dashboard) dashboard.classList.add('hidden');
        if (rawJson) rawJson.classList.add('hidden');
        if (legacy) {
            legacy.innerHTML = `<div class="alert alert-danger">Error: ${error.message}</div>`;
            legacy.classList.remove('hidden');
        }
    } finally {
        showLoading(false);
    }
}

function renderResult() {
    // Show tabs
    document.getElementById('result-tabs').style.visibility = 'visible';
    
    // Default to formatted view
    showResultTab('formatted');
}

function showResultTab(tab) {
    // Update tab UI
    document.querySelectorAll('#result-tabs .nav-link').forEach(el => el.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');

    const dashboard = document.getElementById('resultsContainer');
    const legacy = document.getElementById('legacyResults');
    const rawJson = document.getElementById('rawJson');

    if (tab === 'json') {
        if (rawJson) {
            rawJson.textContent = JSON.stringify(lastResult, null, 2);
            rawJson.classList.remove('hidden');
        }
        if (dashboard) dashboard.classList.add('hidden');
        if (legacy) legacy.classList.add('hidden');
        return;
    }

    if (rawJson) rawJson.classList.add('hidden');

    // Formatted View Logic
    if (currentMode === 'unified') {
        if (legacy) legacy.classList.add('hidden');
        if (dashboard) dashboard.classList.remove('hidden');
        renderDashboard(lastResult);
        return;
    }

    if (dashboard) dashboard.classList.add('hidden');
    if (legacy) legacy.classList.remove('hidden');

    if (currentMode === 'coder') {
        // Coder formatting
        let html = `<h4>Billing Codes</h4>`;

        // Display hybrid pipeline metadata if available
        if (lastResult.hybrid_metadata) {
            const meta = lastResult.hybrid_metadata;
            const difficultyBadge = {
                'high_confidence': 'bg-success',
                'gray_zone': 'bg-warning text-dark',
                'low_confidence': 'bg-danger'
            }[meta.difficulty] || 'bg-secondary';
            const sourceBadge = meta.source === 'ml_rules_fastpath' ? 'bg-success' : 'bg-info';
            const llmBadge = meta.llm_used ? 'bg-warning text-dark' : 'bg-success';

            html += `<div class="alert alert-light border mb-3">
                <h6 class="mb-2"><strong>ML-First Pipeline Metadata</strong></h6>
                <div class="d-flex flex-wrap gap-2 mb-2">
                    <span class="badge ${difficultyBadge}">Difficulty: ${meta.difficulty || 'unknown'}</span>
                    <span class="badge ${sourceBadge}">Source: ${meta.source || 'unknown'}</span>
                    <span class="badge ${llmBadge}">LLM Used: ${meta.llm_used ? 'Yes' : 'No'}</span>
                </div>`;

            if (meta.ml_candidates && meta.ml_candidates.length > 0) {
                html += `<div class="small text-muted mb-1"><strong>ML Candidates:</strong> ${meta.ml_candidates.join(', ')}</div>`;
            }
            if (meta.fallback_reason) {
                html += `<div class="small text-muted mb-1"><strong>Fallback Reason:</strong> ${meta.fallback_reason}</div>`;
            }
            if (meta.rules_error) {
                html += `<div class="small text-danger"><strong>Rules Error:</strong> ${meta.rules_error}</div>`;
            }
            html += `</div>`;
        }

        if (lastResult.codes && lastResult.codes.length > 0) {
             html += `<ul class="list-group mb-3">`;
             lastResult.codes.forEach(code => {
                 html += `<li class="list-group-item d-flex justify-content-between align-items-center">
                            <div>
                                <strong>${code.code || code.cpt}</strong> - ${code.description || 'No description'}
                                ${code.modifiers && code.modifiers.length ? `<br><small class="text-muted">Modifiers: ${code.modifiers.join(', ')}</small>` : ''}
                            </div>
                            <span class="badge bg-primary rounded-pill">${code.quantity || 1}</span>
                          </li>`;
             });
             html += `</ul>`;
        } else {
            html += `<p class="text-muted">No codes generated.</p>`;
        }

        // Display RVU/Financials data if available
        // Always show financials section if it exists (even if empty)
        if (lastResult.financials !== undefined && lastResult.financials !== null) {
            const fin = lastResult.financials;
            const totalWorkRVU = (fin.total_work_rvu !== undefined && fin.total_work_rvu !== null) ? fin.total_work_rvu.toFixed(2) : 'N/A';
            // Use total_facility_payment as the primary estimate
            const totalPayment = (fin.total_facility_payment !== undefined && fin.total_facility_payment !== null) ? fin.total_facility_payment.toFixed(2) : 'N/A';
            
            html += `<hr><h5>RVU & Payment Information</h5>`;
            html += `<div class="card mb-3">`;
            html += `<div class="card-body">`;
            html += `<div class="row mb-3">`;
            html += `<div class="col-md-6"><strong>Total Work RVU:</strong> ${totalWorkRVU}</div>`;
            html += `<div class="col-md-6"><strong>Estimated Payment (Facility):</strong> $${totalPayment}</div>`;
            html += `</div>`;
            
            // Use per_code (list of PerCodeBilling) instead of breakdown
            if (fin.per_code && Array.isArray(fin.per_code) && fin.per_code.length > 0) {
                html += `<hr><h6>Per-Procedure Breakdown</h6>`;
                html += `<table class="table table-sm table-striped">`;
                html += `<thead><tr><th>CPT Code</th><th>Work RVU</th><th>Facility Pay</th></tr></thead>`;
                html += `<tbody>`;
                fin.per_code.forEach(proc => {
                    const workRVU = (proc.work_rvu !== undefined && proc.work_rvu !== null) ? proc.work_rvu.toFixed(2) : 'N/A';
                    // Use allowed_facility_payment
                    const payment = (proc.allowed_facility_payment !== undefined && proc.allowed_facility_payment !== null) ? proc.allowed_facility_payment.toFixed(2) : 'N/A';
                    
                    html += `<tr>`;
                    html += `<td><code>${proc.cpt_code || 'N/A'}</code></td>`;
                    html += `<td>${workRVU}</td>`;
                    html += `<td>$${payment}</td>`;
                    html += `</tr>`;
                });
                html += `</tbody></table>`;
            } else if (fin.total_work_rvu === 0 && fin.total_facility_payment === 0) {
                html += `<p class="text-muted mb-0">No RVU calculations available (no billable codes found).</p>`;
            }
            html += `</div></div>`;
        } else {
            html += `<hr><div class="alert alert-info">RVU calculations not available.</div>`;
        }

        if (lastResult.explanation) {
             html += `<hr><h5>Explanation</h5><pre>${lastResult.explanation}</pre>`;
        }
        
        // Display warnings if any
        if (lastResult.warnings && lastResult.warnings.length > 0) {
            html += `<hr><h5>Warnings</h5>`;
            html += `<div class="alert alert-warning">`;
            lastResult.warnings.forEach(warning => {
                html += `<div>${warning}</div>`;
            });
            html += `</div>`;
        }
        
        if (legacy) legacy.innerHTML = html;

    } else if (currentMode === 'registry') {
        // Replace the generic Object.entries + JSON.stringify behavior with a curated view model.
        const rows = buildRegistryDisplayRows(lastResult || {});
        const isCleanRecord = !!(lastResult?.clean_record || lastResult?.reviewed_record);

        let html = `
            <div class="d-flex align-items-center justify-content-between flex-wrap mb-3">
                <h4 id="registry-header-label" class="mb-0"></h4>
            </div>
            <table class="table table-striped table-sm mb-0">
                <thead><tr><th>Field</th><th>Value</th></tr></thead>
                <tbody id="registry-table-body"></tbody>
            </table>
        `;

        if (lastResult?.evidence) {
            html += `<h5 class="mt-4">Evidence</h5><pre>${JSON.stringify(lastResult.evidence, null, 2)}</pre>`;
        }

        if (legacy) legacy.innerHTML = html;

        const headerLabel = document.getElementById('registry-header-label');
        if (headerLabel) {
            headerLabel.textContent = isCleanRecord ?
                'Registry Record (formatted, human-reviewed)' :
                'Registry Record (formatted)';
        }

        const tbody = document.getElementById('registry-table-body');
        if (!tbody) return;
        tbody.innerHTML = '';

        if (!rows.length) {
            const tr = document.createElement('tr');
            const td = document.createElement('td');
            td.colSpan = 2;
            td.textContent = 'No registry fields available for display.';
            tr.appendChild(td);
            tbody.appendChild(tr);
        } else {
            rows.forEach(row => {
                const tr = document.createElement('tr');

                const tdField = document.createElement('td');
                tdField.textContent = row.field;

                const tdValue = document.createElement('td');
                tdValue.textContent = row.value;

                tr.appendChild(tdField);
                tr.appendChild(tdValue);
                tbody.appendChild(tr);
            });
        }

    } else if (currentMode === 'reporter') {
        const { extraction, verify, render, seed } = lastResult || {};
        const issues = render?.issues || verify?.issues || seed?.issues || [];
        const warnings = render?.warnings || verify?.warnings || seed?.warnings || [];
        const inferenceNotes = render?.inference_notes || verify?.inference_notes || seed?.inference_notes || [];
        const suggestions = render?.suggestions || verify?.suggestions || seed?.suggestions || [];
        const questions = (lastResult?.questions || seed?.questions || reporterBuilderState.questions || []);
        const activeBundle = render?.bundle || verify?.bundle || seed?.bundle || null;

        if (activeBundle) reporterBuilderState.bundle = activeBundle;
        reporterBuilderState.questions = Array.isArray(questions) ? questions : [];

        let html = "";

        html += `<h4>Reporter Flow</h4>`;
        html += `<p class="text-muted small">Registry extraction → seed/verify → questions → JSON Patch render loop</p>`;
        html += renderReporterQuestionsForm(reporterBuilderState.questions);

        if (render?.markdown) {
            html += `<div class="mb-3"><h5>Rendered Markdown</h5><div class="border p-3 bg-white">${marked.parse(render.markdown)}</div></div>`;
        } else if (seed?.markdown) {
            html += `<div class="mb-3"><h5>Rendered Markdown</h5><div class="border p-3 bg-white">${marked.parse(seed.markdown)}</div></div>`;
        } else {
            html += `<div class="alert alert-warning">Report not rendered yet (critical issues or validation warnings must be resolved).</div>`;
        }

        if (issues && issues.length) {
            html += `<h6>Issues</h6><ul class="list-group mb-3">`;
            issues.forEach(issue => {
                html += `<li class="list-group-item d-flex justify-content-between align-items-start">
                    <div>
                        <div><strong>${issue.proc_id || issue.proc_type}</strong>: ${issue.message || issue.field_path}</div>
                        <small class="text-muted">Severity: ${issue.severity}</small>
                    </div>
                    <span class="badge bg-${issue.severity === 'critical' ? 'danger' : 'warning'} text-uppercase">${issue.severity}</span>
                </li>`;
            });
            html += `</ul>`;
        }

        if (warnings && warnings.length) {
            html += `<h6>Warnings</h6><div class="alert alert-warning">${warnings.map(w => `<div>${w}</div>`).join("")}</div>`;
        }

        if (suggestions && suggestions.length) {
            html += `<h6>Suggestions</h6><div class="alert alert-info">${suggestions.map(s => `<div>${s}</div>`).join("")}</div>`;
        }

        if (inferenceNotes && inferenceNotes.length) {
            html += `<h6>Inference Notes</h6><div class="alert alert-info">${inferenceNotes.map(n => `<div>${n}</div>`).join("")}</div>`;
        }

        if (activeBundle) {
            html += `<h6>Bundle</h6><pre class="bg-light p-2 border rounded">${JSON.stringify(activeBundle, null, 2)}</pre>`;
        }

        if (extraction) {
            html += `<h6>Registry Extraction</h6><pre class="bg-light p-2 border rounded">${JSON.stringify(extraction, null, 2)}</pre>`;
        }

        if (legacy) legacy.innerHTML = html;
    }
}

// Check API status on load
fetch('/health')
    .then(r => r.json())
    .then(data => {
        const badge = document.getElementById('api-status');
        if (data.ok) {
            badge.className = 'badge bg-success';
            badge.textContent = 'API Online';
        } else {
            badge.className = 'badge bg-danger';
            badge.textContent = 'API Error';
        }
    })
    .catch(() => {
        const badge = document.getElementById('api-status');
        badge.className = 'badge bg-danger';
        badge.textContent = 'API Offline';
    });

// Ensure reporter template dropdown includes all options even if the HTML was cached
ensureReporterTemplates();
