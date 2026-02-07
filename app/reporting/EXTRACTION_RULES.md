# Procedure Extraction Rules

These rules MUST be followed when extracting procedure data from dictated notes into a ProcedureBundle.

## Rule 1: No Hallucinated Values

**NEVER guess or invent values not explicitly stated in the source text.**

### What to do:
- If a value is not explicitly stated, set the field to `null`
- Add the missing field to `bundle.acknowledged_omissions[proc_id]`
- Do NOT use defaults for clinical values that should be documented

### Examples of values that should be `null` if not stated:
- Ventilation parameters (mode, RR, TV, PEEP, FiO₂)
- Needle gauge
- Number of passes (unless explicitly counted)
- Fluoroscopy use (unless mentioned)
- Lidocaine volumes
- Balloon sizes
- Exact anatomical segments

### Example:
```json
{
  "procedures": [
    {
      "proc_type": "robotic_bronchoscopy_ion",
      "proc_id": "robotic_ion_1",
      "sequence": 1,
      "data": {
        "vent_mode": null,
        "vent_rr": null,
        "vent_tv_ml": null,
        "vent_peep_cm_h2o": null,
        "vent_fio2_pct": null,
        "cbct_performed": true,
        "radial_pattern": "concentric",
        "notes": "8.5 ETT; partial registration"
      }
    }
  ],
  "acknowledged_omissions": {
    "robotic_ion_1": [
      "Ventilation parameters (mode/RR/VT/PEEP/FiO₂) not provided",
      "Exact segmental location not specified"
    ],
    "global": [
      "Referring physician",
      "Pre- and post-op ICD-10 codes"
    ]
  }
}
```

## Rule 2: Chronological Ordering

**Maintain the order procedures appear in the source dictation.**

### What to do:
- Walk through the source text left-to-right
- Assign `sequence` numbers (1, 2, 3...) in order of mention
- Do NOT reorder by CPT code, procedure type, or any other criterion
- The engine will render in sequence order

### Typical chronological flow for robotic bronchoscopy:
1. Airway setup (ETT placement)
2. Registration
3. Navigation to target
4. Radial EBUS confirmation
5. CBCT spin (if used)
6. Tool-in-lesion confirmation
7. Sampling (needle biopsies)
8. Sampling (cryobiopsies if done)
9. Hemostasis confirmation
10. Scope withdrawal

### Example:
```json
{
  "procedures": [
    {"proc_type": "ion_registration_partial", "sequence": 1, ...},
    {"proc_type": "radial_ebus_survey", "sequence": 2, ...},
    {"proc_type": "cbct_assisted_bronchoscopy", "sequence": 3, ...},
    {"proc_type": "transbronchial_needle_aspiration", "sequence": 4, ...},
    {"proc_type": "transbronchial_cryobiopsy", "sequence": 5, ...}
  ]
}
```

## Rule 3: Correct Biopsy Mapping

**Map sampling procedures to the correct macro based on the actual modality used.**

### Mapping rules:

| Dictation Language | Correct Macro | WRONG Macro |
|-------------------|---------------|-------------|
| "cryobiopsy", "cryoprobe", "cryo samples" | `transbronchial_cryobiopsy` | `transbronchial_lung_biopsy` |
| "needle biopsy", "TBNA", "21G needle" | `transbronchial_needle_aspiration` | `transbronchial_lung_biopsy` |
| "forceps biopsy", "forceps samples" | `transbronchial_lung_biopsy` | - |
| "EBUS needle", "EBUS-TBNA" | `linear_ebus_tbna` | `transbronchial_needle_aspiration` |

### Counting rules:

1. **Keep counts separate by modality**:
   - `num_passes` for needle procedures
   - `num_samples` for cryo procedures
   - Do NOT sum across modalities

2. **Do not double-count**:
   - If text says "5 needle passes, 3 cryobiopsies" → create TWO separate procedure entries
   - Do NOT create a single entry with "8 biopsies"

3. **If count is ambiguous**, use `null` and add to acknowledged_omissions

### Example - Correct extraction:
Input: "5 biopsies with 21G Ion needle. ROSE malignant. 3 cryobiopsies obtained."

```json
{
  "procedures": [
    {
      "proc_type": "transbronchial_needle_aspiration",
      "sequence": 4,
      "data": {
        "num_passes": 5,
        "needle_tool": "21G Ion needle",
        "rose_result": "malignant"
      }
    },
    {
      "proc_type": "transbronchial_cryobiopsy",
      "sequence": 5,
      "data": {
        "num_samples": 3
      }
    }
  ]
}
```

### Example - WRONG extraction:
```json
{
  "procedures": [
    {
      "proc_type": "transbronchial_lung_biopsy",
      "data": {
        "num_samples": 8,
        "forceps_tool": "cryoprobe"
      }
    }
  ]
}
```
This is wrong because:
- Combined needle + cryo counts into one number
- Used wrong proc_type (transbronchial_lung_biopsy instead of separate needle/cryo)
- Mislabeled cryoprobe as "forceps_tool"

## Rule 4: Guidance Method Attribution

**Only attribute guidance methods that are explicitly mentioned.**

| If text mentions... | Set guidance to... |
|--------------------|--------------------|
| "fluoroscopy", "under fluoro" | "fluoroscopy" |
| "CBCT", "cone beam" | "cbct" |
| "radial EBUS", "rEBUS" | "radial_ebus" |
| "navigation", "navigated" | "navigation" |
| Nothing about guidance | `null` |

Do NOT default to "fluoroscopy" for biopsy procedures unless explicitly stated.

## Rule 5: Ion/Robotic Bronchoscopy Specifics

For Ion robotic cases, structure data correctly:

1. **Registration** should be a separate procedure entry:
   - `ion_registration_complete` - full registration
   - `ion_registration_partial` - partial/efficiency strategy
   - `ion_registration_drift` - if drift/mismatch noted

2. **Sampling through Ion** can be:
   - Part of `robotic_bronchoscopy_ion.sampling_details` (narrative), OR
   - Separate procedure entries (preferred for counting)

3. **CBCT spins** should be tracked:
   - `cbct_performed: true` if any spin
   - `cbct_for_adjustment: true` if spin was for trajectory correction
   - Include in notes: "TIL confirmed by CBCT"

4. **ETT size** goes in notes field, not vent_mode:
   - WRONG: `"vent_mode": "ETT"`
   - RIGHT: `"notes": "8.5 ETT; ..."`

## Rule 6: Two-Phase Workflow

The extraction system supports a two-phase workflow for iterative refinement:

### Phase 1: Initial Sketch → Bundle + Note + Missing Fields

1. User provides initial dictation/sketch
2. LLM extracts to bundle (with `acknowledged_omissions` for missing fields)
3. Engine renders report + generates "Missing Details" summary
4. UI shows:
   - The rendered note with inline `[placeholder]` blanks
   - A "Missing / incomplete details" list
   - Prompt: _"To fill any of these, provide additional natural language."_

### Phase 2: Clarification Text → Patched Bundle

When user provides additional information:

1. Call `update_bundle()` with existing bundle + new extractions
2. **Rules for Phase 2 updates:**
   - Only fill fields explicitly supported by the new text
   - Don't reorder procedures (preserve sequence)
   - Don't change counts unless `allow_override=True`
   - Don't add new procedures unless `allow_new_procedures=True`
3. Engine re-renders markdown with updated bundle
4. Cleared fields are removed from `acknowledged_omissions`

### Example Phase 2 Update

**User clarification:** _"Segment RB1. Vent VC 14/450/5/40%. Specimens to histology, cultures."_

**Update dict:**
```json
{
  "procedures": [
    {
      "proc_id": "robotic_ion_1",
      "params": {
        "lesion_location": "RB1",
        "vent_params": {
          "mode": "VC",
          "respiratory_rate": 14,
          "tidal_volume": "450 mL",
          "peep": "5 cmH2O",
          "fio2": "40%"
        },
        "testing_types": ["histology", "cultures"]
      }
    }
  ],
  "acknowledged_omissions": {
    "robotic_ion_1": []
  }
}
```

### API Functions

```python
from proc_report.macro_engine import (
    render_bundle_with_summary,  # Phase 1: render + get missing summary
    update_bundle,               # Phase 2: merge updates into existing
    get_missing_fields_summary,  # Get formatted missing fields text
)

# Phase 1
report, missing_summary = render_bundle_with_summary(bundle)

# Phase 2
updated_bundle = update_bundle(
    existing_bundle=bundle,
    updates=extracted_updates,
    allow_override=False,      # Don't overwrite existing values
    allow_new_procedures=False # Don't add new procedures
)
report2 = render_procedure_bundle(updated_bundle)
```

## Summary Checklist

Before finalizing extraction:

- [ ] No invented numerical values (vent settings, counts, sizes)
- [ ] All missing important values added to acknowledged_omissions
- [ ] Procedures ordered by sequence from source text
- [ ] Separate procedure entries for needle vs cryo biopsies
- [ ] Counts not summed across modalities
- [ ] Guidance method only if explicitly mentioned
- [ ] Cryobiopsy mapped to `transbronchial_cryobiopsy`, not `transbronchial_lung_biopsy`
- [ ] Use `update_bundle()` for Phase 2 clarifications (don't rebuild from scratch)
