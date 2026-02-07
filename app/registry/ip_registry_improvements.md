# IP Registry Improvements: Granular Per-Site Data Model

## Executive Summary

This document outlines comprehensive improvements to the IP Registry system to capture granular, per-site/per-node data for all major interventional pulmonology procedures. The key principle is moving from aggregate fields (e.g., `ebus_stations_sampled: ["4R", "7"]`) to detailed per-entity arrays that capture the full clinical context for each site.

---

## 1. EBUS Station-Level Detail (Highest Priority)

### Current Limitation
The existing schema captures EBUS as aggregate data:
- `stations_sampled: ["4R", "7", "11L"]` - no per-station detail
- Single `needle_gauge`, `passes_per_station` - applied globally
- ROSE result is global, not per-station

### Proposed Enhancement

```json
"linear_ebus_stations_detail": {
  "type": ["array", "null"],
  "description": "Per-station EBUS-TBNA data for mediastinal staging",
  "items": {
    "type": "object",
    "properties": {
      "station": {
        "type": "string",
        "enum": ["2R", "2L", "3p", "4R", "4L", "7", "10R", "10L", "11R", "11L", "12R", "12L"],
        "description": "IASLC lymph node station"
      },
      "short_axis_mm": {
        "type": ["number", "null"],
        "minimum": 0,
        "description": "Short-axis diameter in mm"
      },
      "long_axis_mm": {
        "type": ["number", "null"],
        "minimum": 0
      },
      "shape": {
        "type": ["string", "null"],
        "enum": ["oval", "round", "irregular", null],
        "description": "Node shape on EBUS"
      },
      "margin": {
        "type": ["string", "null"],
        "enum": ["distinct", "indistinct", "irregular", null],
        "description": "Node border definition"
      },
      "echogenicity": {
        "type": ["string", "null"],
        "enum": ["homogeneous", "heterogeneous", null],
        "description": "Internal echo pattern"
      },
      "chs_present": {
        "type": ["boolean", "null"],
        "description": "Central hilar structure present (benign marker)"
      },
      "necrosis_present": {
        "type": ["boolean", "null"],
        "description": "Central necrosis or coagulation necrosis sign"
      },
      "calcification_present": {
        "type": ["boolean", "null"]
      },
      "elastography_performed": {
        "type": ["boolean", "null"]
      },
      "elastography_score": {
        "type": ["integer", "null"],
        "minimum": 1,
        "maximum": 5,
        "description": "Elastography score (1-5 scale)"
      },
      "elastography_strain_ratio": {
        "type": ["number", "null"],
        "description": "Strain ratio if measured"
      },
      "elastography_pattern": {
        "type": ["string", "null"],
        "enum": ["predominantly_blue", "blue_green", "green", "predominantly_green", null],
        "description": "Qualitative color pattern"
      },
      "doppler_performed": {
        "type": ["boolean", "null"]
      },
      "doppler_pattern": {
        "type": ["string", "null"],
        "enum": ["avascular", "hilar_vessel", "peripheral", "mixed", null],
        "description": "Vascular pattern on Doppler"
      },
      "morphologic_impression": {
        "type": ["string", "null"],
        "enum": ["benign", "suspicious", "malignant", "indeterminate", null],
        "description": "Overall EBUS morphologic impression (NOT pathology)"
      },
      "sampled": {
        "type": "boolean",
        "default": true,
        "description": "Whether this station was actually sampled"
      },
      "needle_gauge": {
        "type": ["integer", "null"],
        "enum": [19, 21, 22, 25, null]
      },
      "needle_type": {
        "type": ["string", "null"],
        "enum": ["Standard FNA", "FNB/ProCore", "Acquire", "ViziShot Flex", null],
        "description": "Needle type/brand"
      },
      "number_of_passes": {
        "type": ["integer", "null"],
        "minimum": 1,
        "maximum": 10,
        "description": "Number of needle passes at this station"
      },
      "intranodal_forceps_used": {
        "type": ["boolean", "null"],
        "description": "Mini-forceps biopsy performed"
      },
      "rose_performed": {
        "type": ["boolean", "null"]
      },
      "rose_result": {
        "type": ["string", "null"],
        "enum": ["Adequate lymphocytes", "Malignant", "Suspicious for malignancy", "Atypical cells", "Granuloma", "Necrosis only", "Nondiagnostic", "Deferred", null],
        "description": "ROSE interpretation for this station"
      },
      "rose_adequacy": {
        "type": ["boolean", "null"],
        "description": "ROSE confirmed adequate sample"
      },
      "specimen_sent_for": {
        "type": ["array", "null"],
        "items": {
          "type": "string",
          "enum": ["Cytology", "Cell block", "Flow cytometry", "Molecular/NGS", "Culture", "AFB", "Fungal", "Research"]
        }
      },
      "final_pathology": {
        "type": ["string", "null"],
        "description": "Final pathology result for this station"
      },
      "n_stage_contribution": {
        "type": ["string", "null"],
        "enum": ["N0", "N1", "N2", "N3", null],
        "description": "If malignant, what N stage does this station represent"
      },
      "notes": {
        "type": ["string", "null"]
      }
    },
    "required": ["station"]
  }
}
```

### Prompt Instruction for LLM

```python
"linear_ebus_stations_detail": """
Extract detailed per-station EBUS-TBNA data. Create one object per lymph node station that was sampled OR described with morphology.

For each station, extract:
- station: Use IASLC naming (2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L)
- short_axis_mm: Convert "0.54 cm" → 5.4, "5mm" → 5
- shape: Map "oval/elliptical" → "oval", "round/spherical" → "round"
- echogenicity: "homogeneous" or "heterogeneous"
- chs_present: true if "CHS present/preserved", false if "absent/absent CHS"
- morphologic_impression: Based on EBUS features ONLY (not ROSE/path)
- number_of_passes: Extract exact number if stated
- rose_result: ROSE finding for THIS station specifically
- needle_gauge/type: May be same for all stations or vary

Size parsing examples:
- "5.4 mm lymph node" → short_axis_mm: 5.4
- "1.2 x 0.8 cm" → short_axis_mm: 8 (shorter dimension)
- "subcentimeter" → short_axis_mm: null (don't guess)

Morphologic impression rules:
- "benign" if: oval + CHS present + <10mm + homogeneous
- "malignant" if: round + CHS absent + ≥10mm + heterogeneous
- "indeterminate" if: mixed features
- null if: morphology not described

Return empty array [] if no linear EBUS performed or this is radial EBUS only.""".strip()
```

---

## 2. Navigation/Robotic Per-Target Data

### Current Limitation
Navigation procedures with multiple targets are captured as single entities with aggregate data.

### Proposed Enhancement

```json
"navigation_targets": {
  "type": ["array", "null"],
  "description": "Per-target data for navigation/robotic bronchoscopy",
  "items": {
    "type": "object",
    "properties": {
      "target_number": {
        "type": "integer",
        "minimum": 1,
        "description": "Sequential target number in procedure order"
      },
      "target_location_text": {
        "type": "string",
        "description": "Full anatomic description (e.g., 'RUL apical segment')"
      },
      "target_lobe": {
        "type": ["string", "null"],
        "enum": ["RUL", "RML", "RLL", "LUL", "LLL", "Lingula", null]
      },
      "target_segment": {
        "type": ["string", "null"],
        "description": "Segment name or number (e.g., 'apical', 'S1', 'posterior basal')"
      },
      "lesion_size_mm": {
        "type": ["number", "null"],
        "description": "Target lesion size in mm"
      },
      "distance_from_pleura_mm": {
        "type": ["number", "null"],
        "description": "Distance from visceral pleura"
      },
      "bronchus_sign": {
        "type": ["string", "null"],
        "enum": ["Positive", "Negative", "Not assessed", null]
      },
      "ct_characteristics": {
        "type": ["string", "null"],
        "enum": ["Solid", "Part-solid", "Ground-glass", "Cavitary", "Calcified", null]
      },
      "pet_suv_max": {
        "type": ["number", "null"]
      },
      "registration_error_mm": {
        "type": ["number", "null"],
        "description": "CT-to-body registration error for this target"
      },
      "navigation_successful": {
        "type": ["boolean", "null"],
        "description": "Successfully reached target location"
      },
      "rebus_used": {
        "type": ["boolean", "null"],
        "description": "Radial EBUS used for this target"
      },
      "rebus_view": {
        "type": ["string", "null"],
        "enum": ["Concentric", "Eccentric", "Adjacent", "Not visualized", null]
      },
      "rebus_lesion_appearance": {
        "type": ["string", "null"],
        "description": "Radial EBUS appearance (e.g., 'hyperechoic solid lesion')"
      },
      "tool_in_lesion_confirmed": {
        "type": ["boolean", "null"]
      },
      "confirmation_method": {
        "type": ["string", "null"],
        "enum": ["CBCT", "Augmented fluoroscopy", "Fluoroscopy", "Radial EBUS", "None", null]
      },
      "cbct_til_confirmed": {
        "type": ["boolean", "null"],
        "description": "Cone-beam CT confirmed tool-in-lesion"
      },
      "sampling_tools_used": {
        "type": ["array", "null"],
        "items": {
          "type": "string",
          "enum": ["Forceps", "Needle (21G)", "Needle (19G)", "Brush", "Cryoprobe (1.1mm)", "Cryoprobe (1.7mm)", "Cryoprobe (1.9mm)", "NeedleInNeedle"]
        }
      },
      "number_of_forceps_biopsies": {
        "type": ["integer", "null"]
      },
      "number_of_needle_passes": {
        "type": ["integer", "null"]
      },
      "number_of_cryo_biopsies": {
        "type": ["integer", "null"]
      },
      "rose_performed": {
        "type": ["boolean", "null"]
      },
      "rose_result": {
        "type": ["string", "null"]
      },
      "immediate_complication": {
        "type": ["string", "null"],
        "enum": ["None", "Bleeding - mild", "Bleeding - moderate", "Bleeding - severe", "Pneumothorax", null]
      },
      "bleeding_management": {
        "type": ["string", "null"],
        "description": "How bleeding was controlled if applicable"
      },
      "specimen_sent_for": {
        "type": ["array", "null"],
        "items": {
          "type": "string"
        }
      },
      "final_pathology": {
        "type": ["string", "null"]
      },
      "notes": {
        "type": ["string", "null"]
      }
    },
    "required": ["target_number", "target_location_text"]
  }
}
```

---

## 3. CAO Intervention Per-Site (Enhanced)

### Current Schema Additions

```json
"cao_interventions": {
  "type": ["array", "null"],
  "description": "Per-site central airway obstruction intervention data",
  "items": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "enum": ["Trachea - proximal", "Trachea - mid", "Trachea - distal", "Carina", "RMS", "LMS", "BI", "RUL", "RML", "RLL", "LUL", "LLL", "Other"]
      },
      "obstruction_type": {
        "type": ["string", "null"],
        "enum": ["Intraluminal", "Extrinsic", "Mixed", null]
      },
      "etiology": {
        "type": ["string", "null"],
        "enum": ["Malignant - primary lung", "Malignant - metastatic", "Malignant - other", "Benign - post-intubation", "Benign - post-tracheostomy", "Benign - anastomotic", "Benign - inflammatory", "Benign - granulation", "Benign - web/stenosis", "Other", null]
      },
      "length_mm": {
        "type": ["number", "null"],
        "description": "Length of obstruction/stenosis"
      },
      "pre_obstruction_pct": {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 100,
        "description": "Pre-procedure obstruction percentage"
      },
      "post_obstruction_pct": {
        "type": ["integer", "null"],
        "minimum": 0,
        "maximum": 100,
        "description": "Post-procedure residual obstruction"
      },
      "pre_diameter_mm": {
        "type": ["number", "null"],
        "description": "Pre-procedure airway diameter"
      },
      "post_diameter_mm": {
        "type": ["number", "null"],
        "description": "Post-procedure airway diameter"
      },
      "modalities_used": {
        "type": ["array", "null"],
        "items": {
          "type": "object",
          "properties": {
            "modality": {
              "type": "string",
              "enum": ["APC", "Electrocautery - snare", "Electrocautery - knife", "Electrocautery - probe", "Cryotherapy - spray", "Cryotherapy - contact", "Cryoextraction", "Laser - Nd:YAG", "Laser - CO2", "Laser - diode", "Mechanical debulking", "Rigid coring", "Microdebrider", "Balloon dilation", "PDT"]
            },
            "power_setting_watts": {
              "type": ["number", "null"]
            },
            "apc_flow_rate_lpm": {
              "type": ["number", "null"],
              "description": "APC argon flow rate L/min"
            },
            "balloon_diameter_mm": {
              "type": ["number", "null"]
            },
            "balloon_pressure_atm": {
              "type": ["number", "null"]
            },
            "freeze_time_seconds": {
              "type": ["integer", "null"],
              "description": "For cryotherapy"
            },
            "number_of_applications": {
              "type": ["integer", "null"]
            },
            "duration_seconds": {
              "type": ["integer", "null"]
            }
          },
          "required": ["modality"]
        }
      },
      "hemostasis_required": {
        "type": ["boolean", "null"]
      },
      "hemostasis_methods": {
        "type": ["array", "null"],
        "items": {
          "type": "string",
          "enum": ["Cold saline", "Epinephrine", "APC", "Electrocautery", "Balloon tamponade", "Bronchial blocker", "Tranexamic acid"]
        }
      },
      "secretions_present": {
        "type": ["boolean", "null"],
        "description": "Post-obstructive secretions/pus"
      },
      "secretions_drained": {
        "type": ["boolean", "null"]
      },
      "stent_placed": {
        "type": ["boolean", "null"]
      },
      "notes": {
        "type": ["string", "null"]
      }
    },
    "required": ["location"]
  }
}
```

---

## 4. BLVR Per-Valve/Segment Detail

```json
"blvr_valve_placements": {
  "type": ["array", "null"],
  "description": "Individual valve placement data",
  "items": {
    "type": "object",
    "properties": {
      "valve_number": {
        "type": "integer",
        "minimum": 1
      },
      "target_lobe": {
        "type": "string",
        "enum": ["RUL", "RML", "RLL", "LUL", "LLL", "Lingula"]
      },
      "segment": {
        "type": "string",
        "description": "Specific segment (e.g., 'LB1+2', 'LB6', 'apical')"
      },
      "airway_diameter_mm": {
        "type": ["number", "null"],
        "description": "Measured airway diameter"
      },
      "valve_size": {
        "type": "string",
        "description": "Valve size (e.g., '4.0', '4.0-LP', '5.5', '6.0', '7.0')"
      },
      "valve_type": {
        "type": "string",
        "enum": ["Zephyr (Pulmonx)", "Spiration (Olympus)"]
      },
      "deployment_method": {
        "type": ["string", "null"],
        "enum": ["Standard", "Retroflexed", null]
      },
      "deployment_successful": {
        "type": "boolean"
      },
      "seal_confirmed": {
        "type": ["boolean", "null"],
        "description": "Visual confirmation of valve seal"
      },
      "repositioned": {
        "type": ["boolean", "null"],
        "description": "Valve required repositioning"
      },
      "notes": {
        "type": ["string", "null"]
      }
    },
    "required": ["valve_number", "target_lobe", "segment", "valve_size", "valve_type", "deployment_successful"]
  }
},
"blvr_chartis_measurements": {
  "type": ["array", "null"],
  "items": {
    "type": "object",
    "properties": {
      "lobe_assessed": {
        "type": "string",
        "enum": ["RUL", "RML", "RLL", "LUL", "LLL", "Lingula"]
      },
      "segment_assessed": {
        "type": ["string", "null"]
      },
      "measurement_duration_seconds": {
        "type": ["integer", "null"]
      },
      "adequate_seal": {
        "type": ["boolean", "null"]
      },
      "cv_result": {
        "type": "string",
        "enum": ["CV Negative", "CV Positive", "Indeterminate", "Low flow", "No seal", "Aborted"]
      },
      "flow_pattern_description": {
        "type": ["string", "null"]
      },
      "notes": {
        "type": ["string", "null"]
      }
    },
    "required": ["lobe_assessed", "cv_result"]
  }
}
```

---

## 5. Specimen Tracking System

A unified system to track all specimens and link them to their source locations:

```json
"specimens_collected": {
  "type": ["array", "null"],
  "description": "All specimens collected during procedure with source linkage",
  "items": {
    "type": "object",
    "properties": {
      "specimen_number": {
        "type": "integer",
        "minimum": 1
      },
      "source_procedure": {
        "type": "string",
        "enum": ["EBUS-TBNA", "Navigation biopsy", "Endobronchial biopsy", "Transbronchial biopsy", "Transbronchial cryobiopsy", "BAL", "Bronchial wash", "Brushing", "Pleural biopsy", "Pleural fluid", "Other"]
      },
      "source_location": {
        "type": "string",
        "description": "Anatomic location (e.g., '4R', 'RUL apical', 'Left pleura')"
      },
      "collection_tool": {
        "type": ["string", "null"],
        "description": "Tool used (e.g., '22G FNA', 'Forceps', '1.9mm cryoprobe')"
      },
      "specimen_count": {
        "type": ["integer", "null"],
        "minimum": 1
      },
      "specimen_adequacy": {
        "type": ["string", "null"],
        "enum": ["Adequate", "Limited", "Inadequate", "Pending", null]
      },
      "destinations": {
        "type": ["array", "null"],
        "items": {
          "type": "string",
          "enum": ["Histology/Surgical pathology", "Cytology", "Cell block", "Flow cytometry", "Molecular/NGS", "PD-L1", "Bacterial culture", "AFB culture", "Fungal culture", "Viral studies", "Research protocol", "Biobank"]
        }
      },
      "rose_performed": {
        "type": ["boolean", "null"]
      },
      "rose_result": {
        "type": ["string", "null"]
      },
      "final_pathology_diagnosis": {
        "type": ["string", "null"]
      },
      "molecular_markers": {
        "type": ["object", "null"],
        "additionalProperties": true,
        "description": "Key:value pairs for molecular results (e.g., EGFR: L858R)"
      },
      "notes": {
        "type": ["string", "null"]
      }
    },
    "required": ["specimen_number", "source_procedure", "source_location"]
  }
}
```

---

## 6. Thoracoscopy Per-Site Findings

```json
"thoracoscopy_findings_detail": {
  "type": ["array", "null"],
  "items": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "enum": ["Parietal pleura - chest wall", "Parietal pleura - diaphragm", "Parietal pleura - mediastinum", "Visceral pleura", "Lung parenchyma", "Costophrenic angle", "Apex"]
      },
      "finding_type": {
        "type": "string",
        "enum": ["Normal", "Nodules", "Plaques", "Studding", "Mass", "Adhesions - filmy", "Adhesions - dense", "Inflammation", "Thickening", "Trapped lung", "Loculations", "Empyema", "Other"]
      },
      "extent": {
        "type": ["string", "null"],
        "enum": ["Focal", "Multifocal", "Diffuse", null]
      },
      "size_description": {
        "type": ["string", "null"]
      },
      "biopsied": {
        "type": ["boolean", "null"]
      },
      "number_of_biopsies": {
        "type": ["integer", "null"]
      },
      "biopsy_tool": {
        "type": ["string", "null"],
        "enum": ["Rigid forceps", "Flexible forceps", "Cryoprobe", null]
      },
      "impression": {
        "type": ["string", "null"],
        "enum": ["Benign appearing", "Malignant appearing", "Infectious appearing", "Indeterminate", null]
      },
      "notes": {
        "type": ["string", "null"]
      }
    },
    "required": ["location", "finding_type"]
  }
}
```

---

## 7. Cryobiopsy Per-Site Detail

```json
"cryobiopsy_sites": {
  "type": ["array", "null"],
  "description": "Per-site transbronchial cryobiopsy data",
  "items": {
    "type": "object",
    "properties": {
      "site_number": {
        "type": "integer",
        "minimum": 1
      },
      "lobe": {
        "type": "string",
        "enum": ["RUL", "RML", "RLL", "LUL", "LLL", "Lingula"]
      },
      "segment": {
        "type": ["string", "null"]
      },
      "distance_from_pleura": {
        "type": ["string", "null"],
        "enum": [">2cm", "1-2cm", "<1cm", "Not documented", null]
      },
      "fluoroscopy_position": {
        "type": ["string", "null"],
        "description": "Position relative to chest wall on fluoro"
      },
      "radial_ebus_used": {
        "type": ["boolean", "null"]
      },
      "rebus_view": {
        "type": ["string", "null"],
        "description": "Radial EBUS finding (e.g., 'parenchymal pattern, no large vessels')"
      },
      "probe_size_mm": {
        "type": ["number", "null"],
        "enum": [1.1, 1.7, 1.9, 2.4, null]
      },
      "freeze_time_seconds": {
        "type": ["integer", "null"],
        "minimum": 1,
        "maximum": 10
      },
      "number_of_biopsies": {
        "type": ["integer", "null"],
        "minimum": 1
      },
      "specimen_size_mm": {
        "type": ["number", "null"],
        "description": "Approximate specimen size"
      },
      "blocker_used": {
        "type": ["boolean", "null"]
      },
      "blocker_type": {
        "type": ["string", "null"],
        "enum": ["Fogarty", "Arndt", "Cohen", "Cryoprobe sheath", null]
      },
      "bleeding_severity": {
        "type": ["string", "null"],
        "enum": ["None/Scant", "Mild", "Moderate", "Severe", null]
      },
      "bleeding_controlled_with": {
        "type": ["string", "null"]
      },
      "pneumothorax_after_site": {
        "type": ["boolean", "null"]
      },
      "notes": {
        "type": ["string", "null"]
      }
    },
    "required": ["site_number", "lobe"]
  }
}
```

---

## Implementation Recommendations

### 1. Schema Updates (schema.py)

Add Pydantic models for the new nested structures:

```python
class EBUSStationDetail(BaseModel):
    """Per-station EBUS-TBNA data."""
    station: str
    short_axis_mm: float | None = None
    long_axis_mm: float | None = None
    shape: Literal["oval", "round", "irregular"] | None = None
    margin: Literal["distinct", "indistinct", "irregular"] | None = None
    echogenicity: Literal["homogeneous", "heterogeneous"] | None = None
    chs_present: bool | None = None
    necrosis_present: bool | None = None
    elastography_performed: bool | None = None
    elastography_score: int | None = None
    elastography_pattern: str | None = None
    doppler_performed: bool | None = None
    doppler_pattern: str | None = None
    morphologic_impression: Literal["benign", "suspicious", "malignant", "indeterminate"] | None = None
    sampled: bool = True
    needle_gauge: int | None = None
    needle_type: str | None = None
    number_of_passes: int | None = None
    intranodal_forceps_used: bool | None = None
    rose_performed: bool | None = None
    rose_result: str | None = None
    rose_adequacy: bool | None = None
    specimen_sent_for: list[str] | None = None
    final_pathology: str | None = None
    n_stage_contribution: str | None = None
    notes: str | None = None


class NavigationTarget(BaseModel):
    """Per-target navigation/robotic bronchoscopy data."""
    target_number: int
    target_location_text: str
    target_lobe: str | None = None
    target_segment: str | None = None
    lesion_size_mm: float | None = None
    distance_from_pleura_mm: float | None = None
    bronchus_sign: Literal["Positive", "Negative", "Not assessed"] | None = None
    ct_characteristics: str | None = None
    pet_suv_max: float | None = None
    registration_error_mm: float | None = None
    navigation_successful: bool | None = None
    rebus_used: bool | None = None
    rebus_view: Literal["Concentric", "Eccentric", "Adjacent", "Not visualized"] | None = None
    rebus_lesion_appearance: str | None = None
    tool_in_lesion_confirmed: bool | None = None
    confirmation_method: str | None = None
    cbct_til_confirmed: bool | None = None
    sampling_tools_used: list[str] | None = None
    number_of_forceps_biopsies: int | None = None
    number_of_needle_passes: int | None = None
    number_of_cryo_biopsies: int | None = None
    rose_performed: bool | None = None
    rose_result: str | None = None
    immediate_complication: str | None = None
    bleeding_management: str | None = None
    final_pathology: str | None = None
    notes: str | None = None
```

### 2. Prompt Engineering

Update the system prompt to explicitly request per-site arrays:

```python
GRANULAR_EXTRACTION_RULES = """
### CRITICAL: Per-Site Data Extraction

For procedures with multiple sites (EBUS stations, navigation targets, CAO locations), 
create an array with one object per site containing ALL available data for that site.

#### EBUS-TBNA: Create `linear_ebus_stations_detail` array
- One object per station sampled or described
- Include morphology (size, shape, echogenicity, CHS) per station
- Include sampling details (passes, needle, ROSE) per station
- Even if same needle used throughout, repeat in each station object

#### Navigation/Robotic: Create `navigation_targets` array  
- One object per target lesion
- Include target characteristics (size, location, CT features)
- Include confirmation method and result per target
- Include sampling tools and counts per target

#### CAO: Create `cao_interventions` array
- One object per airway location treated
- Include pre/post obstruction percentages per site
- Include modalities array with settings for each modality used at that site

Do NOT aggregate data across sites. Each site should be independently complete.
"""
```

### 3. Validation Rules

Add cross-field validation:

```python
def validate_ebus_stations_detail(data: dict) -> list[str]:
    """Validate per-station EBUS data consistency."""
    errors = []
    stations_detail = data.get("linear_ebus_stations_detail", [])
    stations_sampled = data.get("linear_ebus_stations", [])
    
    if stations_detail and stations_sampled:
        detail_stations = {s["station"] for s in stations_detail if s.get("sampled", True)}
        sampled_set = set(stations_sampled)
        
        if detail_stations != sampled_set:
            missing = sampled_set - detail_stations
            extra = detail_stations - sampled_set
            if missing:
                errors.append(f"Stations in list but missing detail: {missing}")
            if extra:
                errors.append(f"Stations with detail but not in sampled list: {extra}")
    
    return errors
```

---

## Benefits of This Approach

1. **Research-Quality Data**: Per-site granularity enables meaningful clinical research on EBUS morphology, navigation accuracy, CAO outcomes
2. **Quality Metrics**: Track per-station adequacy, per-target diagnostic yield, per-site complication rates
3. **Billing Accuracy**: Detailed per-site data supports accurate CPT coding (different codes for different sites)
4. **Training/QI**: Identify patterns in technique that correlate with outcomes
5. **Pathology Correlation**: Link EBUS morphologic impression to final pathology per-station for ML training
