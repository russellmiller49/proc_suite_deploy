"""Registry data structures with granular per-site models.

This module extends the base schema with detailed per-site/per-node data structures
for EBUS, Navigation, CAO, BLVR, Cryobiopsy, and Thoracoscopy procedures.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# =============================================================================
# Pleural Procedure Overrides
# =============================================================================

class IPCProcedure(BaseModel):
    """Structured model for indwelling pleural catheter actions."""

    model_config = ConfigDict(extra="ignore")

    performed: bool | None = None
    action: Literal["Insertion", "Removal", "Fibrinolytic instillation"] | None = None
    side: Literal["Right", "Left"] | None = None
    catheter_brand: Literal["PleurX", "Aspira", "Rocket", "Other"] | None = None
    indication: Literal[
        "Malignant effusion",
        "Recurrent benign effusion",
        "Hepatic hydrothorax",
        "Heart failure",
        "Other",
    ] | None = None
    tunneled: bool | None = None

    @field_validator("action", mode="before")
    @classmethod
    def normalize_ipc_action(cls, v):
        """Map common IPC action synonyms into the constrained enum."""
        if v is None:
            return v
        s = str(v).strip().lower()
        if "tunneled" in s or "ipc" in s or "indwelling catheter" in s:
            return "Insertion"
        if "removal" in s or "removed" in s or "pull" in s:
            return "Removal"
        if "tpa" in s or "fibrinolytic" in s or "alteplase" in s:
            return "Fibrinolytic instillation"
        return v


class ClinicalContext(BaseModel):
    """Structured clinical context data with normalized bronchus sign."""

    model_config = ConfigDict(extra="ignore")

    class TargetLesion(BaseModel):
        """Structured target lesion disease burden fields (axes, morphology, SUV, location)."""

        model_config = ConfigDict(extra="ignore")

        long_axis_mm: float | None = Field(
            None,
            ge=0,
            description="Long-axis diameter in mm",
        )
        short_axis_mm: float | None = Field(
            None,
            ge=0,
            description="Short-axis diameter in mm",
        )
        craniocaudal_mm: float | None = Field(
            None,
            ge=0,
            description="Third-axis (craniocaudal) diameter in mm when documented",
        )
        morphology: str | None = Field(
            None,
            description="Lesion morphology/CT characteristics (e.g., solid, part-solid, ground-glass, spiculated, cavitary)",
        )
        suv_max: float | None = Field(
            None,
            ge=0,
            description="Maximum SUV on PET if applicable",
        )
        location: str | None = Field(
            None,
            description="Anatomic location of the target lesion (lobe/segment) when documented",
        )
        size_text: str | None = Field(
            None,
            description="Verbatim lesion size string when useful (e.g., '2.5 x 1.7 cm')",
        )

    asa_class: int | None = Field(
        None,
        ge=1,
        le=6,
        description="ASA physical status classification (1-6)",
    )
    ecog_score: int | None = Field(
        None,
        ge=0,
        le=4,
        description="ECOG performance status score (0-4) when explicitly documented",
    )
    ecog_text: str | None = Field(
        None,
        description="Raw ECOG/Zubrod performance status text when not a single integer (e.g., '0-1')",
    )
    primary_indication: str | None = Field(
        None,
        description="Primary indication for the procedure (free text)",
    )
    indication_category: Literal[
        "Lung Cancer Diagnosis",
        "Lung Cancer Staging",
        "Lung Nodule Evaluation",
        "Mediastinal Lymphadenopathy",
        "Infection Workup",
        "ILD Evaluation",
        "Hemoptysis",
        "Airway Obstruction",
        "Pleural Effusion - Diagnostic",
        "Pleural Effusion - Therapeutic",
        "Pneumothorax Management",
        "Empyema/Parapneumonic",
        "BLVR Evaluation",
        "BLVR Treatment",
        "Foreign Body",
        "Tracheobronchomalacia",
        "Stricture/Stenosis",
        "Fistula",
        "Stent Management",
        "Ablation",
        "Other",
    ] | None = None
    radiographic_findings: str | None = Field(
        None,
        description="Relevant radiographic findings",
    )
    lesion_size_mm: float | None = Field(
        None,
        ge=0,
        description="Target lesion size in mm",
    )
    lesion_location: str | None = Field(
        None,
        description="Anatomic location of target lesion",
    )
    pet_avidity: bool | None = Field(
        None,
        description="Whether lesion is PET avid",
    )
    suv_max: float | None = Field(
        None,
        ge=0,
        description="Maximum SUV on PET if applicable",
    )
    bronchus_sign: Literal["Positive", "Negative", "Not assessed"] | None = None
    target_lesion: TargetLesion | None = Field(
        default=None,
        description="Detailed target lesion disease burden fields.",
    )

    @field_validator("bronchus_sign", mode="before")
    @classmethod
    def normalize_bronchus_sign(cls, v):
        if v is None:
            return "Not assessed"
        if isinstance(v, bool):
            return "Positive" if v else "Negative"

        s = str(v).strip().lower()
        if s in {"yes", "y", "present", "positive", "pos", "true", "1"}:
            return "Positive"
        if s in {"no", "n", "absent", "negative", "neg", "false", "0"}:
            return "Negative"
        if s in {"na", "n/a", "not assessed", "unknown", "indeterminate"}:
            return "Not assessed"
        return v

    @field_validator("ecog_score", mode="before")
    @classmethod
    def normalize_ecog_score(cls, v):
        """Normalize ECOG/Zubrod performance status into a single 0-4 integer when explicit."""
        if v is None:
            return None

        # Avoid bool being treated as int
        if isinstance(v, bool):
            return None

        if isinstance(v, int):
            return v if 0 <= v <= 4 else None

        if isinstance(v, float):
            if not v.is_integer():
                return None
            iv = int(v)
            return iv if 0 <= iv <= 4 else None

        s = str(v).strip()
        if not s:
            return None

        lower = s.lower()
        # Do not coerce ranges like "0-1" or "0 to 1" into a single score.
        if re.search(r"\b[0-4]\b\s*(?:-|–|to|/|\+|or)\s*\b[0-4]\b", lower):
            return None

        match = re.search(r"(?i)\b(?:ecog|zubrod)\s*[:=]?\s*([0-4])\b", s)
        if match:
            return int(match.group(1))

        match2 = re.fullmatch(r"\s*([0-4])\s*", s)
        if match2:
            return int(match2.group(1))

        return None


class PatientDemographics(BaseModel):
    """Demographics with normalized gender field."""

    model_config = ConfigDict(extra="ignore")

    age_years: int | None = Field(
        None,
        ge=0,
        le=120,
        description="Patient age in years",
    )
    gender: Literal["Male", "Female", "Other", "Unknown"] | None = None
    height_cm: float | None = Field(
        None,
        ge=50,
        le=250,
        description="Patient height in cm",
    )
    weight_kg: float | None = Field(
        None,
        ge=20,
        le=400,
        description="Patient weight in kg",
    )
    bmi: float | None = Field(
        None,
        ge=10,
        le=80,
        description="Body mass index",
    )
    smoking_status: Literal["Never", "Former", "Current", "Unknown"] | None = None
    pack_years: float | None = Field(
        None,
        ge=0,
        description="Pack-years of smoking history",
    )

    @field_validator("gender", mode="before")
    @classmethod
    def normalize_gender(cls, v):
        if v is None:
            return None
        s = str(v).strip().lower()
        if s in {"m", "male"}:
            return "Male"
        if s in {"f", "female"}:
            return "Female"
        if s in {"other", "nonbinary", "non-binary", "nb"}:
            return "Other"
        if s in {"u", "unknown"}:
            return "Unknown"
        return v


class AirwayStentProcedure(BaseModel):
    """Structured airway stent data with location normalization."""

    model_config = ConfigDict(extra="ignore")

    performed: bool | None = None
    airway_stent_removal: bool = False
    action_type: Literal[
        "placement",
        "removal",
        "revision",
        "assessment_only",
    ] | None = Field(
        default=None,
        description="Normalized stent action classification (derived from action when available).",
    )
    action: Literal[
        "Placement",
        "Removal",
        "Revision/Repositioning",
        "Assessment only",
    ] | None = None
    stent_type: Literal[
        "Silicone - Dumon",
        "Silicone - Hood",
        "Silicone - Novatech",
        "SEMS - Uncovered",
        "SEMS - Covered",
        "SEMS - Partially covered",
        "Hybrid",
        "Y-Stent",
        "Other",
    ] | None = None
    stent_brand: str | None = None
    device_size: str | None = Field(
        default=None,
        description="Verbatim stent/device size when documented (e.g., '14 x 40 mm').",
    )
    diameter_mm: float | None = Field(None, ge=6, le=25)
    length_mm: float | None = Field(None, ge=10, le=100)
    location: Literal[
        "Trachea",
        "Right mainstem",
        "Left mainstem",
        "Bronchus intermedius",
        "RUL",
        "RML",
        "RLL",
        "LUL",
        "LLL",
        "Lingula",
        "Carina (Y)",
        "Other",
    ] | None = None
    indication: Literal[
        "Malignant obstruction",
        "Benign stenosis",
        "Tracheomalacia",
        "Fistula",
        "Post-dilation",
        "Other",
    ] | None = None
    deployment_successful: bool | None = None

    @model_validator(mode="after")
    def derive_action_type(self) -> "AirwayStentProcedure":
        # Keep `action` and `action_type` internally consistent. Some postprocessors
        # (and occasionally the LLM) can update one without clearing the other.
        if self.action is None:
            return self
        if self.action == "Placement":
            expected = "placement"
        elif self.action == "Removal":
            expected = "removal"
        elif self.action == "Revision/Repositioning":
            expected = "revision"
        elif self.action == "Assessment only":
            expected = "assessment_only"
        else:
            expected = None
        if expected and self.action_type != expected:
            self.action_type = expected
        return self

    @field_validator("action", mode="before")
    @classmethod
    def normalize_stent_action(cls, v):
        """Map common free-text stent actions into the constrained enum.

        Self-correction (and occasionally extraction) can emit compound strings like
        "Removal and Insertion", which would otherwise fail Literal validation.
        """
        if v is None:
            return None
        raw = str(v).strip()
        if not raw:
            return None

        allowed = {
            "Placement",
            "Removal",
            "Revision/Repositioning",
            "Assessment only",
        }
        if raw in allowed:
            return raw

        s = raw.lower()

        # Revision covers removal + replacement/exchange semantics (CPT 31638 family).
        if "remov" in s and any(token in s for token in ("insert", "plac", "deploy", "replac", "exchang")):
            return "Revision/Repositioning"
        if any(token in s for token in ("revision", "reposition", "exchange", "replace", "replac")):
            return "Revision/Repositioning"

        if any(token in s for token in ("plac", "insert", "deploy", "implant", "position")):
            return "Placement"
        if any(token in s for token in ("remov", "retriev", "extract", "explant", "pull", "peel", "grasp")):
            return "Removal"

        if any(token in s for token in ("assess", "inspect", "evaluat", "check", "patent", "intact")):
            return "Assessment only"

        # Conservative fallback: preserve pipeline stability by treating as unknown.
        return None

    @field_validator("stent_brand", mode="before")
    @classmethod
    def normalize_stent_brand(cls, v):
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        lower = s.lower()
        if lower in {
            "stent",
            "stents",
            "airway stent",
            "airway stents",
            "y-stent",
            "y stent",
            "y-stents",
            "y stents",
        }:
            return None
        # Drop generic suffixes like "Atrium iCast stent" -> "Atrium iCast"
        s = re.sub(r"(?i)\s+stents?$", "", s).strip()
        if not s:
            return None
        # Normalize common brand shorthands
        if re.fullmatch(r"(?i)icast", s):
            return "Atrium iCast"
        if re.fullmatch(r"(?i)atrium\s+icast", s):
            return "Atrium iCast"
        return s

    @field_validator("location", mode="before")
    @classmethod
    def normalize_stent_location(cls, v):
        if v is None:
            return v
        s = str(v).strip().lower()
        if s == "mainstem":
            return "Other"
        return v


# =============================================================================
# EBUS Per-Station Detail
# =============================================================================

class EBUSStationDetail(BaseModel):
    """Per-station EBUS-TBNA data capturing morphology, sampling, and ROSE."""

    model_config = ConfigDict(extra="ignore")

    # Station identification
    station: str = Field(..., description="IASLC station (2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L, 12R, 12L)")

    # Morphology
    short_axis_mm: float | None = Field(None, ge=0, description="Short-axis diameter in mm")
    long_axis_mm: float | None = Field(None, ge=0, description="Long-axis diameter in mm")
    shape: Literal["oval", "round", "irregular"] | None = None
    margin: Literal["distinct", "indistinct", "irregular"] | None = None
    echogenicity: Literal["homogeneous", "heterogeneous"] | None = None
    chs_present: bool | None = Field(None, description="Central hilar structure present")
    necrosis_present: bool | None = None
    calcification_present: bool | None = None

    # Elastography
    elastography_performed: bool | None = None
    elastography_score: int | None = Field(None, ge=1, le=5)
    elastography_strain_ratio: float | None = None
    elastography_pattern: Literal[
        "predominantly_blue", "blue_green", "green", "predominantly_green"
    ] | None = None

    # Doppler
    doppler_performed: bool | None = None
    doppler_pattern: Literal["avascular", "hilar_vessel", "peripheral", "mixed"] | None = None

    # Morphologic interpretation (separate from pathology)
    morphologic_impression: Literal["benign", "suspicious", "malignant", "indeterminate"] | None = None

    # Sampling details
    sampled: bool = Field(True, description="Whether this station was actually sampled")
    needle_gauge: Literal[19, 21, 22, 25] | None = None
    needle_type: Literal["Standard FNA", "FNB/ProCore", "Acquire", "ViziShot Flex"] | None = None
    number_of_passes: int | None = Field(None, ge=0, le=10)
    intranodal_forceps_used: bool | None = None

    # ROSE
    rose_performed: bool | None = None
    rose_result: Literal[
        "Adequate lymphocytes", "Malignant", "Suspicious for malignancy",
        "Atypical cells", "Granuloma", "Necrosis only", "Nondiagnostic", "Deferred"
    ] | None = None
    lymphocytes_present: bool | None = Field(
        None,
        description="Lymphocytes/lymphoid tissue present for this station when explicitly documented (ROSE/pathology). None when not assessable from note.",
    )
    rose_adequacy: bool | None = None

    @field_validator("needle_gauge", mode="before")
    @classmethod
    def normalize_needle_gauge(cls, v):
        """Parse needle gauge from strings like '22G' or '22-gauge' to integer."""
        if v is None:
            return None
        if isinstance(v, int):
            if v in (19, 21, 22, 25):
                return v
            return None
        s = str(v).upper().replace("G", "").replace("-GAUGE", "").replace("GAUGE", "").strip()
        try:
            gauge = int(s)
            if gauge in (19, 21, 22, 25):
                return gauge
        except ValueError:
            pass
        return None  # Invalid gauge - let validation handle it

    @field_validator("needle_type", mode="before")
    @classmethod
    def normalize_needle_type(cls, v):
        """Map brand names like 'Olympus NA-201SX-4022' to standard categories."""
        if v is None:
            return None
        s = str(v).lower()

        # Map Olympus ViziShot variants
        if any(x in s for x in ["vizishot", "na-u401sx", "na-401"]):
            if "flex" in s:
                return "ViziShot Flex"
            return "Standard FNA"

        # Map standard Olympus FNA needles
        if any(x in s for x in ["na-201", "na-200", "olympus"]):
            return "Standard FNA"

        # Map FNB/ProCore needles (Cook, Medtronic)
        if any(x in s for x in ["procore", "fnb", "core", "echotip"]):
            return "FNB/ProCore"

        # Map Acquire needles (Boston Scientific)
        if any(x in s for x in ["acquire", "boston"]):
            return "Acquire"

        # Generic FNA terminology
        if any(x in s for x in ["fna", "aspiration", "standard"]):
            return "Standard FNA"

        # Return original if it matches the enum
        if v in ("Standard FNA", "FNB/ProCore", "Acquire", "ViziShot Flex"):
            return v

        return None  # Invalid - let validation handle it

    @field_validator("rose_result", mode="before")
    @classmethod
    def normalize_rose_result(cls, v):
        """Map descriptive results like 'POSITIVE - Squamous cell carcinoma' to enum values."""
        if v is None:
            return None

        # If already a valid enum value, return as-is
        valid_values = {
            "Adequate lymphocytes", "Malignant", "Suspicious for malignancy",
            "Atypical cells", "Granuloma", "Necrosis only", "Nondiagnostic", "Deferred"
        }
        if v in valid_values:
            return v

        s = str(v).lower()

        # Map malignant findings
        if any(x in s for x in ["malignant", "positive", "carcinoma", "adenocarcinoma",
                                "squamous", "small cell", "nsclc", "sclc", "tumor", "cancer"]):
            return "Malignant"

        # Map suspicious findings
        if any(x in s for x in ["suspicious", "atypical"]):
            if "malignancy" in s:
                return "Suspicious for malignancy"
            return "Atypical cells"

        # Map granulomatous findings
        if any(x in s for x in ["granuloma", "sarcoid", "non-necrotizing", "nonnecrotizing"]):
            return "Granuloma"

        # Map necrosis
        if "necrosis" in s and ("only" in s or "alone" in s):
            return "Necrosis only"

        # Map benign/lymphocyte findings
        if any(x in s for x in ["lymphocyte", "reactive", "benign", "adequate"]):
            return "Adequate lymphocytes"

        # Map nondiagnostic
        if any(x in s for x in ["nondiagnostic", "non-diagnostic", "inadequate", "insufficient"]):
            return "Nondiagnostic"

        # Map deferred
        if any(x in s for x in ["deferred", "pending", "awaiting"]):
            return "Deferred"

        return None  # Invalid - let validation handle it
    
    # Specimen handling
    specimen_sent_for: list[str] | None = Field(default=None)
    
    # Final results
    final_pathology: str | None = None
    n_stage_contribution: Literal["N0", "N1", "N2", "N3"] | None = None
    
    notes: str | None = None


# =============================================================================
# Navigation Per-Target Detail
# =============================================================================

class NavigationTarget(BaseModel):
    """Per-target data for navigation/robotic bronchoscopy procedures."""
    
    model_config = ConfigDict(extra="ignore")
    
    # Target identification
    target_number: int = Field(..., ge=1, description="Sequential target number")
    target_location_text: str = Field(..., description="Full anatomic description")
    target_lobe: Literal["RUL", "RML", "RLL", "LUL", "LLL", "Lingula"] | None = None
    target_segment: str | None = None
    
    # Target characteristics
    lesion_size_mm: float | None = Field(None, ge=0)
    distance_from_pleura_mm: float | None = Field(None, ge=0)
    bronchus_sign: Literal["Positive", "Negative", "Not assessed"] | None = None
    ct_characteristics: Literal[
        "Solid", "Part-solid", "Ground-glass", "Cavitary", "Calcified"
    ] | None = None
    pet_suv_max: float | None = Field(None, ge=0)
    
    # Navigation performance
    registration_error_mm: float | None = Field(None, ge=0)
    navigation_successful: bool | None = None
    
    # Radial EBUS
    rebus_used: bool | None = None
    rebus_view: Literal["Concentric", "Eccentric", "Adjacent", "Not visualized"] | None = None
    rebus_lesion_appearance: str | None = None
    
    # Tool-in-lesion confirmation
    tool_in_lesion_confirmed: bool | None = None
    confirmation_method: Literal[
        "CBCT", "Augmented fluoroscopy", "Fluoroscopy", "Radial EBUS", "None"
    ] | None = None
    cbct_til_confirmed: bool | None = None
    
    # Sampling
    sampling_tools_used: list[str] | None = Field(default=None)
    number_of_forceps_biopsies: int | None = Field(None, ge=0)
    number_of_needle_passes: int | None = Field(None, ge=0)
    number_of_cryo_biopsies: int | None = Field(None, ge=0)

    # Fiducial marker placement (e.g., for radiation planning)
    fiducial_marker_placed: bool | None = None
    fiducial_marker_details: str | None = None
    
    # ROSE
    rose_performed: bool | None = None
    rose_result: str | None = None
    
    # Complications
    immediate_complication: Literal[
        "None", "Bleeding - mild", "Bleeding - moderate", "Bleeding - severe", "Pneumothorax"
    ] | None = None
    bleeding_management: str | None = None
    
    # Results
    specimen_sent_for: list[str] | None = Field(default=None)
    final_pathology: str | None = None
    
    notes: str | None = None

    @field_validator('bronchus_sign', mode='before')
    @classmethod
    def normalize_bronchus_sign(cls, v):
        """Normalize bronchus sign values from LLM output."""
        if v is True:
            return "Positive"
        if v is False:
            return "Negative"
        if isinstance(v, str):
            v_lower = v.lower().strip()
            if v_lower in ("pos", "positive", "+"):
                return "Positive"
            if v_lower in ("neg", "negative", "-"):
                return "Negative"
            if v_lower in ("not assessed", "n/a", "na", "unknown"):
                return "Not assessed"
        return v


# =============================================================================
# CAO Intervention Per-Site Detail
# =============================================================================

class CAOModalityApplication(BaseModel):
    """Details for a specific modality applied during CAO intervention."""
    
    model_config = ConfigDict(extra="ignore")
    
    modality: Literal[
        "APC", "Electrocautery - snare", "Electrocautery - knife", "Electrocautery - probe",
        "Cryotherapy - spray", "Cryotherapy - contact", "Cryoextraction",
        "Laser - Nd:YAG", "Laser - CO2", "Laser - diode", "Laser",
        "Mechanical debulking", "Rigid coring", "Microdebrider", "Balloon dilation",
        "Balloon tamponade", "PDT", "Iced saline lavage", "Epinephrine instillation",
        "Tranexamic acid instillation", "Suctioning"
    ]
    power_setting_watts: float | None = None
    apc_flow_rate_lpm: float | None = None
    balloon_diameter_mm: float | None = None
    balloon_pressure_atm: float | None = None
    freeze_time_seconds: int | None = None
    number_of_applications: int | None = Field(None, ge=0)
    duration_seconds: int | None = None


class CAOInterventionDetail(BaseModel):
    """Per-site CAO intervention data with modality details."""
    
    model_config = ConfigDict(extra="ignore")
    
    # Location
    location: str = Field(..., description="Airway location (Trachea, RMS, LMS, BI, etc.)")
    
    # Obstruction characterization
    obstruction_type: Literal["Intraluminal", "Extrinsic", "Mixed"] | None = None
    classification: str | None = Field(
        default=None,
        description="Optional obstruction severity classification (e.g., 'Myer-Cotton Grade').",
    )
    etiology: Literal[
        "Malignant - primary lung", "Malignant - metastatic", "Malignant - other",
        "Benign - post-intubation", "Benign - post-tracheostomy", "Benign - anastomotic",
        "Benign - inflammatory", "Benign - infectious", "Benign - granulation",
        "Benign - web/stenosis", "Benign - other", "Infectious", "Other"
    ] | None = None
    length_mm: float | None = Field(None, ge=0)

    # Lesion characteristics (free-text when numeric precision is not possible, e.g. ">50 lesions").
    lesion_morphology: str | None = None
    lesion_count_text: str | None = None
    
    # Pre/Post measurements
    pre_obstruction_pct: int | None = Field(None, ge=0, le=100)
    post_obstruction_pct: int | None = Field(None, ge=0, le=100)
    pre_diameter_mm: float | None = Field(None, ge=0)
    post_diameter_mm: float | None = Field(None, ge=0)
    
    # Treatment modalities
    modalities_applied: list[CAOModalityApplication] | None = Field(default=None)
    
    # Hemostasis
    hemostasis_required: bool | None = None
    hemostasis_methods: list[str] | None = Field(default=None)
    
    # Associated findings
    secretions_present: bool | None = None
    secretions_drained: bool | None = None
    stent_placed_at_site: bool | None = None
    
    notes: str | None = None


# =============================================================================
# BLVR Per-Valve Detail
# =============================================================================

class BLVRValvePlacement(BaseModel):
    """Individual valve placement data for BLVR."""
    
    model_config = ConfigDict(extra="ignore")
    
    valve_number: int = Field(..., ge=1)
    target_lobe: Literal["RUL", "RML", "RLL", "LUL", "LLL", "Lingula"]
    segment: str = Field(..., description="Specific segment (e.g., 'LB1+2', 'LB6')")
    airway_diameter_mm: float | None = Field(None, ge=0)
    valve_size: str = Field(..., description="Valve size (e.g., '4.0', '5.5', '6.0')")
    valve_type: Literal["Zephyr (Pulmonx)", "Spiration (Olympus)"]
    deployment_method: Literal["Standard", "Retroflexed"] | None = None
    deployment_successful: bool
    seal_confirmed: bool | None = None
    repositioned: bool | None = None
    notes: str | None = None


class BLVRChartisMeasurement(BaseModel):
    """Chartis collateral ventilation measurement data."""
    
    model_config = ConfigDict(extra="ignore")
    
    lobe_assessed: Literal["RUL", "RML", "RLL", "LUL", "LLL", "Lingula"]
    segment_assessed: str | None = None
    measurement_duration_seconds: int | None = Field(None, ge=0)
    adequate_seal: bool | None = None
    cv_result: Literal[
        "CV Negative", "CV Positive", "Indeterminate", "Low flow", "No seal", "Aborted"
    ]
    flow_pattern_description: str | None = None
    notes: str | None = None


# =============================================================================
# Cryobiopsy Per-Site Detail
# =============================================================================

class CryobiopsySite(BaseModel):
    """Per-site transbronchial cryobiopsy data."""
    
    model_config = ConfigDict(extra="ignore")
    
    site_number: int = Field(..., ge=1)
    lobe: Literal["RUL", "RML", "RLL", "LUL", "LLL", "Lingula"]
    segment: str | None = None
    distance_from_pleura: Literal[">2cm", "1-2cm", "<1cm", "Not documented"] | None = None
    fluoroscopy_position: str | None = None
    
    # Radial EBUS guidance
    radial_ebus_used: bool | None = None
    rebus_view: str | None = None
    
    # Biopsy details
    probe_size_mm: Literal[1.1, 1.7, 1.9, 2.4] | None = None
    freeze_time_seconds: int | None = Field(None, ge=0, le=10)
    number_of_biopsies: int | None = Field(None, ge=0)
    specimen_size_mm: float | None = Field(None, ge=0)
    
    # Blocker use
    blocker_used: bool | None = None
    blocker_type: Literal["Fogarty", "Arndt", "Cohen", "Cryoprobe sheath"] | None = None
    
    # Complications at site
    bleeding_severity: Literal["None/Scant", "Mild", "Moderate", "Severe"] | None = None
    bleeding_controlled_with: str | None = None
    pneumothorax_after_site: bool | None = None
    
    notes: str | None = None


# =============================================================================
# Thoracoscopy Findings Per-Site
# =============================================================================

class ThoracoscopyFinding(BaseModel):
    """Per-location thoracoscopy/pleuroscopy finding."""
    
    model_config = ConfigDict(extra="ignore")
    
    location: Literal[
        "Parietal pleura - chest wall", "Parietal pleura - diaphragm",
        "Parietal pleura - mediastinum", "Visceral pleura",
        "Lung parenchyma", "Costophrenic angle", "Apex"
    ]
    finding_type: Literal[
        "Normal", "Nodules", "Plaques", "Studding", "Mass",
        "Adhesions - filmy", "Adhesions - dense", "Inflammation",
        "Thickening", "Trapped lung", "Loculations", "Empyema", "Other"
    ]
    extent: Literal["Focal", "Multifocal", "Diffuse"] | None = None
    size_description: str | None = None
    biopsied: bool | None = None
    number_of_biopsies: int | None = Field(None, ge=0)
    biopsy_tool: Literal["Rigid forceps", "Flexible forceps", "Cryoprobe"] | None = None
    impression: Literal[
        "Benign appearing", "Malignant appearing", "Infectious appearing", "Indeterminate"
    ] | None = None
    notes: str | None = None

    @field_validator("biopsy_tool", mode="before")
    @classmethod
    def normalize_thoracoscopy_biopsy_tool(cls, v):
        if v is None:
            return v
        s = str(v).lower()
        if "cryo" in s:
            return "Cryoprobe"
        if "flexible" in s:
            return "Flexible forceps"
        if "rigid" in s:
            return "Rigid forceps"
        if "biopsy forceps" in s:
            return "Rigid forceps"
        return v

    @field_validator("location", mode="before")
    @classmethod
    def normalize_thoracoscopy_location(cls, v):
        if v is None:
            return v
        s = str(v).lower()
        if "pleural space" in s:
            return "Parietal pleura - chest wall"
        if "costophrenic" in s:
            return "Costophrenic angle"
        if "apex" in s:
            return "Apex"
        if "diaphragm" in s:
            return "Parietal pleura - diaphragm"
        if "mediastinum" in s:
            return "Parietal pleura - mediastinum"
        if "visceral" in s or "lung surface" in s:
            return "Visceral pleura"
        return v


# =============================================================================
# Unified Specimen Tracking
# =============================================================================

class SpecimenCollected(BaseModel):
    """Specimen tracking with source linkage."""

    model_config = ConfigDict(extra="ignore")

    specimen_number: int = Field(..., ge=1)
    source_procedure: Literal[
        "EBUS-TBNA", "Navigation biopsy", "Endobronchial biopsy",
        "Transbronchial biopsy", "Transbronchial cryobiopsy",
        "BAL", "Bronchial wash", "Brushing", "Pleural biopsy", "Pleural fluid", "Other"
    ]
    source_location: str = Field(..., description="Anatomic location")
    collection_tool: str | None = None
    specimen_count: int | None = Field(None, ge=0)
    specimen_adequacy: Literal["Adequate", "Limited", "Inadequate", "Pending"] | None = None
    destinations: list[str] | None = Field(default=None)
    rose_performed: bool | None = None
    rose_result: str | None = None
    final_pathology_diagnosis: str | None = None
    molecular_markers: dict[str, Any] | None = Field(default=None)
    notes: str | None = None

    @field_validator("source_procedure", mode="before")
    @classmethod
    def normalize_source_procedure(cls, v):
        """Map descriptive procedure names to standard categories."""
        if v is None:
            return None

        # If already a valid enum value, return as-is
        valid_values = {
            "EBUS-TBNA", "Navigation biopsy", "Endobronchial biopsy",
            "Transbronchial biopsy", "Transbronchial cryobiopsy",
            "BAL", "Bronchial wash", "Brushing", "Pleural biopsy", "Pleural fluid", "Other"
        }
        if v in valid_values:
            return v

        s = str(v).lower()

        # Map EBUS variants
        if any(x in s for x in ["ebus-tbna", "ebus tbna", "ebus", "tbna"]):
            return "EBUS-TBNA"

        # Map navigation variants
        if any(x in s for x in ["navigation", "nav biopsy", "nav-guided", "enb", "robotic"]):
            return "Navigation biopsy"

        # Map endobronchial biopsy variants
        if any(x in s for x in ["endobronchial", "ebx", "forceps biopsy"]):
            return "Endobronchial biopsy"

        # Map transbronchial biopsy variants
        if any(x in s for x in ["transbronchial biopsy", "tbbx", "tblb"]):
            return "Transbronchial biopsy"

        # Map cryobiopsy variants
        if any(x in s for x in ["cryobiopsy", "cryo biopsy", "cryo"]):
            return "Transbronchial cryobiopsy"

        # Map BAL variants
        if any(x in s for x in ["bal", "bronchoalveolar", "lavage"]):
            return "BAL"

        # Map bronchial wash
        if "wash" in s:
            return "Bronchial wash"

        # Map brushing
        if "brush" in s:
            return "Brushing"

        # Map pleural biopsy
        if "pleural biopsy" in s:
            return "Pleural biopsy"

        # Map pleural fluid/thoracentesis
        if any(x in s for x in ["pleural fluid", "thoracentesis", "pleural tap"]):
            return "Pleural fluid"

        # Procedures that don't fit elsewhere
        if any(x in s for x in ["therapeutic aspiration", "aspiration", "suctioning"]):
            return "Other"

        return "Other"  # Fallback for unrecognized procedures


# =============================================================================
# Enhanced Registry Record with Granular Arrays
# =============================================================================

class EnhancedRegistryGranularData(BaseModel):
    """Container for all granular per-site data arrays.
    
    This can be embedded in the main RegistryRecord or used as a separate payload.
    """
    
    model_config = ConfigDict(extra="ignore")
    
    # EBUS per-station
    linear_ebus_stations_detail: list[EBUSStationDetail] | None = Field(default=None)
    
    # Navigation per-target
    navigation_targets: list[NavigationTarget] | None = Field(default=None)
    
    # CAO per-site
    cao_interventions_detail: list[CAOInterventionDetail] | None = Field(default=None)
    
    # BLVR per-valve
    blvr_valve_placements: list[BLVRValvePlacement] | None = Field(default=None)
    blvr_chartis_measurements: list[BLVRChartisMeasurement] | None = Field(default=None)
    
    # Cryobiopsy per-site
    cryobiopsy_sites: list[CryobiopsySite] | None = Field(default=None)
    
    # Thoracoscopy per-location
    thoracoscopy_findings_detail: list[ThoracoscopyFinding] | None = Field(default=None)
    
    # Unified specimen tracking
    specimens_collected: list[SpecimenCollected] | None = Field(default=None)


__all__ = [
    "IPCProcedure",
    "ClinicalContext",
    "PatientDemographics",
    "AirwayStentProcedure",
    # Per-site models
    "EBUSStationDetail",
    "NavigationTarget",
    "CAOModalityApplication",
    "CAOInterventionDetail",
    "BLVRValvePlacement",
    "BLVRChartisMeasurement",
    "CryobiopsySite",
    "ThoracoscopyFinding",
    "SpecimenCollected",
    # Container
    "EnhancedRegistryGranularData",
]
