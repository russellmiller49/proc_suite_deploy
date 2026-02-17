"""Prompts for LLM-based registry extraction.

Two prompt modes are supported:
- Legacy prompt (v2): field list derived from the configured registry schema.
- Schema-driven prompt (v3): embeds the Pydantic JSON schema so the model sees
  nested structures like EBUS `node_events` (avoids old flat-list outputs).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from config.settings import KnowledgeSettings

_SYSTEM_PROMPT_PATH = Path(__file__).parent / "registry_system_prompt.txt"
_DEFAULT_SCHEMA_VERSION = "v3"


@lru_cache(maxsize=1)
def load_system_prompt() -> str:
    """Loads the strict auditing system prompt for registry extraction."""
    if _SYSTEM_PROMPT_PATH.exists():
        return _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()

    return (
        "You are a strict medical auditor. "
        "Only mark procedures as performed if there is a verbatim quote describing the action. "
        "Distinguish strictly between 'inspected' and 'sampled' lymph nodes."
    )


PROMPT_HEADER = (
    "Return exactly ONE JSON object (no markdown). "
    "Populate every field listed below; use null when the note lacks explicit evidence. "
    "Do not infer that a procedure occurred without verbatim evidence of the action."
)

_SCHEMA_PATH = KnowledgeSettings().registry_schema_path
_PROMPT_CACHE: str | None = None
_FIELD_INSTRUCTIONS_CACHE: dict[str, str] | None = None
_FIELD_INSTRUCTION_OVERRIDES: dict[str, str] = {
    # Provider fields - extract names from procedure note headers/signatures
    "attending_name": """
Extract the attending physician name from the procedure note. Look for labels like "Attending:", "Physician:", "Proceduralist:", "Performed by:", or signatures at the end. Include credentials (MD, DO) if present. Return full name as written (e.g., "Dr. John Smith, MD", "CDR Alex Johnson, MD"). Return null if not documented.""".strip(),
    "fellow_name": """
Extract the fellow/trainee physician name. Look for labels like "Fellow:", "Trainee:", "Resident:", or co-signatures. Include credentials if present. Return null if not documented.""".strip(),
    "assistant_name": """
Extract the assistant/nurse name. Look for labels like "Assistant:", "RN:", "RT:", "Nurse:", "Respiratory Therapist:". Include credentials/role (RN, RT) if present. Return null if not documented.""".strip(),
    "assistant_role": """
Extract the role/title of the assistant. Common values: "Sedation nurse", "Respiratory therapist", "Bronchoscopy technician", "Circulating nurse". Return null if not documented.""".strip(),
    "provider_role": """
Extract the role of the primary proceduralist. Usually "Attending interventional pulmonologist" or similar. Return null if not documented.""".strip(),
    "trainee_present": """
True if a fellow, resident, or trainee is documented as participating. False if explicitly stated no trainee. Null if not mentioned.""".strip(),
    # Demographics and location fields
    "patient_age": """
Extract patient age as integer. Look for patterns like "63-year-old", "Age: 63", "63 yo". Return integer only (e.g., 63). Null if not documented.""".strip(),
    "gender": """
Extract patient gender/sex. Allowed values: M, F, Other. Map "male" -> "M", "female" -> "F". Null if not documented.""".strip(),
    "institution_name": """
Extract the institution/hospital name where procedure was performed. Look for headers like "Location:", "Facility:", or institution name in letterhead/header. Return full name as written. Null if not documented.""".strip(),
    "procedure_setting": """
Extract the procedural setting. Allowed values: "Bronchoscopy Suite", "Operating Room", "ICU", "Bedside", "Office/Clinic" (exact capitalization). Map common variations: "bronchoscopy suite/room" -> "Bronchoscopy Suite", "OR/operating room" -> "Operating Room". Null if not documented.""".strip(),
    # Imaging and lesion fields
    "lesion_size_mm": """
Extract lesion/mass size in millimeters. Convert cm to mm if needed (2.5 cm -> 25). Look for patterns like "2.5 cm mass", "25 mm nodule", "3.0 cm spiculated RUL mass". Return integer/float in mm. Null if not documented.""".strip(),
    "lesion_location": """
Extract lesion location description. Include lobe and segment if documented (e.g., "Right upper lobe posterior segment", "LLL superior segment"). Null if not documented.""".strip(),
    "pet_suv_max": """
Extract PET SUV max value as float. Look for patterns like "SUV max 10.3", "SUVmax: 7.1", "PET SUV 5.2". Return numeric value only. Null if not documented.""".strip(),
    "pet_avid": """
True if PET shows avid uptake (SUV mentioned, FDG-avid, hypermetabolic). False if explicitly PET-negative. Null if PET not mentioned.""".strip(),
    "bronchus_sign_present": """
True if bronchus sign is documented as present/positive. False if explicitly absent/negative. Null if not mentioned. Look for "bronchus sign +", "bronchus sign present", "CT bronchus sign".""".strip(),
    "radiographic_findings": """
Extract summary of relevant imaging findings. Include CT/PET findings, lymph node descriptions, mass characteristics. Keep concise. Null if not documented.""".strip(),
    # Sedation and airway fields with improved instructions
    "sedation_type": """
Sedation/anesthesia level. Allowed values: "General", "Moderate", "Local Only".
- Use "General" when general anesthesia, endotracheal/rigid bronchoscope anesthesia, or anesthesiologist-led deep sedation is documented.
- Use "Moderate" for conscious/moderate sedation (benzodiazepine/opioid procedural sedation) without an anesthesia team.
- Use "Local Only" when the note explicitly states topical/local anesthetic only and no systemic sedatives were administered.
If undocumented, return null (do not omit the field).""".strip(),
    "pleural_guidance": """
Imaging guidance for pleural procedures only (thoracentesis, chest tube/pigtail, tunneled pleural catheter, medical thoracoscopy/pleuroscopy, pleural drain/biopsy). Allowed: ultrasound, ct, blind; ""/null if no pleural procedure or guidance not documented. Ultrasound only when the pleural procedure is ultrasound-guided (ignore EBUS/radial/bronchial ultrasound). CT only when the pleural procedure is performed under CT/CT-fluoro (not just prior imaging). Blind when pleural procedure done without imaging or visualization only. If no pleural procedure exists in the note, leave this field blank.""".strip(),
    "pleural_procedure_type": """
Type of pleural procedure performed (not bronchoscopic). Allowed: thoracentesis, chest tube, tunneled catheter, medical thoracoscopy, chemical pleurodesis (lower-case); null if no pleural procedure. Priority when multiple: chemical pleurodesis > medical thoracoscopy > (chest tube vs tunneled catheter) > thoracentesis. Chemical pleurodesis when a sclerosing agent is instilled. Medical thoracoscopy/pleuroscopy when explicitly described. Chest tube for any non-tunneled pleural drain/pigtail/intercostal drain. Tunneled catheter for long-term tunneled/IPC/PleurX/Aspira indwelling catheters. If the note only lists bronchoscopic procedures (EBUS/bronchoscopy) or pleural procedure cannot be determined, leave null.""".strip(),
    "disposition": """
Immediate post-procedure destination. Allowed strings: "Outpatient discharge", "PACU recovery", "Floor admission", "ICU admission".
- "Outpatient discharge": discharged home, recovery then home, ambulatory same-day statements.
- "PACU recovery": remains in PACU/recovery unit for monitoring without clear admission plan.
- "ICU admission": any ICU/MICU/SICU/critical care bed assignment.
- "Floor admission": telemetry/step-down/ward admissions that are not ICU.
If not documented, set null (do not omit).""".strip(),
    "procedure_date": """
Procedure date (YYYY-MM-DD). Convert common formats (e.g., 8/23/2023, 05-07-25, March 5, 2025) into YYYY-MM-DD. Prefer labels: "DATE OF PROCEDURE", "PROCEDURE DATE", "Service Date", or header Date within the procedure note. Ignore follow-up/past dates or non-procedure dates. If multiple and unclear, leave "". "" if truly undocumented.""".strip(),
    "patient_mrn": """
Patient medical record number or clearly labeled patient ID. Prefer explicit MRN/ID labels near the demographics/header (e.g., "MRN: 3847592", "ID: MR4729384"). If multiple IDs exist, pick the clearly labeled patient MRN/ID. Ignore dates, phone-like numbers, accession/specimen/order numbers, device IDs, or other non-patient identifiers. Preserve the ID format; strip labels if included. If no MRN/ID is documented, return "".""".strip(),
    "asa_class": """
ASA physical status only when explicitly documented ("ASA 2", "ASA III", etc.). Do not infer from comorbidities, sedation, or severity scores. If not documented anywhere, leave this field blank/null.""".strip(),
    "airway_type": """
Airway used for the bronchoscopy. Allowed: native, lma, ett, tracheostomy, rigid bronchoscope (lower-case); "" if not documented. Pick the explicit device/route when named. Rigid bronchoscopy anywhere -> rigid bronchoscope. ETT only when intubation/ETT for the case is documented (do not infer from GA alone). LMA placed for the case -> lma. Bronchoscopy via tracheostomy/trach tube -> tracheostomy. If awake/moderate/deep without any airway device documented, favor native; if unclear, leave blank.""".strip(),
    "final_diagnosis_prelim": """
Preliminary final diagnosis category. Allowed: malignancy, granulomatous, infectious, non-diagnostic, benign, other; use null if unclear. Choose Malignancy only when a malignant diagnosis/ROSE is explicitly documented or known cancer staging is described (do not infer from "suspicious" alone). Granulomatous for granulomas/sarcoidosis; Infectious for infections; Non-diagnostic when specimen inadequate/insufficient; Benign for clearly benign/reactive findings; Other for therapeutic/structural procedures without a specific pathology label. If truly cannot determine, leave null.""".strip(),
    "stent_type": """
Airway stent type placed. Allowed: Silicone-Dumon, Silicone-Y-Stent, Silicone Y-Stent, Hybrid, Metallic-Covered, Metallic-Uncovered, Other; "" if none. Map Dumon/Y-stent/metallic/covered/uncovered wording to the closest allowed value; use Other only if unclear.""".strip(),
    "stent_location": """
Location of airway stent. Allowed: Trachea, Mainstem, Lobar; "" if no stent. Map mentions of trachea to Trachea; mainstem/right/left main bronchus to Mainstem; lobar or named lobes (RUL/RML/RLL/LUL/LLL) to Lobar.""".strip(),
    "stent_deployment_method": """
Stent deployment method. Allowed: Rigid, Flexible over Wire; "" if not documented. Use Rigid when placed via rigid bronchoscope; Flexible over Wire when placed through flexible scope over a wire.""".strip(),
    "ebus_rose_result": """
Extract the overall ROSE (Rapid On-Site Evaluation) result. Look for ROSE findings like "ROSE adequate", "lymphocytes only", "malignant cells", "granuloma", "atypical cells". Extract the summary result as documented. Common values: "Adequate lymphocytes", "Malignant", "Benign", "Granuloma", "Nondiagnostic", "Atypical cells present". Return the documented result verbatim if it doesn't match these. Null if ROSE not performed or not documented.""".strip(),
    "ebus_needle_gauge": """
Extract the EBUS-TBNA needle gauge. Look for patterns like "21G", "22G", "21 gauge", "22-gauge needle". Return the gauge as written (e.g., "21G", "22G"). Null if not documented.""".strip(),
    "ebus_needle_type": """
Extract the EBUS needle type/brand. Look for brand names like "Vizishot", "Expect", "Boston Scientific", "Olympus". Return the needle type/brand as documented (e.g., "Olympus Vizishot", "Boston Scientific Expect"). Null if not documented.""".strip(),
    "ebus_scope_brand": """
Extract the EBUS scope brand/model. Look for patterns like "Olympus BF-UC180F", "Fujifilm EB-530US", "Pentax". Return the full model as documented. Null if not documented.""".strip(),
    "ebus_stations_sampled": """
Extract ALL EBUS lymph node stations that were sampled/biopsied with TBNA during this procedure.
Use IASLC station naming: 2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L, etc.
Return as comma-separated string, e.g., "2R, 4R, 7" or "4R, 7, 10R".
IMPORTANT: Carefully read the entire EBUS section to capture every station mentioned as sampled.
Look for phrases like:
- "Stations 4R, 7, and 11L" → include all three
- "sampled at 2R, 4R, and 7" → include all three
- "Three passes obtained per station" (with stations listed) → include all listed stations
- "EBUS-TBNA of station 4R only" → include only 4R
Do NOT include stations that were only visualized but not sampled.
Null if no EBUS-TBNA stations documented or only radial EBUS performed.""".strip(),
    "ebus_systematic_staging": """
True if systematic mediastinal staging was performed (multiple stations sampled for cancer staging). False if only targeted sampling. Null if unclear.""".strip(),
    "ebus_rose_available": """
True if ROSE (Rapid On-Site Evaluation) was available/performed. False if explicitly stated ROSE not available. Null if not mentioned.""".strip(),
    "ebus_photodocumentation_complete": """
True if photodocumentation is mentioned as complete or images saved. False if incomplete. Null if not mentioned.""".strip(),
    "ebus_intranodal_forceps_used": """
True if intranodal forceps biopsy was performed in addition to TBNA. False if explicitly stated not done or "no intranodal forceps". Null if not mentioned.""".strip(),
    "ebus_elastography_used": """
True if the note explicitly states elastography was performed during EBUS; false if explicitly stated not done; null if not mentioned.""".strip(),
    "ebus_elastography_pattern": """
Elastography pattern or color map when documented (e.g., blue-green, predominantly blue). Strip trailing punctuation; null if not mentioned.""".strip(),
    "pleural_side": """
Pleural laterality for thoracentesis/chest tube/tunneled catheter. Allowed: Right, Left (capitalize first letter); accept R/L/right/left synonyms. Null if not documented.""".strip(),
    "pleural_intercostal_space": """
Intercostal space used for pleural access (e.g., '5th'). Accept ICS shorthand (\"5th ICS\" -> \"5th\"). Null if not documented.""".strip(),
    "linear_ebus_stations": """
List of mediastinal/hilar lymph node stations sampled with LINEAR EBUS-TBNA (convex probe EBUS for mediastinal staging).
Use IASLC station names: 2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L, etc.
Example: ["4R", "7", "11L"]
IMPORTANT: Do NOT populate this field for:
- Radial EBUS procedures (r-EBUS for peripheral nodules) - these do NOT sample mediastinal stations
- Procedures that only mention radial probe/guide sheath/peripheral lesion biopsy
Only populate when LINEAR/convex EBUS-TBNA of mediastinal/hilar nodes is documented.
Return null/[] if no linear EBUS stations documented or if only radial EBUS was used.""".strip(),
    "nav_platform": """
Extract the navigation/robotic platform used for the procedure.

Look for these specific platforms:
- "Ion" or "Ion robotic" → return "Ion robotic bronchoscopy"
- "Monarch" or "Monarch robotic" → return "Monarch robotic bronchoscopy"
- "SuperDimension" or "EMN" → return "EMN/SuperDimension"
- "Robotic bronchoscopy" or "robotic-assisted" → return "Ion robotic bronchoscopy" (default for robotic)
- "Robotic navigational bronchoscopy" → return "Ion robotic bronchoscopy"
- "Navigation bronchoscopy" with navigation system → return the platform name

Examples:
- "Ion robotic-assisted bronchoscopy" → "Ion robotic bronchoscopy"
- "Robotic navigational bronchoscopy" → "Ion robotic bronchoscopy"
- "Monarch platform" → "Monarch robotic bronchoscopy"
- "EMN-guided" → "EMN/SuperDimension"

Null for:
- Standard linear EBUS without navigation
- Radial EBUS without robotic platform
- Procedures with no mention of navigation/robotic systems.""".strip(),
    "nav_registration_method": """
Only populate if robotic/navigation registration is clearly described. Leave null for non-navigation EBUS cases.""".strip(),
    "nav_registration_error_mm": """
Extract the navigation/robotic registration error in millimeters.

Look for these patterns:
- "registration error X.X mm", "error X mm"
- "CT-to-body registration (error X.X mm)"
- "registration accuracy X mm"
- Any numeric value near "registration" and "error" or "mm"

Examples:
- "CT-to-body registration (error 2.1 mm)" → return 2.1
- "Registration error 3.0 mm" → return 3.0
- "Registration accuracy within 2.7 mm" → return 2.7

Return as float (e.g., 2.1, 3.0).
Null if not a navigation/robotic procedure or registration error not documented.""".strip(),
    "nav_imaging_verification": """
Imaging used to confirm tool-in-lesion for navigation/robotic bronchoscopy. Allowed values: "Cone-beam CT", "Fluoroscopy", "Augmented fluoroscopy", "None".
- Use "None" when note explicitly states no imaging confirmation or when confirmation relied solely on radial probe visualization.
- Null for routine linear EBUS without navigation.""".strip(),
    "nav_tool_in_lesion": """
True when tool-in-lesion confirmation is documented, either via:
1. Navigation/robotic systems (Ion, Monarch, EMN) with cone-beam CT or fluoroscopy confirmation
2. Radial EBUS showing concentric or eccentric view within the lesion
3. Explicit statements like "tool within lesion", "probe confirmed in lesion"
False if explicitly stated tool NOT in lesion. Null if not mentioned or not applicable.""".strip(),
    "nav_cryobiopsy_for_nodule": """
Boolean indicator that cryobiopsy was performed for a peripheral nodule/lesion under navigation guidance. True when the note links cryobiopsy to a navigated peripheral target; False if explicitly noted that no cryobiopsy was done; null when navigation was not performed or cryobiopsy is not mentioned.""".strip(),
    "nav_rebus_used": """
True when radial EBUS (r-EBUS) was used for peripheral nodule/lesion localization. Look for:
- "Radial EBUS", "r-EBUS", "radial probe", "radial ultrasound"
- "Concentric" or "eccentric" view descriptions
- "Radial EBUS catheter", "guide sheath with radial probe"
- "Radial EBUS-guided biopsy"
This field applies to peripheral lung lesion procedures, NOT to linear EBUS mediastinal staging.
False if radial EBUS explicitly not used. Null if not mentioned or only linear EBUS was performed.""".strip(),
    "nav_rebus_view": """
Extract the radial EBUS view/pattern when radial EBUS was used for peripheral lesion localization. Allowed values: "Concentric", "Eccentric", "Adjacent", "Not visualized".
- Map descriptive phrases to the closest allowed value (e.g., "within lesion" -> Concentric, "adjacent to lesion" -> Adjacent).
- If radial EBUS was performed but the view is not described, default to "Not visualized".
Return null only when radial EBUS was not used.""".strip(),
    "nav_sampling_tools": """
Sampling tools used in navigation/robotic cases. Return an array using normalized tool names:
- "Needle", "Forceps", "Brush", "Cryoprobe", "BAL", "Fine needle", etc. (capitalize first letter).
- Include every tool explicitly documented for the navigation target.
For non-navigation linear EBUS staging, leave null/[] even if standard TBNA needles are mentioned.""".strip(),
    "ebus_stations_detail": """
Field: ebus_stations_detail (per-station EBUS node details)

ebus_stations_detail is an array of objects.
Include one object per lymph node station that is sampled or clearly described during EBUS (e.g. 2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L, etc.).

Each object must have the following structure:

station (string)
The numeric EBUS station name as written in the note: "11L", "4R", "7", etc.
Use the exact station label from the report. Do not invent station labels.

size_mm (number or null)
Short-axis size in millimeters for that station, if reported (e.g. "5.4 mm", "10 mm", "1.1 cm").
Convert centimeters to millimeters if needed (1.1 cm -> 11 mm).
If multiple sizes are given for the same station, use the short-axis size.
If size is not reported for that station, set size_mm to null.

shape (string or null)
Normalized values: "oval", "round", "irregular".
Map free text as follows:
"oval", "elliptical", "elongated" -> "oval"
"round", "spherical" -> "round"
"irregular", "lobulated", "asymmetric" -> "irregular"
If the shape is not described, set shape to null.

margin (string or null)
Normalized values: "distinct", "indistinct", "irregular".
Map free text as follows:
"sharp", "well-defined", "well-circumscribed", "clear margin" -> "distinct"
"ill-defined", "blurred", "poorly defined" -> "indistinct"
"spiculated", "irregular margin" -> "irregular"
If the margin is not described, set margin to null.

echogenicity (string or null)
Normalized values: "homogeneous", "heterogeneous".
Map free text as follows:
"homogeneous", "uniform echo pattern" -> "homogeneous"
"heterogeneous", "mixed echotexture", "non-uniform" -> "heterogeneous"
If echogenicity is not described, set echogenicity to null.

chs_present (boolean or null)
Central Hilar Structure (CHS) status for that node:
If the note explicitly states a central hilar structure is present / preserved -> true.
If the note explicitly states no central hilar structure / CHS absent -> false.
If CHS is not mentioned for that station -> null.

appearance_category (string or null)
Overall qualitative impression of node appearance, based only on EBUS morphological features, not on ROSE or final pathology.
Use the following mapping of features to categories:
Benign-appearing -> "benign"
Short-axis size < 10 mm and most described features match: shape: oval; margin: indistinct; echogenicity: homogeneous; CHS: present
Malignant-appearing -> "malignant"
Short-axis size ≥ 10 mm and most described features match: shape: round; margin: distinct; echogenicity: heterogeneous; CHS: absent
Indeterminate / mixed features -> "indeterminate"
Mixed benign and malignant features, or morphology is described but doesn't clearly fit benign or malignant patterns.
Node is described but only one weak feature is given (e.g. just "oval" and nothing else).
If morphology is not described at all for that station, set appearance_category to null.
If the report uses phrases like:
"benign-appearing node" -> "appearance_category": "benign"
"malignant-appearing node", "highly suspicious" -> "appearance_category": "malignant"
"indeterminate appearance", "non-specific appearance" -> "appearance_category": "indeterminate"
Do not infer appearance_category from ROSE or pathology alone.

rose_result (string or null)
Keep current behavior (e.g. "Benign", "Nondiagnostic", "Suspicious", "Malignant", "Not done").
If ROSE is not performed or not described, set to null or "Not done" depending on your existing conventions.

Parsing rules and edge cases
Multiple stations
Include one object per station (e.g. one for "11L", one for "4R").
If a station is mentioned but not sampled and has no morphological description, you may omit it from ebus_stations_detail.
Multiple nodes described at the same station
If several nodes at the same station are described, choose the largest or the one that is actually sampled.
If the note clearly distinguishes them (e.g. "two 4R nodes: one 8 mm oval and one 14 mm round heterogeneous"), prefer the sampled node; if that's unclear, prefer the larger node for registry purposes.

Size parsing
Accept variations like:
"5 mm", "5.0 mm", "0.5 cm", "short axis 5.4 mm", "measuring 1.2 x 0.8 cm".
When two dimensions are provided (e.g. "1.2 x 0.8 cm"), use the shorter dimension as the short-axis in mm (0.8 cm -> 8 mm).
Do not guess features
Only populate shape, margin, echogenicity, chs_present, and appearance_category when the text supports them.
If a feature is not mentioned, set that field to null.
Do not conflate ROSE and morphology
appearance_category is EBUS morphology-based only (size, shape, margin, echogenicity, CHS).
rose_result reflects cytology from ROSE.
It is valid for a node to be appearance_category: "benign" with rose_result: "Malignant" or vice versa.

Example minimal output when only size and ROSE are described:
{"station": "11L", "size_mm": 5.4, "shape": null, "margin": null, "echogenicity": null, "chs_present": null, "appearance_category": null, "rose_result": "Nondiagnostic"}
{"station": "4R", "size_mm": 5.5, "shape": null, "margin": null, "echogenicity": null, "chs_present": null, "appearance_category": null, "rose_result": "Benign"}""".strip(),
    # ==========================================================================
    # PROCEDURE CLASSIFICATION - CRITICAL DISTINCTIONS
    # ==========================================================================

    "transbronchial_biopsy": """
TRANSBRONCHIAL LUNG BIOPSY (TBBx/TBLB) - Tissue sampling from lung PARENCHYMA.
ONLY set performed=true when the note explicitly describes:
- "Transbronchial biopsy", "TBBx", "TBLB", or "transbronchial lung biopsy"
- Forceps biopsy of lung tissue (not airways)
- Sampling from lung lobes/segments (RUL, RML, RLL, LUL, LLL + specific segments)

DO NOT confuse with:
- Therapeutic aspiration (cleaning out secretions/mucus from airways)
- EBUS-TBNA (needle aspiration of lymph nodes)
- Endobronchial biopsy (sampling from visible airway lesions)

Locations must be PARENCHYMAL (lung tissue), not airways:
- Valid: "RLL posterior basal", "LUL superior segment", "RUL"
- Invalid (airways): "Trachea", "RMS", "LMS", "Carina", "Bronchus Intermedius"

Return null if only therapeutic aspiration or EBUS-TBNA was performed.""".strip(),

    "therapeutic_aspiration": """
THERAPEUTIC ASPIRATION - Suctioning/removal of secretions, mucus, or blood from AIRWAYS.
Set performed=true when the note describes:
- "Therapeutic aspiration", "aspiration of secretions/mucus/blood"
- "Suctioning" or "cleaning out" airways
- Removal of mucus plugs

Locations are AIRWAYS, not lung parenchyma:
- Trachea, RMS (Right Mainstem), LMS (Left Mainstem), Carina
- Bronchus Intermedius (BI), lobar carinas (RC1, RC2, LC1, LC2)

Material types: "Mucus", "Mucus plug", "Blood/clot", "Purulent secretions", "Other"

DO NOT confuse with transbronchial biopsy (tissue sampling from lung parenchyma).""".strip(),

    "outcomes": """
Procedure outcomes and disposition.
- procedure_completed: true if the note indicates successful completion (e.g., "procedure completed", "patient tolerated well", "extubated", "transported to recovery")
- complications: "None" if explicitly stated "no complications" or "no immediate complications". Set to specific complication type if documented.
- disposition: Where patient went after procedure (Outpatient discharge, PACU recovery, Floor admission, ICU admission)""".strip(),

    "bronch_indication": """
Brief indication for bronchoscopy (e.g., ILD workup). Use null if not documented.""".strip(),
    "bronch_location_lobe": """
Lobe targeted for transbronchial biopsies (e.g., RLL).""".strip(),
    "bronch_location_segment": """
Segment targeted for transbronchial biopsies (e.g., lateral basal).""".strip(),
    "bronch_guidance": """
Guidance modality for transbronchial biopsies (e.g., Fluoroscopy).""".strip(),
    "bronch_num_tbbx": """
Number of transbronchial biopsies obtained. Leave null if not documented.""".strip(),
    "bronch_tbbx_tool": """
Tool used for TBBx (e.g., Forceps).""".strip(),
    "bronch_specimen_tests": """
Specimen destinations/tests for TBBx (e.g., Histology, Microbiology).""".strip(),
    "bronch_immediate_complications": """
Immediate complications for TBBx (e.g., none, bleeding, pneumothorax).""".strip(),
    # Complications and outcomes fields
    "ebl_ml": """
Extract estimated blood loss in mL. Look for patterns like "EBL <5 mL", "EBL ~10 mL", "minimal blood loss (<5 mL)". Return integer. Null if not documented.""".strip(),
    "bleeding_severity": """
Extract bleeding severity. Allowed values: "None/Scant", "Mild", "Moderate", "Severe". Map: no bleeding/EBL <5mL/minimal -> "None/Scant", mild/scant -> "Mild". Null if not documented.""".strip(),
    "pneumothorax": """
True if pneumothorax occurred. False if explicitly stated no pneumothorax. Null if not mentioned.""".strip(),
    "hypoxia_respiratory_failure": """
Extract hypoxia/respiratory complication. Allowed values: "None", "Transient", "Escalation of Care", "Post-op Intubation". Map: no hypoxia/none -> "None", brief desaturation -> "Transient". Null if not mentioned.""".strip(),
    "fluoro_time_min": """
Extract fluoroscopy time in minutes. Look for patterns like "fluoro time 2.5 min", "0.0 minutes fluoroscopy". Return float. Null if not documented.""".strip(),
    "radiation_dap_gycm2": """
Extract radiation dose-area product in Gy·cm². Look for patterns like "DAP 3.3 Gy·cm²", "radiation dose 2.1". Return float. Null if not documented.""".strip(),
    # Disposition and follow-up
    "disposition": """
Immediate post-procedure disposition. Normalize to one of: "Discharged home", "PACU recovery", "Floor admission", "ICU admission".
- Map "discharged home", "dc home", "going home" -> "Discharged home"
- Map "PACU", "recovery room", "Phase I" -> "PACU recovery"
- Map "admitted to floor/ward/oncology/telemetry" -> "Floor admission"
- Map "ICU/MICU/SICU/critical care" -> "ICU admission"
Return the normalized value. Null if not documented.""".strip(),
    "follow_up_plan": """
Extract the follow-up plan. Include clinic appointments, pending results review, planned interventions. Keep concise. Null if not documented.""".strip(),
    "final_diagnosis_prelim": """
Preliminary final diagnosis category. Allowed: Malignancy, Granulomatous, Infectious, Non-diagnostic, Benign, Other (capitalize first letter).
- "Malignancy" when malignant cells/cancer confirmed or staging shows positive nodes
- "Granulomatous" for granulomas/sarcoidosis
- "Infectious" for infections identified
- "Non-diagnostic" when specimen inadequate/insufficient
- "Benign" for clearly benign/reactive findings
- "Other" for therapeutic/structural procedures
Null if unclear or pending.""".strip(),
    # Anticoagulation fields
    "anticoag_status": """
Extract anticoagulation status. Document what anticoagulant the patient takes (e.g., "Apixaban for atrial fibrillation", "Warfarin for DVT", "No anticoagulation"). Null if not documented.""".strip(),
    "anticoag_held_preprocedure": """
True if anticoagulation was held before the procedure. False if continued or not on anticoagulation. Null if not mentioned.""".strip(),
    # Anesthesia fields
    "anesthesia_agents": """
Extract anesthesia/sedation medications as comma-separated string. Include all agents mentioned (e.g., "Propofol, fentanyl, lidocaine", "Propofol infusion, fentanyl, rocuronium, sevoflurane"). Null if not documented.""".strip(),
    "sedation_reversal_given": """
True if reversal agent (flumazenil, naloxone) was given. False if explicitly stated no reversal needed. Null if not mentioned.""".strip(),
    "sedation_reversal_agent": """
Extract the specific reversal agent if given (e.g., "Flumazenil", "Naloxone"). Null if no reversal or not documented.""".strip(),
    # Airway fields
    "airway_type": """
Airway used for the procedure. Normalize to: "Native airway", "Endotracheal tube", "Laryngeal mask airway", "Tracheostomy", "Rigid bronchoscope".
- "Native airway" for natural airway, bite block only, nasal cannula without device
- "Endotracheal tube" for ETT/intubated/endotracheal intubation
- "Laryngeal mask airway" for LMA/laryngeal mask
- "Tracheostomy" for procedure via tracheostomy
- "Rigid bronchoscope" for rigid bronchoscopy
Null if not documented.""".strip(),
    "airway_device_size": """
Extract airway device size if documented (e.g., "7.5" for ETT, "4" for LMA). Return as string. Null if not documented.""".strip(),
    "ventilation_mode": """
Extract ventilation mode. Allowed values: "Spontaneous", "Jet Ventilation", "Controlled Mechanical Ventilation". Map: volume control/pressure control/mechanical -> "Controlled Mechanical Ventilation", spontaneous/native breathing -> "Spontaneous", jet -> "Jet Ventilation". Null if not documented.""".strip(),
    # Additional indication/primary fields
    "primary_indication": """
Extract the primary indication for the procedure. Describe concisely (e.g., "Mediastinal staging for right upper lobe lung mass", "Tissue diagnosis of mediastinal adenopathy"). Null if not documented.""".strip(),
    "prior_therapy": """
Extract any prior therapy documented (e.g., "Chemotherapy", "Prior CT-guided biopsy", "None"). Null if not documented.""".strip(),
    # Whole Lung Lavage (WLL) fields
    "wll_volume_instilled_l": """
Extract the total volume of saline instilled during whole lung lavage (WLL) in LITERS.
Look for phrases like:
- "instilled 30L", "30 liters instilled", "total instillation volume 32L"
- Convert if given in mL (30,000 mL = 30L)
Return as float (e.g., 30.0, 32.0). Null if not a WLL procedure or volume not documented.""".strip(),
    "wll_volume_returned_l": """
Extract the total volume of effluent/return during whole lung lavage (WLL) in LITERS.
Look for phrases like:
- "returned 27L", "27 liters recovered", "total return volume 25L"
- "effluent volume 28L"
- Convert if given in mL (27,000 mL = 27L)
Return as float (e.g., 27.0, 25.0). Null if not a WLL procedure or return volume not documented.""".strip(),
    "wll_dlt_used": """
True if a double-lumen tube (DLT) was used for whole lung lavage (WLL).
Look for:
- "Double-lumen tube", "DLT", "double lumen endotracheal tube"
- "Left DLT", "Right DLT", "39 Fr DLT"
False if single-lumen tube explicitly used for WLL. Null if not a WLL procedure or not documented.""".strip(),
    "molecular_testing_requested": """
True if molecular/genetic testing was requested or planned. False if explicitly declined. Null if not mentioned.""".strip(),
    # Pleural pressure fields
    "pleural_opening_pressure_cmh2o": """
Extract the pleural opening pressure in cm H2O. Look for:
- "Opening pressure -8 cm H2O", "initial pressure -12"
- "Pleural pressure at entry: -10 cm H2O"
- May be negative (subatmospheric) for effusions
Return as integer (can be negative). Null if not a pleural procedure or opening pressure not documented.""".strip(),
    "pleural_opening_pressure_measured": """
True if pleural opening pressure was measured during thoracentesis/thoracoscopy.
Look for any mention of opening pressure, initial pressure, or manometry.
False if explicitly stated pressure not measured. Null if not mentioned or not a pleural procedure.""".strip(),
    # Bronchial Thermoplasty fields
    "bt_lobe_treated": """
Extract the lobe(s) treated during bronchial thermoplasty. Return the full lobe name:
- "Right lower lobe" (not RLL)
- "Right middle lobe" (not RML)
- "Right upper lobe" (not RUL)
- "Left upper lobe" (not LUL)
- "Left lower lobe" (not LLL)
- "Bilateral upper lobes" for bilateral sessions
Include partial treatments (e.g., "Right lower lobe" even if partial/aborted).
Null if not a bronchial thermoplasty procedure.""".strip(),
    # Ablation fields
    "ablation_margin_assessed": """
True if post-ablation margin or zone assessment was performed. Look for:
- "CBCT ground glass", "ablation zone confirmed", "margin assessed"
- "Post-ablation imaging showed adequate margins"
- Any assessment of the ablation zone after the procedure
False if no margin assessment documented. Null if not an ablation procedure.""".strip(),
    "ablation_max_temp_c": """
Extract the maximum temperature reached during ablation in Celsius. Look for:
- "Temperature reached 95°C", "max temp 105C", "peak temperature 85°C"
- For RF ablation: typically 60-100°C
- For microwave ablation: typically 100-180°C
- For cryoablation: look for "freeze" cycles or "ice ball" - typical temperature is -40°C to -60°C
  Even if specific temperature not stated, cryoablation freeze cycles imply ~-40°C
Return as integer/float (can be negative for cryoablation). Null if not an ablation procedure or no temperature indication.""".strip(),
    "ablation_modality": """
Extract the ablation modality for PERIPHERAL LUNG NODULE/LESION ABLATION procedures only.
Allowed values:
- "Radiofrequency ablation" or "RFA" - for RF ablation of peripheral nodules
- "Microwave ablation" or "MWA" - for microwave ablation of peripheral nodules
- "Cryoablation" - for cryoablation of peripheral nodules (NOT endobronchial cryotherapy)

IMPORTANT DISTINCTION:
- "Cryoablation" applies ONLY to peripheral lung nodule ablation using cryoprobe for tumor destruction
- Do NOT confuse with "cryotherapy" for endobronchial tumor debulking (CAO procedures)
- Endobronchial cryotherapy/cryodebulking is NOT ablation_modality - leave null for those

Null if:
- Not an ablation procedure
- Procedure is endobronchial cryotherapy/cryodebulking (CAO, not ablation)
- Ablation modality not documented.""".strip(),
    # Navigation/radial EBUS improvements
    "nav_rebus_used": """
IMPORTANT: Return True if ANY radial EBUS terminology appears in the note.

Return True when ANY of these phrases appear (case-insensitive):
- "Radial EBUS", "r-EBUS", "radial probe", "radial ultrasound", "radial EBUS-guided"
- "Radial EBUS catheter", "Radial EBUS probe"
- "Concentric view" / "Eccentric view" **of the lesion/target** (radial EBUS view terminology)
- "Concentric solid lesion pattern" or similar **lesion/target** descriptions
- "Guide sheath" with peripheral biopsy
- For cryobiopsy/ILD: "radial probe", "parenchymal pattern", "absence of vessels"

DO NOT trigger radial EBUS from airway stenosis language:
- "concentric stenosis", "eccentric narrowing", "tracheal stenosis" are NOT radial EBUS evidence.

Radial EBUS is used for:
1. Peripheral nodule/lesion localization
2. Transbronchial cryobiopsy site selection for ILD
3. Any peripheral bronchoscopy with ultrasound confirmation

Return False only if explicitly stated "no radial EBUS" or "radial EBUS not used".
Return Null only if the procedure is purely linear EBUS mediastinal staging with NO mention of radial probe or concentric/eccentric *view of the lesion/target*.

CRITICAL: If you see "Radial EBUS" anywhere in the note, return True.""".strip(),
    "nav_rebus_view": """
IMPORTANT: Extract the radial EBUS view whenever radial EBUS is mentioned.

Look for these patterns and return them:
1. "Concentric" patterns (probe centered in lesion):
   - "Concentric view", "Concentric radial EBUS view", "Concentric solid lesion pattern"
   - Return: "Concentric radial EBUS view of lesion" or as documented

2. "Eccentric" patterns (probe adjacent to lesion):
   - "Eccentric view", "Eccentric radial EBUS view"
   - Return: "Eccentric radial EBUS view of lesion" or as documented

3. For cryobiopsy/ILD procedures:
   - "Parenchymal pattern", "absence of vessels", "vessel-free"
   - Return: "Parenchymal target free of large vessels on radial EBUS" or as documented

4. Other documented views:
   - Return verbatim as documented

CRITICAL: If the note mentions "Concentric" in context of radial EBUS or peripheral biopsy, extract it.
Example: "Concentric solid lesion pattern obtained" → return "Concentric radial EBUS view of lesion"

Null only if radial EBUS not performed OR no view/pattern description found.""".strip(),
    # Linear EBUS stations - prevent hallucination
    "linear_ebus_stations": """
List of mediastinal/hilar lymph node stations sampled with LINEAR EBUS-TBNA.
Use IASLC station names: 2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L, etc.

CRITICAL RULES:
1. ONLY populate when the note EXPLICITLY states that lymph node stations were SAMPLED with TBNA
2. Look for phrases like "TBNA at stations", "sampled at 4R, 7", "needle biopsies of stations"

MUST RETURN null/[] for:
- Radial EBUS procedures (keyword: "Radial EBUS", "r-EBUS", "radial probe")
- Transbronchial cryobiopsy for ILD
- Peripheral nodule biopsies (even with robotic/navigation)
- Any procedure where "Radial EBUS" is mentioned
- Procedures that mention "concentric/eccentric view" **of the lesion/target** (radial EBUS view terminology)

COMMON HALLUCINATION TO AVOID:
- Do NOT add station "7" just because the procedure involved bronchoscopy
- Do NOT add any stations for robotic/navigation peripheral biopsy cases
- Do NOT add stations for cryobiopsy/ILD cases

VERIFICATION: Before returning any stations, confirm the note contains:
- "Linear EBUS" or "EBUS-TBNA" (not just "EBUS" or "Radial EBUS")
- Explicit mention of lymph node station sampling with needle

If unsure, return null/[] - it's better to miss stations than to hallucinate them.""".strip(),

    # ============================================================================
    # GRANULAR PER-SITE DATA EXTRACTION INSTRUCTIONS
    # ============================================================================
    # These fields capture detailed per-site/per-node data for research and QI.
    # When populating, create one array element per site/station/target.
    # ============================================================================

    "granular_data": """
Optional container for granular per-site registry data. This is an object containing arrays for detailed per-site data. Populate when detailed per-site information is documented.

Structure:
{
  "linear_ebus_stations_detail": [...],
  "navigation_targets": [...],
  "cao_interventions_detail": [...],
  "blvr_valve_placements": [...],
  "blvr_chartis_measurements": [...],
  "cryobiopsy_sites": [...],
  "thoracoscopy_findings_detail": [...],
  "specimens_collected": [...]
}

Return null if no granular data is documented. Only populate sub-arrays that have data.""".strip(),

    "granular_data.linear_ebus_stations_detail": """
Per-station EBUS-TBNA data. Create ONE object per lymph node station that was sampled OR described with morphology.

For each station, extract:
- station: Use IASLC naming (2R, 2L, 4R, 4L, 7, 10R, 10L, 11R, 11L, 12R, 12L)
- short_axis_mm: Convert "0.54 cm" → 5.4, "5mm" → 5. For "1.2 x 0.8 cm" use shorter: 8
- shape: Map "oval/elliptical/elongated" → "oval", "round/spherical" → "round", "lobulated/asymmetric" → "irregular"
- margin: Map "well-defined/distinct/sharp" → "distinct", "ill-defined/blurred" → "indistinct"
- echogenicity: "homogeneous" for uniform/homogeneous, "heterogeneous" for mixed/non-uniform
- chs_present: true if "CHS present/preserved", false if "absent CHS/no CHS"
- necrosis_present: true if "necrosis/coagulation necrosis sign" documented
- morphologic_impression: Based on EBUS features ONLY (not ROSE/path):
  - "benign" if: oval + CHS present + <10mm + homogeneous
  - "malignant" if: round + CHS absent + ≥10mm + heterogeneous
  - "suspicious" if: some but not all malignant features
  - "indeterminate" if: mixed features or cannot determine
  - null if: morphology not described
- sampled: true if station was biopsied, false if only visualized
- number_of_passes: Extract exact number if stated (e.g., "3 passes" → 3)
- rose_result: ROSE finding for THIS station specifically (e.g., "Adequate lymphocytes", "Malignant")
- needle_gauge: 19, 21, 22, or 25 if documented
- needle_type: "Standard FNA", "FNB/ProCore", "Acquire", "ViziShot Flex"
- specimen_sent_for: Array of destinations (Cytology, Cell block, Molecular/NGS, etc.)

Return empty array [] if no linear EBUS performed or this is radial EBUS only.""".strip(),

    "granular_data.navigation_targets": """
Per-target data for navigation/robotic bronchoscopy. Create ONE object per target lesion.

For each target, extract:
- target_number: Sequential number (1, 2, 3...)
- target_location_text: Full anatomic description (e.g., "RUL apical segment")
- target_lobe: RUL, RML, RLL, LUL, LLL, or Lingula
- target_segment: Segment name if documented
- lesion_size_mm: Lesion size in mm (convert cm to mm)
- distance_from_pleura_mm: Distance from pleura if documented
- bronchus_sign: "Positive", "Negative", or "Not assessed"
- ct_characteristics: "Solid", "Part-solid", "Ground-glass", "Cavitary", "Calcified"
- pet_suv_max: SUV max value as float
- navigation_successful: true if target was reached, false if aborted
- rebus_used: true if radial EBUS used for this target
- rebus_view: MUST be EXACTLY one of these values: "Concentric", "Eccentric", "Adjacent", "Not visualized" (do NOT use descriptive phrases like "Concentric radial EBUS view of lesion" - use ONLY the single-word enum value)
- tool_in_lesion_confirmed: true if TIL confirmed (CBCT, fluoro, etc.)
- confirmation_method: "CBCT", "Augmented fluoroscopy", "Fluoroscopy", "Radial EBUS", "None"
- sampling_tools_used: Array of tools (Forceps, Needle (21G), Brush, Cryoprobe, etc.)
- number_of_forceps_biopsies: Count of forceps biopsies (use 0 if explicitly stated none taken, null if not documented)
- number_of_needle_passes: Count of needle passes (use 0 if explicitly stated none taken, null if not documented)
- number_of_cryo_biopsies: Count of cryo biopsies (use 0 if explicitly stated none taken, null if not documented)
- rose_performed, rose_result: ROSE for this target

IMPORTANT for Enum fields:
- bronchus_sign: Use ONLY "Positive", "Negative", or "Not assessed" - do NOT use True/False
- rebus_view: Use ONLY "Concentric", "Eccentric", "Adjacent", or "Not visualized"
- immediate_complication: "None", "Bleeding - mild/moderate/severe", "Pneumothorax"
- final_pathology: Final path result if documented

EXAMPLE OUTPUT:
{
  "target_number": 1,
  "target_location_text": "RUL posterior segment",
  "target_lobe": "RUL",
  "lesion_size_mm": 22,
  "rebus_used": true,
  "rebus_view": "Concentric",
  "tool_in_lesion_confirmed": true,
  "confirmation_method": "CBCT",
  "sampling_tools_used": ["Forceps", "Needle"],
  "number_of_forceps_biopsies": 5,
  "immediate_complication": "None"
}

Return empty array [] if no navigation/robotic bronchoscopy.""".strip(),

    "granular_data.cao_interventions_detail": """
Per-site CAO (Central Airway Obstruction) intervention data. Create ONE object per treatment site.

For each site, extract:
- location: Trachea - proximal/mid/distal, Carina, RMS, LMS, BI, RUL, RML, RLL, LUL, LLL
- obstruction_type: "Intraluminal", "Extrinsic", "Mixed"
- etiology: Use only the allowed literal strings:
  * "Malignant - primary lung", "Malignant - metastatic", "Malignant - other"
  * "Benign - post-intubation", "Benign - post-tracheostomy", "Benign - anastomotic"
  * "Benign - inflammatory", "Benign - infectious", "Benign - granulation", "Benign - web/stenosis", "Benign - other"
  * "Infectious", "Other"
- length_mm: Length of obstruction in mm
- pre_obstruction_pct, post_obstruction_pct: Percent obstruction before/after treatment (0-100)
- pre_diameter_mm, post_diameter_mm: Airway diameter before/after
- modalities_applied: Array of objects with:
  - modality: EXACT string from the allowed list:
    ["APC", "Electrocautery - snare", "Electrocautery - knife", "Electrocautery - probe",
     "Cryotherapy - spray", "Cryotherapy - contact", "Cryoextraction",
     "Laser - Nd:YAG", "Laser - CO2", "Laser - diode", "Laser",
     "Mechanical debulking", "Rigid coring", "Microdebrider",
     "Balloon dilation", "Balloon tamponade", "PDT",
     "Iced saline lavage", "Epinephrine instillation",
     "Tranexamic acid instillation", "Suctioning"]
  - power_setting_watts, balloon_diameter_mm, freeze_time_seconds, number_of_applications
- hemostasis_required, hemostasis_methods: Bleeding control needed and methods used
- stent_placed_at_site: true if stent deployed at this location

Return empty array [] if no CAO procedure.""".strip(),

    "granular_data.blvr_valve_placements": """
Per-valve placement data for BLVR. Create ONE object per valve placed.

For each valve, extract:
- valve_number: Sequential number (1, 2, 3...)
- target_lobe: RUL, RML, RLL, LUL, LLL, Lingula
- segment: Specific segment (e.g., "LB1+2", "LB6", "apical")
- airway_diameter_mm: Measured airway diameter
- valve_size: Size as string (e.g., "4.0", "4.0-LP", "5.5", "6.0", "7.0")
- valve_type: "Zephyr (Pulmonx)" or "Spiration (Olympus)"
- deployment_method: "Standard" or "Retroflexed"
- deployment_successful: true/false
- seal_confirmed: true if visual confirmation of seal
- repositioned: true if valve required repositioning

Return empty array [] if no BLVR valve placement.""".strip(),

    "granular_data.blvr_chartis_measurements": """
Chartis collateral ventilation measurements for BLVR. Create ONE object per lobe/segment assessed.

For each measurement:
- lobe_assessed: RUL, RML, RLL, LUL, LLL, Lingula
- segment_assessed: Specific segment if documented
- measurement_duration_seconds: Duration of measurement
- adequate_seal: true if adequate seal obtained
- cv_result: "CV Negative", "CV Positive", "Indeterminate", "Low flow", "No seal", "Aborted"
- flow_pattern_description: Free text description of flow pattern

Return empty array [] if no Chartis performed.""".strip(),

    "granular_data.cryobiopsy_sites": """
Per-site transbronchial cryobiopsy data. Create ONE object per biopsy site.

For each site:
- site_number: Sequential number (1, 2, 3...)
- lobe: RUL, RML, RLL, LUL, LLL, Lingula
- segment: Segment name if documented
- distance_from_pleura: ">2cm", "1-2cm", "<1cm", or "Not documented"
- probe_size_mm: 1.1, 1.7, 1.9, or 2.4
- freeze_time_seconds: Duration of freeze (typically 3-7 seconds)
- number_of_biopsies: Count of biopsies from this site
- specimen_size_mm: Approximate specimen size
- blocker_used: true if Fogarty/balloon blocker used
- blocker_type: "Fogarty", "Arndt", "Cohen", "Cryoprobe sheath"
- bleeding_severity: "None/Scant", "Mild", "Moderate", "Severe"
- bleeding_controlled_with: Method of hemostasis if needed
- pneumothorax_after_site: true if PTX occurred after this site

Return empty array [] if no transbronchial cryobiopsy.""".strip(),

    "granular_data.thoracoscopy_findings_detail": """
Per-location thoracoscopy/pleuroscopy findings. Create ONE object per distinct finding/location.

For each finding:
- location: Parietal pleura - chest wall/diaphragm/mediastinum, Visceral pleura, Lung parenchyma, etc.
- finding_type: Normal, Nodules, Plaques, Studding, Mass, Adhesions - filmy/dense, Inflammation, etc.
- extent: "Focal", "Multifocal", "Diffuse"
- size_description: Size in text form
- biopsied: true if biopsies taken from this site
- number_of_biopsies: Count
- biopsy_tool: "Rigid forceps", "Flexible forceps", "Cryoprobe"
- impression: "Benign appearing", "Malignant appearing", "Infectious appearing", "Indeterminate"

Return empty array [] if no thoracoscopy/pleuroscopy.""".strip(),

    "granular_data.specimens_collected": """
Unified specimen tracking with source linkage. Create ONE object per specimen/specimen jar.

For each specimen:
- specimen_number: Sequential number
- source_procedure: EBUS-TBNA, Navigation biopsy, Endobronchial biopsy, Transbronchial biopsy, Transbronchial cryobiopsy, BAL, Bronchial wash, Brushing, Pleural biopsy, Pleural fluid
- source_location: REQUIRED - Anatomic location of specimen origin (e.g., "4R", "RUL apical", "Left pleura"). Use the source_procedure as the location if specific location is not documented (e.g., "BAL", "Bronchial wash"). NEVER leave null.
- collection_tool: Tool used (e.g., "22G FNA", "Forceps", "1.9mm cryoprobe")
- specimen_count: Number of specimens in this jar/container
- specimen_adequacy: "Adequate", "Limited", "Inadequate", "Pending"
- destinations: Array of test types (Histology, Cytology, Cell block, Flow cytometry, Molecular/NGS, Culture, etc.)
- rose_performed, rose_result: ROSE for this specimen
- final_pathology_diagnosis: Final path result

EXAMPLE OUTPUT:
[
  {"specimen_number": 1, "source_procedure": "EBUS-TBNA", "source_location": "4R", "collection_tool": "22G FNA", "specimen_count": 3, "destinations": ["Cytology", "Cell block"]},
  {"specimen_number": 2, "source_procedure": "Navigation biopsy", "source_location": "RUL apical", "collection_tool": "Forceps", "specimen_count": 5, "destinations": ["Histology", "Molecular/NGS"]},
  {"specimen_number": 3, "source_procedure": "BAL", "source_location": "BAL", "destinations": ["Culture", "Cytology"]}
]

Return empty array [] if no specimens documented.""".strip(),
}


def _load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def _env_flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "y"}


def _normalize_active_families(value: object) -> set[str] | None:
    """Normalize context-provided active families into a set.

    - None -> None (meaning "don't gate")
    - "" -> empty set
    """
    if value is None:
        return None
    if isinstance(value, (set, frozenset)):
        return {str(x) for x in value if str(x)}
    if isinstance(value, list):
        return {str(x) for x in value if str(x)}
    if isinstance(value, str):
        v = value.strip()
        if not v:
            return set()
        return {p.strip() for p in v.split(",") if p.strip()}
    return None


def _build_field_instructions(schema_properties: dict[str, dict]) -> dict[str, str]:
    """Build per-field instruction text from schema properties."""
    instructions: dict[str, str] = {}
    for name, prop in schema_properties.items():
        desc = str(prop.get("description", "") or "").strip()
        enum = prop.get("enum")
        enum_text = f" Allowed values: {', '.join(enum)}." if enum else ""
        text = f"{desc}{enum_text} Use null if not documented."
        instructions[name] = text.strip()

    # Apply curated overrides for fields with frequent errors (only for included fields)
    for k, v in _FIELD_INSTRUCTION_OVERRIDES.items():
        if k in schema_properties:
            instructions[k] = v
    return instructions


def _load_field_instructions() -> dict[str, str]:
    """Build per-field instruction text from the schema description and enums."""

    global _FIELD_INSTRUCTIONS_CACHE
    if _FIELD_INSTRUCTIONS_CACHE is not None:
        return _FIELD_INSTRUCTIONS_CACHE

    schema = _load_schema()
    instructions = _build_field_instructions(schema.get("properties", {}))

    _FIELD_INSTRUCTIONS_CACHE = instructions
    return instructions


FIELD_INSTRUCTIONS: dict[str, str] = _load_field_instructions()


def _resolve_schema_version(context: dict[str, object]) -> str:
    raw = context.get("schema_version") or os.getenv("REGISTRY_SCHEMA_VERSION") or _DEFAULT_SCHEMA_VERSION
    return str(raw).strip().lower() if raw is not None else _DEFAULT_SCHEMA_VERSION


@lru_cache(maxsize=1)
def _registry_record_model_json_schema() -> str:
    # Import lazily to avoid heavy module import cost at startup.
    from app.registry.schema import RegistryRecord

    return json.dumps(RegistryRecord.model_json_schema(), indent=2)


def build_registry_extraction_prompt(note_text: str, context: dict | None = None) -> str:
    """Schema-driven extraction prompt (v3).

    Embeds the full Pydantic JSON schema so the model sees nested structures
    like `procedures_performed.linear_ebus.node_events`.
    """
    context = context or {}
    schema_json = _registry_record_model_json_schema()

    # Optional CPT guidance (hybrid flow only). Extraction-first typically omits this.
    verified_section = ""
    verified_codes = context.get("verified_cpt_codes") or []
    if verified_codes:
        coder_difficulty = context.get("coder_difficulty") or "unknown"
        hybrid_source = context.get("hybrid_source") or "unknown"
        codes_str = ", ".join(sorted(set(str(c) for c in verified_codes)))
        verified_section = (
            "\n--- CPT CODE GUIDANCE ---\n"
            f"Most likely CPT codes (difficulty={coder_difficulty}, source={hybrid_source}): {codes_str}.\n"
            "Use as weak guidance only; do not contradict the note.\n"
            "--- END CPT CODE GUIDANCE ---\n"
        )

    return f"""
You are an expert Clinical Registry Abstractor.
Your goal is to extract structured data from the procedure note below with 100% factual accuracy.

### 🛑 CRITICAL SCHEMA RULES (DO NOT IGNORE)
1. **NO OLD EBUS LIST-ONLY OUTPUTS:** Do not represent EBUS sampling only as a flat `stations_sampled` list. You MUST populate `procedures_performed.linear_ebus.node_events` with `action` and a verbatim `evidence_quote` per station.
2. **EVIDENCE HARD-GATING:** Only set any `performed: true` when there is explicit verbatim text describing the action. If you cannot find it, set `performed: false` (or null).
3. **DO NOT HALLUCINATE:** Do not infer diagnoses (e.g., \"Aspergilloma\") unless explicitly written in the note text. Do not copy details from schema descriptions or examples.
4. **OUTPUT STRICTNESS:** Return ONLY a single valid JSON object that conforms to the schema. No markdown, no code fences, no commentary.

### 🏥 SPECIFIC EXTRACTION RULES
- **EBUS:** If a node is described as \"sized\", \"viewed\", or \"benign ultrasound/sonographic characteristics\" but NOT biopsied/sampled, set `action: \"inspected_only\"` for that station.
- **EBUS Adequacy:** For each `granular_data.linear_ebus_stations_detail[]`, set `lymphocytes_present` only when explicitly stated (e.g., \"adequate lymphocytes\", \"no/scant lymphocytes\", \"blood only\" in ROSE context). Otherwise leave null.
- **Tracheostomy:** If the patient already has a trach and the scope goes through it, set `established_tracheostomy_route: true`. Only set `procedures_performed.percutaneous_tracheostomy.performed: true` if a NEW tracheostomy is created.
- **Rigid/Thermal:** For rigid bronchoscopy, look for \"rigid scope/rigid bronchoscope/rigid barrel\". For thermal ablation, look for \"electrocautery\", \"laser\", \"APC/argon plasma\".
- **ECOG:** Only populate `clinical_context.ecog_score` when explicitly documented (\"ECOG 2\", \"Zubrod 1\"). If a range is documented (\"ECOG 0-1\"), set `clinical_context.ecog_text` and leave `ecog_score` null.
- **Bronchus sign:** Only populate `clinical_context.bronchus_sign` when explicitly documented. Treat "air bronchogram present/absent" as bronchus sign positive/negative. Use ONLY \"Positive\", \"Negative\", or \"Not assessed\" (do NOT use True/False).
- **Bleeding grade:** Only populate `complications.bleeding.bleeding_grade_nashville` when explicitly documented (\"Nashville grade 2\"). Do not infer grade from equipment lists; require bleeding/hemostasis language.
- **Navigation targets (Tier 2):** When navigation/robotic bronchoscopy is performed, populate `granular_data.navigation_targets[]` per target when explicitly documented: `ct_characteristics` (Solid/Part-solid/Ground-glass/Cavitary/Calcified), `distance_from_pleura_mm` (mm; use 0 only when explicitly \"abutting\" pleura), `pet_suv_max` (numeric SUV only when explicitly documented), and `bronchus_sign` (Positive/Negative/Not assessed when explicitly stated).
- **Tool-in-lesion (Tier 2):** Only set `procedures_performed.navigational_bronchoscopy.tool_in_lesion_confirmed` and `confirmation_method` when the note explicitly confirms tool-in-lesion (Radial EBUS/CBCT/Fluoroscopy/Augmented Fluoroscopy/None). Do not infer from equipment lists.
- **Pneumothorax intervention (Tier 2):** If pneumothorax occurred, populate `complications.pneumothorax.intervention[]` only when explicitly documented (Observation/Aspiration/Pigtail catheter/Chest tube/Heimlich valve/Surgery).
{verified_section}

### TARGET JSON SCHEMA:
{schema_json}

### PROCEDURE NOTE:
{note_text}

### OUTPUT:
Return ONLY the valid JSON object.
""".strip()


def _load_registry_prompt() -> str:
    """Load schema from IP_Registry.json and build a concise field guide for the LLM."""

    global _PROMPT_CACHE
    if _PROMPT_CACHE is not None:
        return _PROMPT_CACHE

    instructions = _load_field_instructions()
    lines = ["Return JSON with the following fields (use null if missing):"]
    for name, text in instructions.items():
        # Billing/CPT codes are derived deterministically downstream; do not ask the LLM to populate them.
        if name == "billing":
            continue
        lines.append(f'- "{name}": {text}')

    _PROMPT_CACHE = f"{load_system_prompt()}\n\n{PROMPT_HEADER}\n\n" + "\n".join(lines)
    return _PROMPT_CACHE


@lru_cache(maxsize=64)
def _load_registry_prompt_for_families(active_families_frozen: frozenset[str] | None) -> str:
    """Optional procedure-family gated prompt (disabled by default).

    Enable by setting: REGISTRY_PROMPT_FILTER_BY_FAMILY=1
    """
    if active_families_frozen is None:
        return _load_registry_prompt()

    schema = _load_schema()
    properties = schema.get("properties", {})

    from app.registry.schema_filter import filter_schema_properties

    filtered_properties = filter_schema_properties(properties, set(active_families_frozen))
    instructions = _build_field_instructions(filtered_properties)

    lines = ["Return JSON with the following fields (use null if missing):"]
    for name, text in instructions.items():
        if name == "billing":
            continue
        lines.append(f'- "{name}": {text}')
    return f"{load_system_prompt()}\n\n{PROMPT_HEADER}\n\n" + "\n".join(lines)


def build_registry_prompt(note_text: str, context: dict | None = None) -> str:
    """Build the registry extraction prompt with optional CPT context.

    Args:
        note_text: The procedure note text to extract from.
        context: Optional extraction context with hints from hybrid coder:
            - verified_cpt_codes: List of CPT codes from hybrid coder
            - coder_difficulty: Case difficulty classification (HIGH_CONF/GRAY_ZONE/LOW_CONF)
            - hybrid_source: Source of codes (ml_rules_fastpath, hybrid_llm_fallback)

    Returns:
        Complete prompt string for the LLM.
    """
    context = context or {}
    schema_version = _resolve_schema_version(context)
    if schema_version == "v3":
        return build_registry_extraction_prompt(note_text, context=context)

    active_families = None
    if _env_flag("REGISTRY_PROMPT_FILTER_BY_FAMILY", False):
        active_families = _normalize_active_families(context.get("active_families"))

    prompt_text = _load_registry_prompt_for_families(
        frozenset(active_families) if active_families is not None else None
    )

    # Build verified CPT section if codes are provided
    verified_section = ""
    verified_codes = context.get("verified_cpt_codes") or []
    if verified_codes:
        coder_difficulty = context.get("coder_difficulty") or "unknown"
        hybrid_source = context.get("hybrid_source") or "unknown"
        codes_str = ", ".join(sorted(set(str(c) for c in verified_codes)))

        verified_section = (
            "\n--- CPT CODE GUIDANCE ---\n"
            f"The automated coding system has identified the following CPT codes as "
            f"most likely for this note (difficulty={coder_difficulty}, source={hybrid_source}): {codes_str}.\n\n"
            "Use these as PRIMARY GUIDANCE when determining which registry fields to set:\n"
            "- 31652/31653 (EBUS-TBNA) → set EBUS-related fields\n"
            "- 31624/31625 (BAL) → set BAL-related fields\n"
            "- 31628/31632 (Transbronchial biopsy) → set TBBx fields\n"
            "- 31629/31633 (TBNA) → set peripheral TBNA fields (distinct from EBUS)\n"
            "- 31627 (Navigation) → set navigation fields\n"
            "- 31636/31637 (Stent) → set stent fields\n"
            "- 32555/32556/32557 (Thoracentesis) → set pleural fields\n"
            "- 31647/31648/31649 (BLVR valves) → set BLVR fields\n\n"
            "Do NOT infer procedures that contradict these codes. If the note clearly "
            "contradicts a suggested code, you may explain this and omit that procedure.\n"
            "--- END CPT CODE GUIDANCE ---\n\n"
        )

    return f"{prompt_text}{verified_section}\n\n### PROCEDURE NOTE:\n{note_text}\nJSON:"


__all__ = ["build_registry_prompt", "build_registry_extraction_prompt", "FIELD_INSTRUCTIONS"]
