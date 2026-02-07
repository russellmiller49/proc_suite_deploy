# Interventional Pulmonology Procedural Report Templates

## Overview

This is a comprehensive Jinja2 template system for generating Interventional Pulmonology (IP) procedural reports. The templates cover all major IP procedures and are designed to produce consistent, detailed, and properly formatted documentation.

## Directory Structure

```
ip_templates/
├── templates/
│   ├── base.j2                        # Common macros and utilities
│   ├── main.j2                        # Master import file
│   ├── 01_minor_trach_laryngoscopy.j2 # Minor procedures, tracheostomy, laryngoscopy
│   ├── 02_core_bronchoscopy.j2        # Core diagnostic/therapeutic bronchoscopy
│   ├── 03_navigation_robotic_ebus.j2  # Navigation, robotic, EBUS procedures
│   ├── 04_blvr_cryo.j2                # BLVR and cryotherapy procedures
│   ├── 05_pleural.j2                  # Pleural procedures
│   ├── 06_other_interventions.j2      # WLL, EUS-B, PEG, BPF procedures
│   └── 07_clinical_assessment.j2      # Clinical notes, operative reports, discharge
├── example_usage.py                   # Python examples
└── README.md                          # This documentation
```

## Installation

```bash
pip install jinja2
```

## Quick Start

```python
from jinja2 import Environment, FileSystemLoader

# Set up the environment
env = Environment(loader=FileSystemLoader('templates'))

# Load a template
template = env.get_template('02_core_bronchoscopy.j2')

# Generate a report
report = template.module.bronchoalveolar_lavage(
    lung_segment='RML',
    volume_instilled=120,
    volume_returned=80,
    testing_types=['Cell count', 'Cultures', 'Cytology']
)

print(report)
```

## Template Categories

### 1. Minor Procedures & Tracheostomy (`01_minor_trach_laryngoscopy.j2`)

| Macro | CPT | Description |
|-------|-----|-------------|
| `minor_trach_bleeding()` | 12001 | Control of minor tracheostomy bleeding |
| `chemical_cauterization()` | 17250 | Chemical cauterization of granulation tissue |
| `trach_tube_change()` | 31502 | Tracheostomy tube change |
| `percutaneous_trach()` | 31600 | Percutaneous tracheostomy with bronchoscopy |
| `trach_revision()` | 31613 | Percutaneous tracheostomy revision |
| `tracheobronchoscopy_via_trach()` | 31615 | Tracheobronchoscopy via tracheostomy |
| `trach_decannulation()` | — | Tracheostomy decannulation |
| `trach_downsize()` | 31502 | Tracheostomy downsizing/fenestrated placement |
| `granulation_debridement()` | — | Stoma/tracheal granulation debridement |
| `flexible_laryngoscopy()` | 31575 | Flexible fiberoptic laryngoscopy |

### 2. Core Bronchoscopy Procedures (`02_core_bronchoscopy.j2`)

| Macro | CPT | Description |
|-------|-----|-------------|
| `bronchial_washing()` | 31622 | Bronchial washing |
| `bronchial_brushings()` | 31623 | Bronchial brushings |
| `bronchoalveolar_lavage()` | 31624 | Bronchoalveolar lavage (BAL) |
| `endobronchial_biopsy()` | 31625 | Endobronchial biopsy |
| `transbronchial_lung_biopsy()` | 31628 | Transbronchial lung biopsy (TBLB) |
| `transbronchial_needle_aspiration()` | 31629/31633 | TBNA |
| `balloon_dilation()` | 31630 | Balloon dilation |
| `airway_stent_placement()` | 31631/31636/31637 | Airway stent placement |
| `endobronchial_tumor_destruction()` | 31641 | Tumor destruction (single modality) |
| `tumor_destruction_multimodal()` | 31641 | Tumor destruction (multiple modalities) |
| `therapeutic_aspiration()` | 31645/31646 | Therapeutic aspiration |
| `endobronchial_tumor_excision()` | 31640 | Endobronchial tumor excision |
| `rigid_bronchoscopy()` | — | Rigid bronchoscopy |
| `foreign_body_removal()` | — | Foreign body removal |
| `stent_removal_revision()` | — | Airway stent removal/revision |
| `endobronchial_hemostasis()` | — | Hemostasis for hemoptysis |
| `endobronchial_blocker()` | — | Endobronchial blocker placement |
| `pdt_light_application()` | — | PDT light application |
| `pdt_debridement()` | — | PDT debridement |
| `cryo_extraction_mucus()` | — | Cryo-extraction of mucus casts |
| `awake_foi()` | — | Awake fiberoptic intubation |
| `dlt_confirmation()` | — | DLT placement confirmation |
| `stent_surveillance()` | — | Stent surveillance bronchoscopy |

### 3. Navigation/Robotic/EBUS (`03_navigation_robotic_ebus.j2`)

| Macro | CPT | Description |
|-------|-----|-------------|
| `emn_bronchoscopy()` | 31627 | Electromagnetic navigation bronchoscopy |
| `fiducial_marker_placement()` | 31626 | Fiducial marker placement |
| `radial_ebus_survey()` | 31654 | Radial EBUS survey |
| `robotic_bronchoscopy_ion()` | — | Robotic bronchoscopy (Ion/Intuitive) |
| `ion_registration_complete()` | — | Ion registration - complete |
| `ion_registration_partial()` | — | Ion registration - partial/ssRAB |
| `ion_registration_drift()` | — | Ion registration drift/mismatch |
| `linear_ebus_tbna()` | 31652/31653 | Linear EBUS with TBNA |
| `rebus_guide_sheath_sampling()` | 31654 | Radial EBUS with guide sheath |
| `cbct_assisted_bronchoscopy()` | — | CBCT/augmented fluoroscopy |
| `transbronchial_dye_marking()` | — | Dye marking for surgical localization |
| `robotic_bronchoscopy_monarch()` | — | Robotic bronchoscopy (Monarch/Auris) |
| `ebus_intranodal_forceps_biopsy()` | 31652/31653 | EBUS-guided IFB |
| `ebus_19g_fnb()` | 31652/31653 | EBUS 19G core FNB |

### 4. BLVR & Cryotherapy (`04_blvr_cryo.j2`)

| Macro | CPT | Description |
|-------|-----|-------------|
| `endobronchial_valve_placement()` | 31647/31648 | EBV placement for COPD |
| `endobronchial_valve_removal_exchange()` | — | EBV removal/exchange |
| `post_blvr_protocol()` | — | Post-BLVR management protocol |
| `transbronchial_cryobiopsy()` | 31632 | Transbronchial cryobiopsy |
| `endobronchial_cryoablation()` | 31641 | Endobronchial cryoablation |
| `cryoablation_alternative()` | 31641 | Alternative cryoablation template |

### 5. Pleural Procedures (`05_pleural.j2`)

| Macro | CPT | Description |
|-------|-----|-------------|
| `chest_tube_placement()` | 32551 | Image-guided chest tube |
| `thoracentesis()` | 32555 | Thoracentesis |
| `thoracentesis_manometry()` | 32555 | Thoracentesis with manometry |
| `tpc_placement()` | 32550 | Tunneled pleural catheter placement |
| `tpc_removal()` | 32552 | TPC removal |
| `intrapleural_fibrinolysis()` | 32561/32562 | Intrapleural fibrinolysis |
| `medical_thoracoscopy()` | 32650/32653 | Medical thoracoscopy |
| `pigtail_catheter_placement()` | 32557 | Pigtail catheter placement |
| `transthoracic_needle_biopsy()` | 32408 | Transthoracic needle biopsy |
| `thoravent_placement()` | — | Thoravent placement |
| `chemical_pleurodesis_chest_tube()` | 32560 | Chemical pleurodesis via chest tube |
| `chemical_pleurodesis_ipc()` | — | Chemical pleurodesis via IPC |
| `ipc_exchange()` | — | IPC exchange |
| `chest_tube_exchange()` | — | Chest tube exchange/upsizing |
| `chest_tube_removal()` | — | Chest tube removal |
| `us_guided_pleural_biopsy()` | 32507 | US-guided pleural biopsy |
| `focused_thoracic_ultrasound()` | 76604 | Focused thoracic ultrasound |

### 6. Other Interventions (`06_other_interventions.j2`)

| Macro | CPT | Description |
|-------|-----|-------------|
| `whole_lung_lavage()` | 32997 | Whole lung lavage |
| `eus_b()` | 43237 | EUS-B |
| `paracentesis()` | 49083 | Paracentesis |
| `peg_placement()` | — | PEG placement |
| `peg_removal_exchange()` | — | PEG removal/exchange |
| `bpf_localization()` | — | BPF localization test |
| `ebv_for_air_leak()` | — | EBV for persistent air leak |
| `bpf_sealant_application()` | — | BPF sealant application |

### 7. Clinical Assessment (`07_clinical_assessment.j2`)

| Macro | Description |
|-------|-------------|
| `pre_anesthesia_assessment()` | Pre-procedure sedation assessment |
| `general_bronchoscopy_note()` | General bronchoscopy procedure note |
| `ip_operative_report()` | Full IP operative report |
| `tpc_discharge_instructions()` | TPC discharge instructions |
| `blvr_discharge_instructions()` | BLVR discharge instructions |
| `chest_tube_discharge_instructions()` | Chest tube discharge instructions |
| `peg_discharge_instructions()` | PEG discharge instructions |

## Base Utilities (`base.j2`)

Common macros available for use across all templates:

- `specimen_list(specimens)` - Format specimen list
- `chest_ultrasound_findings(us)` - Format chest ultrasound findings
- `ventilation_parameters(vent)` - Format ventilator settings
- `procedure_tolerance(complications, complications_detail)` - Procedure tolerance statement
- `anesthesia_block(anesthesia)` - Format anesthesia details
- `cxr_ordered(ordered)` - CXR ordered statement
- `staff_present(attending, fellow, rn, rt)` - Staff present documentation
- `consent_block(consent_type)` - Consent documentation
- `timeout_block()` - Timeout documentation
- `rose_result(rose)` - ROSE cytology result

## Advanced Usage

### Combining Multiple Procedures

```python
env = Environment(loader=FileSystemLoader('templates'))
bronch = env.get_template('02_core_bronchoscopy.j2')
ebus = env.get_template('03_navigation_robotic_ebus.j2')

# Generate combined report
report = f"""
PROCEDURE REPORT

{bronch.module.bronchoalveolar_lavage(
    lung_segment='RML',
    volume_instilled=120,
    volume_returned=80,
    testing_types=['Cultures', 'Cytology']
)}

{ebus.module.linear_ebus_tbna(lymph_nodes=[
    {'station': '7', 'size_mm': 15, 'ct_findings': 'enlarged',
     'pet_findings': 'PET avid', 'num_tbna': 4, 'rose': None}
])}
"""
```

### Using Base Utilities

```python
from jinja2 import Environment, FileSystemLoader

env = Environment(loader=FileSystemLoader('templates'))
base = env.get_template('base.j2')

# Use utility macros
specimens = base.module.specimen_list(['Cytology', 'Cell block', 'Cultures'])
vent = base.module.ventilation_parameters({
    'mode': 'Volume Control',
    'respiratory_rate': 14,
    'tidal_volume': '450 mL',
    'peep': '5 cmH2O',
    'fio2': '100%',
    'flow_rate': '40 L/min',
    'mean_pressure': '12 cmH2O'
})
```

## Customization

### Adding Custom Templates

1. Create a new `.j2` file in the `templates/` directory
2. Define your macros using Jinja2 syntax
3. Import in `main.j2` if desired

### Modifying Existing Templates

Templates use standard Jinja2 syntax with:
- `{{ variable }}` for variable substitution
- `{% if condition %}...{% endif %}` for conditionals
- `{% for item in list %}...{% endfor %}` for loops
- `{# comment #}` for comments

## Best Practices

1. **Always verify CPT codes** - Codes marked `[verify]` should be validated against current coding guidelines
2. **Customize patient information** - Replace all placeholder values with actual patient data
3. **Review institutional policies** - Ensure templates comply with local documentation requirements
4. **Validate outputs** - Review generated reports before finalizing

## Contributing

To add new templates or modify existing ones:
1. Follow the established naming conventions
2. Include appropriate CPT codes where applicable
3. Use descriptive parameter names
4. Add documentation for new macros

## License

For internal educational and clinical documentation use. Verify compliance with institutional policies before deployment.
