#!/usr/bin/env python3
"""
Interventional Pulmonology Procedural Report Template System
Example usage of Jinja2 templates for generating IP procedure reports

Author: IP Fellowship Program
Version: 1.0
"""

from jinja2 import Environment, FileSystemLoader
from pathlib import Path
from datetime import datetime


def get_template_env(template_dir: str = "templates") -> Environment:
    """Create and return a Jinja2 environment configured for IP templates."""
    return Environment(
        loader=FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=True
    )


def example_bronchoscopy_with_bal():
    """Example: Generate a bronchoscopy note with BAL."""
    env = get_template_env()
    
    # Load the clinical assessment template
    template = env.get_template("07_clinical_assessment.j2")
    
    # Render the template with data
    report = template.module.general_bronchoscopy_note(
        procedures=['bronchoalveolar lavage', 'bronchial washings'],
        operators=['Smith', 'Jones'],
        indications=['pneumonia workup', 'suspected infection'],
        sedation_type='moderate sedation with midazolam and fentanyl',
        lidocaine_volume=5,
        lidocaine_concentration=2,
        bal_performed=True,
        bal_segment='RML',
        bal_instilled=120,
        bal_returned=80,
        findings={
            'trachea': 'Normal caliber, no lesions',
            'right_lung': 'Patent airways, minimal secretions',
            'left_lung': 'Patent airways, no lesions',
            'mucosa': 'Normal throughout',
            'secretions': 'Minimal clear secretions'
        }
    )
    
    print("=" * 80)
    print("EXAMPLE: Bronchoscopy with BAL")
    print("=" * 80)
    print(report)
    return report


def example_ebus_tbna():
    """Example: Generate an EBUS-TBNA report."""
    env = get_template_env()
    template = env.get_template("03_navigation_robotic_ebus.j2")
    
    # Define lymph node data
    lymph_nodes = [
        {
            'station': '7',
            'size_mm': 15,
            'ct_findings': 'enlarged on CT',
            'pet_findings': 'PET avid (SUV 8.5)',
            'num_tbna': 4,
            'rose': {
                'adequacy': 'adequate',
                'findings': 'malignant cells consistent with adenocarcinoma'
            }
        },
        {
            'station': '4R',
            'size_mm': 12,
            'ct_findings': 'borderline enlarged',
            'pet_findings': 'mildly PET avid (SUV 3.2)',
            'num_tbna': 3,
            'rose': {
                'adequacy': 'adequate',
                'findings': 'benign lymphoid tissue'
            }
        },
        {
            'station': '11R',
            'size_mm': 10,
            'ct_findings': 'normal size',
            'pet_findings': 'not PET avid',
            'num_tbna': 2,
            'rose': {
                'adequacy': 'adequate',
                'findings': 'benign lymphoid tissue'
            }
        }
    ]
    
    report = template.module.linear_ebus_tbna(lymph_nodes=lymph_nodes)
    
    print("=" * 80)
    print("EXAMPLE: EBUS-TBNA Report")
    print("=" * 80)
    print(report)
    return report


def example_robotic_bronchoscopy_ion():
    """Example: Generate a robotic bronchoscopy (Ion) report."""
    env = get_template_env()
    template = env.get_template("03_navigation_robotic_ebus.j2")
    
    vent_params = {
        'mode': 'Volume Control',
        'respiratory_rate': 14,
        'tidal_volume': '450 mL',
        'peep': '5 cmH2O',
        'fio2': '100%',
        'flow_rate': '40 L/min',
        'mean_pressure': '12 cmH2O'
    }
    
    sampling_details = """
Radial EBUS confirmed concentric position within the target lesion.
Sampling performed:
- Transbronchial needle aspiration x 4 passes
- Transbronchial forceps biopsy x 5 samples
- Bronchial brushings x 2

ROSE: Adequate, favor malignancy - consistent with adenocarcinoma
"""
    
    report = template.module.robotic_bronchoscopy_ion(
        vent_params=vent_params,
        sampling_details=sampling_details,
        cbct_performed=True
    )
    
    print("=" * 80)
    print("EXAMPLE: Robotic Bronchoscopy (Ion) Report")
    print("=" * 80)
    print(report)
    return report


def example_tpc_placement():
    """Example: Generate a tunneled pleural catheter placement report."""
    env = get_template_env()
    template = env.get_template("05_pleural.j2")
    
    us_findings = """
Hemithorax: Right
Pleural Effusion Volume: Large
Echogenicity: Anechoic
Loculations: None
Diaphragmatic Motion: Diminished
Lung sliding before procedure: Present
Lung sliding post procedure: Present
Lung consolidation/atelectasis: Absent
Pleura: Thick
"""
    
    pleural_pressures = {
        'opening': -2,
        'at_500': -8,
        'at_1000': -15,
        'at_1500': -22
    }
    
    report = template.module.tpc_placement(
        side='Right',
        intercostal_space='5th',
        location='mid-axillary line',
        tunnel_length_cm=8,
        exit_site_location='6th intercostal space anterior axillary line',
        volume_removed=1500,
        fluid_appearance='Serosanguinous',
        lidocaine_volume=20,
        us_findings=us_findings,
        pleural_pressures=pleural_pressures,
        drainage_device='Capped',
        suction=False,
        specimens=['Cell count', 'Chemistry', 'Cytology', 'Culture'],
        cxr_ordered=True,
        sutured=True
    )
    
    print("=" * 80)
    print("EXAMPLE: Tunneled Pleural Catheter Placement")
    print("=" * 80)
    print(report)
    return report


def example_blvr_valve_placement():
    """Example: Generate a BLVR valve placement report."""
    env = get_template_env()
    template = env.get_template("04_blvr_cryo.j2")
    
    valve_details = [
        {'segment': 'LB1+2', 'size': '5.5mm'},
        {'segment': 'LB3', 'size': '5.0mm'},
        {'segment': 'LB4+5', 'size': '6.0mm'},
    ]
    
    report = template.module.endobronchial_valve_placement(
        balloon_occlusion=True,
        chartis_used=True,
        air_leak_confirmed=True,
        lobes_treated=['Left Upper'],
        valve_brand='Zephyr',
        valve_details=valve_details
    )
    
    print("=" * 80)
    print("EXAMPLE: BLVR Valve Placement")
    print("=" * 80)
    print(report)
    return report


def example_transbronchial_cryobiopsy():
    """Example: Generate a transbronchial cryobiopsy report."""
    env = get_template_env()
    template = env.get_template("04_blvr_cryo.j2")
    
    report = template.module.transbronchial_cryobiopsy(
        forceps_tool='1.9mm Cryoprobe',
        lung_segment='RLL posterior basal segment (B10)',
        num_samples=5,
        testing_types=['Histopathology', 'IHC panel', 'Cultures'],
        blocker_used=True,
        blocker_volume_cc=6,
        rebus_performed=True
    )
    
    print("=" * 80)
    print("EXAMPLE: Transbronchial Cryobiopsy")
    print("=" * 80)
    print(report)
    return report


def example_operative_report():
    """Example: Generate a full IP operative report."""
    env = get_template_env()
    template = env.get_template("07_clinical_assessment.j2")
    
    report = template.module.ip_operative_report(
        procedure_date=datetime.now().strftime("%m/%d/%Y"),
        referring_physician="Dr. Primary Care",
        patient_name="John Doe",
        patient_age=65,
        patient_sex="male",
        indication="right upper lobe mass suspicious for malignancy",
        consent_type='standard',
        preop_dx=['ICD10: R91.1 - Solitary pulmonary nodule', 'ICD10: Z87.891 - History of tobacco use'],
        postop_dx=['ICD10: C34.11 - Malignant neoplasm of RUL', 'ICD10: R91.1 - Solitary pulmonary nodule'],
        cpt_codes=['31652 - EBUS-TBNA', '31623 - Bronchial brushings', '31624 - BAL'],
        modifier_codes=['-59'],
        hcpcs_codes=None,
        attending="Dr. IP Attending",
        fellow="Dr. IP Fellow",
        rn="RN Name",
        rt="RT Name",
        anesthesia_type="General anesthesia",
        anesthesia_details="propofol/fentanyl/rocuronium",
        equipment_used="Olympus BF-UC180F linear EBUS bronchoscope, 22G EBUS needle",
        ebl="Minimal (<5mL)",
        complications='None',
        patient_position="Supine",
        airway_findings="""
Vocal cords: Normal mobility bilaterally
Trachea: Normal caliber, no lesions
Carina: Sharp, normal position
Right lung: All segments patent, no endobronchial lesions
Left lung: All segments patent, no endobronchial lesions
RUL mass: External compression visible at RB1""",
        procedure_details="""
Linear EBUS was performed. Station 4R lymph node (12mm) was identified and sampled x 4 passes.
Station 7 lymph node (18mm) was identified and sampled x 4 passes.
Station 11R lymph node (8mm) was identified and sampled x 3 passes.

ROSE cytology was performed with preliminary results:
- Station 4R: Benign lymphoid tissue
- Station 7: Malignant cells, favor adenocarcinoma
- Station 11R: Benign lymphoid tissue

BAL was performed in the RUL with 60mL instilled and 40mL returned.
Bronchial brushings were obtained from RUL.""",
        specimens=['EBUS-TBNA 4R x4', 'EBUS-TBNA 7 x4', 'EBUS-TBNA 11R x3', 'BAL RUL', 'Brushings RUL'],
        impression_plan="""
EBUS-TBNA with preliminary ROSE suggesting adenocarcinoma in station 7.
Final pathology pending. Will discuss at tumor board.
Recommend PET-CT for complete staging if not already done.
Follow-up in IP clinic for results discussion and treatment planning."""
    )
    
    print("=" * 80)
    print("EXAMPLE: Full IP Operative Report")
    print("=" * 80)
    print(report)
    return report


def example_trach_change():
    """Example: Generate a tracheostomy tube change note."""
    env = get_template_env()
    template = env.get_template("01_minor_trach_laryngoscopy.j2")
    
    report = template.module.trach_tube_change(
        old_trach_type='Shiley',
        old_size='8.0',
        new_trach_type='Shiley',
        new_size='6.0'
    )
    
    print("=" * 80)
    print("EXAMPLE: Tracheostomy Tube Change")
    print("=" * 80)
    print(report)
    return report


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("INTERVENTIONAL PULMONOLOGY TEMPLATE EXAMPLES")
    print("=" * 80 + "\n")
    
    examples = [
        ("Tracheostomy Tube Change", example_trach_change),
        ("Bronchoscopy with BAL", example_bronchoscopy_with_bal),
        ("EBUS-TBNA", example_ebus_tbna),
        ("Robotic Bronchoscopy (Ion)", example_robotic_bronchoscopy_ion),
        ("Tunneled Pleural Catheter", example_tpc_placement),
        ("BLVR Valve Placement", example_blvr_valve_placement),
        ("Transbronchial Cryobiopsy", example_transbronchial_cryobiopsy),
        ("Full Operative Report", example_operative_report),
    ]
    
    for name, func in examples:
        try:
            print(f"\n>>> Running: {name}")
            func()
            print()
        except Exception as e:
            print(f"Error in {name}: {e}")
    
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
