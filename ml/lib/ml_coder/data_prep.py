"""
Data preparation module for registry ML training.

Builds clean training CSVs from golden JSONs and existing registry data
with patient-level splitting to support Silver Standard training
(Train on Synthetic+Real, Test on Real).

Key design goals:
- Single Source of Truth: In production, labels are derived using the canonical implementation.
  Here, we map Golden JSON structured data (CPT codes + Registry Fields) into the target
  boolean schema found in registry_train.csv.
- Extraction-First: Uses structured evidence from JSONs as ground-truth labels for the
  synthetic partition.

Registry-First Alternative:
  For the registry-first approach that extracts labels directly from the nested
  registry_entry structure (rather than CPT-based derivation), use:

      from ml.lib.ml_coder.registry_data_prep import prepare_registry_training_splits
      train_df, val_df, test_df = prepare_registry_training_splits()
"""

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import KnowledgeSettings
from ml.lib.ml_coder.registry_label_schema import REGISTRY_LABELS
from ml.lib.ml_coder.valid_ip_codes import VALID_IP_CODES

# Canonical boolean procedure columns (30) for registry training.
BOOLEAN_COLUMNS = list(REGISTRY_LABELS)

# ---------------------------------------------------------------------------
# Legacy ML-coder data_prep public contract
# ---------------------------------------------------------------------------
#
# Many callers (including tests and API code) import the following symbols from
# `ml.lib.ml_coder.data_prep`. A recent refactor shifted registry training code
# into this module and broke the older ML-coder training contract.
#
# The functions below restore that contract while keeping the registry training
# pipeline intact.

EDGE_SOURCE_NAME = "synthetic_edge_case_notes_with_registry.jsonl"


def _extract_codes(entry: dict) -> list[str]:
    """
    Backward-compatible CPT code extractor.

    Preference order (when present and non-empty):
    1) entry["coding_review"]["final_cpt_codes"] (list)
    2) entry["coding_review"]["cpt_summary"]["final_codes"] (list)
    3) entry["coding_review"]["cpt_summary"] keys when it's a dict keyed by code
    4) entry["coding_review"]["cpt_summary"] list of objects with {"code": ...}
    5) entry["cpt_codes"]
    """
    if not isinstance(entry, dict):
        return []

    fallback = entry.get("cpt_codes") or []
    fallback = [str(c).strip() for c in fallback if str(c).strip()]

    coding_review = entry.get("coding_review")
    if not isinstance(coding_review, dict):
        return fallback

    final_codes = coding_review.get("final_cpt_codes")
    if isinstance(final_codes, list) and final_codes:
        return [str(c).strip() for c in final_codes if str(c).strip()]

    cpt_summary = coding_review.get("cpt_summary")
    if isinstance(cpt_summary, dict):
        summary_final = cpt_summary.get("final_codes")
        if isinstance(summary_final, list) and summary_final:
            return [str(c).strip() for c in summary_final if str(c).strip()]

        # If cpt_summary is keyed by code, return those keys.
        keys = [str(k).strip() for k in cpt_summary.keys() if str(k).strip()]
        if keys and all(any(ch.isdigit() for ch in k) for k in keys):
            return keys

    if isinstance(cpt_summary, list):
        codes = []
        for item in cpt_summary:
            if isinstance(item, dict) and item.get("code"):
                code = str(item["code"]).strip()
                if code:
                    codes.append(code)
        if codes:
            return codes

    return fallback


def _build_label_matrix(
    df: pd.DataFrame,
    code_column: str = "verified_cpt_codes",
) -> tuple[np.ndarray, list[str]]:
    """Build a multi-hot label matrix from a comma-separated CPT code column."""
    if df.empty:
        return np.zeros((0, 0), dtype=int), []

    all_codes: set[str] = set()
    parsed_rows: list[list[str]] = []
    for value in df[code_column].fillna("").astype(str).tolist():
        codes = [c.strip() for c in value.split(",") if c.strip()]
        parsed_rows.append(codes)
        all_codes.update(codes)

    sorted_codes = sorted(all_codes)
    code_to_idx = {c: i for i, c in enumerate(sorted_codes)}
    y = np.zeros((len(df), len(sorted_codes)), dtype=int)
    for row_idx, codes in enumerate(parsed_rows):
        for code in codes:
            y[row_idx, code_to_idx[code]] = 1

    return y, sorted_codes


def _enforce_encounter_grouping(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    mrn_col: str = "patient_mrn",
    date_col: str = "procedure_date",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Ensure all rows from the same encounter (MRN+date) stay in a single split.

    If an encounter is present in both splits, move the entire encounter to the
    split that already contains the majority of its rows (ties go to train).
    """
    train_in = np.asarray(train_idx)
    test_in = np.asarray(test_idx)
    was_2d = train_in.ndim == 2

    train_flat = set(train_in.flatten().tolist())
    test_flat = set(test_in.flatten().tolist())

    # Build encounter -> row indices mapping
    enc_to_rows: dict[tuple[str, str], list[int]] = {}
    for row_i, (mrn, date) in enumerate(zip(df[mrn_col].astype(str), df[date_col].astype(str))):
        enc_to_rows.setdefault((mrn, date), []).append(row_i)

    new_train: set[int] = set()
    new_test: set[int] = set()

    for rows in enc_to_rows.values():
        in_train = sum(1 for r in rows if r in train_flat)
        in_test = sum(1 for r in rows if r in test_flat)
        if in_train == 0 and in_test == 0:
            continue
        if in_train >= in_test:
            new_train.update(rows)
        else:
            new_test.update(rows)

    # Defensive: ensure disjoint
    overlap = new_train & new_test
    if overlap:
        new_test.difference_update(overlap)

    train_out = np.array(sorted(new_train), dtype=int)
    test_out = np.array(sorted(new_test), dtype=int)
    if was_2d:
        train_out = train_out.reshape(-1, 1)
        test_out = test_out.reshape(-1, 1)

    return train_out, test_out


def _infer_label_columns(df: pd.DataFrame) -> list[str]:
    candidate = [c for c in df.columns if str(c).startswith("label_")]
    if candidate:
        return candidate

    metadata = {
        "note_id",
        "encounter_id",
        "patient_mrn",
        "procedure_date",
        "source_file",
        "text",
        "note_text",
        "raw_text",
        "verified_cpt_codes",
        "is_edge_case",
    }
    inferred: list[str] = []
    for col in df.columns:
        if col in metadata:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        values = set(series.unique().tolist())
        if values.issubset({0, 1, True, False}):
            inferred.append(col)
    return inferred


def stratified_split(
    df: pd.DataFrame,
    label_columns: list[str] | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Backward-compatible stratified split used by ML-coder training tests.

    Returns (train_idx, test_idx, all_codes).
    """
    if df.empty:
        return [], [], []

    rng = np.random.RandomState(random_state)

    using_code_column = "verified_cpt_codes" in df.columns and label_columns is None
    if using_code_column:
        y, all_codes = _build_label_matrix(df, code_column="verified_cpt_codes")
    else:
        if label_columns is None:
            label_columns = _infer_label_columns(df)
        all_codes = list(label_columns)
        y = df[label_columns].fillna(0).astype(int).to_numpy()

    X = np.arange(len(df)).reshape(-1, 1)
    try:
        from skmultilearn.model_selection import IterativeStratification

        splitter = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[test_size, 1 - test_size],
        )
        train_idx_arr, test_idx_arr = next(splitter.split(X, y))
        train_idx_arr = np.asarray(train_idx_arr).reshape(-1, 1)
        test_idx_arr = np.asarray(test_idx_arr).reshape(-1, 1)
    except Exception:
        indices = np.arange(len(df))
        rng.shuffle(indices)
        n_test = max(1, int(round(len(df) * test_size)))
        test_idx_arr = indices[:n_test].reshape(-1, 1)
        train_idx_arr = indices[n_test:].reshape(-1, 1)

    train_idx_arr, test_idx_arr = _enforce_encounter_grouping(df, train_idx_arr, test_idx_arr)
    return train_idx_arr.flatten().tolist(), test_idx_arr.flatten().tolist(), all_codes


def _build_dataframe() -> pd.DataFrame:
    """
    Legacy hook used by `prepare_training_and_eval_splits`.

    Tests monkeypatch this function; production callers should migrate to the
    registry-first data prep pipeline.
    """
    return pd.DataFrame()


def prepare_training_and_eval_splits(
    output_dir: Path | str = "data/ml_training",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create `train.csv`, `test.csv`, and `edge_cases_holdout.csv` in `output_dir`.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = _build_dataframe().copy()
    if df.empty:
        edge_df = df.copy()
        train_df = df.copy()
        test_df = df.copy()
    else:
        if "is_edge_case" not in df.columns:
            df["is_edge_case"] = df.get("source_file", "").astype(str) == EDGE_SOURCE_NAME
        if "source_file" in df.columns:
            df["is_edge_case"] = df["is_edge_case"] | (df["source_file"].astype(str) == EDGE_SOURCE_NAME)

        edge_df = df[df["is_edge_case"] == True].copy()  # noqa: E712
        main_df = df[df["is_edge_case"] == False].copy()  # noqa: E712

        train_idx, test_idx, _ = stratified_split(
            main_df,
            label_columns=None,
            test_size=test_size,
            random_state=random_state,
        )
        train_df = main_df.iloc[train_idx].copy()
        test_df = main_df.iloc[test_idx].copy()

    train_df.to_csv(output_path / "train.csv", index=False)
    test_df.to_csv(output_path / "test.csv", index=False)
    edge_df.to_csv(output_path / "edge_cases_holdout.csv", index=False)

    return train_df, test_df, edge_df


__all__ = [
    "VALID_IP_CODES",
    "EDGE_SOURCE_NAME",
    "_extract_codes",
    "_build_label_matrix",
    "_enforce_encounter_grouping",
    "_build_dataframe",
    "stratified_split",
    "prepare_training_and_eval_splits",
]

def derive_booleans_from_json(entry: dict) -> dict:
    """
    Derives boolean flags from a golden JSON entry using CPT codes
    and registry_entry fields to match the registry_train.csv schema.
    """
    # Normalize CPT codes to a set of integers
    raw_cpts = entry.get('cpt_codes', [])
    cpt_codes = set()
    for c in raw_cpts:
        try:
            cpt_codes.add(int(c))
        except (ValueError, TypeError):
            continue

    reg = entry.get('registry_entry', {})
    
    # Initialize row with 0s
    row = {col: 0 for col in BOOLEAN_COLUMNS}
    
    # --- Heuristic Mapping Logic based on CPTs and Registry Fields ---

    # 1. Diagnostic Bronchoscopy
    # Set to 1 if specific diagnostic bronchoscopy codes are present.
    # Note: BLVR (31647) and Therapeutic codes do not trigger this on their own.
    diagnostic_cpts = [
        31622, 31623, 31624, 31625, 31626, 31627, 31628, 31629, 
        31651, 31652, 31653, 31654
    ]
    if any(c in cpt_codes for c in diagnostic_cpts):
        row['diagnostic_bronchoscopy'] = 1

    # 2. BAL (31624)
    if 31624 in cpt_codes:
        row['bal'] = 1
        
    # 3. Bronchial Wash (31622 implies basic bronch, wash often bundled)
    if 31622 in cpt_codes:
        row['bronchial_wash'] = 1
        
    # 4. Brushings (31623)
    if 31623 in cpt_codes:
        row['brushings'] = 1
        
    # 5. Endobronchial Biopsy (31625)
    if 31625 in cpt_codes:
        row['endobronchial_biopsy'] = 1
        
    # 6. Transbronchial Biopsy (31628, 31632)
    if 31628 in cpt_codes or 31632 in cpt_codes:
        row['transbronchial_biopsy'] = 1

    # 7. Linear EBUS (31651, 31652, 31653)
    if any(c in cpt_codes for c in [31651, 31652, 31653]) or reg.get('linear_ebus_stations'):
        row['linear_ebus'] = 1
        
    # 8. TBNA Conventional (31629)
    # Logic: If 31629 is present AND Linear EBUS codes are NOT present, it is conventional.
    if 31629 in cpt_codes:
        if row['linear_ebus'] == 0:
            row['tbna_conventional'] = 1
            
    # 9. Radial EBUS (31654)
    if 31654 in cpt_codes or reg.get('nav_rebus_used') is True:
        row['radial_ebus'] = 1
        
    # 10. Navigational Bronchoscopy (31627)
    if 31627 in cpt_codes or reg.get('nav_platform'):
        row['navigational_bronchoscopy'] = 1
        
    # 11. Transbronchial Cryobiopsy
    # Distinct from TBBX in registry. Look for explicit registry flag or keyword.
    if reg.get('nav_cryobiopsy_for_nodule') is True:
        row['transbronchial_cryobiopsy'] = 1
        
    # 12. Therapeutic Aspiration (31645, 31646)
    if any(c in cpt_codes for c in [31645, 31646]):
        row['therapeutic_aspiration'] = 1
        
    # 13. Foreign Body Removal (31635)
    if 31635 in cpt_codes or reg.get('fb_object_type'):
        row['foreign_body_removal'] = 1
        
    # 14. Airway Dilation (31630, 31631, 31634)
    if any(c in cpt_codes for c in [31630, 31631, 31634]):
        row['airway_dilation'] = 1
        
    # 15. Airway Stent (31636, 31637, 31638)
    if any(c in cpt_codes for c in [31636, 31637, 31638]):
        row['airway_stent'] = 1
        
    # 16. Peripheral Ablation
    # Rely on the registry flag 'ablation_peripheral_performed'
    if reg.get('ablation_peripheral_performed') is True:
        row['peripheral_ablation'] = 1
        
    # 17. Thermal Ablation (Central)
    # 31641 is "Destruction of tumor". If not peripheral, assume central/thermal.
    elif 31641 in cpt_codes:
        row['thermal_ablation'] = 1
        
    # 18. Cryotherapy (Central)
    # Usually implies central airway treatment if not peripheral.
    # Check modality text.
    modality = str(reg.get('ablation_modality', '')).lower()
    if 'cryo' in modality and row['peripheral_ablation'] == 0:
        row['cryotherapy'] = 1
        # Note: Cryo vs thermal can be ambiguous; we don't forcibly clear thermal_ablation here.
        # Based on CSV row 6 (Cryoablation): Peripheral=1, Thermal=0, Cryo=0 (as it's peripheral).
        # This matches the logic above (elif 31641).
    
    # 19. BLVR (31647-31651)
    if any(c in cpt_codes for c in [31647, 31648, 31649, 31651]) or reg.get('blvr_valve_type'):
        row['blvr'] = 1
        
    # 20. Whole Lung Lavage (32997)
    if 32997 in cpt_codes or reg.get('wll_volume_instilled_l'):
        row['whole_lung_lavage'] = 1

    # 21. Percutaneous Tracheostomy (31600, 31601, 31612)
    if any(c in cpt_codes for c in [31600, 31601, 31612]):
        row["percutaneous_tracheostomy"] = 1

    # 22. PEG Insertion (43246, 49440)
    if any(c in cpt_codes for c in [43246, 49440]):
        row["peg_insertion"] = 1
        
    # 23. Rigid Bronchoscopy (31600, 31601)
    if 31600 in cpt_codes or 31601 in cpt_codes:
        row['rigid_bronchoscopy'] = 1
        
    # 24. Medical Thoracoscopy (32601)
    if 32601 in cpt_codes:
        row['medical_thoracoscopy'] = 1
        
    # 25. Pleurodesis (32650)
    if 32650 in cpt_codes or reg.get('pleurodesis_performed') is True:
        row['pleurodesis'] = 1
        
    # 26. Thoracentesis (32554, 32555)
    if 32554 in cpt_codes or 32555 in cpt_codes:
        row['thoracentesis'] = 1
        
    # 27. Chest Tube (32551)
    if 32551 in cpt_codes or str(reg.get('pleural_procedure_type')) == 'Chest Tube':
        row['chest_tube'] = 1

    # 28. IPC (32550)
    if 32550 in cpt_codes:
        row['ipc'] = 1
        
    # 29. Fibrinolytic Therapy (32560)
    if 32560 in cpt_codes:
        row['fibrinolytic_therapy'] = 1
        
    return row

def main():
    # 1. Load Real Data
    real_csv_path = 'data/ml_training/registry_train.csv'
    
    print(f"Loading real data from {real_csv_path}...")
    try:
        if os.path.exists(real_csv_path):
            df_real = pd.read_csv(real_csv_path)
        else:
            print(f"Warning: {real_csv_path} not found. Proceeding with empty real dataframe.")
            df_real = pd.DataFrame(columns=['note_text'] + BOOLEAN_COLUMNS)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    # 2. Split Real Data (Train/Test)
    # Test set is purely Real data (Silver Standard methodology)
    if len(df_real) > 5:
        train_real, test_real = train_test_split(df_real, test_size=0.2, random_state=42)
    else:
        print("Warning: Not enough real data to split. Using all for training.")
        train_real = df_real
        test_real = pd.DataFrame(columns=df_real.columns)
    
    # 3. Load Synthetic Data (Golden JSONs)
    synthetic_rows = []
    # Search in the canonical golden extractions directory
    golden_dir = KnowledgeSettings().kb_path.parent / "golden_extractions_final"
    json_files = glob.glob(str(golden_dir / 'golden_*.json'))
    
    print(f"Loading synthetic data from {len(json_files)} JSON files...")
    
    for json_file in json_files:
        try:
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)
                # Ensure data is a list
                if isinstance(data, dict):
                    data = [data]
                    
                for entry in data:
                    note_text = entry.get('note_text', '')
                    if not note_text: 
                        continue
                    
                    # Derive labels from structure
                    labels = derive_booleans_from_json(entry)
                    
                    row_data = {'note_text': note_text}
                    row_data.update(labels)
                    synthetic_rows.append(row_data)
        except Exception as e:
            print(f"Warning: Failed to process {json_file}: {e}")
                
    if synthetic_rows:
        df_synthetic = pd.DataFrame(synthetic_rows)
        # Fill missing columns in synthetic with 0
        for col in BOOLEAN_COLUMNS:
            if col not in df_synthetic.columns:
                df_synthetic[col] = 0
    else:
        print("Warning: No synthetic data found.")
        df_synthetic = pd.DataFrame(columns=['note_text'] + BOOLEAN_COLUMNS)

    # 4. Combine Synthetic + Real Train
    # Ensure column order matches
    cols = ['note_text'] + BOOLEAN_COLUMNS
    
    # Filter only columns that exist (intersection of schema and df)
    valid_cols = [c for c in cols if c in df_synthetic.columns]
    df_synthetic = df_synthetic[valid_cols]
    
    # Reindex real data to ensure matching columns
    train_real = train_real.reindex(columns=cols, fill_value=0)
    test_real = test_real.reindex(columns=cols, fill_value=0)
    df_synthetic = df_synthetic.reindex(columns=cols, fill_value=0)
    
    df_train_final = pd.concat([df_synthetic, train_real], ignore_index=True)
    
    # 5. Save Outputs
    output_dir = Path("processed_data")
    output_dir.mkdir(exist_ok=True)
    
    train_path = output_dir / 'train.csv'
    test_path = output_dir / 'test.csv'
    
    df_train_final.to_csv(train_path, index=False)
    test_real.to_csv(test_path, index=False)
    
    print("-" * 30)
    print(f"Processed {len(df_real)} real records and {len(df_synthetic)} synthetic records.")
    print(f"Train Set (Synthetic + 80% Real): {len(df_train_final)} rows.")
    print(f"Test Set (20% Real):              {len(test_real)} rows.")
    print("-" * 30)
    print(f"Files saved to {output_dir}")

if __name__ == '__main__':
    main()


# =============================================================================
# V2 Booleans Integration
# =============================================================================
# Import the canonical boolean field list and extraction function from
# app/registry/v2_booleans.py for backward compatibility.

from app.registry.v2_booleans import (  # noqa: E402, I001
    PROCEDURE_BOOLEAN_FIELDS,
    extract_v2_booleans,
)

# Alias for backward compatibility with existing code
# Keep as list (not tuple) to match original PROCEDURE_BOOLEAN_FIELDS type
REGISTRY_TARGET_FIELDS = list(PROCEDURE_BOOLEAN_FIELDS)


def _extract_registry_booleans(entry: dict) -> dict:
    """Wrapper for extract_v2_booleans for backward compatibility.

    Args:
        entry: Registry entry dictionary from golden JSON.

    Returns:
        Dict mapping field names to 0/1 values.
    """
    return extract_v2_booleans(entry)


def _filter_rare_registry_labels(
    labels: list[list[int]],
    min_count: int = 5,
) -> tuple[list[list[int]], list[str]]:
    """Filter out labels with fewer than min_count positive examples.

    Args:
        labels: List of label vectors (each vector is a list of 0/1 ints).
        min_count: Minimum positive count required to keep a label.

    Returns:
        Tuple of (filtered_labels, kept_field_names).
    """
    if not labels:
        return [], []

    import numpy as np

    # Convert to array for easier computation
    arr = np.array(labels)
    n_labels = arr.shape[1] if arr.ndim > 1 else 0

    if n_labels == 0:
        return [], []

    # Count positives for each label
    counts = arr.sum(axis=0)

    # Find which labels to keep
    keep_mask = counts >= min_count
    kept_indices = np.where(keep_mask)[0]

    # Filter labels and get field names
    if len(kept_indices) == 0:
        return [], []

    filtered = arr[:, kept_indices].tolist()
    kept_names = [REGISTRY_TARGET_FIELDS[i] for i in kept_indices]

    return filtered, kept_names


# =============================================================================
# Registry-First Data Prep Re-exports
# =============================================================================
# Import registry-first functions for convenient access from this module.
# These are imported last to avoid circular import issues.

from .registry_data_prep import (  # noqa: E402, F401, I001
    prepare_registry_training_splits,
    RegistryLabelExtractor,
    ALL_PROCEDURE_LABELS,
    extract_records_from_golden_dir,
    stratified_split as registry_stratified_split,
    filter_rare_labels,
)
