"""
Registry-First Data Preparation Module

This module provides functions for preparing ML training data from golden JSON files
using the registry-first approach. The key function is `prepare_registry_training_splits()`
which is referenced in CLAUDE.md as the entry point for generating training data.

FEATURES:
- 3-tier extraction with hydration (structured -> CPT -> keyword)
- Uses authoritative extract_v2_booleans() from v2_booleans.py
- Adds label_source and label_confidence metadata columns
- Achieves <5% all-zero label rows vs 80% with old approach

Integration with existing codebase:
    - Place this file at: ml/lib/ml_coder/registry_data_prep.py
    - Import in data_prep.py: from .registry_data_prep import prepare_registry_training_splits
    - Or use standalone: python -m ml.lib.ml_coder.registry_data_prep

Example:
    from ml.lib.ml_coder.registry_data_prep import prepare_registry_training_splits

    train_df, val_df, test_df = prepare_registry_training_splits()
    train_df.to_csv("data/ml_training/registry_train.csv", index=False)
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import KnowledgeSettings
from ml.lib.ml_coder.registry_label_schema import REGISTRY_LABELS, compute_encounter_id
from ml.lib.ml_coder.registry_label_constraints import apply_label_constraints

logger = logging.getLogger(__name__)


# =============================================================================
# Canonical Procedure Flags (V2 Schema)
# =============================================================================
# Canonical procedure label list (single import point for ordering).
ALL_PROCEDURE_LABELS = list(REGISTRY_LABELS)

# Legacy lists for backward compatibility
BRONCHOSCOPY_LABELS = [
    "diagnostic_bronchoscopy",
    "bal",
    "bronchial_wash",
    "brushings",
    "endobronchial_biopsy",
    "tbna_conventional",
    "linear_ebus",
    "radial_ebus",
    "navigational_bronchoscopy",
    "transbronchial_biopsy",
    "transbronchial_cryobiopsy",
    "therapeutic_aspiration",
    "foreign_body_removal",
    "airway_dilation",
    "airway_stent",
    "thermal_ablation",
    "cryotherapy",
    "blvr",
    "peripheral_ablation",
    "bronchial_thermoplasty",
    "whole_lung_lavage",
    "rigid_bronchoscopy",
]

PLEURAL_LABELS = [
    "thoracentesis",
    "chest_tube",
    "ipc",
    "medical_thoracoscopy",
    "pleurodesis",
    "pleural_biopsy",
    "fibrinolytic_therapy",
]

# Alias mapping for V2 ↔ V3 schema compatibility
LABEL_ALIASES = {
    "ebus_linear": "linear_ebus",
    "ebus_radial": "radial_ebus",
    "navigation": "navigational_bronchoscopy",
    "tbna": "tbna_conventional",
    "tbb": "transbronchial_biopsy",
    "tbb_cryo": "transbronchial_cryobiopsy",
    "stent": "airway_stent",
    "dilation": "airway_dilation",
    "ablation_thermal": "thermal_ablation",
    "ablation_cryo": "cryotherapy",
    "ablation_peripheral": "peripheral_ablation",
    "thermoplasty": "bronchial_thermoplasty",
    "wll": "whole_lung_lavage",
    "rigid": "rigid_bronchoscopy",
    "thoraco": "medical_thoracoscopy",
    "ipc_placement": "ipc",
    "tube": "chest_tube",
    "tap": "thoracentesis",
}

# Source priority for deduplication (higher = better)
SOURCE_PRIORITY = {
    "human": 4,
    "structured": 3,
    "cpt": 2,
    "keyword": 1,
    "legacy": 0,
    "empty": -1,
}


def deduplicate_records(
    records: list[dict],
    key_field: str = "note_text",
    priority_field: str = "label_source",
    confidence_field: str = "label_confidence",
) -> tuple[list[dict], dict[str, Any]]:
    """Remove duplicate records, keeping highest-priority source.

    When same note_text appears multiple times with different labels:
    1. Group by note_text hash
    2. Keep record with highest priority source (structured > cpt > keyword)
    3. If same source, keep highest confidence
    4. Track stats on removed duplicates

    Args:
        records: List of record dicts with note_text, label_source, etc.
        key_field: Field to detect duplicates on (default: note_text)
        priority_field: Field indicating source priority
        confidence_field: Field for confidence tiebreaker

    Returns:
        Tuple of (deduplicated_records, dedup_stats)
    """
    from collections import defaultdict

    # Group records by content hash
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        content = record.get(key_field, "")
        content_hash = hashlib.md5(content.encode()).hexdigest()
        groups[content_hash].append(record)

    # Select best record from each group
    deduped = []
    stats = {
        "total_input": len(records),
        "unique_texts": len(groups),
        "duplicates_removed": 0,
        "conflicts_by_source": defaultdict(int),
    }

    for _content_hash, group in groups.items():
        if len(group) == 1:
            deduped.append(group[0])
        else:
            # Sort by priority (desc), then confidence (desc)
            sorted_group = sorted(
                group,
                key=lambda r: (
                    SOURCE_PRIORITY.get(r.get(priority_field, ""), 0),
                    r.get(confidence_field, 0),
                ),
                reverse=True,
            )
            best = sorted_group[0]
            deduped.append(best)

            # Track conflict stats
            stats["duplicates_removed"] += len(group) - 1
            sources = set(r.get(priority_field, "unknown") for r in group)
            if len(sources) > 1:
                stats["conflicts_by_source"][frozenset(sources)] += 1

    stats["total_output"] = len(deduped)
    # Convert defaultdict to regular dict for JSON serialization
    stats["conflicts_by_source"] = {
        " vs ".join(sorted(k)): v
        for k, v in stats["conflicts_by_source"].items()
    }
    return deduped, stats


@dataclass
class RegistryExtractionResult:
    """Result from extracting training data from golden JSONs."""

    df: pd.DataFrame
    label_columns: list[str]
    stats: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


class RegistryLabelExtractor:
    """Extracts boolean procedure labels from nested registry structures.

    Handles multiple schema versions (V2 flat, V3 granular) and various
    nesting patterns found in golden JSON files.
    """

    # Paths to search for procedure flags in registry structure
    SEARCH_PATHS = [
        [],  # Top level
        ["procedures_performed"],
        ["procedures_performed", "bronchoscopy"],
        ["procedures_performed", "pleural"],
        ["granular_data"],
        ["granular_data", "bronchoscopy"],
        ["granular_data", "pleural"],
    ]

    def __init__(self, labels: list[str] = None, aliases: dict[str, str] = None):
        self.labels = labels or ALL_PROCEDURE_LABELS
        self.aliases = aliases or LABEL_ALIASES

    def _get_nested(self, data: dict, path: list[str]) -> dict | None:
        """Navigate to a nested dict by path."""
        current = data
        for key in path:
            if not isinstance(current, dict):
                return None
            current = current.get(key)
        return current if isinstance(current, dict) else None

    def _normalize_key(self, key: str) -> str:
        """Convert alias key to canonical label name."""
        return self.aliases.get(key, key)

    def _extract_bool(self, value: Any) -> bool | None:
        """Extract boolean from value (handles dict with 'performed' key)."""
        if isinstance(value, bool):
            return value
        if isinstance(value, dict):
            # Check for "performed" key
            if "performed" in value:
                return value["performed"] is True
            # Check for any True boolean in the dict
            return any(v is True for v in value.values() if isinstance(v, bool))
        return None

    def extract(self, registry: dict[str, Any]) -> dict[str, int]:
        """Extract all procedure labels from a registry entry.

        Args:
            registry: Registry entry dictionary

        Returns:
            Dict mapping label names to binary values (0/1)
        """
        result = {label: 0 for label in self.labels}

        for path in self.SEARCH_PATHS:
            section = self._get_nested(registry, path) if path else registry
            if not section:
                continue

            for key, value in section.items():
                canonical = self._normalize_key(key)
                if canonical not in self.labels:
                    continue

                extracted = self._extract_bool(value)
                if extracted is True:
                    result[canonical] = 1

        return result


def _generate_encounter_id(text: str) -> str:
    """Generate stable encounter ID for grouping (text-derived)."""
    return compute_encounter_id(text)


def _load_human_labels_csv(path: Path, labels: list[str]) -> list[dict[str, Any]]:
    """Load a human-labeled registry CSV into record dicts for Tier-0 merge."""
    df = pd.read_csv(path)
    if df.empty:
        return []
    if "note_text" not in df.columns:
        raise ValueError(f"Human labels CSV missing required column 'note_text': {path}")

    # Normalize/ensure label columns exist and are {0,1}.
    for label in labels:
        if label not in df.columns:
            df[label] = 0
        df[label] = pd.to_numeric(df[label], errors="coerce").fillna(0).clip(0, 1).astype(int)

    if "label_confidence" not in df.columns:
        df["label_confidence"] = 1.0
    else:
        df["label_confidence"] = (
            pd.to_numeric(df["label_confidence"], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        )

    if "label_source" not in df.columns:
        df["label_source"] = "human"
    else:
        df["label_source"] = df["label_source"].fillna("human").astype(str)

    if "encounter_id" not in df.columns:
        df["encounter_id"] = df["note_text"].fillna("").astype(str).map(_generate_encounter_id)
    else:
        df["encounter_id"] = df["encounter_id"].fillna("").astype(str)
        missing = df["encounter_id"].str.len() == 0
        if missing.any():
            df.loc[missing, "encounter_id"] = (
                df.loc[missing, "note_text"].fillna("").astype(str).map(_generate_encounter_id)
            )

    if "source_file" not in df.columns:
        df["source_file"] = path.name
    else:
        df["source_file"] = df["source_file"].fillna(path.name).astype(str)

    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        note_text = str(row.get("note_text") or "").strip()
        if not note_text:
            continue
        record: dict[str, Any] = {
            "note_text": note_text,
            "encounter_id": str(row.get("encounter_id") or _generate_encounter_id(note_text)),
            "source_file": str(row.get("source_file") or path.name),
            "label_source": "human",
            "label_confidence": float(row.get("label_confidence") or 1.0),
        }
        record.update({label: int(row.get(label, 0)) for label in labels})
        apply_label_constraints(record)
        records.append(record)
    return records


def _load_golden_json(path: Path) -> list[dict] | None:
    """Load a golden JSON file with error handling.

    Returns a list of entries (handles both single dict and list formats).
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        # Normalize to list format
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            logger.warning(f"Unexpected data type in {path.name}: {type(data)}")
            return None
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to load {path.name}: {e}")
        return None


def extract_records_from_golden_dir(
    golden_dir: Path,
    extractor: RegistryLabelExtractor = None,
    min_text_length: int = 50,
    use_hydration: bool = True,
) -> tuple[list[dict], dict[str, Any]]:
    """Extract training records from all golden JSONs in a directory.

    Uses 3-tier extraction with hydration by default:
    1. Tier 1: Structured extraction from registry_entry (confidence 0.95)
    2. Tier 2: CPT-based derivation from cpt_codes (confidence 0.80)
    3. Tier 3: Keyword hydration from note_text (confidence 0.60)

    Args:
        golden_dir: Directory containing golden_*.json files
        extractor: Label extractor instance (deprecated, use_hydration=True instead)
        min_text_length: Minimum note text length to include
        use_hydration: If True, use 3-tier hydration (default). If False, use
                       legacy RegistryLabelExtractor.

    Returns:
        Tuple of (records list, statistics dict)
    """
    # Import hydration function here to avoid circular imports
    from .label_hydrator import extract_labels_with_hydration

    # Legacy extractor as fallback
    extractor = extractor or RegistryLabelExtractor()

    stats = {
        "total_files": 0,
        "total_entries": 0,
        "successful": 0,
        "skipped_no_text": 0,
        "skipped_no_registry": 0,
        "skipped_empty_labels": 0,
        "parse_errors": 0,
        "label_counts": Counter(),
        # Hydration tier statistics
        "tier_structured": 0,
        "tier_cpt": 0,
        "tier_keyword": 0,
        "tier_empty": 0,
    }

    records = []
    json_files = sorted(golden_dir.glob("golden_*.json"))

    for path in json_files:
        stats["total_files"] += 1

        entries = _load_golden_json(path)
        if entries is None:
            stats["parse_errors"] += 1
            continue

        # Process each entry in the file
        for entry in entries:
            stats["total_entries"] += 1

            if not isinstance(entry, dict):
                continue

            # Extract note text
            note_text = entry.get("note_text") or entry.get("text") or entry.get("note")
            if not note_text or not isinstance(note_text, str):
                stats["skipped_no_text"] += 1
                continue

            note_text = note_text.strip()
            if len(note_text) < min_text_length:
                stats["skipped_no_text"] += 1
                continue

            # Extract labels using hydration or legacy extractor
            if use_hydration:
                result = extract_labels_with_hydration(entry, note_text)
                labels = result.labels
                label_source = result.source
                label_confidence = result.confidence

                # Track tier statistics
                stats[f"tier_{label_source}"] += 1
            else:
                # Legacy extraction (deprecated)
                registry = (
                    entry.get("registry_entry")
                    or entry.get("registry")
                    or entry.get("extraction")
                )
                if not registry or not isinstance(registry, dict):
                    stats["skipped_no_registry"] += 1
                    continue

                labels = extractor.extract(registry)
                label_source = "legacy"
                label_confidence = 0.5

            apply_label_constraints(labels, note_text=note_text)

            # Require at least one positive label
            if not any(v == 1 for v in labels.values()):
                stats["skipped_empty_labels"] += 1
                continue

            # Build record with metadata
            record = {
                "note_text": note_text,
                "encounter_id": _generate_encounter_id(note_text),
                "source_file": path.name,
                "label_source": label_source,
                "label_confidence": label_confidence,
                **labels,
            }
            records.append(record)
            stats["successful"] += 1

            # Update label counts
            for label, value in labels.items():
                if value == 1:
                    stats["label_counts"][label] += 1

    # Deduplicate records when using hydration pipeline
    if use_hydration and records:
        records, dedup_stats = deduplicate_records(records)
        stats["dedup"] = dedup_stats
        if dedup_stats["duplicates_removed"] > 0:
            logger.info(
                f"Deduplication: {dedup_stats['duplicates_removed']} duplicates removed, "
                f"{dedup_stats['total_output']} unique records"
            )

    return records, stats


def stratified_split(
    df: pd.DataFrame,
    label_columns: list[str],
    group_column: str = "encounter_id",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform multi-label stratified split with encounter grouping.

    Uses iterative stratification when skmultilearn is available,
    otherwise falls back to random split with encounter grouping.

    Args:
        df: Input DataFrame
        label_columns: Binary label column names
        group_column: Column for encounter-level grouping
        train_size: Training set fraction
        val_size: Validation set fraction
        test_size: Test set fraction
        random_state: Random seed

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    np.random.seed(random_state)

    # Get unique encounters
    encounters = df[group_column].unique()
    n_encounters = len(encounters)

    # Build encounter-level label matrix
    enc_to_labels = {}
    for enc_id in encounters:
        mask = df[group_column] == enc_id
        label_vec = tuple(int(df.loc[mask, col].max()) for col in label_columns)
        enc_to_labels[enc_id] = label_vec

    enc_array = np.array(encounters)
    label_matrix = np.array([enc_to_labels[e] for e in encounters])

    # Try skmultilearn for proper stratification
    try:
        from skmultilearn.model_selection import IterativeStratification

        # Split train vs rest
        # Note: IterativeStratification doesn't accept random_state in newer sklearn
        strat1 = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[1 - train_size, train_size],
        )
        train_idx, rest_idx = next(strat1.split(enc_array.reshape(-1, 1), label_matrix))

        train_encounters = set(enc_array[train_idx])
        rest_enc = enc_array[rest_idx]
        rest_labels = label_matrix[rest_idx]

        # Split val vs test
        val_frac = val_size / (val_size + test_size)
        strat2 = IterativeStratification(
            n_splits=2,
            order=2,
            sample_distribution_per_fold=[1 - val_frac, val_frac],
        )
        val_idx, test_idx = next(strat2.split(rest_enc.reshape(-1, 1), rest_labels))

        val_encounters = set(rest_enc[val_idx])
        test_encounters = set(rest_enc[test_idx])

        logger.info("Using skmultilearn iterative stratification")

    except (ImportError, ValueError) as e:
        logger.warning(f"skmultilearn stratification failed ({e}), using random split")

        np.random.shuffle(enc_array)
        n_train = int(n_encounters * train_size)
        n_val = int(n_encounters * val_size)

        train_encounters = set(enc_array[:n_train])
        val_encounters = set(enc_array[n_train:n_train + n_val])
        test_encounters = set(enc_array[n_train + n_val:])

    train_df = df[df[group_column].isin(train_encounters)].copy()
    val_df = df[df[group_column].isin(val_encounters)].copy()
    test_df = df[df[group_column].isin(test_encounters)].copy()

    return train_df, val_df, test_df


def filter_rare_labels(
    df: pd.DataFrame,
    label_columns: list[str],
    min_count: int = 5,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Remove labels with fewer than min_count positive examples.

    Args:
        df: Input DataFrame
        label_columns: Label column names
        min_count: Minimum required positive examples

    Returns:
        Tuple of (filtered_df, remaining_labels, dropped_labels)
    """
    remaining = []
    dropped = []

    for col in label_columns:
        if df[col].sum() >= min_count:
            remaining.append(col)
        else:
            dropped.append(col)

    if dropped:
        df = df.drop(columns=dropped)
        logger.warning(f"Dropped {len(dropped)} rare labels: {dropped}")

    return df, remaining, dropped


def prepare_registry_training_splits(
    golden_dir: Path | str | None = None,
    human_labels_csv: Path | str | None = None,
    min_label_count: int = 5,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main entry point for registry-first training data preparation.

    This function:
    1. Scans all golden_*.json files in the golden directory
    2. Extracts note text and 30 boolean procedure flags
    3. Filters rare labels (< min_label_count examples)
    4. Performs iterative multi-label stratification
    5. Ensures encounter-level grouping (no data leakage)

    Args:
        golden_dir: Directory with golden_*.json files. Defaults to common
                    golden-extraction subdirectories under the KB directory.
        human_labels_csv: Optional CSV of human labels to merge as Tier-0
        min_label_count: Minimum positive examples required per label
        train_ratio: Training set fraction (default 0.70)
        val_ratio: Validation set fraction (default 0.15)
        test_ratio: Test set fraction (default 0.15)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df) DataFrames

    Raises:
        FileNotFoundError: If golden directory doesn't exist
        ValueError: If no valid records could be extracted

    Example:
        >>> train_df, val_df, test_df = prepare_registry_training_splits()
        >>> train_df.to_csv("data/ml_training/registry_train.csv", index=False)
    """
    # Resolve golden directory
    if golden_dir is None:
        knowledge_dir = KnowledgeSettings().kb_path.parent
        candidates = [
            knowledge_dir / "golden_extractions_final",
            knowledge_dir / "golden_extractions_scrubbed",
            knowledge_dir / "golden_extractions",
        ]
        for candidate in candidates:
            if candidate.exists():
                golden_dir = candidate
                break
        else:
            raise FileNotFoundError(
                "No golden extractions directory found. "
                "Expected one of: " + ", ".join(str(c) for c in candidates)
            )
    else:
        golden_dir = Path(golden_dir)

    if not golden_dir.exists():
        raise FileNotFoundError(f"Golden directory not found: {golden_dir}")

    logger.info(f"Loading golden JSONs from: {golden_dir}")

    # Extract records with hydration
    records, stats = extract_records_from_golden_dir(golden_dir, use_hydration=True)

    # Tier-0 merge: human labels (highest priority).
    if human_labels_csv:
        human_path = Path(human_labels_csv)
        if human_path.exists():
            human_records = _load_human_labels_csv(
                human_path,
                labels=ALL_PROCEDURE_LABELS,
            )
            if human_records:
                logger.info(
                    "Loaded %d human-labeled records from %s",
                    len(human_records),
                    human_path,
                )
                records = [*human_records, *records]
                records, dedup_stats = deduplicate_records(records)
                stats["dedup_with_human"] = dedup_stats
        else:
            logger.warning("Human labels CSV not found (skipping): %s", human_path)

    if not records:
        raise ValueError(
            f"No valid records extracted. Stats: "
            f"total={stats['total_files']}, "
            f"entries={stats.get('total_entries', 0)}, "
            f"no_text={stats['skipped_no_text']}, "
            f"empty_labels={stats['skipped_empty_labels']}"
        )

    logger.info(
        f"Extracted {len(records)} records from {stats['total_files']} files "
        f"({stats.get('total_entries', 0)} total entries)"
    )

    # Log hydration tier statistics
    total_with_labels = stats["successful"]
    tier_stats = {
        "structured": stats.get("tier_structured", 0),
        "cpt": stats.get("tier_cpt", 0),
        "keyword": stats.get("tier_keyword", 0),
        "empty": stats.get("tier_empty", 0),
    }

    denom = max(1, total_with_labels)
    logger.info(
        "Label extraction tiers: structured=%d (%.1f%%), cpt=%d (%.1f%%), "
        "keyword=%d (%.1f%%), empty=%d (skipped)",
        tier_stats["structured"],
        100 * tier_stats["structured"] / denom,
        tier_stats["cpt"],
        100 * tier_stats["cpt"] / denom,
        tier_stats["keyword"],
        100 * tier_stats["keyword"] / denom,
        tier_stats["empty"],
    )

    # Log deduplication stats if present
    if "dedup" in stats:
        dedup = stats["dedup"]
        logger.info(
            f"Deduplication: {dedup['duplicates_removed']} removed "
            f"({100*dedup['duplicates_removed']/max(1,dedup['total_input']):.1f}%), "
            f"{dedup['total_output']} unique records"
        )
        if dedup.get("conflicts_by_source"):
            logger.info(f"  Conflicts: {dedup['conflicts_by_source']}")

    # Create DataFrame
    df = pd.DataFrame(records)

    # Filter rare labels
    df, remaining_labels, dropped = filter_rare_labels(
        df, ALL_PROCEDURE_LABELS, min_count=min_label_count
    )

    logger.info(f"Using {len(remaining_labels)} labels after filtering")

    # Perform stratified split
    train_df, val_df, test_df = stratified_split(
        df,
        label_columns=remaining_labels,
        group_column="encounter_id",
        train_size=train_ratio,
        val_size=val_ratio,
        test_size=test_ratio,
        random_state=random_state,
    )

    # Log split statistics
    logger.info(
        f"Split complete: train={len(train_df)}, "
        f"val={len(val_df)}, test={len(test_df)}"
    )

    return train_df, val_df, test_df


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for registry data preparation."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Prepare registry-first ML training data from golden JSONs"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Golden extractions directory",
    )
    parser.add_argument(
        "--human-labels-csv",
        type=Path,
        default=None,
        help="Optional CSV of human registry labels to merge as Tier-0",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ml_training"),
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--prefix",
        default="registry",
        help="Prefix for output files",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum label count",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    train_df, val_df, test_df = prepare_registry_training_splits(
        golden_dir=args.input_dir,
        human_labels_csv=args.human_labels_csv,
        min_label_count=args.min_count,
        random_state=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(args.output_dir / f"{args.prefix}_train.csv", index=False)
    val_df.to_csv(args.output_dir / f"{args.prefix}_val.csv", index=False)
    test_df.to_csv(args.output_dir / f"{args.prefix}_test.csv", index=False)

    print(f"Written to {args.output_dir}:")
    print(f"  {args.prefix}_train.csv ({len(train_df)} rows)")
    print(f"  {args.prefix}_val.csv ({len(val_df)} rows)")
    print(f"  {args.prefix}_test.csv ({len(test_df)} rows)")


if __name__ == "__main__":
    main()
