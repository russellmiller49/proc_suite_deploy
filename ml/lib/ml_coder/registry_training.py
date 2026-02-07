"""Training utilities for the registry multi-label classifier.

This module provides:
- load_registry_csv: Load registry training/test CSVs
- build_registry_pipeline: Create TF-IDF + calibrated logistic regression pipeline
- optimize_label_thresholds: Find F1-optimal thresholds per label
- train_registry_model: Train and persist model artifacts
- evaluate_registry_model: Evaluate model on test set
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from app.common.logger import get_logger
from ml.lib.ml_coder.preprocessing import NoteTextCleaner

logger = get_logger("ml_coder.registry_training")

# Default paths for registry model artifacts
MODELS_DIR = Path("data/models")
REGISTRY_PIPELINE_PATH = MODELS_DIR / "registry_classifier.pkl"
REGISTRY_MLB_PATH = MODELS_DIR / "registry_mlb.pkl"
REGISTRY_THRESHOLDS_PATH = MODELS_DIR / "registry_thresholds.json"
REGISTRY_METRICS_PATH = MODELS_DIR / "registry_metrics.json"

# Training data paths
TRAIN_CSV_PATH = Path("data/ml_training/registry_train.csv")
TEST_CSV_PATH = Path("data/ml_training/registry_test.csv")


def load_registry_csv(path: Path) -> tuple[list[str], np.ndarray, list[str]]:
    """Load registry training/test CSV file.

    Expected CSV format:
    - First column: note_text (clinical note text)
    - Remaining columns: boolean label columns (0/1)

    Args:
        path: Path to CSV file

    Returns:
        Tuple of (texts, labels_matrix, label_names)
        - texts: List of note text strings
        - labels_matrix: numpy array of shape (n_samples, n_labels)
        - label_names: List of label column names
    """
    df = pd.read_csv(path)

    if df.empty:
        msg = f"Registry training file {path} is empty."
        raise ValueError(msg)

    if "note_text" not in df.columns:
        msg = f"Registry training file {path} missing required 'note_text' column."
        raise ValueError(msg)

    # Extract texts
    texts = df["note_text"].fillna("").astype(str).tolist()

    # Label columns are all columns except note_text
    label_cols = [c for c in df.columns if c != "note_text"]
    if not label_cols:
        msg = f"Registry training file {path} has no label columns."
        raise ValueError(msg)

    # Extract label matrix
    y = df[label_cols].fillna(0).astype(int).to_numpy()

    logger.info(
        "Loaded %d samples with %d labels from %s",
        len(texts),
        len(label_cols),
        path,
    )

    return texts, y, label_cols


def build_registry_pipeline() -> Pipeline:
    """Build TF-IDF + calibrated logistic regression multi-label pipeline.

    Pipeline stages:
    1. clean: Remove boilerplate text (signatures, disclaimers)
    2. tfidf: TF-IDF vectorization with unigrams and bigrams
    3. clf: Calibrated logistic regression (OneVsRest for multi-label)

    The calibrated classifier provides reliable probability estimates
    for downstream threshold-based classification.

    Returns:
        Sklearn Pipeline ready for fitting
    """
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )

    base_lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",  # Important for rare procedures
        max_iter=1000,
    )

    # Use cv=3 for calibration
    # Fall back to cv=2 if some labels have too few samples
    calibrated = CalibratedClassifierCV(
        estimator=base_lr,
        cv=3,
        method="sigmoid",
    )

    classifier = OneVsRestClassifier(calibrated, n_jobs=-1)

    return Pipeline(
        steps=[
            ("clean", NoteTextCleaner()),
            ("tfidf", vectorizer),
            ("clf", classifier),
        ]
    )


def optimize_label_thresholds(
    model: Pipeline,
    texts: Sequence[str],
    y_true: np.ndarray,
    label_names: Sequence[str],
    grid_points: int = 17,
) -> dict[str, float]:
    """Find F1-optimal threshold per label on a validation set.

    Searches a coarse grid of thresholds [0.1, 0.9] for each label
    and selects the threshold that maximizes F1 score.

    Args:
        model: Fitted Pipeline with predict_proba method
        texts: Validation texts
        y_true: True labels matrix (n_samples, n_labels)
        label_names: List of label names (in column order)
        grid_points: Number of threshold points to search

    Returns:
        Dict mapping label name to optimal threshold
    """
    logger.info("Optimizing thresholds on %d validation samples", len(texts))

    # Get probability predictions
    proba = model.predict_proba(list(texts))

    thresholds: dict[str, float] = {}
    threshold_grid = np.linspace(0.1, 0.9, grid_points)

    for idx, label in enumerate(label_names):
        # Handle different predict_proba output formats
        # OneVsRestClassifier with CalibratedClassifierCV returns (n_samples, n_labels)
        if isinstance(proba, list):
            # proba is list of (n_samples, 2) arrays
            if proba[idx].shape[1] > 1:
                label_proba = proba[idx][:, 1]
            else:
                label_proba = proba[idx][:, 0]
        else:
            # proba is (n_samples, n_labels)
            label_proba = proba[:, idx]

        y_label_true = y_true[:, idx]

        # Skip threshold optimization if no positive samples
        if y_label_true.sum() == 0:
            thresholds[label] = 0.5
            logger.warning("Label '%s' has no positive samples in validation; using default 0.5", label)
            continue

        best_thresh = 0.5
        best_f1 = -1.0

        for t in threshold_grid:
            y_pred = (label_proba >= t).astype(int)
            f1 = f1_score(y_label_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        thresholds[label] = best_thresh
        logger.debug("Label '%s': optimal threshold=%.2f, F1=%.3f", label, best_thresh, best_f1)

    # Log summary statistics
    thresh_values = list(thresholds.values())
    logger.info(
        "Threshold optimization complete: min=%.2f, median=%.2f, max=%.2f",
        min(thresh_values),
        np.median(thresh_values),
        max(thresh_values),
    )

    return thresholds


def train_registry_model(
    train_csv: Path | str = TRAIN_CSV_PATH,
    models_dir: Path | str = MODELS_DIR,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, MultiLabelBinarizer, dict[str, float]]:
    """Train registry classifier and persist artifacts.

    Args:
        train_csv: Path to registry training CSV
        models_dir: Directory to save model artifacts
        val_size: Fraction of training data for validation (threshold tuning)
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (fitted_pipeline, label_binarizer, thresholds)
    """
    train_csv = Path(train_csv)
    models_dir = Path(models_dir)

    logger.info("Starting registry model training from %s", train_csv)

    # Load training data
    texts, y, label_names = load_registry_csv(train_csv)

    # Split into train/validation for threshold optimization
    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        y,
        test_size=val_size,
        random_state=random_state,
    )

    logger.info(
        "Training split: %d train, %d validation",
        len(X_train),
        len(X_val),
    )

    # Build and fit pipeline
    model = build_registry_pipeline()

    logger.info("Fitting model on %d training samples with %d labels", len(X_train), len(label_names))
    model.fit(X_train, y_train)

    # Optimize thresholds on validation set
    thresholds = optimize_label_thresholds(model, X_val, y_val, label_names)

    # Create MLB for label ordering
    mlb = MultiLabelBinarizer(classes=label_names)
    mlb.fit([label_names])  # Just to set .classes_

    # Persist artifacts
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, models_dir / "registry_classifier.pkl")
    logger.info("Saved classifier to %s", models_dir / "registry_classifier.pkl")

    joblib.dump(mlb, models_dir / "registry_mlb.pkl")
    logger.info("Saved MLB to %s", models_dir / "registry_mlb.pkl")

    with open(models_dir / "registry_thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    logger.info("Saved thresholds to %s", models_dir / "registry_thresholds.json")

    return model, mlb, thresholds


def evaluate_registry_model(
    test_csv: Path | str = TEST_CSV_PATH,
    model_path: Path | str | None = None,
    mlb_path: Path | str | None = None,
    thresholds_path: Path | str | None = None,
    output_path: Path | str | None = None,
) -> dict[str, Any]:
    """Evaluate trained registry model on test set.

    Args:
        test_csv: Path to test CSV
        model_path: Path to trained pipeline pickle
        mlb_path: Path to MultiLabelBinarizer pickle
        thresholds_path: Path to thresholds JSON
        output_path: Optional path to save metrics JSON

    Returns:
        Dictionary with per-label metrics and aggregate metrics
    """
    test_csv = Path(test_csv)
    model_path = Path(model_path) if model_path else REGISTRY_PIPELINE_PATH
    mlb_path = Path(mlb_path) if mlb_path else REGISTRY_MLB_PATH
    thresholds_path = Path(thresholds_path) if thresholds_path else REGISTRY_THRESHOLDS_PATH

    logger.info("Loading model from %s", model_path)
    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)

    with open(thresholds_path) as f:
        thresholds = json.load(f)

    # Load test data
    texts, y_true, label_names = load_registry_csv(test_csv)

    logger.info("Evaluating on %d test samples", len(texts))

    # Get probability predictions
    proba = model.predict_proba(texts)

    # Apply per-label thresholds
    y_pred = np.zeros_like(y_true)
    for idx, label in enumerate(label_names):
        thresh = thresholds.get(label, 0.5)
        if isinstance(proba, list):
            label_proba = proba[idx][:, 1] if proba[idx].shape[1] > 1 else proba[idx][:, 0]
        else:
            label_proba = proba[:, idx]
        y_pred[:, idx] = (label_proba >= thresh).astype(int)

    # Per-label metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_label_metrics = {}
    for i, label in enumerate(label_names):
        per_label_metrics[label] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predictions": int(y_pred[:, i].sum()),
            "threshold": thresholds.get(label, 0.5),
        }

    # Aggregate metrics
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    metrics = {
        "n_samples": len(texts),
        "n_labels": len(label_names),
        "macro": {
            "precision": float(macro_p),
            "recall": float(macro_r),
            "f1": float(macro_f1),
        },
        "micro": {
            "precision": float(micro_p),
            "recall": float(micro_r),
            "f1": float(micro_f1),
        },
        "per_label": per_label_metrics,
    }

    # Log summary
    logger.info("Evaluation complete:")
    logger.info("  Macro F1: %.3f", macro_f1)
    logger.info("  Micro F1: %.3f", micro_f1)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", output_path)

    return metrics


def train_and_evaluate(
    train_csv: Path | str = TRAIN_CSV_PATH,
    test_csv: Path | str = TEST_CSV_PATH,
    models_dir: Path | str = MODELS_DIR,
    metrics_path: Path | str = REGISTRY_METRICS_PATH,
) -> dict[str, Any]:
    """Train model and evaluate on test set.

    Convenience function that chains train_registry_model and evaluate_registry_model.

    Args:
        train_csv: Path to training CSV
        test_csv: Path to test CSV
        models_dir: Directory for model artifacts
        metrics_path: Path to save evaluation metrics

    Returns:
        Evaluation metrics dictionary
    """
    train_registry_model(train_csv=train_csv, models_dir=models_dir)
    return evaluate_registry_model(test_csv=test_csv, output_path=metrics_path)


__all__ = [
    "load_registry_csv",
    "build_registry_pipeline",
    "optimize_label_thresholds",
    "train_registry_model",
    "evaluate_registry_model",
    "train_and_evaluate",
    "REGISTRY_PIPELINE_PATH",
    "REGISTRY_MLB_PATH",
    "REGISTRY_THRESHOLDS_PATH",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train registry ML classifier")
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=TRAIN_CSV_PATH,
        help="Path to training CSV",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=TEST_CSV_PATH,
        help="Path to test CSV",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory for model artifacts",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model after training",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Registry ML Model Training")
    print("=" * 60)

    if args.evaluate:
        metrics = train_and_evaluate(
            train_csv=args.train_csv,
            test_csv=args.test_csv,
            models_dir=args.models_dir,
        )
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"Macro F1: {metrics['macro']['f1']:.3f}")
        print(f"Micro F1: {metrics['micro']['f1']:.3f}")
        print("\nPer-label F1 scores:")
        for label, m in sorted(metrics["per_label"].items(), key=lambda x: x[1]["f1"], reverse=True):
            print(f"  {label:35s}: F1={m['f1']:.3f} (P={m['precision']:.3f}, R={m['recall']:.3f}, thresh={m['threshold']:.2f})")
    else:
        model, mlb, thresholds = train_registry_model(
            train_csv=args.train_csv,
            models_dir=args.models_dir,
        )
        print("\nTraining complete!")
        print(f"Model saved to: {args.models_dir / 'registry_classifier.pkl'}")
        print(f"MLB saved to: {args.models_dir / 'registry_mlb.pkl'}")
        print(f"Thresholds saved to: {args.models_dir / 'registry_thresholds.json'}")
        print("\nOptimized thresholds:")
        for label, thresh in sorted(thresholds.items()):
            print(f"  {label:35s}: {thresh:.2f}")
