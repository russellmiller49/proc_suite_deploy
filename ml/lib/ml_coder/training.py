"""Training utilities for the CPT multi-label classifier."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

from app.common.logger import get_logger
from ml.lib.ml_coder.preprocessing import NoteTextCleaner
from ml.lib.ml_coder.utils import clean_cpt_codes

logger = get_logger("ml_coder.training")

MODELS_DIR = Path("data/models")
PIPELINE_PATH = MODELS_DIR / "cpt_classifier.pkl"
MLB_PATH = MODELS_DIR / "mlb.pkl"


def _load_training_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    """Load and clean note/cpt rows from the provided CSV file."""

    df = pd.read_csv(csv_path)
    if df.empty:
        msg = f"Training file {csv_path} is empty."
        raise ValueError(msg)

    required_cols = {"note_text", "verified_cpt_codes"}
    missing = required_cols - set(df.columns)
    if missing:
        cols = ", ".join(sorted(missing))
        msg = f"Training file is missing required columns: {cols}"
        raise ValueError(msg)

    cleaned = df.dropna(subset=["note_text", "verified_cpt_codes"]).copy()
    cleaned["note_text"] = cleaned["note_text"].astype(str)
    cleaned["verified_cpt_codes"] = cleaned["verified_cpt_codes"].apply(clean_cpt_codes)
    cleaned = cleaned[cleaned["verified_cpt_codes"].map(bool)]

    texts = cleaned["note_text"].tolist()
    labels = cleaned["verified_cpt_codes"].tolist()

    if not texts:
        msg = "No valid training rows remain after cleaning the CSV."
        raise ValueError(msg)

    logger.info("Loaded %s training rows from %s", len(texts), csv_path)
    return texts, labels


def _build_pipeline() -> Pipeline:
    """
    Create the scikit-learn pipeline used for training/prediction.

    Pipeline stages:
    1. clean: Remove boilerplate text (signatures, disclaimers)
    2. tfidf: TF-IDF vectorization with unigrams, bigrams, trigrams
    3. clf: Calibrated logistic regression (OneVsRest for multi-label)

    The calibrated classifier provides reliable probability estimates
    for downstream confidence thresholding and LLM fallback decisions.
    """
    vectorizer = TfidfVectorizer(
        max_features=3000,
        stop_words="english",
        ngram_range=(1, 3),
        min_df=2,
    )

    base_lr = LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=1000,
    )

    # Use cv=2 to support rare codes with few examples
    # Codes with <10 examples will fall back to LLM anyway per hybrid_policy
    calibrated = CalibratedClassifierCV(
        estimator=base_lr,
        cv=2,
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


MIN_SAMPLES_PER_CODE = 2  # Minimum samples needed for calibrated CV


def _filter_rare_codes(
    labels: list[list[str]], min_samples: int = MIN_SAMPLES_PER_CODE
) -> tuple[list[list[str]], set[str]]:
    """
    Filter out codes that appear in fewer than min_samples examples.

    These very rare codes will be handled by LLM fallback per hybrid_policy.

    Returns:
        Filtered labels and set of removed codes
    """
    from collections import Counter

    code_counts = Counter(code for label_list in labels for code in label_list)
    rare_codes = {code for code, count in code_counts.items() if count < min_samples}

    if rare_codes:
        logger.warning(
            "Filtering %d codes with <%d samples (will use LLM fallback): %s",
            len(rare_codes),
            min_samples,
            sorted(rare_codes),
        )

    filtered_labels = [
        [code for code in label_list if code not in rare_codes] for label_list in labels
    ]

    return filtered_labels, rare_codes


def train_model(csv_path: str | Path) -> tuple[Pipeline, MultiLabelBinarizer]:
    """Train the classifier on the supplied CSV and persist the artifacts."""

    csv_file = Path(csv_path)
    logger.info("Starting ML training from %s", csv_file)

    texts, labels = _load_training_rows(csv_file)

    # Filter codes with too few samples for calibrated CV
    labels, rare_codes = _filter_rare_codes(labels)

    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(labels)

    pipeline = _build_pipeline()
    pipeline.fit(texts, y)
    logger.info("Model fit complete. Persisting artifacts to %s", MODELS_DIR)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)
    joblib.dump(mlb, MLB_PATH)

    logger.info("Saved classifier to %s", PIPELINE_PATH)
    logger.info("Saved label binarizer to %s", MLB_PATH)

    return pipeline, mlb


def _compute_reliability_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """
    Compute reliability/calibration curve data.

    Groups predictions into probability bins and computes
    the actual positive rate (precision) for each bin.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_precisions = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            precision = y_true[mask].mean()
            bin_centers.append((lo + hi) / 2)
            bin_precisions.append(float(precision))
            bin_counts.append(int(mask.sum()))

    return {
        "bin_centers": bin_centers,
        "bin_precisions": bin_precisions,
        "bin_counts": bin_counts,
    }


def evaluate_model(
    test_csv: str | Path,
    model_path: str | Path | None = None,
    mlb_path: str | Path | None = None,
    output_path: str | Path | None = None,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate trained model on test set and generate metrics.

    Args:
        test_csv: Path to test CSV with note_text and verified_cpt_codes
        model_path: Path to trained pipeline (defaults to PIPELINE_PATH)
        mlb_path: Path to label binarizer (defaults to MLB_PATH)
        output_path: Optional path to save metrics JSON
        threshold: Probability threshold for positive prediction

    Returns:
        Dictionary with per-code metrics and reliability curves
    """
    model_path = Path(model_path) if model_path else PIPELINE_PATH
    mlb_path = Path(mlb_path) if mlb_path else MLB_PATH

    logger.info("Loading model from %s", model_path)
    pipeline = joblib.load(model_path)
    mlb = joblib.load(mlb_path)

    texts, labels = _load_training_rows(Path(test_csv))
    y_true = mlb.transform(labels)

    logger.info("Running predictions on %d test samples", len(texts))
    y_prob = pipeline.predict_proba(texts)
    y_pred = (y_prob >= threshold).astype(int)

    # Per-code metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    code_metrics = {}
    for i, code in enumerate(mlb.classes_):
        code_metrics[code] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
            "predictions": int(y_pred[:, i].sum()),
        }

    # Macro/micro averages
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )

    # Reliability curves (flattened across all codes)
    reliability = _compute_reliability_curve(y_true.ravel(), y_prob.ravel())

    metrics = {
        "threshold": threshold,
        "n_samples": len(texts),
        "n_codes": len(mlb.classes_),
        "macro": {"precision": float(macro_p), "recall": float(macro_r), "f1": float(macro_f1)},
        "micro": {"precision": float(micro_p), "recall": float(micro_r), "f1": float(micro_f1)},
        "per_code": code_metrics,
        "reliability_curve": reliability,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved metrics to %s", output_path)

    return metrics


def train_and_evaluate(
    train_csv: str | Path = "data/ml_training/train.csv",
    test_csv: str | Path = "data/ml_training/test.csv",
    metrics_path: str | Path = "data/models/metrics.json",
) -> dict[str, Any]:
    """
    Train model and immediately evaluate on test set.

    Convenience function that chains train_model and evaluate_model.

    Returns:
        Evaluation metrics dictionary
    """
    train_model(train_csv)
    return evaluate_model(test_csv, output_path=metrics_path)


__all__ = [
    "train_model",
    "evaluate_model",
    "train_and_evaluate",
    "PIPELINE_PATH",
    "MLB_PATH",
]
