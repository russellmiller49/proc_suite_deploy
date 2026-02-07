"""Evaluation script for ActionPredictor against ground truth test data.

Computes per-field precision, recall, F1 and overall metrics.

Usage:
    python -m app.registry.ml.evaluate
    python -m app.registry.ml.evaluate --test-data data/ml_training/registry_test.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from app.common.logger import get_logger
from app.registry.ml import ActionPredictor, ClinicalActions

logger = get_logger("registry.ml.evaluate")

# Map between training data columns and ClinicalActions fields
FIELD_MAPPING = {
    "diagnostic_bronchoscopy": lambda a: a.diagnostic_bronchoscopy,
    "brushings": lambda a: a.brushings.performed,
    "tbna_conventional": lambda a: a.biopsy.tbna_conventional_performed,
    "linear_ebus": lambda a: a.ebus.performed and not a.navigation.radial_ebus_used,
    "radial_ebus": lambda a: a.navigation.radial_ebus_used,
    "navigational_bronchoscopy": lambda a: a.navigation.performed,
    "transbronchial_biopsy": lambda a: a.biopsy.transbronchial_performed,
    "transbronchial_cryobiopsy": lambda a: a.biopsy.cryobiopsy_performed,
    "foreign_body_removal": lambda a: a.therapeutic.foreign_body_removal_performed,
    "airway_stent": lambda a: a.stent.performed,
    "thermal_ablation": lambda a: a.cao.thermal_ablation_performed,
    "blvr": lambda a: a.blvr.performed,
    "peripheral_ablation": lambda a: False,  # Not yet extracted
    "bronchial_thermoplasty": lambda a: getattr(a.therapeutic, 'bronchial_thermoplasty_performed', False),
    "whole_lung_lavage": lambda a: getattr(a.therapeutic, 'whole_lung_lavage_performed', False),
    "rigid_bronchoscopy": lambda a: a.rigid_bronchoscopy,
    "thoracentesis": lambda a: a.pleural.thoracentesis_performed,
    "chest_tube": lambda a: a.pleural.chest_tube_performed,
    "ipc": lambda a: a.pleural.ipc_performed,
    "medical_thoracoscopy": lambda a: a.pleural.thoracoscopy_performed,
    "pleurodesis": lambda a: a.pleural.pleurodesis_performed,
}


@dataclass
class FieldMetrics:
    """Metrics for a single field."""
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.tp + self.fn == 0:
            return 0.0
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    @property
    def support(self) -> int:
        """Number of actual positives."""
        return self.tp + self.fn


@dataclass
class EvaluationResult:
    """Full evaluation results."""
    field_metrics: dict[str, FieldMetrics] = field(default_factory=dict)
    total_samples: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def micro_precision(self) -> float:
        tp = sum(m.tp for m in self.field_metrics.values())
        fp = sum(m.fp for m in self.field_metrics.values())
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @property
    def micro_recall(self) -> float:
        tp = sum(m.tp for m in self.field_metrics.values())
        fn = sum(m.fn for m in self.field_metrics.values())
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def macro_f1(self) -> float:
        f1s = [m.f1 for m in self.field_metrics.values() if m.support > 0]
        return sum(f1s) / len(f1s) if f1s else 0.0


def evaluate_action_predictor(
    test_data_path: str | Path,
    max_samples: int | None = None,
    verbose: bool = False,
) -> EvaluationResult:
    """Evaluate ActionPredictor against labeled test data.

    Args:
        test_data_path: Path to CSV with note_text and label columns
        max_samples: Limit number of samples (for quick testing)
        verbose: Print per-sample results

    Returns:
        EvaluationResult with per-field and aggregate metrics
    """
    predictor = ActionPredictor()
    result = EvaluationResult()

    # Initialize metrics for all fields
    for field_name in FIELD_MAPPING:
        result.field_metrics[field_name] = FieldMetrics()

    test_data_path = Path(test_data_path)
    logger.info(f"Loading test data from {test_data_path}")

    with open(test_data_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if max_samples:
        rows = rows[:max_samples]

    logger.info(f"Evaluating {len(rows)} samples...")

    for i, row in enumerate(rows):
        if (i + 1) % 100 == 0:
            logger.info(f"  Processed {i + 1}/{len(rows)} samples")

        note_text = row.get("note_text", "")
        if not note_text:
            result.errors.append(f"Row {i}: Empty note_text")
            continue

        try:
            prediction = predictor.predict(note_text)
            actions = prediction.actions

            for field_name, extractor in FIELD_MAPPING.items():
                # Ground truth (convert string "1"/"0" to bool)
                gt_value = row.get(field_name, "0")
                ground_truth = gt_value in ("1", "True", "true", 1, True)

                # Prediction
                try:
                    predicted = extractor(actions)
                except Exception:
                    predicted = False

                metrics = result.field_metrics[field_name]
                if ground_truth and predicted:
                    metrics.tp += 1
                elif ground_truth and not predicted:
                    metrics.fn += 1
                elif not ground_truth and predicted:
                    metrics.fp += 1
                else:
                    metrics.tn += 1

            result.total_samples += 1

        except Exception as e:
            result.errors.append(f"Row {i}: {e}")

    return result


def print_evaluation_report(result: EvaluationResult) -> None:
    """Print formatted evaluation report."""
    print("\n" + "=" * 70)
    print("ActionPredictor Evaluation Report")
    print("=" * 70)

    print(f"\nTotal samples evaluated: {result.total_samples}")
    if result.errors:
        print(f"Errors encountered: {len(result.errors)}")

    print("\n" + "-" * 70)
    print(f"{'Field':<30} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
    print("-" * 70)

    # Sort by support (number of positives) descending
    sorted_fields = sorted(
        result.field_metrics.items(),
        key=lambda x: x[1].support,
        reverse=True,
    )

    for field_name, metrics in sorted_fields:
        print(
            f"{field_name:<30} "
            f"{metrics.precision:>8.3f} "
            f"{metrics.recall:>8.3f} "
            f"{metrics.f1:>8.3f} "
            f"{metrics.support:>8}"
        )

    print("-" * 70)
    print(f"\n{'Micro-averaged:':<30} "
          f"{result.micro_precision:>8.3f} "
          f"{result.micro_recall:>8.3f} "
          f"{result.micro_f1:>8.3f}")
    print(f"{'Macro-averaged F1:':<30} {result.macro_f1:>26.3f}")

    # Identify worst performing fields
    print("\n" + "-" * 70)
    print("Fields needing improvement (F1 < 0.7, support > 10):")
    for field_name, metrics in sorted_fields:
        if metrics.f1 < 0.7 and metrics.support > 10:
            print(f"  - {field_name}: F1={metrics.f1:.3f}, "
                  f"Precision={metrics.precision:.3f}, Recall={metrics.recall:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ActionPredictor")
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/ml_training/registry_test.csv",
        help="Path to test CSV",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    args = parser.parse_args()

    result = evaluate_action_predictor(
        test_data_path=args.test_data,
        max_samples=args.max_samples,
        verbose=args.verbose,
    )

    print_evaluation_report(result)

    return result


if __name__ == "__main__":
    main()
