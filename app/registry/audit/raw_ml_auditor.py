"""RAW-ML auditor for extraction-first registry pipeline.

Critical invariant:
This auditor must use raw ML output only via MLCoderPredictor.classify_case(raw_note_text)
and must never call SmartHybridOrchestrator.get_codes() or any rules validation / LLM fallback.

Phase 7 Addition: RegistryFlagAuditor for flag-based auditing using the multi-label
registry flag model (roberta_pm3_registry). This maps predicted flags to possible CPT codes.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml.lib.ml_coder.predictor import CaseClassification, CodePrediction, MLCoderPredictor
from ml.lib.ml_coder.thresholds import CaseDifficulty

logger = logging.getLogger(__name__)

# Flag-to-CPT mapping for discrepancy reporting
# Maps registry boolean flags to the CPT codes they might generate
FLAG_TO_CPT_MAP: dict[str, list[str]] = {
    "bal": ["31624"],
    "brushings": ["31623"],
    "endobronchial_biopsy": ["31625"],
    "transbronchial_biopsy": ["31625", "31628", "31632"],
    "transbronchial_cryobiopsy": ["31628", "31632"],
    "tbna_conventional": ["31629", "31633"],
    "peripheral_tbna": ["31629", "31633"],
    "linear_ebus": ["31652", "31653"],
    "radial_ebus": ["31654"],
    "navigational_bronchoscopy": ["31627"],
    "fiducial_placement": ["31626"],
    "therapeutic_aspiration": ["31645", "31646"],
    "foreign_body_removal": ["31635"],
    "airway_dilation": ["31630", "31631"],
    "airway_stent": ["31636", "31637", "31638"],
    "thermal_ablation": ["31641"],
    "cryotherapy": ["31641"],
    "blvr": ["31647", "31648", "31649", "31651"],
    "peripheral_ablation": ["31641", "31651"],
    "bronchial_thermoplasty": ["31660", "31661"],
    "whole_lung_lavage": ["32997"],
    "rigid_bronchoscopy": ["31603"],
    "diagnostic_bronchoscopy": ["31622"],
    "bronchial_wash": ["31622"],
    # Pleural procedures
    "thoracentesis": ["32554", "32555"],
    "chest_tube": ["32551"],
    "ipc": ["32550"],
    "medical_thoracoscopy": ["32601", "32604", "32606", "32607", "32608", "32609", "32650"],
    "pleurodesis": ["32560", "32650"],
    "pleural_biopsy": ["32400", "32405", "32604", "32609"],
    "fibrinolytic_therapy": ["32561", "32562"],
}


@dataclass(frozen=True)
class RawMLAuditConfig:
    use_buckets: bool = True
    top_k: int = 25
    min_prob: float = 0.50
    self_correct_min_prob: float = 0.95

    @classmethod
    def from_env(cls) -> "RawMLAuditConfig":
        def _as_bool(name: str, default: str) -> bool:
            return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y"}

        def _as_int(name: str, default: int) -> int:
            try:
                return int(os.getenv(name, str(default)).strip())
            except ValueError:
                return default

        def _as_float(name: str, default: float) -> float:
            try:
                return float(os.getenv(name, str(default)).strip())
            except ValueError:
                return default

        return cls(
            use_buckets=_as_bool("REGISTRY_ML_AUDIT_USE_BUCKETS", "1"),
            top_k=_as_int("REGISTRY_ML_AUDIT_TOP_K", 25),
            min_prob=_as_float("REGISTRY_ML_AUDIT_MIN_PROB", 0.50),
            self_correct_min_prob=_as_float("REGISTRY_ML_SELF_CORRECT_MIN_PROB", 0.95),
        )


class RawMLAuditor:
    def __init__(self, predictor: MLCoderPredictor | None = None) -> None:
        self._predictor: MLCoderPredictor | None = predictor
        self._load_error: str | None = None
        if self._predictor is None:
            try:
                self._predictor = MLCoderPredictor()
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"{type(exc).__name__}: {exc}"
                logger.warning("RAW-ML predictor unavailable: %s", self._load_error)

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def is_loaded(self) -> bool:
        """Return True if the auditor is ready to serve predictions."""
        return self._predictor is not None

    def warm(self) -> "RawMLAuditor":
        """Eagerly warm underlying artifacts if any are lazily loaded."""
        if self._predictor is None:
            return self
        try:
            # If the predictor has its own warm method, use it.
            warm = getattr(self._predictor, "warm", None)
            if callable(warm):
                warm()
        except Exception:  # noqa: BLE001
            pass
        return self

    def classify(self, raw_note_text: str) -> CaseClassification:
        if self._predictor is None:
            return CaseClassification(
                predictions=[],
                high_conf=[],
                gray_zone=[],
                difficulty=CaseDifficulty.LOW_CONF,
            )
        return self._predictor.classify_case(raw_note_text)

    def audit_predictions(
        self, cls: CaseClassification, cfg: RawMLAuditConfig | None = None
    ) -> list[CodePrediction]:
        cfg = cfg or RawMLAuditConfig.from_env()

        if cfg.use_buckets:
            return list(cls.high_conf) + list(cls.gray_zone)

        preds = cls.predictions[: cfg.top_k]
        return [p for p in preds if p.prob >= cfg.min_prob]

    def self_correct_triggers(
        self, cls: CaseClassification, cfg: RawMLAuditConfig | None = None
    ) -> list[CodePrediction]:
        cfg = cfg or RawMLAuditConfig.from_env()
        return [p for p in cls.high_conf if p.prob >= cfg.self_correct_min_prob]


@dataclass
class RegistryFlagPrediction:
    """A single flag prediction from the registry flag model."""

    flag_name: str
    probability: float
    threshold: float
    is_predicted: bool

    @property
    def possible_cpt_codes(self) -> list[str]:
        """Return CPT codes that could be generated by this flag."""
        return FLAG_TO_CPT_MAP.get(self.flag_name, [])


@dataclass
class RegistryFlagClassification:
    """Classification result from the registry flag model."""

    predictions: dict[str, RegistryFlagPrediction]
    raw_probabilities: dict[str, float]

    @property
    def predicted_flags(self) -> dict[str, bool]:
        """Return {flag_name: True/False} based on thresholds."""
        return {name: pred.is_predicted for name, pred in self.predictions.items()}

    @property
    def high_confidence_flags(self) -> list[str]:
        """Return flags predicted with high confidence (>= 0.9)."""
        return [name for name, pred in self.predictions.items() if pred.is_predicted and pred.probability >= 0.9]

    def flags_to_cpt_codes(self) -> list[str]:
        """Map all predicted flags to possible CPT codes."""
        codes: set[str] = set()
        for flag_name, pred in self.predictions.items():
            if pred.is_predicted:
                codes.update(pred.possible_cpt_codes)
        return sorted(codes)


class RegistryFlagAuditor:
    """Auditor that uses the registry flag multi-label model for flag-based auditing.

    This auditor loads the roberta_pm3_registry model and predicts registry boolean flags
    directly from procedure note text. It can be used to:
    1. Audit derived CPT codes by comparing to ML-predicted flags
    2. Identify discrepancies between extraction and ML predictions
    3. Generate flag-based audit warnings

    The model artifacts are expected to be co-located in the model directory:
    - tokenizer/
    - pytorch_model.bin (or model.safetensors)
    - registry_label_fields.json
    - thresholds.json
    """

    DEFAULT_MODEL_DIR = Path("data/models/roberta_pm3_registry")

    def __init__(
        self,
        model_dir: Path | str | None = None,
        lazy_load: bool = True,
    ) -> None:
        """Initialize the registry flag auditor.

        Args:
            model_dir: Path to model directory containing artifacts. Defaults to
                       data/models/roberta_pm3_registry.
            lazy_load: If True, defer model loading until first classify() call.
        """
        self._model_dir = Path(model_dir) if model_dir else self.DEFAULT_MODEL_DIR
        self._model: Any = None
        self._tokenizer: Any = None
        self._label_fields: list[str] = []
        self._thresholds: dict[str, float] = {}
        self._loaded = False

        if not lazy_load:
            self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Load model artifacts from model directory."""
        if self._loaded:
            return

        # Check if model directory exists
        if not self._model_dir.exists():
            logger.warning(f"Model directory not found: {self._model_dir}")
            self._loaded = True
            return

        # Load label fields
        label_fields_path = self._model_dir / "registry_label_fields.json"
        if label_fields_path.exists():
            with open(label_fields_path) as f:
                self._label_fields = json.load(f)
            logger.info(f"Loaded {len(self._label_fields)} label fields")
        else:
            logger.warning(f"Label fields not found: {label_fields_path}")

        # Load thresholds
        thresholds_path = self._model_dir / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path) as f:
                self._thresholds = json.load(f)
            logger.info(f"Loaded {len(self._thresholds)} thresholds")
        else:
            logger.warning(f"Thresholds not found: {thresholds_path}")

        # Try to load model and tokenizer (optional - may not be available in all envs)
        try:
            import torch
            from transformers import AutoTokenizer

            tokenizer_path = self._model_dir / "tokenizer"
            if tokenizer_path.exists():
                self._tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
                logger.info("Loaded tokenizer")

            # Check for model files
            model_file = self._model_dir / "pytorch_model.bin"
            safetensors_file = self._model_dir / "model.safetensors"
            if model_file.exists() or safetensors_file.exists():
                # Import the model class if available
                try:
                    from ml.scripts.train_roberta_pm3 import RoBERTaPM3MultiLabel

                    num_labels = len(self._label_fields) if self._label_fields else 30
                    self._model = RoBERTaPM3MultiLabel.from_pretrained(
                        self._model_dir, num_labels
                    )
                    self._model.eval()
                    logger.info("Loaded registry flag model")
                except ImportError:
                    logger.warning("Could not import RoBERTaPM3MultiLabel - model not loaded")
        except ImportError:
            logger.info("PyTorch/transformers not available - using thresholds-only mode")

        self._loaded = True

    def is_loaded(self) -> bool:
        """Return True if artifacts have been loaded (even if model weights are unavailable)."""
        return self._loaded

    def warm(self) -> "RegistryFlagAuditor":
        """Eagerly load artifacts (tokenizer/model/thresholds) so first request is fast."""
        self._load_artifacts()
        return self

    def classify(self, raw_note_text: str) -> RegistryFlagClassification:
        """Classify procedure note text to predict registry flags.

        Args:
            raw_note_text: The raw procedure note text.

        Returns:
            RegistryFlagClassification with predictions for all flags.
        """
        self._load_artifacts()

        # If model is available, run inference
        if self._model is not None and self._tokenizer is not None:
            return self._classify_with_model(raw_note_text)

        # Fallback: return empty predictions (model not available)
        logger.debug("Model not loaded - returning empty classification")
        predictions = {}
        for flag_name in self._label_fields:
            threshold = self._thresholds.get(flag_name, 0.5)
            predictions[flag_name] = RegistryFlagPrediction(
                flag_name=flag_name,
                probability=0.0,
                threshold=threshold,
                is_predicted=False,
            )
        return RegistryFlagClassification(predictions=predictions, raw_probabilities={})

    def _classify_with_model(self, raw_note_text: str) -> RegistryFlagClassification:
        """Run model inference to classify text."""
        import torch

        # Tokenize
        encoding = self._tokenizer(
            raw_note_text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
        )

        # Run inference
        with torch.no_grad():
            outputs = self._model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
            )
            probs = torch.sigmoid(outputs).squeeze().cpu().numpy()

        # Build predictions
        raw_probabilities: dict[str, float] = {}
        predictions: dict[str, RegistryFlagPrediction] = {}

        for i, flag_name in enumerate(self._label_fields):
            prob = float(probs[i]) if i < len(probs) else 0.0
            threshold = self._thresholds.get(flag_name, 0.5)
            raw_probabilities[flag_name] = prob
            predictions[flag_name] = RegistryFlagPrediction(
                flag_name=flag_name,
                probability=prob,
                threshold=threshold,
                is_predicted=prob >= threshold,
            )

        return RegistryFlagClassification(predictions=predictions, raw_probabilities=raw_probabilities)

    def get_predicted_flags(self, raw_note_text: str) -> dict[str, bool]:
        """Convenience method to get {flag_name: True/False} dict."""
        classification = self.classify(raw_note_text)
        return classification.predicted_flags

    def audit_derived_codes(
        self,
        raw_note_text: str,
        derived_codes: list[str],
    ) -> list[str]:
        """Audit derived CPT codes against ML-predicted flags.

        Args:
            raw_note_text: The procedure note text.
            derived_codes: CPT codes derived from deterministic rules.

        Returns:
            List of audit warning strings for discrepancies.
        """
        classification = self.classify(raw_note_text)
        ml_possible_codes = set(classification.flags_to_cpt_codes())
        derived_set = set(derived_codes)

        warnings: list[str] = []

        # Find high-confidence flags that suggest codes not in derived set
        for flag_name in classification.high_confidence_flags:
            flag_codes = FLAG_TO_CPT_MAP.get(flag_name, [])
            missing_codes = [c for c in flag_codes if c not in derived_set]
            if missing_codes:
                pred = classification.predictions[flag_name]
                warnings.append(
                    f"ML predicts {flag_name} (p={pred.probability:.2f}) -> "
                    f"possible codes {missing_codes} not in derived codes"
                )

        return warnings


__all__ = [
    "RawMLAuditor",
    "RawMLAuditConfig",
    "RegistryFlagAuditor",
    "RegistryFlagClassification",
    "RegistryFlagPrediction",
    "FLAG_TO_CPT_MAP",
]
