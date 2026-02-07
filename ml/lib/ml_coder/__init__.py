"""Machine learning utilities for the CPT coder and registry predictor."""

from .predictor import MLCoderService, MLCoderPredictor
from .registry_predictor import (
    RegistryMLPredictor,
    RegistryFieldPrediction,
    RegistryCaseClassification,
)
from .registry_training import (
    train_registry_model,
    evaluate_registry_model,
    train_and_evaluate as train_and_evaluate_registry,
)

__all__ = [
    # CPT predictor
    "MLCoderService",
    "MLCoderPredictor",
    # Registry predictor
    "RegistryMLPredictor",
    "RegistryFieldPrediction",
    "RegistryCaseClassification",
    # Registry training
    "train_registry_model",
    "evaluate_registry_model",
    "train_and_evaluate_registry",
]
