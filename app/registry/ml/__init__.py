"""Registry ML module for extraction-first architecture.

This module provides atomic clinical action extraction from procedure notes,
serving as the foundation for deterministic CPT code derivation.

Key components:
- ClinicalActions: Structured representation of all clinical actions
- ActionPredictor: Extracts clinical actions with evidence and confidence
- ActionResult: Single action extraction result with evidence spans

Usage:
    from app.registry.ml import ActionPredictor, ClinicalActions

    predictor = ActionPredictor()
    result = predictor.predict(note_text)

    # Access extracted actions
    if result.actions.ebus.performed:
        print(f"EBUS stations: {result.actions.ebus.stations}")

    # Check confidence and evidence
    for field, extraction in result.field_extractions.items():
        print(f"{field}: {extraction.value} (conf={extraction.confidence:.2f})")
"""

from app.registry.ml.models import (
    ActionResult,
    ClinicalActions,
    EBUSActions,
    BiopsyActions,
    NavigationActions,
    BALActions,
    BrushingsActions,
    PleuralActions,
    CAOActions,
    StentActions,
    BLVRActions,
    ComplicationActions,
    PredictionResult,
)
from app.registry.ml.action_predictor import ActionPredictor

__all__ = [
    "ActionPredictor",
    "ActionResult",
    "ClinicalActions",
    "EBUSActions",
    "BiopsyActions",
    "NavigationActions",
    "BALActions",
    "BrushingsActions",
    "PleuralActions",
    "CAOActions",
    "StentActions",
    "BLVRActions",
    "ComplicationActions",
    "PredictionResult",
]
