"""Shared registry field-path fragments for warning reconciliation."""

from __future__ import annotations

LINEAR_EBUS_PERFORMED = "procedures_performed.linear_ebus.performed=true"
TBNA_CONVENTIONAL_PERFORMED = "procedures_performed.tbna_conventional.performed=true"
PERIPHERAL_TBNA_PERFORMED = "procedures_performed.peripheral_tbna.performed=true"
BRUSHINGS_PERFORMED = "procedures_performed.brushings.performed=true"
TRANSBRONCHIAL_BIOPSY_PERFORMED = "procedures_performed.transbronchial_biopsy.performed=true"
BRONCHIAL_WASH_PERFORMED = "procedures_performed.bronchial_wash.performed=true"

__all__ = [
    "LINEAR_EBUS_PERFORMED",
    "TBNA_CONVENTIONAL_PERFORMED",
    "PERIPHERAL_TBNA_PERFORMED",
    "BRUSHINGS_PERFORMED",
    "TRANSBRONCHIAL_BIOPSY_PERFORMED",
    "BRONCHIAL_WASH_PERFORMED",
]
