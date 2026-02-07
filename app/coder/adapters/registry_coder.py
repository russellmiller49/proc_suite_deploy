"""Registry-Based CPT Coder - Deterministic code derivation from clinical actions.

This module implements the extraction-first architecture where CPT codes are
derived deterministically from structured registry data (ClinicalActions),
rather than predicted probabilistically from raw text.

Architecture:
    Text → ActionPredictor → ClinicalActions → RegistryBasedCoder → CPT Codes

Benefits:
- Auditable: "We billed 31653 because registry.ebus.stations.count >= 3"
- Deterministic: Same input always produces same output
- Evidence-backed: Each code linked to specific registry fields

Usage:
    from app.registry.ml import ActionPredictor, ClinicalActions
    from app.coder.adapters.registry_coder import RegistryBasedCoder

    predictor = ActionPredictor()
    coder = RegistryBasedCoder()

    result = predictor.predict(note_text)
    codes = coder.derive_codes(result.actions)

    for code in codes:
        print(f"{code.code}: {code.description}")
        print(f"  Rationale: {code.rationale}")
        print(f"  Evidence: {code.evidence_fields}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.common.logger import get_logger
from app.registry.ml.models import ClinicalActions
from app.coder.domain_rules import (
    apply_addon_family_rules,
    apply_ebus_aspiration_bundles,
    apply_thoracentesis_ipc_bundles,
    apply_all_ncci_bundles,
)


logger = get_logger("coder.adapters.registry_coder")


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class DerivedCode:
    """A CPT code derived from registry data with full audit trail.

    Attributes:
        code: CPT code string (e.g., "31653")
        description: Human-readable description
        rationale: Explanation of why this code was derived
        evidence_fields: Registry fields that support this code
        confidence: Confidence level (always 1.0 for deterministic derivation)
        is_add_on: Whether this is an add-on code
        requires_primary: If add-on, which primary codes it can accompany
    """

    code: str
    description: str
    rationale: str
    evidence_fields: list[str] = field(default_factory=list)
    confidence: float = 1.0
    is_add_on: bool = False
    requires_primary: list[str] = field(default_factory=list)


@dataclass
class DerivationResult:
    """Complete result from registry-based code derivation.

    Attributes:
        codes: List of derived CPT codes
        bundled_codes: Codes removed due to bundling rules
        bundling_reasons: Explanations for bundled codes
        warnings: Any warnings during derivation
    """

    codes: list[DerivedCode]
    bundled_codes: list[str] = field(default_factory=list)
    bundling_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def get_code_list(self) -> list[str]:
        """Return just the CPT code strings."""
        return [c.code for c in self.codes]


# =============================================================================
# CPT Code Definitions
# =============================================================================

# Bronchoscopy codes
CPT_CODES = {
    # Diagnostic bronchoscopy
    "31622": "Bronchoscopy, diagnostic, with cell washing",
    "31623": "Bronchoscopy with brushing or protected brushings",
    "31624": "Bronchoscopy with bronchoalveolar lavage",
    "31625": "Bronchoscopy with bronchial or endobronchial biopsy(s)",
    # Navigation add-on
    "31627": "Bronchoscopy with computer-assisted navigation (add-on)",
    # Transbronchial biopsy
    "31628": "Bronchoscopy with transbronchial lung biopsy(s), single lobe",
    "31629": "Bronchoscopy with transbronchial needle aspiration biopsy(s)",
    "+31632": "Bronchoscopy with transbronchial lung biopsy(s), each additional lobe (add-on)",
    "+31633": "Bronchoscopy with transbronchial needle aspiration, each additional lobe (add-on)",
    # EBUS
    "31652": "Bronchoscopy with EBUS-TBNA, 1-2 lymph node stations",
    "31653": "Bronchoscopy with EBUS-TBNA, 3 or more lymph node stations",
    "+31654": "Bronchoscopy with transendoscopic ultrasound during EBUS-TBNA (add-on)",
    # Stent
    "31631": "Bronchoscopy with tracheal dilation and stent placement",
    "31636": "Bronchoscopy with bronchial stent placement, initial",
    "+31637": "Bronchoscopy with bronchial stent placement, each additional (add-on)",
    # Therapeutic
    "31638": "Bronchoscopy with balloon bronchoplasty",
    "31640": "Bronchoscopy with excision of tumor",
    "31641": "Bronchoscopy with destruction of tumor or relief of stenosis",
    "31645": "Bronchoscopy with therapeutic aspiration, initial",
    "31646": "Bronchoscopy with therapeutic aspiration, subsequent",
    # BLVR
    "31647": "Bronchoscopy with balloon occlusion for BLVR assessment",
    "+31651": "Bronchoscopy with endobronchial valve insertion, each (add-on)",
    # Pleural procedures
    "32550": "Insertion of indwelling tunneled pleural catheter with cuff",
    "32554": "Thoracentesis without imaging guidance",
    "32555": "Thoracentesis with imaging guidance",
    "32556": "Pleural drainage, percutaneous, with insertion of catheter without imaging",
    "32557": "Pleural drainage, percutaneous, with insertion of catheter with imaging",
    # Thoracoscopy
    "32601": "Thoracoscopy, diagnostic, lungs/pericardium/mediastinal space",
    "32606": "Thoracoscopy, diagnostic, with biopsy(s) of lung infiltrate(s)",
    "32650": "Thoracoscopy with pleurodesis",
}

# Add-on codes and their required primaries
ADD_ON_CODES = {
    "31627": ["31622", "31623", "31624", "31625", "31628", "31629", "31652", "31653"],
    "+31632": ["31628"],
    "+31633": ["31629"],
    "+31637": ["31636", "31631"],
    "+31651": ["31647"],
    "+31654": ["31652", "31653"],
}


# =============================================================================
# Registry-Based Coder Implementation
# =============================================================================


class RegistryBasedCoder:
    """Derives CPT codes deterministically from structured clinical actions.

    This coder implements the extraction-first architecture where:
    1. ClinicalActions are extracted from procedure notes (via ActionPredictor)
    2. CPT codes are derived using deterministic rules
    3. Each code includes rationale and evidence for audit

    Unlike ML-based coders, this produces deterministic results with
    full transparency into why each code was selected.
    """

    def __init__(self, apply_bundling: bool = True) -> None:
        """Initialize the registry-based coder.

        Args:
            apply_bundling: Whether to apply NCCI bundling rules (default True)
        """
        self.apply_bundling = apply_bundling
        self._version = "registry_coder_v1.0"

    @property
    def version(self) -> str:
        """Return the coder version."""
        return self._version

    def derive_codes(self, actions: ClinicalActions) -> DerivationResult:
        """Derive CPT codes from clinical actions.

        This is the main entry point for extraction-first CPT coding.

        Args:
            actions: Structured clinical actions from ActionPredictor

        Returns:
            DerivationResult with derived codes, bundling info, and warnings
        """
        codes: list[DerivedCode] = []
        warnings: list[str] = []

        # Derive codes for each procedure type
        codes.extend(self._derive_ebus_codes(actions))
        codes.extend(self._derive_biopsy_codes(actions))
        codes.extend(self._derive_sampling_codes(actions))
        codes.extend(self._derive_navigation_codes(actions, codes))
        codes.extend(self._derive_pleural_codes(actions))
        codes.extend(self._derive_cao_codes(actions))
        codes.extend(self._derive_stent_codes(actions))
        codes.extend(self._derive_blvr_codes(actions))

        # Add diagnostic bronchoscopy if any bronchoscopic procedure but no other codes
        if actions.diagnostic_bronchoscopy and not codes:
            codes.append(
                DerivedCode(
                    code="31622",
                    description=CPT_CODES["31622"],
                    rationale="Diagnostic bronchoscopy with no other billable procedures",
                    evidence_fields=["diagnostic_bronchoscopy"],
                )
            )

        # Apply bundling rules
        bundled_codes: list[str] = []
        bundling_reasons: list[str] = []

        if self.apply_bundling and codes:
            codes, bundled_codes, bundling_reasons = self._apply_bundling_rules(codes)

        return DerivationResult(
            codes=codes,
            bundled_codes=bundled_codes,
            bundling_reasons=bundling_reasons,
            warnings=warnings,
        )

    # =========================================================================
    # EBUS Code Derivation
    # =========================================================================

    def _derive_ebus_codes(self, actions: ClinicalActions) -> list[DerivedCode]:
        """Derive EBUS-TBNA codes based on station count.

        CPT Rules:
        - 31652: EBUS-TBNA, 1-2 lymph node stations
        - 31653: EBUS-TBNA, 3 or more lymph node stations
        """
        if not actions.ebus.performed:
            return []

        station_count = actions.ebus.station_count
        stations_str = ", ".join(actions.ebus.stations) if actions.ebus.stations else "unknown"

        if station_count >= 3:
            return [
                DerivedCode(
                    code="31653",
                    description=CPT_CODES["31653"],
                    rationale=f"EBUS-TBNA with {station_count} stations ({stations_str}) >= 3",
                    evidence_fields=["ebus.performed", "ebus.stations"],
                )
            ]
        elif station_count >= 1:
            return [
                DerivedCode(
                    code="31652",
                    description=CPT_CODES["31652"],
                    rationale=f"EBUS-TBNA with {station_count} station(s) ({stations_str}) < 3",
                    evidence_fields=["ebus.performed", "ebus.stations"],
                )
            ]
        else:
            # EBUS performed but no stations documented
            return [
                DerivedCode(
                    code="31652",
                    description=CPT_CODES["31652"],
                    rationale="EBUS performed but station count not documented; defaulting to 31652",
                    evidence_fields=["ebus.performed"],
                    confidence=0.8,  # Lower confidence due to missing station data
                )
            ]

    # =========================================================================
    # Biopsy Code Derivation
    # =========================================================================

    def _derive_biopsy_codes(self, actions: ClinicalActions) -> list[DerivedCode]:
        """Derive biopsy codes based on biopsy type and location.

        CPT Rules:
        - 31625: Bronchial or endobronchial biopsy
        - 31628: Transbronchial lung biopsy, single lobe
        - +31632: Transbronchial lung biopsy, each additional lobe (add-on)
        """
        codes: list[DerivedCode] = []

        # Transbronchial biopsy
        if actions.biopsy.transbronchial_performed:
            sites = actions.biopsy.transbronchial_sites
            lobe_count = len(set(sites)) if sites else 1
            sites_str = ", ".join(sites) if sites else "unspecified"

            # Primary code for first lobe
            codes.append(
                DerivedCode(
                    code="31628",
                    description=CPT_CODES["31628"],
                    rationale=f"Transbronchial biopsy performed at {sites_str}",
                    evidence_fields=["biopsy.transbronchial_performed", "biopsy.transbronchial_sites"],
                )
            )

            # Add-on for additional lobes
            if lobe_count > 1:
                for _ in range(lobe_count - 1):
                    codes.append(
                        DerivedCode(
                            code="+31632",
                            description=CPT_CODES["+31632"],
                            rationale=f"Additional lobe biopsied ({lobe_count} lobes total)",
                            evidence_fields=["biopsy.transbronchial_sites"],
                            is_add_on=True,
                            requires_primary=["31628"],
                        )
                    )

        # Endobronchial biopsy (if not already covered by transbronchial)
        if actions.biopsy.endobronchial_performed and not actions.biopsy.transbronchial_performed:
            codes.append(
                DerivedCode(
                    code="31625",
                    description=CPT_CODES["31625"],
                    rationale="Endobronchial biopsy performed",
                    evidence_fields=["biopsy.endobronchial_performed"],
                )
            )

        # Cryobiopsy (uses same code as transbronchial biopsy)
        if actions.biopsy.cryobiopsy_performed and not actions.biopsy.transbronchial_performed:
            sites = actions.biopsy.cryobiopsy_sites
            sites_str = ", ".join(sites) if sites else "unspecified"
            codes.append(
                DerivedCode(
                    code="31628",
                    description=CPT_CODES["31628"],
                    rationale=f"Transbronchial cryobiopsy performed at {sites_str}",
                    evidence_fields=["biopsy.cryobiopsy_performed", "biopsy.cryobiopsy_sites"],
                )
            )

        return codes

    # =========================================================================
    # Sampling Code Derivation (BAL, Brushings, Wash)
    # =========================================================================

    def _derive_sampling_codes(self, actions: ClinicalActions) -> list[DerivedCode]:
        """Derive sampling codes for BAL, brushings, and bronchial wash.

        CPT Rules:
        - 31624: Bronchoalveolar lavage
        - 31623: Bronchial brushings
        - 31622: Diagnostic bronchoscopy with wash (only if no other codes)
        """
        codes: list[DerivedCode] = []

        # BAL
        if actions.bal.performed:
            sites_str = ", ".join(actions.bal.sites) if actions.bal.sites else "unspecified site"
            codes.append(
                DerivedCode(
                    code="31624",
                    description=CPT_CODES["31624"],
                    rationale=f"Bronchoalveolar lavage performed at {sites_str}",
                    evidence_fields=["bal.performed", "bal.sites"],
                )
            )

        # Brushings
        if actions.brushings.performed:
            sites_str = ", ".join(actions.brushings.sites) if actions.brushings.sites else "unspecified site"
            codes.append(
                DerivedCode(
                    code="31623",
                    description=CPT_CODES["31623"],
                    rationale=f"Bronchial brushings performed at {sites_str}",
                    evidence_fields=["brushings.performed", "brushings.sites"],
                )
            )

        # Bronchial wash - only code if no other sampling performed
        # (wash is bundled into 31622 diagnostic bronchoscopy)

        return codes

    # =========================================================================
    # Navigation Code Derivation
    # =========================================================================

    def _derive_navigation_codes(
        self, actions: ClinicalActions, existing_codes: list[DerivedCode]
    ) -> list[DerivedCode]:
        """Derive navigation add-on code.

        CPT Rules:
        - 31627: Computer-assisted navigation (add-on only)
        - Requires a primary bronchoscopy procedure
        """
        if not actions.navigation.performed:
            return []

        # Check if we have a valid primary code
        primary_codes = {c.code for c in existing_codes}
        valid_primaries = set(ADD_ON_CODES["31627"])

        if not (primary_codes & valid_primaries):
            # No valid primary - cannot bill navigation add-on
            logger.warning(
                "Navigation performed but no valid primary bronchoscopy code; "
                "31627 requires primary procedure"
            )
            return []

        platform = actions.navigation.platform or "unspecified"
        rationale = f"Computer-assisted navigation using {platform}"

        if actions.navigation.is_robotic:
            rationale += " (robotic-assisted)"

        return [
            DerivedCode(
                code="31627",
                description=CPT_CODES["31627"],
                rationale=rationale,
                evidence_fields=["navigation.performed", "navigation.platform"],
                is_add_on=True,
                requires_primary=ADD_ON_CODES["31627"],
            )
        ]

    # =========================================================================
    # Pleural Code Derivation
    # =========================================================================

    def _derive_pleural_codes(self, actions: ClinicalActions) -> list[DerivedCode]:
        """Derive pleural procedure codes.

        CPT Rules:
        - 32554/32555: Thoracentesis (without/with imaging)
        - 32550: Tunneled pleural catheter (IPC)
        - 32650: Thoracoscopy with pleurodesis
        - 32601: Diagnostic thoracoscopy
        """
        codes: list[DerivedCode] = []

        # Thoracentesis
        if actions.pleural.thoracentesis_performed:
            # Default to with imaging (32555) as most thoracenteses use ultrasound
            codes.append(
                DerivedCode(
                    code="32555",
                    description=CPT_CODES["32555"],
                    rationale="Thoracentesis performed (assuming imaging guidance)",
                    evidence_fields=["pleural.thoracentesis_performed"],
                )
            )

        # IPC (tunneled pleural catheter)
        if actions.pleural.ipc_performed:
            action = actions.pleural.ipc_action or "insertion"
            codes.append(
                DerivedCode(
                    code="32550",
                    description=CPT_CODES["32550"],
                    rationale=f"Tunneled pleural catheter {action}",
                    evidence_fields=["pleural.ipc_performed", "pleural.ipc_action"],
                )
            )

        # Chest tube
        if actions.pleural.chest_tube_performed:
            codes.append(
                DerivedCode(
                    code="32556",
                    description=CPT_CODES["32556"],
                    rationale="Chest tube placement",
                    evidence_fields=["pleural.chest_tube_performed"],
                )
            )

        # Thoracoscopy
        if actions.pleural.thoracoscopy_performed:
            if actions.pleural.pleurodesis_performed:
                codes.append(
                    DerivedCode(
                        code="32650",
                        description=CPT_CODES["32650"],
                        rationale="Thoracoscopy with pleurodesis",
                        evidence_fields=["pleural.thoracoscopy_performed", "pleural.pleurodesis_performed"],
                    )
                )
            else:
                codes.append(
                    DerivedCode(
                        code="32601",
                        description=CPT_CODES["32601"],
                        rationale="Diagnostic thoracoscopy",
                        evidence_fields=["pleural.thoracoscopy_performed"],
                    )
                )
        elif actions.pleural.pleurodesis_performed:
            # Pleurodesis without thoracoscopy (e.g., via chest tube)
            codes.append(
                DerivedCode(
                    code="32650",
                    description=CPT_CODES["32650"],
                    rationale="Pleurodesis performed",
                    evidence_fields=["pleural.pleurodesis_performed"],
                )
            )

        return codes

    # =========================================================================
    # CAO (Central Airway Obstruction) Code Derivation
    # =========================================================================

    def _derive_cao_codes(self, actions: ClinicalActions) -> list[DerivedCode]:
        """Derive CAO therapeutic codes.

        CPT Rules:
        - 31641: Destruction of tumor or relief of stenosis
        - 31638: Balloon bronchoplasty
        """
        codes: list[DerivedCode] = []

        # Thermal ablation or cryotherapy → destruction code
        if actions.cao.thermal_ablation_performed or actions.cao.cryotherapy_performed:
            modalities = []
            if actions.cao.thermal_ablation_performed:
                modalities.append("thermal ablation")
            if actions.cao.cryotherapy_performed:
                modalities.append("cryotherapy")

            codes.append(
                DerivedCode(
                    code="31641",
                    description=CPT_CODES["31641"],
                    rationale=f"Airway tumor destruction via {', '.join(modalities)}",
                    evidence_fields=[
                        "cao.thermal_ablation_performed",
                        "cao.cryotherapy_performed",
                    ],
                )
            )

        # Balloon dilation
        if actions.cao.dilation_performed:
            codes.append(
                DerivedCode(
                    code="31638",
                    description=CPT_CODES["31638"],
                    rationale="Balloon bronchoplasty for airway dilation",
                    evidence_fields=["cao.dilation_performed"],
                )
            )

        return codes

    # =========================================================================
    # Stent Code Derivation
    # =========================================================================

    def _derive_stent_codes(self, actions: ClinicalActions) -> list[DerivedCode]:
        """Derive airway stent codes.

        CPT Rules:
        - 31631: Tracheal stent placement
        - 31636: Bronchial stent placement, initial
        - +31637: Bronchial stent placement, additional (add-on)
        """
        if not actions.stent.performed:
            return []

        location = actions.stent.location or "unspecified"
        action = actions.stent.action or "placement"

        # Determine if tracheal or bronchial based on location
        is_tracheal = "trache" in location.lower() if location else False

        if is_tracheal:
            return [
                DerivedCode(
                    code="31631",
                    description=CPT_CODES["31631"],
                    rationale=f"Tracheal stent {action} at {location}",
                    evidence_fields=["stent.performed", "stent.location", "stent.action"],
                )
            ]
        else:
            return [
                DerivedCode(
                    code="31636",
                    description=CPT_CODES["31636"],
                    rationale=f"Bronchial stent {action} at {location}",
                    evidence_fields=["stent.performed", "stent.location", "stent.action"],
                )
            ]

    # =========================================================================
    # BLVR Code Derivation
    # =========================================================================

    def _derive_blvr_codes(self, actions: ClinicalActions) -> list[DerivedCode]:
        """Derive BLVR (bronchoscopic lung volume reduction) codes.

        CPT Rules:
        - 31647: Balloon occlusion for assessment (Chartis)
        - +31651: Endobronchial valve insertion, each valve (add-on)
        """
        codes: list[DerivedCode] = []

        # Chartis assessment
        if actions.blvr.chartis_performed:
            codes.append(
                DerivedCode(
                    code="31647",
                    description=CPT_CODES["31647"],
                    rationale="Chartis assessment for BLVR evaluation",
                    evidence_fields=["blvr.chartis_performed"],
                )
            )

        # Valve placement
        if actions.blvr.performed and actions.blvr.valve_count:
            valve_count = actions.blvr.valve_count
            target_lobe = actions.blvr.target_lobe or "unspecified lobe"

            # Each valve is billed separately with +31651
            for i in range(valve_count):
                codes.append(
                    DerivedCode(
                        code="+31651",
                        description=CPT_CODES["+31651"],
                        rationale=f"Endobronchial valve #{i+1} placed in {target_lobe}",
                        evidence_fields=["blvr.performed", "blvr.valve_count", "blvr.target_lobe"],
                        is_add_on=True,
                        requires_primary=["31647"],
                    )
                )

        return codes

    # =========================================================================
    # Bundling Rules
    # =========================================================================

    def _apply_bundling_rules(
        self, codes: list[DerivedCode]
    ) -> tuple[list[DerivedCode], list[str], list[str]]:
        """Apply NCCI bundling rules to derived codes.

        Returns:
            Tuple of (kept_codes, bundled_codes, reasons)
        """
        code_strings = [c.code for c in codes]
        bundled: list[str] = []
        reasons: list[str] = []

        # Apply EBUS-Aspiration bundles
        ebus_result = apply_ebus_aspiration_bundles(code_strings)
        bundled.extend(ebus_result.removed_codes)
        reasons.extend([r[2] for r in ebus_result.bundle_reasons])
        code_strings = ebus_result.kept_codes

        # Apply Thoracentesis-IPC bundles
        ipc_result = apply_thoracentesis_ipc_bundles(code_strings)
        bundled.extend(ipc_result.removed_codes)
        reasons.extend([r[2] for r in ipc_result.bundle_reasons])
        code_strings = ipc_result.kept_codes

        # Apply add-on family rules
        family_result = apply_addon_family_rules(code_strings)
        code_strings = family_result.converted_codes
        if family_result.conversions:
            reasons.extend([c[2] for c in family_result.conversions])

        # Filter original codes to only kept ones
        kept_code_set = set(code_strings)
        kept_codes = [c for c in codes if c.code in kept_code_set]

        return kept_codes, bundled, reasons


# =============================================================================
# Convenience Functions
# =============================================================================


def derive_codes_from_actions(actions: ClinicalActions) -> DerivationResult:
    """Convenience function to derive CPT codes from clinical actions.

    Args:
        actions: ClinicalActions from ActionPredictor

    Returns:
        DerivationResult with derived codes
    """
    coder = RegistryBasedCoder()
    return coder.derive_codes(actions)


__all__ = [
    "RegistryBasedCoder",
    "DerivedCode",
    "DerivationResult",
    "derive_codes_from_actions",
]
