# neurons/symbolic/extractor.py

"""
Basic SymbolicMetadataExtractor for Phase 0.

Lightweight rule-based extraction. Can be upgraded later with ModelingToolkit.
"""

from typing import Dict, Any

from .symbolic_models import SymbolicMetadata


class SymbolicMetadataExtractor:
    """
    Extracts basic symbolic features from a challenge definition.

    Phase 0 version is rule-based. Future versions can use ModelingToolkit.jl
    or more advanced symbolic analysis.
    """

    def extract(self, challenge_id: str, challenge_config: Dict[str, Any]) -> SymbolicMetadata:
        physics_class = challenge_config.get("physics_class", "unknown")

        symmetries = []
        conservation_laws = []
        boundary_types = []
        coupling_terms = []
        dimensionless_groups = []

        # Simple rule-based extraction
        if physics_class in ["elliptic", "poisson", "darcy"]:
            conservation_laws.append("divergence_free_or_mass_conservation")
            boundary_types.append(challenge_config.get("boundary_type", "dirichlet"))

        elif physics_class == "hyperbolic":
            conservation_laws.append("conservation_law")
            symmetries.append("translation_invariance_possible")

        elif physics_class == "parabolic":
            conservation_laws.append("energy_conservation")

        elif physics_class in ["elasticity", "multi_physics", "thermo_elasticity"]:
            conservation_laws.append("momentum_conservation")
            conservation_laws.append("energy_conservation")
            coupling_terms.append("thermo_mechanical_coupling")

        return SymbolicMetadata(
            challenge_id=challenge_id,
            symmetries=symmetries,
            conservation_laws=conservation_laws,
            boundary_types=boundary_types,
            coupling_terms=coupling_terms,
            dimensionless_groups=dimensionless_groups,
            extracted_at="phase0_rule_based",
            metadata={"physics_class": physics_class},
        )
