# neurons/stress/well_generator.py

"""
Well-based Stress Generator for Hydrogen.

Samples relevant slices from The Well dataset and applies physics-preserving augmentations.
Follows docs/STRESS_TEST_DESIGN.md
"""

import random
from typing import List, Optional

from .base_generator import BaseStressGenerator, stress_registry
from .stress_models import StressVariant, StressSource


class WellStressGenerator(BaseStressGenerator):
    """
    Generates stress variants by sampling from The Well dataset.

    Currently supports mapping physics classes to relevant Well datasets.
    """

    # Mapping of physics classes to relevant Well datasets
    WELL_DATASET_MAP = {
        "hyperbolic": ["turbulence", "viscoelastic"],
        "elliptic": ["active_matter"],
        "parabolic": ["acoustic_scattering"],
        "multi_physics": ["active_matter", "viscoelastic"],
    }

    def generate(
        self,
        challenge_id: str,
        physics_class: str,
        seed: int,
        difficulty: float = 0.5
    ) -> List[StressVariant]:

        rng = random.Random(seed)
        variants = []

        datasets = self.WELL_DATASET_MAP.get(physics_class, [])
        if not datasets:
            return variants

        # Number of Well-based variants scales with difficulty
        num_variants = max(2, int(3 + difficulty * 4))

        for i in range(num_variants):
            dataset = rng.choice(datasets)

            # Simulate sampling a slice (in real implementation this would load actual data)
            slice_id = f"{dataset}_slice_{rng.randint(0, 999)}"

            variant = StressVariant(
                variant_id=f"well_{physics_class}_{i}",
                source=StressSource.WELL,
                parameters={
                    "well_dataset": dataset,
                    "slice_id": slice_id,
                },
                well_dataset=dataset,
                difficulty=difficulty,
                metadata={
                    "physics_justification": (
                        f"Sampled from The Well ({dataset}) to test generalization "
                        "on real simulation data outside the procedural distribution."
                    ),
                },
            )
            variants.append(variant)

        return variants


# Register for supported physics classes
for physics_class in WellStressGenerator.WELL_DATASET_MAP.keys():
    stress_registry.register(physics_class, WellStressGenerator())
