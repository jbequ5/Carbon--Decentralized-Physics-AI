# neurons/stress/procedural_generator.py

"""
Procedural Stress Generator for Hydrogen - Full Phase 0 Coverage (Deep).

All physics classes now have deep, nuanced implementations comparable to Burgers.
Follows docs/STRESS_TEST_DESIGN.md
"""

import random
from typing import List

from .base_generator import BaseStressGenerator, stress_registry
from .stress_models import StressVariant, StressSource


class ProceduralStressGenerator(BaseStressGenerator):
    """
    Generates procedural stress variants based on physics class.
    Deep implementations for all Phase 0 problem types.
    """

    def generate(self, challenge_id: str, physics_class: str, seed: int, difficulty: float = 0.5):
        rng = random.Random(seed)
        variants = []

        if physics_class in ["elliptic", "poisson", "darcy"]:
            variants.extend(self._generate_elliptic_stress(rng, difficulty))

        elif physics_class == "hyperbolic":
            variants.extend(self._generate_burgers_stress(rng, difficulty))

        elif physics_class == "parabolic":
            variants.extend(self._generate_heat_stress(rng, difficulty))

        elif physics_class in ["incompressible", "navier_stokes"]:
            variants.extend(self._generate_ns_stress(rng, difficulty))

        elif physics_class == "elasticity":
            variants.extend(self._generate_elasticity_stress(rng, difficulty))

        elif physics_class in ["multi_physics", "thermo_elasticity"]:
            variants.extend(self._generate_thermo_elasticity_stress(rng, difficulty))

        return variants

    # ============================================================
    # Elliptic (Poisson / Darcy) - Deep
    # ============================================================

    def _generate_elliptic_stress(self, rng, difficulty):
        variants = []
        num = max(3, int(4 + difficulty * 6))

        for i in range(num):
            source_amplitude = 1.0 + difficulty * rng.uniform(0.5, 5.0)
            source_location_variation = difficulty * rng.uniform(0.1, 0.6)
            boundary_strength = 1.0 + difficulty * rng.uniform(-0.3, 0.5)
            coeff_smoothness = max(0.1, 1.0 - difficulty * rng.uniform(0.2, 0.8))
            coeff_discontinuity = difficulty * rng.uniform(0.0, 0.7)

            parameters = {
                "source_amplitude": round(source_amplitude, 3),
                "source_location_variation": round(source_location_variation, 3),
                "boundary_strength": round(boundary_strength, 3),
                "coeff_smoothness": round(coeff_smoothness, 3),
                "coeff_discontinuity": round(coeff_discontinuity, 3),
            }

            variant = StressVariant(
                variant_id=f"elliptic_proc_{i}",
                source=StressSource.PROCEDURAL,
                parameters=parameters,
                difficulty=difficulty,
                metadata={
                    "physics_justification": (
                        "Varying source strength/location, boundary conditions, and "
                        "coefficient field regularity/discontinuity to stress conservation, "
                        "maximum principle, and solution regularity."
                    )
                },
            )
            variants.append(variant)

        return variants

    # ============================================================
    # Burgers (Hyperbolic) - Deep
    # ============================================================

    def _generate_burgers_stress(self, rng, difficulty):
        variants = []
        num = max(3, int(5 + difficulty * 6))

        for i in range(num):
            shock_strength = 1.0 + difficulty * rng.uniform(0.5, 3.5)
            hf_amplitude = difficulty * rng.uniform(0.01, 0.18)
            rollout_steps = int(50 + difficulty * rng.uniform(50, 250))
            viscosity = max(0.0005, 0.01 * (1 + rng.uniform(-0.4, 0.4)))
            initial_condition_noise = difficulty * rng.uniform(0.0, 0.12)

            parameters = {
                "shock_strength": round(shock_strength, 3),
                "hf_amplitude": round(hf_amplitude, 4),
                "rollout_steps": rollout_steps,
                "viscosity": round(viscosity, 6),
                "initial_condition_noise": round(initial_condition_noise, 4),
            }

            variant = StressVariant(
                variant_id=f"burgers_proc_{i}",
                source=StressSource.PROCEDURAL,
                parameters=parameters,
                difficulty=difficulty,
                metadata={
                    "physics_justification": (
                        "Varying shock strength, high-frequency content, rollout length, "
                        "viscosity, and initial condition noise to stress conservation, "
                        "shock capturing, and long-term stability."
                    )
                },
            )
            variants.append(variant)

        return variants

    # ============================================================
    # Heat (Parabolic) - Deep
    # ============================================================

    def _generate_heat_stress(self, rng, difficulty):
        variants = []
        num = max(3, int(4 + difficulty * 6))

        for i in range(num):
            forcing_amplitude = difficulty * rng.uniform(0.5, 4.0)
            forcing_frequency = 0.5 + difficulty * rng.uniform(0.0, 2.5)
            conductivity_variation = difficulty * rng.uniform(0.1, 0.9)
            rollout_steps = int(40 + difficulty * rng.uniform(40, 180))
            initial_condition_roughness = difficulty * rng.uniform(0.0, 0.3)

            parameters = {
                "forcing_amplitude": round(forcing_amplitude, 3),
                "forcing_frequency": round(forcing_frequency, 2),
                "conductivity_variation": round(conductivity_variation, 3),
                "rollout_steps": rollout_steps,
                "initial_condition_roughness": round(initial_condition_roughness, 3),
            }

            variant = StressVariant(
                variant_id=f"heat_proc_{i}",
                source=StressSource.PROCEDURAL,
                parameters=parameters,
                difficulty=difficulty,
                metadata={
                    "physics_justification": (
                        "Varying time-dependent forcing amplitude/frequency, conductivity "
                        "variation, rollout length, and initial condition roughness to "
                        "stress energy conservation and long-term decay behavior."
                    )
                },
            )
            variants.append(variant)

        return variants

    # ============================================================
    # Navier-Stokes (Incompressible Flow) - Deep
    # ============================================================

    def _generate_ns_stress(self, rng, difficulty):
        variants = []
        num = max(3, int(5 + difficulty * 5))

        for i in range(num):
            reynolds = 40 + difficulty * rng.uniform(15, 90)  # Still laminar
            geometry_scale = 1.0 + difficulty * rng.uniform(-0.25, 0.5)
            boundary_perturbation = difficulty * rng.uniform(0.02, 0.2)
            initial_vorticity_noise = difficulty * rng.uniform(0.0, 0.15)
            forcing_strength = difficulty * rng.uniform(0.0, 0.8)

            parameters = {
                "reynolds": round(reynolds, 1),
                "geometry_scale": round(geometry_scale, 3),
                "boundary_perturbation": round(boundary_perturbation, 3),
                "initial_vorticity_noise": round(initial_vorticity_noise, 3),
                "forcing_strength": round(forcing_strength, 3),
            }

            variant = StressVariant(
                variant_id=f"ns_proc_{i}",
                source=StressSource.PROCEDURAL,
                parameters=parameters,
                difficulty=difficulty,
                metadata={
                    "physics_justification": (
                        "Varying Reynolds number, geometry, boundary conditions, initial "
                        "vorticity, and weak forcing to stress divergence-free condition, "
                        "energy stability, and momentum conservation."
                    )
                },
            )
            variants.append(variant)

        return variants

    # ============================================================
    # Elasticity - Deep
    # ============================================================

    def _generate_elasticity_stress(self, rng, difficulty):
        variants = []
        num = max(3, int(4 + difficulty * 5))

        for i in range(num):
            young_modulus_variation = difficulty * rng.uniform(0.15, 0.8)
            poisson_ratio = 0.22 + difficulty * rng.uniform(-0.08, 0.15)
            boundary_displacement = difficulty * rng.uniform(0.05, 0.35)
            material_anisotropy = difficulty * rng.uniform(0.0, 0.6)
            body_force_strength = difficulty * rng.uniform(0.0, 0.4)

            parameters = {
                "young_modulus_variation": round(young_modulus_variation, 3),
                "poisson_ratio": round(poisson_ratio, 3),
                "boundary_displacement": round(boundary_displacement, 3),
                "material_anisotropy": round(material_anisotropy, 3),
                "body_force_strength": round(body_force_strength, 3),
            }

            variant = StressVariant(
                variant_id=f"elasticity_proc_{i}",
                source=StressSource.PROCEDURAL,
                parameters=parameters,
                difficulty=difficulty,
                metadata={
                    "physics_justification": (
                        "Varying stiffness, Poisson ratio, boundary displacement, material "
                        "anisotropy, and body forces to stress equilibrium, compatibility, "
                        "and boundary satisfaction."
                    )
                },
            )
            variants.append(variant)

        return variants

    # ============================================================
    # Thermo-Elasticity (Multi-physics) - Deep
    # ============================================================

    def _generate_thermo_elasticity_stress(self, rng, difficulty):
        variants = []
        num = max(3, int(5 + difficulty * 5))

        for i in range(num):
            thermal_expansion = 1e-5 * (1 + difficulty * rng.uniform(0.3, 2.0))
            coupling_strength = difficulty * rng.uniform(0.4, 1.4)
            temperature_variation = difficulty * rng.uniform(15, 100)
            mechanical_damping = max(0.0, 0.3 - difficulty * rng.uniform(0.0, 0.4))
            heat_source_from_deformation = difficulty * rng.uniform(0.0, 0.7)

            parameters = {
                "thermal_expansion": round(thermal_expansion, 8),
                "coupling_strength": round(coupling_strength, 3),
                "temperature_variation": round(temperature_variation, 1),
                "mechanical_damping": round(mechanical_damping, 3),
                "heat_source_from_deformation": round(heat_source_from_deformation, 3),
            }

            variant = StressVariant(
                variant_id=f"thermo_elasticity_proc_{i}",
                source=StressSource.PROCEDURAL,
                parameters=parameters,
                difficulty=difficulty,
                metadata={
                    "physics_justification": (
                        "Varying thermal expansion, coupling strength, temperature loading, "
                        "mechanical damping, and deformation-induced heat source to stress "
                        "thermo-mechanical energy exchange and coupled conservation laws."
                    )
                },
            )
            variants.append(variant)

        return variants


# Register for all supported physics classes
for pc in ["elliptic", "poisson", "darcy", "hyperbolic", "parabolic",
           "incompressible", "navier_stokes", "elasticity",
           "multi_physics", "thermo_elasticity"]:
    stress_registry.register(pc, ProceduralStressGenerator())
