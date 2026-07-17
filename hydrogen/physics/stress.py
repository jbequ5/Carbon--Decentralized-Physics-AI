"""Stress Test Engine with UQ Calibration Stress.

Now includes explicit UQ calibration checking on hidden stress sets.
We learn from uncertainty quality (good calibration can be rewarded).
"""

import hashlib
import random
from typing import Dict, Any, Tuple

import numpy as np
import torch


from .gates import evaluate_all_gates, compute_relative_l2_error


def _derive_seed(challenge_id: str, salt: str = "hydrogen_stress") -> int:
    combined = f"{challenge_id}:{salt}".encode()
    return int(hashlib.sha256(combined).hexdigest(), 16) % (2**32)


def generate_stress_conditions(
    challenge_id: str,
    backbone: str = "PINO",
    difficulty: float = 1.0,
    salt: str = "hydrogen_stress_v1",
) -> Dict[str, Any]:
    seed = _derive_seed(challenge_id, salt)
    rng = random.Random(seed)

    conditions = {
        "challenge_id": challenge_id,
        "backbone": backbone,
        "difficulty": difficulty,
        "seed": seed,
    }

    conditions["time_horizon_multiplier"] = 1.5 + difficulty * rng.uniform(0.5, 2.5)
    conditions["parameter_perturbation"] = rng.uniform(0.1, 0.7) * difficulty
    conditions["noise_level"] = rng.uniform(0.005, 0.05) * difficulty
    conditions["resolution_scale"] = rng.choice([1.0, 1.5, 2.0])

    if "ns_2d" in challenge_id or "navier" in challenge_id:
        conditions["reynolds_multiplier"] = 1.3 + rng.uniform(0.4, 2.2) * difficulty
        conditions["forcing_perturbation"] = rng.uniform(0.05, 0.3)
        conditions["divergence_tolerance"] = 1e-3 * (1 + difficulty * 0.5)
    elif "burgers" in challenge_id:
        conditions["viscosity_multiplier"] = rng.uniform(0.2, 3.0) * difficulty
    elif "darcy" in challenge_id:
        conditions["permeability_contrast"] = 80 + rng.uniform(30, 500) * difficulty
    elif "heat" in challenge_id:
        conditions["diffusivity_multiplier"] = rng.uniform(0.3, 2.5) * difficulty

    return conditions


def run_hard_gates(
    results: Dict[str, torch.Tensor],
    pde_type: str,
    conditions: Dict[str, Any],
) -> Tuple[bool, Dict[str, Any]]:
    hard_pass, gate_details = evaluate_all_gates(results, pde_type=pde_type)

    if not hard_pass:
        return False, gate_details

    difficulty = conditions.get("difficulty", 1.0)

    if pde_type == "navier_stokes":
        if "velocity" in results or "u_pred" in results:
            div_norm = results.get("divergence_norm", torch.tensor(0.0))
            if div_norm > conditions.get("divergence_tolerance", 1e-3):
                gate_details["divergence_free"] = False
                return False, gate_details

        if "energy_trajectory" in results:
            energy = results["energy_trajectory"]
            if len(energy) > 5:
                drift = abs(energy[-1] - energy[0]) / (abs(energy[0]) + 1e-8)
                if drift > 0.08 * difficulty:
                    gate_details["ns_long_term_stability"] = False
                    return False, gate_details

    if pde_type == "burgers":
        if "energy_trajectory" in results:
            energy = results["energy_trajectory"]
            if len(energy) > 5:
                if torch.any(energy < -0.1):
                    gate_details["burgers_negative_energy"] = False
                    return False, gate_details

    if "energy_trajectory" in results:
        energy = results["energy_trajectory"]
        if len(energy) > 8:
            drift = abs(energy[-1] - energy[0]) / (abs(energy[0]) + 1e-8)
            if drift > 0.12 * difficulty:
                gate_details["stress_rollout_stability"] = False
                return False, gate_details

    return True, gate_details


def compute_stress_metrics(
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    conditions: Dict[str, Any],
    pde_type: str,
    results: Dict[str, torch.Tensor] = None,
) -> Dict[str, float]:
    metrics = {}

    base_error = compute_relative_l2_error(u_pred, u_true).item()
    metrics["stress_l2_error"] = base_error

    param_pert = conditions.get("parameter_perturbation", 0.2)
    metrics["generalization_gap"] = base_error * (1 + param_pert)
    metrics["ood_sensitivity"] = base_error * conditions.get("noise_level", 0.02) * 12

    difficulty = conditions.get("difficulty", 1.0)

    if pde_type == "navier_stokes":
        metrics["divergence_violation"] = base_error * 0.8 * difficulty
        metrics["long_term_dissipation"] = max(0.0, 1.0 - base_error * 4 * difficulty)
    elif pde_type == "burgers":
        metrics["shock_capturing_quality"] = max(0.0, 1.0 - base_error * 6)
        metrics["viscosity_robustness"] = max(0.0, 1.0 - base_error * 3.5 * difficulty)
    elif pde_type == "darcy":
        metrics["heterogeneous_robustness"] = max(0.0, 1.0 - base_error * 2.5)
    elif pde_type == "heat":
        metrics["diffusion_robustness"] = max(0.0, 1.0 - base_error * 3)

    # === UQ Calibration Stress ===
    if results is not None:
        uncertainty = results.get("uncertainty", results.get("std", results.get("ensemble_std", None)))
        if uncertainty is not None and isinstance(uncertainty, torch.Tensor):
            # Simple calibration proxy: correlation between error and uncertainty
            error_map = torch.abs(u_pred - u_true)
            if error_map.numel() > 10 and uncertainty.numel() > 10:
                # Flatten and compute correlation
                err_flat = error_map.flatten().detach().cpu().numpy()
                unc_flat = uncertainty.flatten().detach().cpu().numpy()
                if np.std(unc_flat) > 1e-6:
                    corr = np.corrcoef(err_flat, unc_flat)[0, 1]
                    metrics["uq_error_correlation"] = float(corr)
                    # Good calibration → positive correlation
                    metrics["uq_calibration_score"] = max(0.0, min(1.0, (corr + 1) / 2))
                else:
                    metrics["uq_calibration_score"] = 0.5

    return metrics


def run_stress_test(
    challenge_id: str,
    results: Dict[str, torch.Tensor],
    u_pred: torch.Tensor,
    u_true: torch.Tensor,
    pde_type: str,
    difficulty: float = 1.0,
    salt: str = "hydrogen_stress_v1",
) -> Dict[str, Any]:
    conditions = generate_stress_conditions(challenge_id, difficulty=difficulty, salt=salt)

    hard_pass, gate_details = run_hard_gates(results, pde_type, conditions)

    if not hard_pass:
        return {
            "hard_pass": False,
            "gate_details": gate_details,
            "stress_metrics": {},
            "final_stress_score": 0.0,
            "conditions": conditions,
        }

    stress_metrics = compute_stress_metrics(u_pred, u_true, conditions, pde_type, results=results)

    base_error = stress_metrics.get("stress_l2_error", 0.6)
    uq_calib = stress_metrics.get("uq_calibration_score", 0.5)

    # Reward good UQ calibration slightly
    final_stress_score = max(0.0, 1.0 - base_error * 2.6 + (uq_calib - 0.5) * 0.3)

    return {
        "hard_pass": True,
        "gate_details": gate_details,
        "stress_metrics": stress_metrics,
        "final_stress_score": final_stress_score,
        "conditions": conditions,
    }
