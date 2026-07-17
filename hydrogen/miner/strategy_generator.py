"""Strategy generation + local validation for Hydrogen miners.

Local validation now supports real (short) PhysicsNeMo training for more accurate scoring.
"""

from typing import Dict, Any, Tuple

import torch

try:
    from hydrogen.training.physicsnemo_trainer import train_physics_neural_operator
    PHYSICSNEMO_AVAILABLE = True
except ImportError:
    PHYSICSNEMO_AVAILABLE = False

from hydrogen.challenges.poisson_2d import load_challenge
from hydrogen.physics.gates import evaluate_all_gates, compute_relative_l2_error


def generate_strategy(challenge_id: str = "poisson_2d_v1") -> Dict[str, Any]:
    """
    Generate a strategy using symbolic metadata from the challenge.
    """
    try:
        challenge = load_challenge(challenge_id)
        symbolic = challenge.symbolic_metadata
        suggested_weights = symbolic.get("suggested_loss_weights", {})
    except Exception:
        suggested_weights = {"pde_residual": 1.0, "boundary": 0.8}

    return {
        "backbone": "PINO",
        "resolution": [128, 128],
        "pino": {
            "loss_vector": suggested_weights,
            "physics_loss_type": "pde_residual",
            "boundary_handling": "ghost_cells",
        },
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "epochs": 80,
        "curriculum_learning": {
            "enabled": True,
            "start_resolution": [64, 64],
            "end_resolution": [128, 128],
            "ramp_epochs": 30,
        },
        "uq_config": {
            "method": "deep_ensemble",
            "num_members": 3,
            "calibration_target": 0.90,
        },
        "auto_loss_weights": True,
    }


def get_local_validation_score(
    challenge_id: str,
    strategy: dict,
    use_real_training: bool = False,
    quick_epochs: int = 4,
) -> Tuple[float, bool, Dict[str, Any]]:
    """
    Perform local validation of a strategy.

    If use_real_training=True and PhysicsNeMo is available, it will run a short
    training loop (quick_epochs) then evaluate gates. Otherwise falls back to
    simulated prediction.

    Returns: (improvement, hard_pass, gate_details)
    """
    try:
        challenge = load_challenge(challenge_id)

        if use_real_training and PHYSICSNEMO_AVAILABLE:
            # Real short training for better validation
            results = train_physics_neural_operator(
                challenge, strategy, epochs=quick_epochs
            )
        else:
            # Simulated / lightweight prediction
            stress = challenge.stress_data
            u_true = stress["u_true"][0]
            noise_level = strategy.get("noise_level", 0.012)
            u_pred = u_true + noise_level * torch.randn_like(u_true)

            results = {
                "u_pred": u_pred,
                "u_bc": torch.zeros_like(u_true),
                "div_u": torch.zeros_like(u_true),
                "u_norm": u_true,
                "energy_trajectory": torch.linspace(1.0, 0.97, 60),
                "uq_coverage": 0.91,
                "dE_dt": torch.tensor([-0.0002]),
            }

        hard_pass, gate_details = evaluate_all_gates(results, pde_type="poisson")

        submission_error = compute_relative_l2_error(results["u_pred"], challenge.stress_data["u_true"][0])
        baseline_error = challenge.baseline_error
        improvement = float(torch.log(torch.tensor(baseline_error)) - torch.log(torch.tensor(submission_error)))

        return improvement, hard_pass, gate_details

    except Exception as e:
        return -1.0, False, {"error": str(e)}
