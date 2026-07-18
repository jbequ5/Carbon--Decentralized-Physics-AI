"""SOTA Strategy Generator for Hydrogen.

Produces rich, validator-complete strategy JSONs.
"""

from typing import Dict, Any, Optional

from hydrogen.challenges import load_challenge
from hydrogen.backbones import list_backbones


def generate_strategy(
    challenge_id: str = "poisson_2d_v1",
    backbone: Optional[str] = None,
    use_curriculum: bool = True,
    use_uq: bool = True,
) -> Dict[str, Any]:
    """
    Generates a rich, SOTA strategy JSON that the validator can fully consume.
    """
    try:
        challenge = load_challenge(challenge_id)
        symbolic = getattr(challenge, "symbolic_metadata", {})
        suggested_weights = symbolic.get("suggested_loss_weights", {
            "pde_residual": 1.0,
            "boundary": 0.8,
            "initial_condition": 0.6,
        })
        chosen_backbone = backbone or getattr(challenge, "get_backbone", lambda: "physicsnemo_fno")()
        resolution = getattr(challenge, "resolution", [128, 128])
    except Exception:
        suggested_weights = {
            "pde_residual": 1.0,
            "boundary": 0.8,
            "initial_condition": 0.6,
        }
        chosen_backbone = backbone or "physicsnemo_fno"
        resolution = [128, 128]

    strategy = {
        "backbone": chosen_backbone,
        "resolution": list(resolution),
        "pino": {
            "loss_vector": suggested_weights,
            "physics_loss_type": "pde_residual",
            "boundary_handling": "ghost_cells",
            "initial_condition_weight": suggested_weights.get("initial_condition", 0.6),
        },
        "optimizer": "AdamW",
        "learning_rate": 0.0008,
        "weight_decay": 1e-4,
        "epochs": 100,
        "batch_size": 8,
        "curriculum_learning": {
            "enabled": use_curriculum,
            "start_resolution": [64, 64],
            "end_resolution": list(resolution),
            "ramp_epochs": 35,
            "difficulty_schedule": "linear",
        },
        "uq_config": {
            "enabled": use_uq,
            "method": "deep_ensemble",
            "num_members": 4,
            "calibration_target": 0.92,
            "dropout_rate": 0.05,
        },
        "data_split": {
            "train": "benchmark_train",
            "stress_holdout": "procedural_hidden",
            "benchmark": "benchmark_test",
        },
        "auto_loss_weights": True,
        "physics_gate_strictness": 0.9,
    }

    return strategy


def list_available_backbones():
    return list_backbones()
