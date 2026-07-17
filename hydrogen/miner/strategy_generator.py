"""Strategy generation logic for Hydrogen miners.

Uses symbolic metadata from challenges to generate good starting strategies.
This is where domain intelligence lives for the miner.
"""

from typing import Dict, Any

from hydrogen.challenges.poisson_2d import load_challenge


def generate_strategy(challenge_id: str = "poisson_2d_v1") -> Dict[str, Any]:
    """
    Generate a strategy dict for a given challenge.

    Currently uses symbolic metadata to set sensible loss weights.
    TODO: Add more sophisticated logic (curriculum, UQ method, etc.)
    """
    try:
        challenge = load_challenge(challenge_id)
        symbolic = challenge.symbolic_metadata
        suggested_weights = symbolic.get("suggested_loss_weights", {})
    except Exception:
        suggested_weights = {"pde_residual": 1.0, "boundary": 0.8}

    strategy = {
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

    return strategy


def get_local_validation_score(challenge_id: str, strategy: dict) -> float:
    """
    Optional: Run a quick local dry-run validation before submitting.
    Returns estimated improvement or -1 if it fails.
    """
    # TODO: Wire up to hydrogen.physics.gates + a lightweight forward pass
    # For now return a placeholder
    return 0.03  # Dummy positive improvement
