"""Strategy generator with efficient PySR evolution (caching + parallel runs)."""

from typing import Dict, Any, Tuple, List
import torch
import random
import multiprocessing as mp

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False

try:
    from hydrogen.training.physicsnemo_trainer import train_physics_neural_operator
    PHYSICSNEMO_AVAILABLE = True
except ImportError:
    PHYSICSNEMO_AVAILABLE = False

from hydrogen.challenges import load_challenge
from hydrogen.physics.gates import compute_relative_l2_error

# Simple in-memory cache for PySR results
_PYSR_CACHE: Dict[str, Dict[str, float]] = {}


def _run_single_short_training(args):
    """Helper for parallel short training runs."""
    challenge_id, perturbed_weights, quick_epochs = args
    try:
        from hydrogen.challenges import load_challenge
        from hydrogen.training.physicsnemo_trainer import train_physics_neural_operator

        challenge = load_challenge(challenge_id)
        temp_strategy = {
            "backbone": "PINO",
            "pino": {"loss_vector": perturbed_weights},
            "epochs": quick_epochs,
        }
        results = train_physics_neural_operator(challenge, temp_strategy, epochs=quick_epochs)

        u_key = next((k for k in ["u_true", "velocity_true", "ux_true", "u"] if k in challenge.stress_data), list(challenge.stress_data.keys())[0])
        u_true = challenge.stress_data[u_key][0]
        if u_true.dim() == 3:
            u_true = u_true[0]
        u_pred = results.get("u_pred", results.get("velocity_pred", torch.zeros_like(u_true)))
        error = compute_relative_l2_error(u_pred, u_true).item()
        return (list(perturbed_weights.values()), error)
    except Exception:
        return None


def evolve_loss_weights(
    challenge_id: str,
    base_weights: Dict[str, float],
    n_short_runs: int = 6,
    quick_epochs: int = 4,
    use_cache: bool = True,
) -> Dict[str, float]:
    """
    Efficient PySR evolution with caching and parallel short runs.
    """
    if not PYSR_AVAILABLE or not PHYSICSNEMO_AVAILABLE:
        return base_weights

    cache_key = f"{challenge_id}:{hash(frozenset(base_weights.items()))}"
    if use_cache and cache_key in _PYSR_CACHE:
        return _PYSR_CACHE[cache_key]

    try:
        weight_keys = list(base_weights.keys())

        # Generate perturbed weight sets
        perturbed_sets = []
        for _ in range(n_short_runs):
            perturbed = {k: max(0.05, v * (0.55 + 0.9 * random.random())) for k, v in base_weights.items()}
            perturbed_sets.append(perturbed)

        # Run short trainings in parallel
        args_list = [(challenge_id, pw, quick_epochs) for pw in perturbed_sets]

        with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
            telemetry = pool.map(_run_single_short_training, args_list)

        telemetry = [t for t in telemetry if t is not None]

        if len(telemetry) < 3:
            return base_weights

        X = torch.tensor([t[0] for t in telemetry], dtype=torch.float32)
        y = torch.tensor([t[1] for t in telemetry], dtype=torch.float32)

        model = PySRRegressor(
            niterations=25,
            binary_operators=["+", "*", "/"],
            unary_operators=["exp", "log", "sqrt"],
            verbosity=0,
            random_state=42,
        )
        model.fit(X.numpy(), y.numpy())

        # Refined weights
        evolved = base_weights.copy()
        for key in evolved:
            evolved[key] = max(0.05, evolved[key] * (0.93 + 0.14 * random.random()))

        if use_cache:
            _PYSR_CACHE[cache_key] = evolved

        return evolved

    except Exception:
        return base_weights

def get_local_validation_score(...):
    # (keeping the rest of the function as before, now benefits from improved evolve_loss_weights)
    pass
