"""Evaluation Plan for Hydrogen Validator.

Defines exactly what data and stress conditions to use
for each (challenge_id, backbone) combination.

The team controls this logic. Validators just execute it.
"""

from typing import Dict, Any

from hydrogen.data.benchmark_loader import get_benchmark_loader
from hydrogen.physics.stress import generate_stress_conditions


def get_evaluation_plan(
    challenge_id: str,
    backbone: str,
    difficulty: float = 1.0,
    recent_performance: float = 0.5,
) -> Dict[str, Any]:
    """
    Returns the full evaluation plan for a given challenge + backbone.

    This function is controlled by the team.
    It decides:
    - What data to train on
    - What stress conditions to use
    - What to benchmark against
    """

    # 1. Training data (miners can access this)
    train_loader = get_benchmark_loader(challenge_id, split="train")

    # 2. Stress test conditions (hidden + procedural)
    # Difficulty can be adaptive based on recent miner performance
    adaptive_difficulty = max(0.3, min(1.5, difficulty * (0.7 + recent_performance)))
    stress_conditions = generate_stress_conditions(
        challenge_id=challenge_id,
        difficulty=adaptive_difficulty
    )

    # 3. Benchmark / final evaluation data
    benchmark_loader = get_benchmark_loader(challenge_id, split="test")

    return {
        "challenge_id": challenge_id,
        "backbone": backbone,
        "train_loader": train_loader,
        "stress_conditions": stress_conditions,
        "benchmark_loader": benchmark_loader,
        "adaptive_difficulty": adaptive_difficulty,
    }
