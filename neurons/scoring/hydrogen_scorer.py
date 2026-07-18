"""HydrogenScorer (Improved)

Improved version with better model loading and diagnostics.
"""

from typing import Dict, Any

import bittensor as bt

from hydrogen.backbones import get_model
from hydrogen.evaluation.plan import get_evaluation_plan
from hydrogen.training.trainer import train_model
from hydrogen.physics.stress import run_stress_test


class HydrogenScorer:
    def __init__(self, config: bt.config):
        self.config = config

    def score_strategy(self, uid: int, hotkey: str, strategy: dict) -> float:
        challenge_id = strategy.get("challenge_id") or self._get_default_challenge(uid)
        backbone = strategy.get("backbone", "physicsnemo_fno")

        plan = get_evaluation_plan(challenge_id, backbone)

        eval_result = self._evaluate(strategy, plan, backbone)

        return eval_result.get("final_score", 0.0)

    def _evaluate(self, strategy: dict, plan: dict, backbone: str) -> Dict[str, Any]:
        try:
            model = get_model(backbone=backbone)

            train_result = train_model(
                model=model,
                train_loader=plan.get("train_loader"),
                strategy=strategy,
                epochs=strategy.get("epochs", 50),
            )

            stress_result = run_stress_test(
                challenge_id=plan.get("challenge_id", "unknown"),
                results=train_result,
                u_pred=train_result.get("u_pred"),
                u_true=plan.get("u_true"),
                pde_type=plan.get("pde_type", "generic"),
            )

            final_score = stress_result.get("final_stress_score", 0.0)

            return {
                "final_score": final_score,
                "stress_result": stress_result,
                "train_result": train_result,
            }

        except Exception as e:
            bt.logging.error(f"Evaluation failed for strategy: {e}")
            return {"final_score": 0.0}

    def _get_default_challenge(self, uid: int) -> str:
        challenges = getattr(self.config, "active_challenges", ["poisson_2d_v1"])
        return challenges[uid % len(challenges)]
