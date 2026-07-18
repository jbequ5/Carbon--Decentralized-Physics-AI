"""Hydrogen Validator with full multi-backbone awareness."""

import time
import numpy as np
import torch
import bittensor as bt

from hydrogen.base.validator import BaseValidatorNeuron
from hydrogen.protocol import StrategySynapse
from hydrogen.challenges import load_challenge
from hydrogen.physics.stress import run_stress_test
from hydrogen.training.physicsnemo_trainer import train_physics_neural_operator


class Validator(BaseValidatorNeuron):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.scores = {}
        self.moving_average_alpha = 0.15
        self.current_challenge_index = 0
        self.challenge_ids = [
            "poisson_2d_v1",
            "darcy_2d_v1",
            "burgers_v1",
            "heat_v1",
            "elasticity_2d_v1",
            "ns_2d_laminar_v1",
        ]
        self.use_benchmark = True
        bt.logging.info("Hydrogen Validator with multi-backbone support active.")

    async def forward(self):
        bt.logging.info("Starting validation round...")

        challenge_id = self.challenge_ids[self.current_challenge_index]
        self.current_challenge_index = (self.current_challenge_index + 1) % len(self.challenge_ids)

        axons = [a for a in self.metagraph.axons if a.hotkey != self.wallet.hotkey.ss58_address]
        if not axons:
            return

        synapse = StrategySynapse(challenge_id=challenge_id)
        responses = await self.dendrite(axons=axons, synapse=synapse, timeout=30)

        round_scores = {}
        improvements = []

        for response in responses:
            if not response or not getattr(response, "accepted", False):
                continue
            strategy = getattr(response, "strategy", None)
            if not strategy:
                continue

            validation = self.validate_submission(challenge_id, strategy)
            hotkey = response.dendrite.hotkey if response.dendrite else None

            if hotkey and validation.get("hard_pass"):
                score = validation["score"]
                improvement = validation.get("improvement", 0.0)
                round_scores[hotkey] = score
                improvements.append((hotkey, improvement))

                stress_result = validation.get("stress_result", {})
                final_stress_score = stress_result.get("final_stress_score", score)

                # Log backbone being used
                backbone = strategy.get("backbone", "unknown")
                bt.logging.info(f"{hotkey[:8]} validated on {challenge_id} using backbone={backbone}")

        self._update_scores(round_scores, improvements)

        if self.scores:
            self._set_weights()

    def _update_scores(self, round_scores: dict, improvements: list):
        for hotkey, score in round_scores.items():
            if hotkey in self.scores:
                self.scores[hotkey] = (1 - self.moving_average_alpha) * self.scores[hotkey] + self.moving_average_alpha * score
            else:
                self.scores[hotkey] = score

        if improvements:
            sorted_improvements = sorted(improvements, key=lambda x: x[1], reverse=True)
            top_k = sorted_improvements[:4]
            for hotkey, improvement in top_k:
                if improvement > 0 and hotkey in self.scores:
                    self.scores[hotkey] *= 1.15

    def _set_weights(self):
        try:
            hotkeys = []
            weights = []
            for hotkey, score in self.scores.items():
                if hotkey in self.metagraph.hotkeys:
                    uid = self.metagraph.hotkeys.index(hotkey)
                    hotkeys.append(uid)
                    weights.append(max(0.01, score))

            if not hotkeys:
                return

            weights = np.array(weights, dtype=np.float32)
            weights = weights / weights.sum()

            self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=hotkeys,
                weights=weights,
            )

        except Exception as e:
            bt.logging.error(f"Weight setting failed: {e}")

    def validate_submission(self, challenge_id: str, strategy: dict):
        challenge = load_challenge(challenge_id, use_benchmark=self.use_benchmark)
        baseline_error = challenge.baseline_error

        # Train using the backbone specified in the strategy
        results = train_physics_neural_operator(challenge, strategy, epochs=6)

        backbone = strategy.get("backbone", "unknown")

        if "ns_2d" in challenge_id or "navier" in challenge_id:
            pde_type = "navier_stokes"
        elif "burgers" in challenge_id:
            pde_type = "burgers"
        elif "darcy" in challenge_id:
            pde_type = "darcy"
        elif "heat" in challenge_id:
            pde_type = "heat"
        elif "elasticity" in challenge_id:
            pde_type = "elasticity"
        else:
            pde_type = "poisson"

        u_key = next(
            (k for k in ["u_true", "velocity_true", "ux_true", "u"] if k in challenge.stress_data),
            list(challenge.stress_data.keys())[0]
        )
        u_true = challenge.stress_data[u_key][0]
        if u_true.dim() == 3:
            u_true = u_true[0]

        u_pred = results.get("u_pred", results.get("velocity_pred", torch.zeros_like(u_true)))
        public_error = compute_relative_l2_error(u_pred, u_true)
        public_improvement = float(torch.log(torch.tensor(baseline_error)) - torch.log(torch.tensor(public_error)))

        stress_result = run_stress_test(
            challenge_id=challenge_id,
            results=results,
            u_pred=u_pred,
            u_true=u_true,
            pde_type=pde_type,
            difficulty=1.0,
        )

        if not stress_result["hard_pass"]:
            return {
                "score": 0.0,
                "improvement": 0.0,
                "hard_pass": False,
                "stress_result": stress_result,
            }

        stress_score = stress_result.get("final_stress_score", 0.5)

        stress_weight = 0.35 + min(0.25, public_improvement * 0.8)
        stress_weight = min(0.65, max(0.35, stress_weight))
        public_weight = 1.0 - stress_weight

        final_score = (public_improvement * public_weight) + (stress_score * stress_weight)

        return {
            "score": max(0.0, final_score),
            "improvement": public_improvement,
            "hard_pass": True,
            "stress_result": stress_result,
            "data_source": getattr(challenge, "data_source", "unknown"),
            "backbone_used": backbone,
        }


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Hydrogen Validator running...")
            time.sleep(30)
