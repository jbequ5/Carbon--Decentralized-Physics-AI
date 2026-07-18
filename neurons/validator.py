"""Hydrogen Validator with robust 75/25 emission weight logic.

Includes better edge-case handling and clear logging for Top-2 stipend.
"""

import time
import numpy as np
import torch
import bittensor as bt

from hydrogen.base.validator import BaseValidatorNeuron
from hydrogen.protocol import StrategySynapse
from hydrogen.challenges import load_challenge
from hydrogen.physics.stress import run_stress_test
from hydrogen.emission.mechanics import (
    update_leaderboard,
    get_yuma_weights,
    get_or_create_state,
)


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
        bt.logging.info("Hydrogen Validator with robust Top-2 stipend logic.")

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

                was_breakthrough, msg = update_leaderboard(
                    challenge_id=challenge_id,
                    hotkey=hotkey,
                    new_score=final_stress_score,
                )

                if was_breakthrough:
                    bt.logging.warning(f"🚀 BREAKTHROUGH on {challenge_id}! {msg}")

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
        """
        Applies the decaying Top-2 stipend on-chain with good robustness and logging.
        """
        try:
            weight_dict = {}

            bt.logging.info("=== Top-2 Stipend Weight Calculation ===")

            for challenge_id in self.challenge_ids:
                state = get_or_create_state(challenge_id)

                # Log current leaders for visibility
                leader = state.leader_hotkey[:8] if state.leader_hotkey else "None"
                second = state.second_hotkey[:8] if state.second_hotkey else "None"
                bt.logging.info(
                    f"{challenge_id}: #1={leader}, #2={second}, "
                    f"best_score={state.current_best_score:.4f}, "
                    f"epochs_no_improve={state.epochs_without_improvement}"
                )

                challenge_weights = get_yuma_weights(
                    challenge_id=challenge_id,
                    total_stipend_pool=1.0,
                )

                for hotkey, w in challenge_weights.items():
                    weight_dict[hotkey] = weight_dict.get(hotkey, 0.0) + w

            if not weight_dict:
                bt.logging.warning("No Top-2 leaders found. Skipping weight set.")
                return

            # Build uids and weights
            hotkeys = []
            weights_list = []
            for hotkey, w in weight_dict.items():
                if hotkey in self.metagraph.hotkeys:
                    uid = self.metagraph.hotkeys.index(hotkey)
                    hotkeys.append(uid)
                    weights_list.append(float(w))

            if not hotkeys:
                bt.logging.warning("No valid hotkeys in Top-2. Skipping weight set.")
                return

            weights = np.array(weights_list, dtype=np.float32)

            # Normalize
            total_weight = weights.sum()
            if total_weight > 1e-8:
                weights = weights / total_weight
            else:
                bt.logging.warning("Total weight too small. Skipping weight set.")
                return

            # Final safety cap
            weights = np.clip(weights, 0.001, 1.0)
            weights = weights / weights.sum()

            bt.logging.info(f"Setting weights for {len(hotkeys)} miners (Top-2 stipend).")

            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=hotkeys,
                weights=weights,
            )

        except Exception as e:
            bt.logging.error(f"Weight setting failed: {e}")

    def validate_submission(self, challenge_id: str, strategy: dict):
        from hydrogen.physics.gates import evaluate_all_gates, compute_relative_l2_error
        from hydrogen.training.physicsnemo_trainer import train_physics_neural_operator

        challenge = load_challenge(challenge_id, use_benchmark=self.use_benchmark)
        baseline_error = challenge.baseline_error

        results = train_physics_neural_operator(challenge, strategy, epochs=6)

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
        }


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Hydrogen Validator running...")
            time.sleep(30)
