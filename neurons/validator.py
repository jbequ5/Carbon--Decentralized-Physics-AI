"""Hydrogen Validator - Robust version with better error handling and sampling."""

import time
import random
import numpy as np
import bittensor as bt

from hydrogen.base.validator import BaseValidatorNeuron
from hydrogen.protocol import StrategySynapse
from hydrogen.challenges import load_challenge


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
        self.sample_size = 20  # Number of miners to query per round
        bt.logging.info("Hydrogen Validator initialized (robust mode).")

    async def forward(self):
        bt.logging.info("Starting validation round...")

        challenge_id = self.challenge_ids[self.current_challenge_index]
        self.current_challenge_index = (self.current_challenge_index + 1) % len(self.challenge_ids)

        # Robust miner sampling
        all_axons = [a for a in self.metagraph.axons if a.hotkey != self.wallet.hotkey.ss58_address]
        if not all_axons:
            bt.logging.warning("No miners available.")
            return

        sample_size = min(self.sample_size, len(all_axons))
        axons = random.sample(all_axons, sample_size)

        bt.logging.info(f"Querying {len(axons)} miners for challenge {challenge_id}...")

        synapse = StrategySynapse(challenge_id=challenge_id)

        try:
            responses = await self.dendrite(
                axons=axons,
                synapse=synapse,
                timeout=25,  # Slightly reduced timeout for robustness
            )
        except Exception as e:
            bt.logging.error(f"Dendrite query failed: {e}")
            return

        round_scores = {}
        improvements = []

        for response in responses:
            try:
                if response is None:
                    continue
                if not getattr(response, "accepted", False):
                    continue

                strategy = getattr(response, "strategy", None)
                if strategy is None:
                    continue

                validation = self.validate_submission(challenge_id, strategy)
                hotkey = response.dendrite.hotkey if response.dendrite else None

                if hotkey and validation.get("hard_pass"):
                    score = validation["score"]
                    improvement = validation.get("improvement", 0.0)
                    round_scores[hotkey] = score
                    improvements.append((hotkey, improvement))

                    data_src = validation.get("data_source", "unknown")
                    bt.logging.info(f"{hotkey[:8]} on {challenge_id} ({data_src}): score={score:.4f}")

            except Exception as e:
                bt.logging.warning(f"Error processing response: {e}")
                continue

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

            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=hotkeys,
                weights=weights,
            )
            bt.logging.info(f"Weights set successfully.")

        except Exception as e:
            bt.logging.error(f"Weight setting failed: {e}")

    def validate_submission(self, challenge_id: str, strategy: dict):
        from hydrogen.physics.gates import evaluate_all_gates, compute_relative_l2_error
        from hydrogen.training.physicsnemo_trainer import train_physics_neural_operator

        try:
            challenge = load_challenge(challenge_id, use_benchmark=self.use_benchmark)
        except Exception as e:
            bt.logging.error(f"Failed to load challenge {challenge_id}: {e}")
            return {"score": 0.0, "improvement": 0.0, "hard_pass": False}

        baseline_error = challenge.baseline_error

        try:
            results = train_physics_neural_operator(challenge, strategy, epochs=6)
        except Exception as e:
            bt.logging.warning(f"Training failed for {challenge_id}: {e}")
            return {"score": 0.0, "improvement": 0.0, "hard_pass": False}

        # pde_type routing
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

        try:
            hard_pass, gate_details = evaluate_all_gates(results, pde_type=pde_type)
        except Exception as e:
            bt.logging.warning(f"Gate evaluation failed: {e}")
            return {"score": 0.0, "improvement": 0.0, "hard_pass": False}

        if not hard_pass:
            return {"score": 0.0, "improvement": 0.0, "hard_pass": False}

        # Flexible field access
        try:
            u_key = next(
                (k for k in ["u_true", "velocity_true", "ux_true", "u"] if k in challenge.stress_data),
                list(challenge.stress_data.keys())[0]
            )
            u_true = challenge.stress_data[u_key][0]
            u_pred = results.get("u_pred", results.get("velocity_pred", torch.zeros_like(u_true)))

            if u_true.dim() == 3:
                u_true = u_true[0]

            submission_error = compute_relative_l2_error(u_pred, u_true)
            improvement = float(torch.log(torch.tensor(baseline_error)) - torch.log(torch.tensor(submission_error)))

            return {
                "score": max(0.0, improvement),
                "improvement": improvement,
                "hard_pass": True,
                "data_source": getattr(challenge, "data_source", "unknown"),
            }
        except Exception as e:
            bt.logging.warning(f"Error computing improvement: {e}")
            return {"score": 0.0, "improvement": 0.0, "hard_pass": False}


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Hydrogen Validator running...")
            time.sleep(30)
