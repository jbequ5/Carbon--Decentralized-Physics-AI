"""Hydrogen Validator with weight setting and basic emissions logic.

After each validation round, the validator aggregates scores and sets
weights on-chain based on miner performance.
"""

import time
import numpy as np
import bittensor as bt

from hydrogen.base.validator import BaseValidatorNeuron
from hydrogen.protocol import StrategySynapse


class Validator(BaseValidatorNeuron):
    """
    Hydrogen Validator with scoring + weight setting.
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        self.scores = {}  # hotkey -> moving average score
        self.moving_average_alpha = 0.1  # How much new scores affect the average
        bt.logging.info("Hydrogen Validator initialized with weight setting.")

    async def forward(self):
        bt.logging.info("Starting validation round...")

        challenge_id = "poisson_2d_v1"

        axons = [
            axon for axon in self.metagraph.axons
            if axon.hotkey != self.wallet.hotkey.ss58_address
        ]

        if not axons:
            bt.logging.warning("No miners to query.")
            return

        synapse = StrategySynapse(challenge_id=challenge_id)

        responses = await self.dendrite(
            axons=axons,
            synapse=synapse,
            timeout=30,
        )

        round_scores = {}  # hotkey -> score this round

        for response in responses:
            if response is None or not getattr(response, "accepted", False):
                continue

            strategy = getattr(response, "strategy", None)
            if strategy is None:
                continue

            validation = self.validate_submission(challenge_id, strategy)
            hotkey = response.dendrite.hotkey if response.dendrite else None

            if hotkey:
                score = validation.get("score", 0.0)
                round_scores[hotkey] = score

                bt.logging.info(
                    f"Miner {hotkey[:8]}: score={score:.4f}, hard_pass={validation.get('hard_pass')}"
                )

        # Update moving averages
        self._update_scores(round_scores)

        # Set weights periodically
        if len(self.scores) > 0:
            self._set_weights()

    def _update_scores(self, round_scores: dict):
        """Update moving average scores for miners."""
        for hotkey, new_score in round_scores.items():
            if hotkey in self.scores:
                # Moving average
                old = self.scores[hotkey]
                self.scores[hotkey] = (1 - self.moving_average_alpha) * old + self.moving_average_alpha * new_score
            else:
                self.scores[hotkey] = new_score

    def _set_weights(self):
        """Set weights on-chain based on current scores."""
        try:
            # Get UIDs for hotkeys we have scores for
            hotkeys = list(self.scores.keys())
            uids = []
            weights = []

            for hotkey in hotkeys:
                if hotkey in self.metagraph.hotkeys:
                    uid = self.metagraph.hotkeys.index(hotkey)
                    uids.append(uid)
                    weights.append(max(0.0, self.scores[hotkey]))

            if not uids:
                return

            # Normalize weights
            weights = np.array(weights, dtype=np.float32)
            if weights.sum() > 0:
                weights = weights / weights.sum()

            # Set weights
            result = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=self.config.netuid,
                uids=uids,
                weights=weights,
            )

            bt.logging.info(f"Set weights on-chain. Success: {result}")

        except Exception as e:
            bt.logging.error(f"Failed to set weights: {e}")

    def validate_submission(self, challenge_id: str, strategy: dict):
        from hydrogen.challenges.poisson_2d import load_challenge
        from hydrogen.physics.gates import evaluate_all_gates, compute_relative_l2_error
        from hydrogen.training.physicsnemo_trainer import train_physics_neural_operator

        challenge = load_challenge(challenge_id)
        baseline_error = challenge.baseline_error

        results = train_physics_neural_operator(challenge, strategy, epochs=6)
        hard_pass, gate_details = evaluate_all_gates(results, pde_type="poisson")

        if not hard_pass:
            return {"score": 0.0, "hard_pass": False, "gate_details": gate_details}

        u_pred = results["u_pred"]
        u_true = challenge.stress_data["u_true"][0]
        submission_error = compute_relative_l2_error(u_pred, u_true)
        improvement = float(torch.log(torch.tensor(baseline_error)) - torch.log(torch.tensor(submission_error)))

        return {
            "score": max(0.0, improvement),
            "improvement": improvement,
            "hard_pass": True,
            "gate_details": gate_details,
        }


if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info("Hydrogen Validator running...")
            time.sleep(30)
