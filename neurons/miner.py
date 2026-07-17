"""Hydrogen Miner Neuron (Improved).

Now includes real strategy generation using symbolic metadata
and optional local validation before submission.
"""

import time
import typing
import bittensor as bt

from hydrogen.protocol import StrategySynapse
from hydrogen.base.miner import BaseMinerNeuron
from hydrogen.miner.strategy_generator import generate_strategy, get_local_validation_score


class Miner(BaseMinerNeuron):
    """
    Hydrogen Miner implementation with intelligent strategy generation.
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        bt.logging.info("Hydrogen Miner initialized with strategy generator.")

    async def forward(self, synapse: StrategySynapse) -> StrategySynapse:
        bt.logging.info(f"Received request for challenge: {synapse.challenge_id}")

        if synapse.strategy is None:
            # Generate a good strategy using symbolic priors
            strategy = generate_strategy(synapse.challenge_id)

            # Optional: Run local validation / dry-run
            estimated_improvement = get_local_validation_score(synapse.challenge_id, strategy)
            if estimated_improvement > 0:
                bt.logging.info(f"Local validation passed. Est. improvement: {estimated_improvement:.4f}")

            synapse.strategy = strategy
            synapse.accepted = True
            synapse.message = "Strategy generated using symbolic metadata"
        else:
            # This is a submission from another miner or validator confirmation
            synapse.accepted = True
            synapse.message = "Strategy submission acknowledged"
            synapse.submission_id = f"sub_{int(time.time())}"

        return synapse

    async def blacklist(self, synapse: StrategySynapse) -> typing.Tuple[bool, str]:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return True, "Missing dendrite or hotkey"

        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
            return True, "Unrecognized hotkey"

        return False, "Hotkey recognized"

    async def priority(self, synapse: StrategySynapse) -> float:
        if synapse.dendrite is None or synapse.dendrite.hotkey is None:
            return 0.0
        try:
            uid = self.metagraph.hotkeys.index(synapse.dendrite.hotkey)
            return float(self.metagraph.S[uid])
        except Exception:
            return 0.0


if __name__ == "__main__":
    with Miner() as miner:
        while True:
            bt.logging.info("Hydrogen Miner running...")
            time.sleep(10)
