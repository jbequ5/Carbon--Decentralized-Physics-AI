"""Hydrogen Miner Neuron (with real local validation).

The miner now generates strategies using symbolic metadata and
runs local physics gate validation before responding.
"""

import time
import typing
import bittensor as bt

from hydrogen.protocol import StrategySynapse
from hydrogen.base.miner import BaseMinerNeuron
from hydrogen.miner.strategy_generator import generate_strategy, get_local_validation_score


class Miner(BaseMinerNeuron):
    """
    Hydrogen Miner with intelligent strategy generation + local validation.
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        bt.logging.info("Hydrogen Miner initialized with local validation.")

    async def forward(self, synapse: StrategySynapse) -> StrategySynapse:
        bt.logging.info(f"Received request for challenge: {synapse.challenge_id}")

        if synapse.strategy is None:
            # Generate strategy using symbolic metadata
            strategy = generate_strategy(synapse.challenge_id)

            # Run real local validation using physics gates
            improvement, hard_pass, gate_details = get_local_validation_score(
                synapse.challenge_id, strategy
            )

            if hard_pass:
                bt.logging.info(
                    f"Local validation PASSED. Est. improvement: {improvement:+.4f}"
                )
                synapse.strategy = strategy
                synapse.accepted = True
                synapse.message = f"Strategy validated locally (improvement ~{improvement:.3f})"
            else:
                bt.logging.warning("Local validation FAILED. Adjusting strategy...")
                # In a real implementation we could mutate the strategy here
                synapse.strategy = strategy
                synapse.accepted = False
                synapse.message = "Local validation failed - strategy may need tuning"

        else:
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
