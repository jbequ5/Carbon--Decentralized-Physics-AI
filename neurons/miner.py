"""Hydrogen Miner with sophisticated local validation.

Miner can now optionally run short real training during local validation
for higher quality strategy vetting before submission.
"""

import time
import typing
import bittensor as bt

from hydrogen.protocol import StrategySynapse
from hydrogen.base.miner import BaseMinerNeuron
from hydrogen.miner.strategy_generator import generate_strategy, get_local_validation_score


class Miner(BaseMinerNeuron):
    """
    Hydrogen Miner with symbolic strategy generation + real local validation.
    """

    def __init__(self, config=None):
        super().__init__(config=config)
        bt.logging.info("Hydrogen Miner initialized (sophisticated local validation enabled).")

    async def forward(self, synapse: StrategySynapse) -> StrategySynapse:
        bt.logging.info(f"Received request for challenge: {synapse.challenge_id}")

        if synapse.strategy is None:
            strategy = generate_strategy(synapse.challenge_id)

            # Sophisticated local validation (can use real short training)
            improvement, hard_pass, gate_details = get_local_validation_score(
                synapse.challenge_id,
                strategy,
                use_real_training=False,   # Set to True if you want short real training
                quick_epochs=4,
            )

            if hard_pass:
                bt.logging.info(f"Local validation PASSED. Est. improvement: {improvement:+.4f}")
                synapse.strategy = strategy
                synapse.accepted = True
                synapse.message = f"Validated locally (est. improvement {improvement:.3f})"
            else:
                bt.logging.warning("Local validation FAILED gates check.")
                synapse.strategy = strategy
                synapse.accepted = False
                synapse.message = "Local validation failed - consider adjusting strategy"

        else:
            synapse.accepted = True
            synapse.message = "Strategy submission received"
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
