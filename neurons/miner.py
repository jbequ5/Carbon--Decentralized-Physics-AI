"""Hydrogen Miner (Bittensor neuron skeleton - Phase 0 MVP).

Lightweight - no GPU. Submit strategy JSONs.

TODOs: CLI for challenges/baseline/priors, local validation helper, submission, reward tracking.
"""

import argparse
import bittensor as bt


def main():
    parser = argparse.ArgumentParser(description="Hydrogen Miner")
    parser.add_argument("--netuid", type=int, default=107)
    parser.add_argument("--wallet.name", type=str, default="miner")
    parser.add_argument("--wallet.hotkey", type=str, default="default")
    args = parser.parse_args()

    bt.logging.info("Hydrogen Miner starting (Phase 0 scaffold)...")

    # TODO: strategy generation, local dry-run with gates, dendrite submit, results query

    bt.logging.info("Miner ready. Next: implement strategy submission + local validation.")

if __name__ == "__main__":
    main()
