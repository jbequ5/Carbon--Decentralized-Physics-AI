# Hydrogen Emissions Overview (Current State)

**Last Updated:** July 2026

## Current Emissions Model

Hydrogen currently uses **standard Yuma Consensus** miner emissions.

- The validator computes weights using the `ChallengeWinnerTracker`.
- Weights are submitted via the standard `subtensor.set_weights()` call.
- Miner rewards are determined by Yuma Consensus (stake-weighted median + clipping + normalization).

There is **no custom hybrid bounty layer** active at this time.

## Planned Future Model (Not Implemented Yet)

A hybrid emissions model was discussed with the following goals:

- **75% Breakthrough Bounties** — Large rewards when a miner sets a new record on hidden stress tests (combined Physics + Robustness + Accuracy).
- **25% Decaying Top-2 Stipend** — Ongoing but decaying reward for current leaders, requiring continued improvement.
- Daily/round-based reset with unclaimed funds rolling to a treasury.

This model is **not active**. The current implementation uses only standard Yuma emissions.

## Why Standard Yuma for Now

- Simpler and more compatible with the broader Bittensor network.
- Easier to reason about and audit.
- Allows the team to focus on core scoring and challenge winner tracking first.

## Weight Distribution Logic (Active)

The `ChallengeWinnerTracker` produces weights with the following characteristics:

- Strong preference for current leaders of active challenges (winner-heavy).
- Small participation "dust" for recent miners who do not win.
- Exponential decay on old performance.
- Only miners who beat the current best **combined score** on a challenge receive meaningful credit.

## Summary

| Aspect                    | Current Status          | Notes |
|---------------------------|-------------------------|-------|
| Base emissions            | Standard Yuma           | Via `set_weights()` |
| Custom bounties / stipends| Not active              | Planned for future |
| Per-challenge rewards     | Partially supported     | Via tracker per-challenge logic |
| Daily/round reset         | Supported via tracker   | Exponential decay + round reset |

For the most up-to-date behavior, refer to `ChallengeWinnerTracker.get_weights()` and the validator's `_set_weights()` method.
