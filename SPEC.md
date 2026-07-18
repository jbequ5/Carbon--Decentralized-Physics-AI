# Hydrogen Subnet Specification

**Status:** Active Development (July 2026)

Hydrogen is a Bittensor subnet (SN63) focused on decentralized Physics-Informed Neural Operator (PINO) training for complex engineering simulations.

## Core Goals

- Discover high-quality training strategies for physics-respecting neural surrogates.
- Use hidden stress testing + hard physics gates to evaluate robustness.
- Maintain strong anti-gaming properties through hidden evaluation data and adaptive difficulty.
- Support multiple parallel challenges while keeping reward distribution reasonably balanced.

## Current Architecture (as of July 2026)

### 1. Scoring System

The validator uses a multi-objective scorer with three high-level objectives:

| Objective            | Default Weight | Description |
|----------------------|----------------|-------------|
| `physics_fidelity`   | 45%            | How well the model respects physics laws (residuals, divergence, conservation, boundaries) |
| `robustness`         | 30%            | Performance under stress conditions + long-term stability |
| `accuracy`           | 25%            | Performance on benchmark / hold-out data |

A **combined final score** is computed from these three. Only submissions that beat the current best combined score on a challenge receive meaningful weight.

### 2. ChallengeWinnerTracker

Inspired by Minos-style round-only winner logic:

- Tracks the best combined score **per challenge**.
- Uses **exponential decay** on old performance.
- Only miners who set a new best combined score on a challenge contribute strongly to weights.
- Includes **participation dust** for recent miners.
- Produces a **winner-heavy + dust** weight distribution for `set_weights()`.

### 3. Validator

- Owns `HydrogenScorer` and `ChallengeWinnerTracker`.
- Retrieves strategies via `StrategyStore`.
- Scores submissions and updates the tracker per challenge.
- Computes weights using the tracker and submits via standard `set_weights()`.
- Supports dry-run mode.

### 4. Strategy Handling

- `StrategyStore` abstraction with `LocalFileStrategyStore` implementation.
- Designed to be easily swapped for a platform-backed store later.
- Currently uses local JSON files for development/testing.

### 5. Emissions

Currently uses **standard Yuma Consensus** miner emissions via `set_weights()`.

A more advanced hybrid emissions model (Breakthrough Bounties + Decaying Stipends + treasury rollover) is planned but **not active** at this time.

## Validator-Private Components

To maintain strong anti-gaming properties, the following should remain **validator-private** (not directly accessible to miners or agents):

- Hidden stress test conditions and evaluation data
- Exact physics gate implementations and thresholds
- Full combined scoring logic and internal weight calculations
- `ChallengeWinnerTracker` internal state and leader tracking logic
- Raw per-miner evaluation traces and detailed physics violation data
- Landscape Agent raw data and learned model (only noisy/aggregated outputs should be exposed)

Miners and agents are expected to operate with **partial information**. This asymmetry is intentional and is one of Hydrogen’s core anti-gaming mechanisms.

Publicly exposed information should be limited to:
- Challenge descriptions
- High-level scoring objectives (Physics, Robustness, Accuracy)
- Miners’ own scores and metrics
- Noisy or aggregated priors / leader information

## Key Components

| Component                        | Location                                      | Status |
|----------------------------------|-----------------------------------------------|--------|
| `HydrogenScorer`                 | `neurons/scoring/hydrogen_scorer.py`          | Active |
| `ChallengeWinnerTracker`         | `neurons/scoring/challenge_winner_tracker.py` | Active |
| `Validator`                      | `neurons/validator.py`                        | Active |
| `StrategyStore`                  | `neurons/strategy/strategy_store.py`          | Active |
| MCP Server                       | `hydrogen/miner/mcp_server.py`                | Active |

## Current Limitations

- Strategy retrieval is still primarily local-file based.
- No hybrid bounty/treasury layer yet.
- Single-validator focused (multi-validator consistency not yet stress-tested).
- Emissions are standard Yuma only.

## Future Work (Not in current scope)

- Hybrid emissions model (75/25 Breakthrough Bounties + Decaying Top-2 Stipend)
- Per-challenge bounty pools with daily/round reset + treasury rollover
- Stronger anti-self-dealing / capture protections
- Full platform-backed strategy store
- Multi-validator testing and canonical ranking integration

## Versioning

This document reflects the state of the codebase after integration of the `ChallengeWinnerTracker` and `StrategyStore` (July 2026).
