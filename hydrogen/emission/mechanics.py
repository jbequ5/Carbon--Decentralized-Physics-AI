"""Core emission mechanics for Hydrogen (75/25 model).

Follows Bittensor conventions:
- Uses standard concepts from Yuma Consensus for the stipend part.
- Breakthrough bounties are event-driven (detected by validator/Landscape).
- All calculations are per-challenge.
"""

from typing import Dict, Any, Tuple

import numpy as np


# ============================================================
# Configuration
# ============================================================

BREAKTHROUGH_BOUNTY_SHARE = 0.75
TOP2_STIPEND_SHARE = 0.25

# How much better than current best to count as breakthrough
BREAKTHROUGH_THRESHOLD = 0.06  # 6% improvement in stress score

# Decay rate for the Top-2 stipend (per epoch without improvement)
STIPEND_DECAY_RATE = 0.45


# ============================================================
# Data Structures
# ============================================================

class ChallengeState:
    """Tracks state for one challenge."""

    def __init__(self):
        self.current_best_score = 0.0
        self.leader_hotkey = None
        self.second_hotkey = None
        self.leader_stipend_share = 0.0
        self.second_stipend_share = 0.0
        self.epochs_without_improvement = 0


# In-memory state per challenge (in production this would be persisted)
CHALLENGE_STATES: Dict[str, ChallengeState] = {}


def get_or_create_state(challenge_id: str) -> ChallengeState:
    if challenge_id not in CHALLENGE_STATES:
        CHALLENGE_STATES[challenge_id] = ChallengeState()
    return CHALLENGE_STATES[challenge_id]


def is_breakthrough(
    new_score: float,
    current_best: float,
    threshold: float = BREAKTHROUGH_THRESHOLD,
) -> bool:
    """Check if new score constitutes a breakthrough."""
    if current_best <= 0:
        return new_score > 0.1  # first meaningful result
    improvement = (new_score - current_best) / (current_best + 1e-8)
    return improvement >= threshold


def update_leaderboard(
    challenge_id: str,
    hotkey: str,
    new_score: float,
) -> Tuple[bool, str]:
    """
    Update leaderboard for a challenge.
    Returns (is_breakthrough, message)
    """
    state = get_or_create_state(challenge_id)

    was_breakthrough = False
    message = "No change"

    if is_breakthrough(new_score, state.current_best_score):
        was_breakthrough = True
        state.current_best_score = new_score
        state.leader_hotkey = hotkey
        state.epochs_without_improvement = 0
        message = f"New record set by {hotkey[:8]}!"
    else:
        state.epochs_without_improvement += 1

    # Update #2 if this hotkey is not the leader but beats current #2
    # (simplified logic for now)
    if hotkey != state.leader_hotkey:
        if state.second_hotkey is None or new_score > state.current_best_score * 0.95:
            state.second_hotkey = hotkey

    return was_breakthrough, message


def calculate_top2_stipend(
    challenge_id: str,
    total_stipend_pool: float,
) -> Dict[str, float]:
    """
    Calculate decaying stipend shares for Top 2.
    """
    state = get_or_create_state(challenge_id)

    if state.leader_hotkey is None:
        return {}

    # Apply decay
    decay_factor = (1 - STIPEND_DECAY_RATE) ** state.epochs_without_improvement

    leader_share = 0.72 * decay_factor
    second_share = 0.28 * decay_factor

    total = leader_share + second_share
    if total > 0:
        leader_share /= total
        second_share /= total

    result = {}
    if state.leader_hotkey:
        result[state.leader_hotkey] = leader_share * total_stipend_pool
    if state.second_hotkey:
        result[state.second_hotkey] = second_share * total_stipend_pool

    return result


def calculate_breakthrough_bounty(
    challenge_id: str,
    total_bounty_pool: float,
) -> float:
    """
    For now, return the full pool when a breakthrough happens.
    In production this would be more sophisticated (cooldowns, accumulation, etc).
    """
    return total_bounty_pool
