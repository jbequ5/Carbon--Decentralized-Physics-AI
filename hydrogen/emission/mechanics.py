"""Core emission mechanics for Hydrogen (75/25 model).

Configurable Top-2 split and Yuma-friendly weight generation.
"""

from typing import Dict, Any, Tuple

import numpy as np


# ============================================================
# Configuration (can be overridden)
# ============================================================

BREAKTHROUGH_BOUNTY_SHARE = 0.75
TOP2_STIPEND_SHARE = 0.25

BREAKTHROUGH_THRESHOLD = 0.06
STIPEND_DECAY_RATE = 0.45
BREAKTHROUGH_COOLDOWN_EPOCHS = 8

# Default Top-2 split (can be changed)
DEFAULT_LEADER_SHARE = 0.72
DEFAULT_SECOND_SHARE = 0.28


# ============================================================
# Data Structures
# ============================================================

class ChallengeState:
    def __init__(self):
        self.current_best_score = 0.0
        self.leader_hotkey = None
        self.second_hotkey = None
        self.leader_stipend_share = 0.0
        self.second_stipend_share = 0.0
        self.epochs_without_improvement = 0
        self.epochs_since_last_breakthrough = 0
        self.accumulated_bounty_pool = 0.0


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
    if current_best <= 0:
        return new_score > 0.1
    improvement = (new_score - current_best) / (current_best + 1e-8)
    return improvement >= threshold


def update_leaderboard(
    challenge_id: str,
    hotkey: str,
    new_score: float,
) -> Tuple[bool, str]:
    state = get_or_create_state(challenge_id)

    was_breakthrough = False
    message = "No change"

    state.epochs_since_last_breakthrough += 1

    if is_breakthrough(new_score, state.current_best_score):
        if state.epochs_since_last_breakthrough >= BREAKTHROUGH_COOLDOWN_EPOCHS:
            was_breakthrough = True
            state.current_best_score = new_score
            state.leader_hotkey = hotkey
            state.epochs_without_improvement = 0
            state.epochs_since_last_breakthrough = 0
            message = f"New record set by {hotkey[:8]}!"
        else:
            message = f"Strong result, but cooldown active ({state.epochs_since_last_breakthrough}/{BREAKTHROUGH_COOLDOWN_EPOCHS})"
    else:
        state.epochs_without_improvement += 1

    if hotkey != state.leader_hotkey:
        if state.second_hotkey is None or new_score > state.current_best_score * 0.95:
            state.second_hotkey = hotkey

    return was_breakthrough, message


def calculate_top2_stipend(
    challenge_id: str,
    total_stipend_pool: float,
    leader_ratio: float = DEFAULT_LEADER_SHARE,
    second_ratio: float = DEFAULT_SECOND_SHARE,
) -> Dict[str, float]:
    """
    Calculate decaying Top-2 stipend with configurable split.
    """
    state = get_or_create_state(challenge_id)

    if state.leader_hotkey is None:
        return {}

    decay_factor = (1 - STIPEND_DECAY_RATE) ** state.epochs_without_improvement

    leader_share = leader_ratio * decay_factor
    second_share = second_ratio * decay_factor

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


def get_yuma_weights(
    challenge_id: str,
    total_stipend_pool: float = 1.0,
    leader_ratio: float = DEFAULT_LEADER_SHARE,
    second_ratio: float = DEFAULT_SECOND_SHARE,
) -> Dict[str, float]:
    """
    Returns weights ready for Yuma Consensus / set_weights.
    Normalized so they sum to total_stipend_pool.
    """
    return calculate_top2_stipend(
        challenge_id=challenge_id,
        total_stipend_pool=total_stipend_pool,
        leader_ratio=leader_ratio,
        second_ratio=second_ratio,
    )


def calculate_breakthrough_bounty(
    challenge_id: str,
    total_bounty_pool: float,
    was_breakthrough: bool,
) -> float:
    state = get_or_create_state(challenge_id)

    state.accumulated_bounty_pool += total_bounty_pool

    if was_breakthrough:
        payout = state.accumulated_bounty_pool
        state.accumulated_bounty_pool = 0.0
        return payout
    else:
        return 0.0
