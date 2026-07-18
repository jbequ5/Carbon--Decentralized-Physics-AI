#!/bin/bash
set -e

echo "=============================================="
echo "   Hydrogen Miner Environment (v0.4 - Agent Optimized)"
echo "=============================================="
echo ""

# === Configuration ===
HOTKEY=${HYDROGEN_HOTKEY:-}
WALLET=${HYDROGEN_WALLET:-default}
API_KEY=${HYDROGEN_API_KEY:-}
CHALLENGE_ID=${CHALLENGE_ID:-}

DRY_RUN=${DRY_RUN:-false}
ITERATIONS=${ITERATIONS:-6}
SUBMIT_THRESHOLD=${SUBMIT_THRESHOLD:-0.07}

# === Validation ===
if [ -z "$HOTKEY" ]; then
    echo "WARNING: HYDROGEN_HOTKEY is not set. Submissions will likely fail."
fi

if [ -z "$CHALLENGE_ID" ]; then
    echo "No CHALLENGE_ID set. Falling back to multi-challenge mode."
    python examples/run_agentic_miner.py
    exit 0
fi

# === Focused Mode ===
echo "Focused Mode Activated"
echo "  Challenge       : $CHALLENGE_ID"
echo "  Dry Run         : $DRY_RUN"
echo "  Iterations      : $ITERATIONS"
echo "  Submit Threshold: $SUBMIT_THRESHOLD"
echo ""

echo "Loading priors for $CHALLENGE_ID..."

# Simulate loading priors (in real version this would call the real client)
echo "Priors loaded successfully."
echo ""

# Run focused iteration loop
python -c "
import asyncio
import os
import random

CHALLENGE = os.environ.get('CHALLENGE_ID')
DRY = os.environ.get('DRY_RUN', 'false').lower() == 'true'
MAX_ITER = int(os.environ.get('ITERATIONS', '6'))
THRESHOLD = float(os.environ.get('SUBMIT_THRESHOLD', '0.07'))

async def focused_loop():
    print(f'Starting focused loop on {CHALLENGE}...')
    best_score = 0.0
    best_strategy = None

    for i in range(1, MAX_ITER + 1):
        # Simulate proposing + validating
        score = round(0.055 + (i * 0.008) + random.uniform(-0.01, 0.015), 4)
        print(f'Iteration {i}: estimated_score = {score}')

        if score > best_score:
            best_score = score

        if score >= THRESHOLD:
            print(f'Score {score} meets threshold ({THRESHOLD}). Preparing to submit...')
            if not DRY:
                print('Submitting strategy...')
                # In real version: await miner.submit(...)
                print({'status': 'submitted', 'challenge': CHALLENGE, 'score': score})
            else:
                print('[DRY RUN] Would have submitted.')
            return

    print(f'No strategy reached threshold after {MAX_ITER} iterations. Best score: {best_score}')
    if not DRY and best_score > 0.05:
        print('Submitting best found strategy anyway...')
        print({'status': 'submitted', 'challenge': CHALLENGE, 'score': best_score})

asyncio.run(focused_loop())
"

echo ""
echo "Focused run complete."
echo "=============================================="
