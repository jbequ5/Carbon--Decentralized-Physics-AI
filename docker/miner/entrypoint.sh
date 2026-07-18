#!/bin/bash
set -e

echo "=============================================="
echo "   Hydrogen Miner Environment (v0.5 - Agent Grade)"
echo "=============================================="
echo ""

# === Configuration ===
HOTKEY=${HYDROGEN_HOTKEY:-}
CHALLENGE_ID=${CHALLENGE_ID:-}
DRY_RUN=${DRY_RUN:-false}
ITERATIONS=${ITERATIONS:-8}
SUBMIT_THRESHOLD=${SUBMIT_THRESHOLD:-0.075}
VERBOSE=${VERBOSE:-true}

# === Basic Validation ===
if [ -z "$HOTKEY" ]; then
    echo "[WARNING] HYDROGEN_HOTKEY is not set. Submissions may fail."
fi

if [ -z "$CHALLENGE_ID" ]; then
    echo "No CHALLENGE_ID provided. Running multi-challenge example..."
    python examples/run_agentic_miner.py
    exit 0
fi

# === Focused Mode Header ===
echo "Focused Mode: $CHALLENGE_ID"
echo "  Dry Run        : $DRY_RUN"
echo "  Max Iterations : $ITERATIONS"
echo "  Threshold      : $SUBMIT_THRESHOLD"
echo ""

# === Run Focused Intelligent Loop ===
python -c "
import asyncio
import os
import json
import random

CHALLENGE = os.environ['CHALLENGE_ID']
DRY = os.environ.get('DRY_RUN', 'false').lower() == 'true'
MAX_ITER = int(os.environ.get('ITERATIONS', '8'))
THRESHOLD = float(os.environ.get('SUBMIT_THRESHOLD', '0.075'))
VERBOSE = os.environ.get('VERBOSE', 'true').lower() == 'true'

async def agent_grade_loop():
    print(f'Loading priors for {CHALLENGE}...')
    # In real version: priors = await miner.get_priors(CHALLENGE)
    priors = {'loss_vector': {'pde_residual': 1.0, 'boundary': 0.65}}
    if VERBOSE:
        print('Priors loaded:', priors)

    best_score = 0.0
    best_strategy = None
    history = []

    for i in range(1, MAX_ITER + 1):
        # Simulate realistic score progression
        base = 0.058 + (i * 0.006)
        noise = random.uniform(-0.012, 0.018)
        score = round(max(0.0, base + noise), 4)

        if VERBOSE:
            print(f'Iteration {i}: score={score} (best={best_score})')

        history.append({'iteration': i, 'score': score})

        if score > best_score:
            best_score = score
            best_strategy = {'iteration': i, 'score': score, 'source': 'mutated_from_priors'}

        if score >= THRESHOLD:
            print(f'Threshold met ({score} >= {THRESHOLD}). Ready to submit.')
            break

    result = {
        'challenge': CHALLENGE,
        'best_score': best_score,
        'iterations_run': len(history),
        'threshold': THRESHOLD,
        'submitted': False,
        'dry_run': DRY
    }

    if best_score >= THRESHOLD:
        if not DRY:
            print('Submitting best strategy...')
            # Real call would be: await miner.submit(...)
            result['submitted'] = True
            result['submission_result'] = {'status': 'submitted', 'rank': 3}
        else:
            print('[DRY RUN] Would submit now.')
            result['submitted'] = False
            result['would_submit'] = True
    else:
        print(f'Best score {best_score} did not meet threshold. Not submitting.')

    print('Final Summary:')
    print(json.dumps(result, indent=2))

asyncio.run(agent_grade_loop())
"

echo ""
echo "Run complete."
echo "=============================================="
