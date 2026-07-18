#!/bin/bash
set -e

echo "=============================================="
echo "   Hydrogen Miner Environment (v0.7)"
echo "=============================================="
echo ""

# === Configuration (Miner-controlled) ===
CHALLENGE_ID=${CHALLENGE_ID:-}
DRY_RUN=${DRY_RUN:-false}
ITERATIONS=${ITERATIONS:-8}
SUBMIT_THRESHOLD=${SUBMIT_THRESHOLD:-0.075}

if [ -z "$CHALLENGE_ID" ]; then
    echo "No CHALLENGE_ID set. Running multi-challenge example..."
    python examples/run_agentic_miner.py
    exit 0
fi

# === Focused Mode ===
echo "Focused Mode: $CHALLENGE_ID"
echo "  Dry Run         : $DRY_RUN"
echo "  Iterations      : $ITERATIONS"
echo "  Threshold       : $SUBMIT_THRESHOLD"
echo "  Note: Priors include system-controlled noise (anti-gaming)"
echo ""

python -c "
import asyncio
import os
import json
import random

CHALLENGE = os.environ['CHALLENGE_ID']
DRY = os.environ.get('DRY_RUN', 'false').lower() == 'true'
MAX_ITER = int(os.environ.get('ITERATIONS', '8'))
THRESHOLD = float(os.environ.get('SUBMIT_THRESHOLD', '0.075'))

async def improved_agent_loop():
    print(f'Loading priors for {CHALLENGE} (with system noise)...')

    # Simulated priors with internal noise (miner cannot control noise level)
    base_priors = {'pde_residual': 1.0, 'boundary': 0.67, 'conservation_mass': 1.18}
    noisy_priors = {k: round(v * (1 + random.gauss(0, 0.035)), 3) for k, v in base_priors.items()}

    print('Priors (noisy):', noisy_priors)

    best_score = 0.0
    best_strategy = None
    history = []

    for i in range(1, MAX_ITER + 1):
        progress = i / MAX_ITER
        base = 0.056 + (progress * 0.048)
        noise = random.uniform(-0.014, 0.017)
        score = round(max(0.0, base + noise), 4)

        print(f'Iteration {i}/{MAX_ITER}: score = {score}')
        history.append({'iter': i, 'score': score})

        if score > best_score:
            best_score = score
            best_strategy = {
                'challenge': CHALLENGE,
                'score': score,
                'priors_used': noisy_priors,
                'iteration': i
            }

        if score >= THRESHOLD:
            print(f'Threshold met at iteration {i}.')
            break

    result = {
        'challenge': CHALLENGE,
        'best_score': round(best_score, 4),
        'iterations': len(history),
        'threshold': THRESHOLD,
        'submitted': False,
        'dry_run': DRY
    }

    if best_score >= THRESHOLD:
        if not DRY:
            print('Submitting best strategy...')
            result['submitted'] = True
            result['submission'] = {'status': 'submitted'}
        else:
            print('[DRY RUN] Would submit.')
            result['would_submit'] = True
    else:
        print(f'Best score {best_score} below threshold. Not submitting.')

    # Attach the actual best strategy
    if best_strategy:
        result['best_strategy'] = best_strategy

    # Intelligent recommendations
    recs = []
    if best_score < 0.065:
        recs.append('Try increasing physics_loss_weight or tuning loss_vector.')
    elif best_score > 0.095:
        recs.append('Excellent result. Consider trying a different backbone next run.')
    else:
        recs.append('Solid run. Small mutations around current priors may help.')

    result['recommended_next_actions'] = recs

    print('\n=== Final JSON Summary ===')
    print(json.dumps(result, indent=2))

    print('\nRecommended Next Actions:')
    for r in recs:
        print(' -', r)

asyncio.run(improved_agent_loop())
"

echo ""
echo "Run complete."
echo "=============================================="
