"""Landscape Agent with CATE + Novelty-Aware Distillation.

Implements both heterogeneous treatment effects and improved novelty scoring.
"""

import time
from typing import Dict, Any, List, Optional

import numpy as np

from .causal_knowledge_base import CausalKnowledgeBase
from .storage import save_symbolic_artifact, load_symbolic_artifacts
from hydrogen.specialist.distillation import distill_strategy_to_specialist, regression_test_specialist
from hydrogen.specialist.bank import SpecialistBank

bank = SpecialistBank()


class LandscapeAgent:
    def __init__(self, storage_dir: str = "./data/landscape"):
        self.storage_dir = storage_dir
        self.kb = CausalKnowledgeBase(storage_dir=storage_dir)
        self.last_update = None

    def ingest_observation(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        features: Dict[str, float] = None,
        treatment: Dict[str, float] = None,
        outcome: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.kb.add_observation(
            challenge_id=challenge_id,
            backbone=backbone,
            features=features,
            treatment=treatment,
            outcome=outcome,
            metadata=metadata,
        )

    def run_daily_update(self, challenge_ids: List[str] = None):
        if challenge_ids is None:
            challenge_ids = [
                "poisson_2d_v1",
                "darcy_2d_v1",
                "burgers_v1",
                "heat_v1",
                "elasticity_2d_v1",
                "ns_2d_laminar_v1",
            ]

        print(f"[Landscape] Running daily causal + symbolic update...")

        for challenge_id in challenge_ids:
            for backbone in ["PINO"]:
                causal_result = self.kb.estimate_causal_effects(
                    challenge_id, backbone=backbone
                )
                if causal_result.get("status") == "success":
                    print(f"  [{challenge_id}/{backbone}] Causal ATE: {causal_result.get('ate', 0):.4f}")

                # Also estimate heterogeneous effects
                cate_result = self.kb.estimate_heterogeneous_effects(
                    challenge_id, backbone=backbone
                )
                if cate_result.get("status") == "success":
                    print(f"  [{challenge_id}/{backbone}] CATE estimated (high vs low feature)")

        self.last_update = int(time.time())
        print("[Landscape] Daily update complete.")

    def propose_distillation_candidates(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Novelty + Causal aware candidate selection.
        """
        candidates = []

        weight_artifacts = load_symbolic_artifacts(
            artifact_type="evolved_loss_weights",
            challenge_id=challenge_id,
            limit=25,
        )

        causal = self.kb.causal_estimates.get(
            self.kb._make_key(challenge_id, backbone), {}
        )
        causal_ate = causal.get("ate", 0.0)

        if causal_ate <= 0.01:
            print(f"[{challenge_id}] Weak causal signal. Skipping distillation.")
            return []

        for i, artifact in enumerate(weight_artifacts[:top_k]):
            content = artifact.get("content", {})
            loss_vector = content.get("loss_vector", {})

            # Simple novelty score: average absolute deviation from current best
            novelty = 0.0
            if loss_vector:
                # Compare to a simple uniform prior as baseline
                avg_dev = np.mean([abs(v - 1.0) for v in loss_vector.values()])
                novelty = min(1.0, avg_dev)

            combined_score = causal_ate * 0.7 + novelty * 0.3

            candidate = {
                "rank": i + 1,
                "challenge_id": challenge_id,
                "backbone": backbone,
                "loss_vector": loss_vector,
                "causal_ate": causal_ate,
                "novelty_score": novelty,
                "combined_score": combined_score,
                "recommended_for_distillation": True,
            }
            candidates.append(candidate)

        # Sort by combined causal + novelty score
        candidates = sorted(candidates, key=lambda x: x["combined_score"], reverse=True)
        return candidates[:top_k]

    def run_full_daily_cycle(self):
        print("\n[Landscape] === Starting Full Daily Cycle (Causal + Novelty) ===")

        self.run_daily_update()

        challenges = [
            "poisson_2d_v1",
            "darcy_2d_v1",
            "burgers_v1",
            "heat_v1",
            "elasticity_2d_v1",
            "ns_2d_laminar_v1",
        ]

        for challenge_id in challenges:
            candidates = self.propose_distillation_candidates(challenge_id)

            if not candidates:
                continue

            print(f"[{challenge_id}] Distilling {len(candidates)} cause + novelty backed candidates...")

            for candidate in candidates:
                strategy = {
                    "backbone": candidate["backbone"],
                    "pino": {
                        "loss_vector": candidate.get("loss_vector", {})
                    },
                }

                specialist = distill_strategy_to_specialist(
                    challenge_id=challenge_id,
                    strategy=strategy,
                )

                test_result = regression_test_specialist(
                    specialist=specialist,
                    challenge_id=challenge_id,
                )

                if test_result.get("regression_passed"):
                    print(f"  ✓ Registered specialist {specialist['specialist_id']}")
                    bank.register(specialist)

                    save_symbolic_artifact(
                        artifact_type="specialist",
                        challenge_id=challenge_id,
                        content=specialist,
                        metadata={
                            "source": "LandscapeAgent",
                            "causal_ate": candidate.get("causal_ate"),
                            "novelty_score": candidate.get("novelty_score"),
                        },
                    )

        print("[Landscape] === Daily Cycle Complete ===\n")

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_update": self.last_update,
            "storage_dir": self.storage_dir,
            "causal_estimates_tracked": len(self.kb.causal_estimates),
        }
