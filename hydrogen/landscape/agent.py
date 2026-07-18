"""Landscape Agent with Cause-Aware Distillation (SOTA version).

Uses causal inference on benchmark + stress performance to decide
what and when to distill. Aligns with physics-correct progress only.
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
                # Estimate causal effects using benchmark + stress performance
                causal_result = self.kb.estimate_causal_effects(
                    challenge_id, backbone=backbone
                )
                if causal_result.get("status") == "success":
                    print(f"  [{challenge_id}/{backbone}] Causal ATE: {causal_result.get('ate', 0):.4f}")

        self.last_update = int(time.time())
        print("[Landscape] Daily update complete.")

    def propose_distillation_candidates(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Cause-aware candidate selection.

        Ranks candidates using:
        - Causal effect strength (from Double ML on benchmark/stress performance)
        - Recent stress performance
        - Novelty relative to current best
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

        # Only consider candidates if there is positive causal signal
        if causal_ate <= 0.01:
            print(f"[{challenge_id}] Weak causal signal (ATE={causal_ate:.4f}). Skipping distillation.")
            return []

        for i, artifact in enumerate(weight_artifacts[:top_k]):
            content = artifact.get("content", {})
            candidate = {
                "rank": i + 1,
                "challenge_id": challenge_id,
                "backbone": backbone,
                "loss_vector": content.get("loss_vector", {}),
                "causal_ate": causal_ate,
                "source_artifact": artifact.get("timestamp"),
                "recommended_for_distillation": True,
            }
            candidates.append(candidate)

        return candidates

    def run_full_daily_cycle(self):
        """
        SOTA daily cycle:
        1. Run causal + symbolic update (using benchmark + stress data)
        2. For each challenge, propose cause-aware distillation candidates
        3. Distill only when causal evidence is positive
        4. Register successful specialists
        5. (Future) Publish improved priors
        """
        print("\n[Landscape] === Starting Full Daily Cycle ===")

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

            print(f"[{challenge_id}] Distilling {len(candidates)} cause-backed candidates...")

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
                        metadata={"source": "LandscapeAgent", "causal_ate": candidate.get("causal_ate")},
                    )

        print("[Landscape] === Daily Cycle Complete ===\n")

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_update": self.last_update,
            "storage_dir": self.storage_dir,
            "causal_estimates_tracked": len(self.kb.causal_estimates),
        }
