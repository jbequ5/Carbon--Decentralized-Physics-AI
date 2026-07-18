"""Landscape Agent with Specialist Bank integration.

`distill_top_candidates()` now automatically registers successful specialists.
"""

import time
from typing import Dict, Any, List, Optional

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

        print(f"[Landscape] Running daily update for {len(challenge_ids)} challenges...")

        for challenge_id in challenge_ids:
            for backbone in ["PINO"]:
                causal_result = self.kb.estimate_causal_effects(
                    challenge_id, backbone=backbone
                )
                if causal_result.get("status") == "success":
                    print(f"  [{challenge_id}/{backbone}] Causal ATE updated")

        self.last_update = int(time.time())
        print("[Landscape] Daily update complete.")

    def propose_improved_priors(
        self,
        challenge_id: str,
        backbone: str = "PINO",
    ) -> Dict[str, Any]:
        return self.kb.get_best_priors(challenge_id, backbone=backbone)

    def get_publishable_priors(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        noise_level: float = 0.03,
    ) -> Dict[str, Any]:
        return self.kb.get_publishable_priors(
            challenge_id, backbone=backbone, noise_level=noise_level
        )

    def propose_distillation_candidates(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        candidates = []

        weight_artifacts = load_symbolic_artifacts(
            artifact_type="evolved_loss_weights",
            challenge_id=challenge_id,
            limit=20,
        )

        scoring_artifacts = load_symbolic_artifacts(
            artifact_type="pysr_scoring",
            challenge_id=challenge_id,
            limit=10,
        )

        causal = self.kb.causal_estimates.get(
            self.kb._make_key(challenge_id, backbone), {}
        )

        for i, artifact in enumerate(weight_artifacts[:top_k]):
            content = artifact.get("content", {})
            candidate = {
                "rank": i + 1,
                "challenge_id": challenge_id,
                "backbone": backbone,
                "loss_vector": content.get("loss_vector", {}),
                "causal_ate": causal.get("ate", 0.0),
                "source_artifact": artifact.get("timestamp"),
                "recommended_for_distillation": True,
            }
            candidates.append(candidate)

        if scoring_artifacts:
            best_scoring = scoring_artifacts[0]
            candidates.append({
                "rank": len(candidates) + 1,
                "challenge_id": challenge_id,
                "backbone": backbone,
                "type": "scoring_expression",
                "expression": best_scoring.get("content", {}).get("expression"),
                "recommended_for_distillation": True,
            })

        return candidates[:top_k]

    def distill_top_candidates(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        top_k: int = 2,
    ) -> List[Dict[str, Any]]:
        print(f"[Landscape] Distilling top candidates for {challenge_id}...")

        candidates = self.propose_distillation_candidates(
            challenge_id, backbone=backbone, top_k=top_k
        )

        distilled_specialists = []

        for candidate in candidates:
            if candidate.get("type") == "scoring_expression":
                continue

            strategy = {
                "backbone": backbone,
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

            specialist["regression_test"] = test_result

            if test_result.get("regression_passed"):
                print(f"  ✓ Specialist {specialist['specialist_id']} passed regression")
                distilled_specialists.append(specialist)

                # Already registered by distill_strategy_to_specialist
                save_symbolic_artifact(
                    artifact_type="specialist",
                    challenge_id=challenge_id,
                    content=specialist,
                    metadata={"source": "LandscapeAgent"},
                )
            else:
                print(f"  ✗ Specialist {specialist['specialist_id']} failed regression")

        return distilled_specialists

    def get_status(self) -> Dict[str, Any]:
        return {
            "last_update": self.last_update,
            "storage_dir": self.storage_dir,
            "causal_estimates_tracked": len(self.kb.causal_estimates),
        }
