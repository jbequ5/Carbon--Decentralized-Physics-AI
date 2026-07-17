"""Landscape Agent - SOTA starting state for Hydrogen.

The Landscape is the central intelligence of the subnet.
It collects fragments, runs causal inference (Double ML), integrates
symbolic knowledge (PySR), and proposes improved priors/baselines.

Design goals:
- Challenge + backbone aware
- High-signal causal reasoning
- Clean interface for miners/validators
- Extensible for future specialist distillation
"""

import time
from typing import Dict, Any, List, Optional

from .causal_knowledge_base import CausalKnowledgeBase
from .storage import save_symbolic_artifact, load_symbolic_artifacts


class LandscapeAgent:
    """
    The Landscape Agent orchestrates causal + symbolic knowledge.

    Current capabilities (SOTA starting state):
    - Ingest strategy fragments / observations
    - Run Double ML causal inference per challenge + backbone
    - Integrate PySR symbolic discoveries
    - Propose improved priors for miners
    - Maintain versioned knowledge
    """

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
        """Ingest a new training/validation observation."""
        self.kb.add_observation(
            challenge_id=challenge_id,
            backbone=backbone,
            features=features,
            treatment=treatment,
            outcome=outcome,
            metadata=metadata,
        )

    def run_daily_update(self, challenge_ids: List[str] = None):
        """
        Run the daily causal + symbolic update cycle.

        This is the main heartbeat of the Landscape.
        """
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
                # 1. Update causal estimates
                causal_result = self.kb.estimate_causal_effects(
                    challenge_id, backbone=backbone
                )

                if causal_result.get("status") == "success":
                    print(f"  [{challenge_id}/{backbone}] Causal ATE updated: {causal_result.get('ate', 0):.4f}")

                # 2. Could trigger PySR re-analysis here in future

        self.last_update = int(time.time())
        print("[Landscape] Daily update complete.")

    def propose_improved_priors(
        self,
        challenge_id: str,
        backbone: str = "PINO",
    ) -> Dict[str, Any]:
        """
        Propose the current best priors based on causal + symbolic evidence.

        This is what miners should ideally build from (via published noisy version).
        """
        return self.kb.get_best_priors(challenge_id, backbone=backbone)

    def get_publishable_priors(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        noise_level: float = 0.03,
    ) -> Dict[str, Any]:
        """
        Return noisy priors suitable for daily public publication.
        """
        return self.kb.get_publishable_priors(
            challenge_id, backbone=backbone, noise_level=noise_level
        )

    def get_status(self) -> Dict[str, Any]:
        """Return current status of the Landscape."""
        return {
            "last_update": self.last_update,
            "storage_dir": self.storage_dir,
            "causal_estimates_tracked": len(self.kb.causal_estimates),
        }
