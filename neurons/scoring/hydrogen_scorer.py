# neurons/scoring/hydrogen_scorer.py

"""
HydrogenScorer - Multi-objective scoring with integrated stress testing.

Now fully wired with StressEvaluator for hidden stress assessment.
"""

from typing import Any, Dict, Optional

from neurons.stress.stress_evaluator import StressEvaluator

from neurons.stress.stress_models import StressTestSet


class HydrogenScorer:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.stress_evaluator = StressEvaluator(self)

    # ============================================================
    # Existing scoring methods (simplified for integration)
    # ============================================================

    def compute_physics_fidelity(self, metrics: Dict[str, float]) -> float:
        # Placeholder - real implementation would compute residuals, conservation, etc.
        return metrics.get("physics_fidelity", 0.8)

    def compute_robustness(self, metrics: Dict[str, float]) -> float:
        return metrics.get("robustness", 0.75)

    def compute_accuracy(self, metrics: Dict[str, float]) -> float:
        return metrics.get("accuracy", 0.85)

    def check_hard_gates(self, metrics: Dict[str, float]) -> list:
        violations = []
        if metrics.get("mass_conservation_error", 0) > 1e-3:
            violations.append("mass_conservation")
        if metrics.get("energy_dissipation_rate", 0) > 1e-4:
            violations.append("energy_dissipation")
        if metrics.get("boundary_error", 0) > 1e-3:
            violations.append("boundary")
        return violations

    def apply_soft_gates(self, metrics: Dict[str, float]) -> Dict[str, float]:
        penalties = {}
        if metrics.get("symmetry_error", 0) > 0.05:
            penalties["symmetry"] = min(0.3, (metrics["symmetry_error"] - 0.05) * 5)
        return penalties

    # ============================================================
    # New: Integrated stress evaluation
    # ============================================================

    def evaluate_with_stress(
        self,
        model: Any,
        stress_set: StressTestSet,
        base_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on hidden stress set and combine with base metrics.

        Returns full scoring result including stress contribution.
        """
        stress_results = self.stress_evaluator.evaluate(model, stress_set)

        # If hard gate failed, overall stress contribution is 0
        if stress_results["hard_gate_failures"]:
            stress_contribution = 0.0
        else:
            stress_contribution = stress_results["stress_score_contribution"]

        # Combine with base metrics (from holdout/benchmark)
        base = base_metrics or {}
        physics = self.compute_physics_fidelity(base)
        robustness = self.compute_robustness(base)
        accuracy = self.compute_accuracy(base)

        # Final combined score (simple weighted for now)
        combined = (
            0.45 * physics +
            0.30 * robustness +
            0.25 * accuracy
        ) * (0.7 + 0.3 * stress_contribution)   # Stress modulates final score

        return {
            "physics_fidelity": physics,
            "robustness": robustness,
            "accuracy": accuracy,
            "combined_score": combined,
            "stress_results": stress_results,
            "hard_gate_failures": stress_results["hard_gate_failures"],
        }

    def score_strategy(
        self,
        model: Any,
        stress_set: Optional[StressTestSet] = None,
        base_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Main entry point.
        If stress_set is provided, runs full stress evaluation.
        """
        if stress_set is not None:
            return self.evaluate_with_stress(model, stress_set, base_metrics)
        else:
            # Fallback: just base scoring (no hidden stress)
            base = base_metrics or {}
            return {
                "physics_fidelity": self.compute_physics_fidelity(base),
                "robustness": self.compute_robustness(base),
                "accuracy": self.compute_accuracy(base),
                "combined_score": 0.8,  # placeholder
            }
