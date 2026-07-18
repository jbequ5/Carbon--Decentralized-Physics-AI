"""Causal Knowledge Base with Heterogeneous Treatment Effects (CATE).

Now supports estimating how treatment effects vary across contexts
(e.g. high vs low Reynolds, different permeability ranges).
"""

import json
import os
import time
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

try:
    from econml.dml import LinearDML
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False

try:
    from pysr import PySRRegressor
    PYSR_AVAILABLE = True
except ImportError:
    PYSR_AVAILABLE = False


from .storage import save_symbolic_artifact, load_symbolic_artifacts


class CausalKnowledgeBase:
    def __init__(self, storage_dir: str = "./data/landscape"):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.causal_estimates: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def _make_key(self, challenge_id: str, backbone: str = "PINO") -> Tuple[str, str]:
        return (challenge_id, backbone)

    def add_observation(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        features: Dict[str, float] = None,
        treatment: Dict[str, float] = None,
        outcome: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        features = features or {}
        treatment = treatment or {}

        artifact = {
            "challenge_id": challenge_id,
            "backbone": backbone,
            "features": features,
            "treatment": treatment,
            "outcome": outcome,
            "metadata": metadata or {},
            "timestamp": int(time.time()),
        }

        save_symbolic_artifact(
            artifact_type="causal_observation",
            challenge_id=challenge_id,
            content=artifact,
            metadata={"source": "training_run", "backbone": backbone},
        )

    def estimate_causal_effects(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        treatment_key: str = "pde_residual_weight",
        min_samples: int = 30,
    ) -> Dict[str, Any]:
        key = self._make_key(challenge_id, backbone)
        artifacts = load_symbolic_artifacts(
            artifact_type="causal_observation",
            challenge_id=challenge_id,
            limit=1000,
        )

        filtered = [
            a for a in artifacts
            if a.get("content", {}).get("backbone", "default") == backbone
        ]
        if len(filtered) < min_samples:
            filtered = artifacts

        if len(filtered) < min_samples:
            return {
                "status": "insufficient_data",
                "n_observations": len(filtered),
                "challenge_id": challenge_id,
                "backbone": backbone,
            }

        X, T, Y = [], [], []
        for art in filtered:
            content = art.get("content", {})
            if treatment_key not in content.get("treatment", {}):
                continue

            feat_vec = list(content.get("features", {}).values())
            t_val = content["treatment"][treatment_key]
            y_val = content["outcome"]

            X.append(feat_vec)
            T.append(t_val)
            Y.append(y_val)

        if len(Y) < min_samples:
            return {"status": "insufficient_data_after_filtering", "n": len(Y)}

        X = np.array(X)
        T = np.array(T).ravel()
        Y = np.array(Y)

        method_used = "simple_residual_on_residual"
        ate = 0.0

        if ECONML_AVAILABLE:
            try:
                model = LinearDML(model_y="auto", model_t="auto", random_state=42)
                model.fit(Y, T, X=X)
                ate = float(model.ate(X))
                method_used = "LinearDML"
            except Exception:
                pass

        if method_used == "simple_residual_on_residual":
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_predict

            mu_y = cross_val_predict(Ridge(), X, Y, cv=5)
            mu_t = cross_val_predict(Ridge(), X, T, cv=5)
            residual_y = Y - mu_y
            residual_t = T - mu_t
            ate = np.mean(residual_y * residual_t) / (np.var(residual_t) + 1e-8)

        result = {
            "status": "success",
            "method": method_used,
            "ate": ate,
            "n_samples": len(Y),
            "challenge_id": challenge_id,
            "backbone": backbone,
            "treatment": treatment_key,
            "timestamp": int(time.time()),
        }

        self.causal_estimates[key] = result
        save_symbolic_artifact(
            artifact_type="causal_estimate",
            challenge_id=challenge_id,
            content=result,
            metadata={"backbone": backbone, "source": "CausalKnowledgeBase"},
        )
        return result

    def estimate_heterogeneous_effects(
        self,
        challenge_id: str,
        backbone: str = "PINO",
        treatment_key: str = "pde_residual_weight",
        min_samples: int = 40,
    ) -> Dict[str, Any]:
        """
        Estimate Heterogeneous Treatment Effects (CATE).

        Returns average treatment effect + simple stratification
        by feature values (e.g. high vs low Reynolds).
        """
        key = self._make_key(challenge_id, backbone)
        artifacts = load_symbolic_artifacts(
            artifact_type="causal_observation",
            challenge_id=challenge_id,
            limit=1000,
        )

        filtered = [a for a in artifacts if a.get("content", {}).get("backbone", "default") == backbone]
        if len(filtered) < min_samples:
            return {"status": "insufficient_data", "n": len(filtered)}

        # Build dataset
        data = []
        for art in filtered:
            content = art.get("content", {})
            if treatment_key not in content.get("treatment", {}):
                continue
            row = {
                "treatment": content["treatment"][treatment_key],
                "outcome": content["outcome"],
            }
            row.update(content.get("features", {}))
            data.append(row)

        if len(data) < min_samples:
            return {"status": "insufficient_data_after_filtering", "n": len(data)}

        import pandas as pd
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import cross_val_predict

        df = pd.DataFrame(data)
        feature_cols = [c for c in df.columns if c not in ["treatment", "outcome"]]

        if not feature_cols:
            return {"status": "no_features"}

        X = df[feature_cols].values
        T = df["treatment"].values
        Y = df["outcome"].values

        # Simple CATE via interaction model
        X_interact = np.column_stack([X, X * T.reshape(-1, 1)])

        mu_y = cross_val_predict(Ridge(), X_interact, Y, cv=5)
        residual_y = Y - mu_y

        # Estimate interaction effects
        from sklearn.linear_model import LinearRegression
        interaction_model = LinearRegression().fit(X_interact, residual_y)

        # Average treatment effect
        ate = np.mean(T * residual_y) / (np.var(T) + 1e-8)

        # Simple stratification (e.g. by first feature)
        if len(feature_cols) > 0:
            feat_name = feature_cols[0]
            high_mask = X[:, 0] > np.median(X[:, 0])
            ate_high = np.mean(T[high_mask] * residual_y[high_mask]) / (np.var(T[high_mask]) + 1e-8)
            ate_low = np.mean(T[~high_mask] * residual_y[~high_mask]) / (np.var(T[~high_mask]) + 1e-8)
        else:
            ate_high = ate_low = ate

        result = {
            "status": "success",
            "method": "interaction_model_cate",
            "ate": float(ate),
            "ate_high": float(ate_high),
            "ate_low": float(ate_low),
            "feature": feature_cols[0] if feature_cols else None,
            "n_samples": len(Y),
            "challenge_id": challenge_id,
            "backbone": backbone,
        }

        save_symbolic_artifact(
            artifact_type="cate_estimate",
            challenge_id=challenge_id,
            content=result,
            metadata={"source": "CausalKnowledgeBase"},
        )
        return result
