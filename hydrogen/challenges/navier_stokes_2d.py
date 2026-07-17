"""Navier-Stokes 2D Laminar Challenge (Phase 0).

Incompressible NS: ∂u/∂t + (u·∇)u = -∇p/ρ + ν∇^{2}u
                              ∇·u = 0

This is the most important fluid challenge. Strong emphasis on mass conservation.
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple
import numpy as np
import torch


@dataclass
class Challenge:
    challenge_id: str
    problem: str
    dim: int
    resolution: Tuple[int, int]
    train_data: Dict[str, torch.Tensor]
    holdout_data: Dict[str, torch.Tensor]
    stress_data: Dict[str, torch.Tensor]
    symbolic_metadata: Dict[str, Any]
    baseline_error: float


def get_symbolic_metadata() -> Dict[str, Any]:
    return {
        "governing_pde": "∂u/∂t + (u·∇)u = -∇p + ν∇^{2}u,   ∇·u = 0",
        "symmetries": ["translation", "rotation", "galilean"],
        "conservation_laws": ["mass", "momentum"],
        "dimensionless_groups": {"Reynolds_number": "laminar"},
        "boundary_types": ["periodic", "no-slip"],
        "suggested_loss_weights": {
            "pde_residual": 1.0,
            "advection": 1.4,
            "viscous": 1.0,
            "pressure": 0.8,
            "divergence": 2.0,   # Very important for incompressible
            "boundary": 0.7,
        },
        "validity_domain": {"reynolds": [10, 200]},
        "hard_constraints": ["incompressible", "divergence_free"],
    }


def generate_ns_2d_data(
    resolution: Tuple[int, int] = (64, 64),
    seed: int = 42,
    n_samples: int = 4,
    reynolds: float = 100.0,
) -> Dict[str, torch.Tensor]:
    """
    Generate 2D incompressible NS data (laminar regime).

    Uses Taylor-Green vortex style initial condition + simple evolution.
    Good enough for MVP. Real version should use high-order CFD or PhysicsNeMo.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    nx, ny = resolution
    x = torch.linspace(0, 2*np.pi, nx)
    y = torch.linspace(0, 2*np.pi, ny)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Taylor-Green vortex initial condition (divergence free)
    u =  np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    p = 0.25 * (np.cos(2*X) + np.cos(2*Y))

    viscosity = 2 * np.pi / reynolds   # rough scaling

    # Very simple pseudo-spectral style evolution (demo quality)
    u_field = torch.from_numpy(u).float()
    v_field = torch.from_numpy(v).float()

    # Add small noise
    u_field += 0.02 * torch.randn_like(u_field)
    v_field += 0.02 * torch.randn_like(v_field)

    # Store as channels
    velocity = torch.stack([u_field, v_field], dim=0)  # (2, H, W)

    return {
        "x": X.unsqueeze(0).repeat(n_samples, 1, 1),
        "y": Y.unsqueeze(0).repeat(n_samples, 1, 1),
        "velocity_true": velocity.unsqueeze(0).repeat(n_samples, 1, 1, 1),
        "pressure_true": torch.from_numpy(p).float().unsqueeze(0).repeat(n_samples, 1, 1),
        "reynolds": torch.full((n_samples, 1, 1), reynolds),
    }


def load_challenge(challenge_id: str = "ns_2d_laminar_v1") -> Challenge:
    if challenge_id != "ns_2d_laminar_v1":
        raise ValueError(f"Unknown challenge: {challenge_id}")

    resolution = (64, 64)
    full_data = generate_ns_2d_data(resolution=resolution, n_samples=4)

    train_data = {k: v[:2] for k, v in full_data.items()}
    holdout_data = {k: v[2:3] for k, v in full_data.items()}
    stress_data = {k: v[3:4] for k, v in full_data.items()}

    baseline_error = 0.18  # Significantly harder

    return Challenge(
        challenge_id=challenge_id,
        problem="navier_stokes",
        dim=2,
        resolution=resolution,
        train_data=train_data,
        holdout_data=holdout_data,
        stress_data=stress_data,
        symbolic_metadata=get_symbolic_metadata(),
        baseline_error=baseline_error,
    )


def get_baseline_error(challenge_id: str = "ns_2d_laminar_v1") -> float:
    return load_challenge(challenge_id).baseline_error
