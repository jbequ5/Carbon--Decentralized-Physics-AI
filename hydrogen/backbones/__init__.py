"""Backbone registry for Hydrogen.

Supports multiple neural operator implementations:
- PhysicsNeMo (NVIDIA)
- NeuralOperator library (standard open-source)
"""

from .registry import get_backbone, list_available_backbones, register_backbone

__all__ = ["get_backbone", "list_available_backbones", "register_backbone"]
