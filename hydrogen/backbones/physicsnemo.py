"""Wrapper for PhysicsNeMo models (placeholder for now)."""

import torch.nn as nn

# In a full integration, we would wrap PhysicsNeMo FNO / DeepONet here.
# For now this is a stub to keep the registry clean.

def create_physicsnemo_fno(**kwargs) -> nn.Module:
    # TODO: Proper PhysicsNeMo FNO wrapper
    raise NotImplementedError("PhysicsNeMo backbone wrapper not yet implemented")
