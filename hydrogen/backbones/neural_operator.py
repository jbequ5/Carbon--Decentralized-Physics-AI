"""Wrapper for the NeuralOperator library."""

import torch.nn as nn

from neuralop.models import FNO, DeepONet


def create_fno(
    in_channels: int = 3,
    out_channels: int = 1,
    modes: int = 16,
    width: int = 64,
    n_layers: int = 4,
    **kwargs,
) -> nn.Module:
    return FNO(
        in_channels=in_channels,
        out_channels=out_channels,
        modes=modes,
        width=width,
        n_layers=n_layers,
        **kwargs,
    )


def create_deeptonet(
    branch_net: nn.Module,
    trunk_net: nn.Module,
    output_dim: int = 1,
    **kwargs,
) -> nn.Module:
    return DeepONet(
        branch_net=branch_net,
        trunk_net=trunk_net,
        output_dim=output_dim,
        **kwargs,
    )
