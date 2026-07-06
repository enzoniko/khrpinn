"""Residual-branch models (verbatim from the frozen ICLR scripts).

KANResidual matches the config-ablation variant (configurable grid_size), which
is a strict superset of the parameter-ablation/BPTT variant (grid_size=5).
"""

import os
import sys

import torch
import torch.nn as nn

_ICLR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'iclr')
if _ICLR_DIR not in sys.path:
    sys.path.insert(0, _ICLR_DIR)

from kan import KAN  # noqa: E402  (frozen efficient-kan implementation)


class MLPResidual(nn.Module):
    """Standard MLP baseline: Linear -> ReLU -> Linear -> ReLU -> Linear"""

    def __init__(self, input_dim=2, hidden_dims=[32, 32], output_dim=1):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())

        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        """Forward pass. Accepts kwargs for compatibility (e.g., update_grid is ignored)."""
        return self.net(x)


class KANResidual(nn.Module):
    """Multi-Layer B-Spline KAN residual branch."""

    def __init__(self, layers_hidden=[2, 8, 1], scale_base=1.0, scale_spline=1.0,
                 grid_size=5, **kan_kwargs):
        super().__init__()
        self.kan = KAN(
            layers_hidden=layers_hidden,
            grid_size=grid_size,
            spline_order=3,
            scale_noise=0.1,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
            **kan_kwargs
        )

    def forward(self, x, update_grid=False):
        return self.kan(x, update_grid=update_grid)

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return self.kan.regularization_loss(regularize_activation, regularize_entropy)
