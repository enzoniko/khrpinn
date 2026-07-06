"""HRPINN cell with hard constraints (verbatim from kan_hrpinn_paper_bptt.py)."""

import torch
import torch.nn as nn


class HRPINNCell(nn.Module):
    """HRPINN cell with hard-constrained physics."""

    def __init__(self, system, residual_network, normalizer, dt=0.05):
        super().__init__()
        self.system = system
        self.residual_network = residual_network
        self.normalizer = normalizer
        self.dt = dt

        self.register_buffer('norm_min', torch.tensor(normalizer.min_vals, dtype=torch.float32))
        self.register_buffer('norm_range', torch.tensor(normalizer.range, dtype=torch.float32))

    def forward(self, state_normalized):
        """Forward pass with hard constraints."""
        min_vals = self.norm_min.to(state_normalized.device)
        range_vals = self.norm_range.to(state_normalized.device)
        state = (state_normalized + 1) * range_vals / 2 + min_vals

        # Known physics (hard-constrained)
        dv_known = self.system.known_acceleration(state)
        if not isinstance(dv_known, torch.Tensor):
            dv_known = torch.tensor(dv_known, dtype=state.dtype, device=state.device)
        if dv_known.dim() == 0:
            dv_known = dv_known.unsqueeze(0)
        if dv_known.dim() == 1:
            dv_known = dv_known.unsqueeze(1)

        # Residual (learned)
        dv_residual = self.residual_network(state_normalized)

        # Total acceleration
        dv_total = dv_known + dv_residual

        # Integration step
        x, v = state[:, 0:1], state[:, 1:2]
        x_next = x + self.dt * v
        v_next = v + self.dt * dv_total

        next_state = torch.cat([x_next, v_next], dim=1)
        next_state_normalized = 2 * (next_state - min_vals) / range_vals - 1

        return next_state_normalized
