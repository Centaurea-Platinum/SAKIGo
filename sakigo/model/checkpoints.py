"""Checkpoint interop helpers for the unified model."""

from __future__ import annotations

import torch


def remap_legacy_scalar_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map a legacy ScalarSakiGoModel state dict onto the unified group_size=1 SakiGoNet.

    The only shape difference is ScalarLinear1x1 weights [out, in] becoming
    GroupLinear1x1 weights [out, in, 1]. rule_mlp (nn.Linear) stays 2-D and
    must not be touched.
    """
    remapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if (
            value.ndim == 2
            and key.endswith(".weight")
            and not key.startswith("rule_mlp.")
        ):
            remapped[key] = value.unsqueeze(-1)
        else:
            remapped[key] = value
    return remapped
