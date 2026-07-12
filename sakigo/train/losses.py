from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sakigo.constants import HEADS


@dataclass(frozen=True)
class LossWeights:
    wdl: float = 1.0
    score: float = 1.0
    policy: float = 1.0
    budget: float = 1.0

    def as_dict(self) -> dict[str, float]:
        return {
            "wdl": self.wdl,
            "score": self.score,
            "policy": self.policy,
            "budget": self.budget,
        }


def masked_soft_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Branchless: an all-false mask yields 0 with zero gradients (no host sync, graph-capturable).
    log_probs = F.log_softmax(logits, dim=-1)
    per_record = -(target * log_probs).sum(dim=-1)
    weights = mask.float()
    return (per_record * weights).sum() / weights.sum().clamp_min(1.0)


def masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    per_record = F.smooth_l1_loss(
        prediction.reshape(target.shape),
        target,
        reduction="none",
    ).reshape(target.shape[0], -1).mean(dim=-1)
    weights = mask.float()
    return (per_record * weights).sum() / weights.sum().clamp_min(1.0)


def compute_head_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        "wdl": masked_soft_cross_entropy(
            output["wdl_logits"],
            batch["wdl_target"],
            batch["wdl_mask"],
        ),
        "score": masked_smooth_l1(
            output["score"],
            batch["score_target"],
            batch["score_mask"],
        ),
        "policy": masked_soft_cross_entropy(
            output["policy_logits"],
            batch["policy_target"],
            batch["policy_mask"],
        ),
        "budget": masked_soft_cross_entropy(
            output["budget_logits"],
            batch["budget_target"],
            batch["budget_mask"],
        ),
    }


def weighted_total_loss(
    head_losses: dict[str, torch.Tensor],
    weights: LossWeights,
) -> torch.Tensor:
    weight_map = weights.as_dict()
    total = None
    for head in HEADS:
        value = head_losses[head] * float(weight_map[head])
        total = value if total is None else total + value
    if total is None:
        raise ValueError("no losses were provided")
    return total
