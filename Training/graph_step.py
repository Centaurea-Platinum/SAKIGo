from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

import torch

from Training.losses import LossWeights, compute_head_losses, weighted_total_loss


class GraphedTrainStep:
    """Full-train-step CUDA graph capture: forward, losses, backward, clip, optimizer step.

    The first `warmup_steps` calls run eagerly on a side stream (they are real,
    counted training steps). The next call captures the whole step into a CUDA
    graph; it and every later call execute by copying the batch into static
    buffers and replaying the graph.

    Requirements: model and batches on CUDA, fixed batch shape (one board size,
    constant batch size), optimizer built with capturable=True, and no host
    syncs inside forward/loss (the masked losses are branchless).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_weights: LossWeights,
        grad_clip: float,
        amp_dtype: torch.dtype | None,
        warmup_steps: int = 3,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.grad_clip = grad_clip
        self.amp_dtype = amp_dtype
        self.warmup_steps = max(1, warmup_steps)
        self._calls = 0
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static: dict[str, torch.Tensor] | None = None
        self._static_output: dict[str, torch.Tensor] | None = None
        self._static_losses: dict[str, torch.Tensor] | None = None
        self._static_loss: torch.Tensor | None = None

    def _autocast(self) -> ContextManager:
        if self.amp_dtype is None:
            return nullcontext()
        # cache_enabled=False keeps warmup and captured kernels consistent (per AMP+graphs docs).
        return torch.autocast("cuda", dtype=self.amp_dtype, cache_enabled=False)

    def _forward_loss(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        with self._autocast():
            output = self.model(batch["board"], batch["rules"])
            head_losses = compute_head_losses(output, batch)
            loss = weighted_total_loss(head_losses, self.loss_weights)
        return output, head_losses, loss

    def _optimize(self, loss: torch.Tensor) -> None:
        loss.backward()
        if self.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

    def _eager_warmup_step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            # set_to_none=False keeps .grad buffers alive at stable addresses for capture.
            self.optimizer.zero_grad(set_to_none=False)
            output, head_losses, loss = self._forward_loss(batch)
            self._optimize(loss)
        torch.cuda.current_stream().wait_stream(side)
        # Detach so the side-stream autograd graph is freed before the next step
        # (a kept-alive AccumulateGrad node would warn and can disturb capture).
        return (
            {key: value.detach() for key, value in output.items()},
            {key: value.detach() for key, value in head_losses.items()},
            loss.detach(),
        )

    def _capture(self, batch: dict[str, torch.Tensor]) -> None:
        self._static = {key: value.clone() for key, value in batch.items()}
        torch.cuda.synchronize()
        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self.optimizer.zero_grad(set_to_none=False)
            output, head_losses, loss = self._forward_loss(self._static)
            self._optimize(loss)
        self._static_output = output
        self._static_losses = head_losses
        self._static_loss = loss

    def step(
        self, batch: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        self._calls += 1
        if self._graph is None:
            if self._calls <= self.warmup_steps:
                return self._eager_warmup_step(batch)
            self._capture(batch)
        assert self._static is not None
        if batch.keys() != self._static.keys():
            raise ValueError("cuda-graphs batch keys changed after capture")
        for key, value in batch.items():
            self._static[key].copy_(value, non_blocking=True)
        assert self._graph is not None
        self._graph.replay()
        assert self._static_output is not None and self._static_losses is not None and self._static_loss is not None
        return self._static_output, self._static_losses, self._static_loss
