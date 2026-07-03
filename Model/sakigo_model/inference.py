from __future__ import annotations

import torch
from torch import nn


class SakiGoInference:
    """Frozen inference wrapper with optional bf16 and CUDA graph replay."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        use_cuda_graph: bool = False,
        warmup: int = 3,
    ) -> None:
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.dtype = dtype
        self.model = model.to(device=self.device, dtype=dtype).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.use_cuda_graph = use_cuda_graph
        self.warmup = warmup
        self._graph_key: tuple[object, ...] | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_board: torch.Tensor | None = None
        self._static_rules: torch.Tensor | None = None
        self._static_output: dict[str, torch.Tensor] | None = None
        if self.use_cuda_graph and self.device.type != "cuda":
            raise ValueError("CUDA graph replay requires a CUDA device")

    def _prepare(self, board: torch.Tensor, rules: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            board.to(device=self.device, dtype=self.dtype),
            rules.to(device=self.device, dtype=self.dtype),
        )

    def _key(self, board: torch.Tensor, rules: torch.Tensor) -> tuple[object, ...]:
        return (
            tuple(board.shape),
            tuple(rules.shape),
            self.dtype,
            self.device.type,
            self.device.index,
        )

    def _capture(self, board: torch.Tensor, rules: torch.Tensor) -> None:
        static_board = board.clone()
        static_rules = rules.clone()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(max(self.warmup, 1)):
                self.model(static_board, static_rules)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            static_output = self.model(static_board, static_rules)
        torch.cuda.synchronize(self.device)
        self._graph_key = self._key(board, rules)
        self._graph = graph
        self._static_board = static_board
        self._static_rules = static_rules
        self._static_output = static_output

    def __call__(self, board: torch.Tensor, rules: torch.Tensor) -> dict[str, torch.Tensor]:
        board, rules = self._prepare(board, rules)
        if not self.use_cuda_graph:
            with torch.no_grad():
                return self.model(board, rules)
        key = self._key(board, rules)
        if self._graph_key != key:
            self._capture(board, rules)
        if (
            self._graph is None
            or self._static_board is None
            or self._static_rules is None
            or self._static_output is None
        ):
            raise RuntimeError("CUDA graph was not captured")
        self._static_board.copy_(board)
        self._static_rules.copy_(rules)
        self._graph.replay()
        return {key: value.clone() for key, value in self._static_output.items()}
