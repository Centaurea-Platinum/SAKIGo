"""Map-style dataset + balanced batch sampler over prepared tensor shards.

Standard PyTorch pieces: `PreparedDataset` (memmap-backed `Dataset`),
`RulesetBalancedBatchSampler` (one board size per batch, near-equal ruleset
counts, without-replacement cycling per (size, ruleset)), `collate_prepared`
(contract batch layout), and `make_dataloader` (workers + pin_memory).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from sakigo.data.prepare import load_manifest
from sakigo.data.records import _d4_transform_planes

_ARRAY_NAMES = (
    "board_planes",
    "rule_features",
    "ply",
    "ruleset_code",
    "wdl",
    "wdl_mask",
    "score",
    "score_mask",
    "ownership",
    "ownership_mask",
    "policy",
    "policy_mask",
    "budget",
    "budget_mask",
    "legal_mask",
    "legal_available",
)


class _Group:
    """Lazily-opened memmaps for one (split, board_size) directory."""

    def __init__(self, directory: Path, board_size: int, count: int) -> None:
        self.directory = directory
        self.board_size = board_size
        self.count = count
        self._arrays: dict[str, np.ndarray] | None = None

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        if self._arrays is None:
            self._arrays = {
                name: np.load(self.directory / f"{name}.npy", mmap_mode="r")
                for name in _ARRAY_NAMES
            }
        return self._arrays


class PreparedDataset(Dataset[dict[str, np.ndarray]]):
    """Samples from prepared shards; global index spans all groups of one split.

    With augment_d4=True a random dihedral transform is applied per fetch
    (pass entry stays last). Worker RNG is derived from the DataLoader's
    per-worker torch seed, so runs are reproducible given a seeded loader.
    """

    def __init__(self, prepared_dir: Path, split: str, augment_d4: bool = False) -> None:
        manifest = load_manifest(prepared_dir)
        self.split = split
        self.augment_d4 = augment_d4
        self.ruleset_keys: list[str] = list(manifest["ruleset_keys"])
        self.groups: list[_Group] = []
        for group in manifest["groups"]:
            if group["split"] != split:
                continue
            self.groups.append(
                _Group(prepared_dir / group["directory"], int(group["board_size"]), int(group["count"]))
            )
        if not self.groups:
            raise ValueError(f"no groups for split {split!r} in {prepared_dir}")
        self._offsets: list[int] = []
        total = 0
        for group in self.groups:
            self._offsets.append(total)
            total += group.count
        self._total = total
        self._rng: random.Random | None = None

    def __len__(self) -> int:
        return self._total

    @property
    def board_sizes(self) -> list[int]:
        return sorted({group.board_size for group in self.groups})

    def group_of(self, index: int) -> tuple[_Group, int]:
        if not 0 <= index < self._total:
            raise IndexError(index)
        position = 0
        for offset, group in zip(self._offsets, self.groups):
            if index < offset + group.count:
                return group, index - offset
            position = offset
        raise IndexError(index)  # pragma: no cover - unreachable

    def board_size_of(self, index: int) -> int:
        group, _ = self.group_of(index)
        return group.board_size

    def ruleset_code_of(self, index: int) -> int:
        group, row = self.group_of(index)
        return int(group.arrays["ruleset_code"][row])

    def _worker_rng(self) -> random.Random:
        if self._rng is None:
            worker = get_worker_info()
            seed = torch.initial_seed() if worker is None else worker.seed
            self._rng = random.Random(seed & 0xFFFF_FFFF_FFFF_FFFF)
        return self._rng

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        group, row = self.group_of(index)
        arrays = group.arrays
        sample = {name: np.asarray(arrays[name][row]) for name in _ARRAY_NAMES}
        sample["board_size"] = np.int64(group.board_size)
        if self.augment_d4:
            transform = self._worker_rng().randrange(8)
            if transform:
                size = group.board_size
                sample["board_planes"] = _d4_transform_planes(sample["board_planes"], transform)
                for key in ("ownership",):
                    sample[key] = _d4_transform_planes(
                        sample[key].reshape(size, size), transform
                    ).reshape(-1)
                for key in ("policy", "budget", "legal_mask"):
                    vector = sample[key].copy()
                    vector[:-1] = _d4_transform_planes(
                        vector[:-1].reshape(size, size), transform
                    ).reshape(-1)
                    sample[key] = vector
        return sample


class RulesetBalancedBatchSampler(Sampler[list[int]]):
    """One board size per batch; near-equal per-batch ruleset counts.

    Per (board size, ruleset) index bags are sampled without replacement and
    reshuffled independently on exhaustion (epoch-like coverage). Board size
    is chosen per batch by the given weights. Infinite by default (step-based
    training); pass `length` for a bounded sampler.
    """

    def __init__(
        self,
        dataset: PreparedDataset,
        batch_size: int,
        seed: int,
        board_weights: Mapping[int, float] | None = None,
        length: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.length = length
        self._rng = random.Random(seed)
        # bags[(size, ruleset_code)] = list of global indices
        bags: dict[tuple[int, int], list[int]] = {}
        offset = 0
        for group in dataset.groups:
            codes = np.asarray(group.arrays["ruleset_code"])
            for code in np.unique(codes):
                indices = (np.nonzero(codes == code)[0] + offset).tolist()
                bags[(group.board_size, int(code))] = indices
            offset += group.count
        self._pools = bags
        self._queues: dict[tuple[int, int], list[int]] = {}
        self._sizes = sorted({size for size, _ in bags})
        self._weights = [
            (board_weights or {}).get(size, 1.0) for size in self._sizes
        ]
        if any(weight <= 0 for weight in self._weights):
            raise ValueError("board weights must be positive")

    def _pop(self, key: tuple[int, int]) -> int:
        queue = self._queues.get(key)
        if not queue:
            queue = self._pools[key].copy()
            self._rng.shuffle(queue)
            self._queues[key] = queue
        return queue.pop()

    def _next_batch(self) -> list[int]:
        size = self._rng.choices(self._sizes, weights=self._weights, k=1)[0]
        codes = [code for board, code in self._pools if board == size]
        order: list[int] = []
        while len(order) < self.batch_size:
            cycle = codes.copy()
            self._rng.shuffle(cycle)
            order.extend(cycle[: self.batch_size - len(order)])
        return [self._pop((size, code)) for code in order]

    def __iter__(self) -> Iterator[list[int]]:
        emitted = 0
        while self.length is None or emitted < self.length:
            yield self._next_batch()
            emitted += 1

    def __len__(self) -> int:
        if self.length is None:
            raise TypeError("infinite sampler has no length")
        return self.length


class FixedBatchSampler(Sampler[list[int]]):
    """Replays a precomputed batch list; every iteration yields identical batches.

    Used for fixed-subset validation: freeze the first `count` batches of a
    seeded RulesetBalancedBatchSampler so every evaluation measures the same
    samples (smooth step-to-step deltas, at the cost of never covering the rest
    of the val set).
    """

    def __init__(self, batches: list[list[int]]) -> None:
        if not batches:
            raise ValueError("FixedBatchSampler needs at least one batch")
        self.batches = batches

    @classmethod
    def freeze(cls, sampler: RulesetBalancedBatchSampler, count: int) -> "FixedBatchSampler":
        iterator = iter(sampler)
        return cls([next(iterator) for _ in range(count)])

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def collate_prepared(samples: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    """Assemble the contract batch layout from prepared samples."""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    board_size = int(samples[0]["board_size"])
    if any(int(sample["board_size"]) != board_size for sample in samples):
        raise ValueError("batches must contain one board size")

    def stack(name: str, dtype: type | None = None) -> torch.Tensor:
        array = np.stack([sample[name] for sample in samples])
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return torch.from_numpy(array)

    batch: dict[str, torch.Tensor] = {
        "board": stack("board_planes", np.float32),
        "rules": stack("rule_features", np.float32),
        "ply": stack("ply"),
        "wdl_target": stack("wdl"),
        "wdl_mask": stack("wdl_mask"),
        "score_target": stack("score").unsqueeze(1),
        "score_mask": stack("score_mask"),
        "ownership_target": stack("ownership"),
        "ownership_mask": stack("ownership_mask"),
        "policy_target": stack("policy"),
        "policy_mask": stack("policy_mask"),
        "budget_target": stack("budget"),
        "budget_mask": stack("budget_mask"),
        "legal_mask": stack("legal_mask"),
        "legal_mask_available": stack("legal_available"),
    }
    return batch


def make_dataloader(
    dataset: PreparedDataset,
    batch_sampler: Sampler[list[int]],
    *,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 0,
    persistent_workers: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_prepared,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        persistent_workers=persistent_workers and num_workers > 0,
    )
