"""Map-style dataset + size-grouped batch sampler over prepared tensor shards.

Standard PyTorch pieces: `PreparedDataset` (memmap-backed `Dataset`),
`SizeGroupedBatchSampler` (one randomly selected board size per batch, with
    records shuffled together regardless of ruleset), `collate_prepared`
(contract batch layout), and `make_dataloader` (workers + pin_memory).
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass
import random
from pathlib import Path
from typing import Any, Iterator, Mapping

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
        group_index = bisect_right(self._offsets, index) - 1
        group = self.groups[group_index]
        return group, index - self._offsets[group_index]

    def board_size_of(self, index: int) -> int:
        group, _ = self.group_of(index)
        return group.board_size

    def ruleset_code_of(self, index: int) -> int:
        group, row = self.group_of(index)
        return int(group.arrays["ruleset_code"][row])

    def validation_cohorts(self) -> list["ValidationCohort"]:
        """Return exact index sets for every board-size/ruleset pair."""
        cohorts: list[ValidationCohort] = []
        offset = 0
        for group in self.groups:
            codes = np.asarray(group.arrays["ruleset_code"])
            for code in sorted(int(value) for value in np.unique(codes)):
                rows = np.flatnonzero(codes == code)
                cohorts.append(
                    ValidationCohort(
                        board_size=group.board_size,
                        ruleset_code=code,
                        ruleset_key=self.ruleset_keys[code],
                        indices=tuple(int(offset + row) for row in rows),
                    )
                )
            offset += group.count
        return sorted(cohorts, key=lambda item: (item.board_size, item.ruleset_key))

    def _worker_rng(self) -> random.Random:
        if self._rng is None:
            worker = get_worker_info()
            seed = torch.initial_seed() if worker is None else worker.seed
            self._rng = random.Random(seed & 0xFFFF_FFFF_FFFF_FFFF)
        return self._rng

    def augmentation_state_dict(self) -> dict[str, object] | None:
        if self._rng is None:
            return None
        return {"rng": self._rng.getstate()}

    def load_augmentation_state_dict(self, state: Mapping[str, object] | None) -> None:
        if state is None:
            self._rng = None
            return
        rng_state = state.get("rng")
        if not isinstance(rng_state, (tuple, list)) or len(rng_state) != 3:
            raise ValueError("invalid augmentation RNG state")
        self._rng = random.Random()
        self._rng.setstate((int(rng_state[0]), tuple(rng_state[1]), rng_state[2]))

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
                for key in ("policy", "budget", "legal_mask"):
                    vector = sample[key].copy()
                    vector[:-1] = _d4_transform_planes(
                        vector[:-1].reshape(size, size), transform
                    ).reshape(-1)
                    sample[key] = vector
        return sample

    def fetch_batch(self, indices: list[int]) -> dict[str, torch.Tensor] | list[dict[str, np.ndarray]]:
        """Fetch one sampler-emitted batch with fewer Python round trips.

        Our batch sampler already emits one board size per batch, so the common
        path can gather rows from a single group and return the contract batch
        directly. Augmentation remains per-sample and falls back to the scalar
        path so each record gets an independent transform.
        """
        if not indices:
            return []
        if self.augment_d4:
            return [self[index] for index in indices]

        first_group, _ = self.group_of(indices[0])
        rows: list[int] = []
        for index in indices:
            group, row = self.group_of(index)
            if group is not first_group:
                return [self[item] for item in indices]
            rows.append(row)
        arrays = first_group.arrays
        gathered = {name: np.asarray(arrays[name][rows]) for name in _ARRAY_NAMES}
        gathered["board_size"] = np.full(len(indices), first_group.board_size, dtype=np.int64)
        return batch_from_prepared_arrays(gathered)

    def __getitems__(self, indices: list[int]) -> dict[str, torch.Tensor] | list[dict[str, np.ndarray]]:
        """PyTorch DataLoader batched-fetch hook."""
        return self.fetch_batch(indices)


class SizeGroupedBatchSampler(Sampler[list[int]]):
    """Shuffle size-compatible batches and their records without rule quotas.

    A tensor batch must have one spatial shape, but rulesets are deliberately
    not balanced or stratified. All records for the selected size share one
    without-replacement queue, so their natural dataset proportions and random
    ordering determine each batch's ruleset mix. The default size schedule is a
    shuffled bag with tickets proportional to the available records, so uneven
    size populations are preserved and consecutive equal sizes are allowed.
    Explicit ``board_weights`` retain weighted independent size draws. Infinite
    by default; pass ``length`` for a bounded sampler.
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
        pools: dict[int, list[int]] = {}
        offset = 0
        for group in dataset.groups:
            pools.setdefault(group.board_size, []).extend(
                range(offset, offset + group.count)
            )
            offset += group.count
        self._pools = pools
        undersized = {
            size: len(values)
            for size, values in pools.items()
            if len(values) < batch_size
        }
        if undersized:
            details = ", ".join(
                f"{size}x{size}={count}" for size, count in sorted(undersized.items())
            )
            raise ValueError(
                f"batch_size {batch_size} exceeds prepared records for {details}"
            )
        self._queues: dict[int, list[int]] = {}
        self._sizes = sorted(pools)
        self._weighted_sizes = board_weights is not None
        self._weights = [
            (board_weights or {}).get(size, 1.0) for size in self._sizes
        ]
        self._size_queue: list[int] = []
        self._ticket_counts = {
            size: max(
                1,
                (len(self._pools[size]) + self.batch_size - 1) // self.batch_size,
            )
            for size in self._sizes
        }
        if any(weight <= 0 for weight in self._weights):
            raise ValueError("board weights must be positive")

    def _refill(self, size: int, *, exclude: set[int] | None = None) -> list[int]:
        queue = self._pools[size].copy()
        self._rng.shuffle(queue)
        if exclude:
            # ``pop`` reads from the end. Put fresh indices there so wrapping a
            # pool cannot repeat a record inside the same full batch.
            repeated = [index for index in queue if index in exclude]
            fresh = [index for index in queue if index not in exclude]
            queue = repeated + fresh
        self._queues[size] = queue
        return queue

    def state_dict(self) -> dict[str, object]:
        return {
            "sampler_version": 1,
            "batch_size": self.batch_size,
            "pool_counts": {size: len(self._pools[size]) for size in self._sizes},
            "weighted_sizes": self._weighted_sizes,
            "weights": list(self._weights),
            "rng": self._rng.getstate(),
            "queues": [
                {"size": size, "values": list(values)}
                for size, values in sorted(self._queues.items())
            ],
            "size_queue": list(self._size_queue),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if type(state.get("sampler_version")) is not int or state.get(
            "sampler_version"
        ) != 1:
            raise ValueError("unsupported sampler state version")
        if type(state.get("batch_size")) is not int or state.get(
            "batch_size"
        ) != self.batch_size:
            raise ValueError("sampler state batch size does not match")
        expected_counts = {size: len(self._pools[size]) for size in self._sizes}
        raw_counts = state.get("pool_counts")
        if not isinstance(raw_counts, Mapping) or any(
            type(size) is not int or type(count) is not int
            for size, count in raw_counts.items()
        ) or dict(raw_counts) != expected_counts:
            raise ValueError("sampler state does not match prepared data")
        if type(state.get("weighted_sizes")) is not bool or state.get(
            "weighted_sizes"
        ) != self._weighted_sizes:
            raise ValueError("sampler weighting mode does not match")
        raw_weights = state.get("weights")
        if (
            not isinstance(raw_weights, (tuple, list))
            or any(type(value) not in (int, float) for value in raw_weights)
            or [float(value) for value in raw_weights]
            != [float(value) for value in self._weights]
        ):
            raise ValueError("sampler board weights do not match")
        rng_state = state.get("rng")
        if not isinstance(rng_state, (tuple, list)) or len(rng_state) != 3:
            raise ValueError("invalid sampler RNG state")
        self._rng.setstate((int(rng_state[0]), tuple(rng_state[1]), rng_state[2]))
        queues: dict[int, list[int]] = {}
        raw_queues = state.get("queues", [])
        if not isinstance(raw_queues, list):
            raise ValueError("invalid sampler queue state")
        for item in raw_queues:
            if not isinstance(item, Mapping):
                raise ValueError("invalid sampler queue entry")
            raw_size = item.get("size")
            raw_values = item.get("values")
            if not isinstance(raw_size, int):
                raise ValueError("invalid sampler queue size")
            size = int(raw_size)
            if size not in self._pools or not isinstance(raw_values, (tuple, list)):
                raise ValueError(f"sampler state does not match prepared data for size {size}")
            if any(type(value) is not int for value in raw_values):
                raise ValueError(f"sampler queue contains a non-integer for size {size}")
            values = list(raw_values)
            if len(values) != len(set(values)) or not set(values).issubset(
                set(self._pools[size])
            ):
                raise ValueError(f"sampler queue contains an unknown index for size {size}")
            queues[size] = values
        self._queues = queues

        raw_size_queue = state.get("size_queue", [])
        if not isinstance(raw_size_queue, (tuple, list)):
            raise ValueError("invalid sampler size queue state")
        if any(type(size) is not int for size in raw_size_queue):
            raise ValueError("sampler size queue contains a non-integer")
        size_queue = list(raw_size_queue)
        if not set(size_queue).issubset(self._sizes):
            raise ValueError("sampler size queue contains an unknown board size")
        if self._weighted_sizes and size_queue:
            raise ValueError("weighted sampler state cannot contain size tickets")
        remaining_tickets = Counter(size_queue)
        if any(
            remaining_tickets[size] > self._ticket_counts[size]
            for size in self._sizes
        ):
            raise ValueError("sampler size queue contains too many tickets")
        self._size_queue = size_queue

    def _next_size(self) -> int:
        if self._weighted_sizes:
            return self._rng.choices(self._sizes, weights=self._weights, k=1)[0]
        if not self._size_queue:
            for size in self._sizes:
                self._size_queue.extend([size] * self._ticket_counts[size])
            self._rng.shuffle(self._size_queue)
        return self._size_queue.pop()

    def _next_batch(self) -> list[int]:
        size = self._next_size()
        batch: list[int] = []
        while len(batch) < self.batch_size:
            queue = self._queues.get(size)
            if not queue:
                queue = self._refill(size, exclude=set(batch))
            take = min(self.batch_size - len(batch), len(queue))
            batch.extend(queue.pop() for _ in range(take))
        return batch

    def __iter__(self) -> Iterator[list[int]]:
        emitted = 0
        while self.length is None or emitted < self.length:
            yield self._next_batch()
            emitted += 1

    def __len__(self) -> int:
        if self.length is None:
            raise TypeError("infinite sampler has no length")
        return self.length


# Compatibility for existing configs/checkpoints importing the old name.
RulesetBalancedBatchSampler = SizeGroupedBatchSampler


@dataclass(frozen=True)
class ValidationCohort:
    board_size: int
    ruleset_code: int
    ruleset_key: str
    indices: tuple[int, ...]

    @property
    def ruleset_name(self) -> str:
        return self.ruleset_key.split("|", 1)[0]

    @property
    def komi(self) -> str:
        return self.ruleset_key.rsplit("|", 1)[-1]

    @property
    def metric_name(self) -> str:
        komi = self.komi.replace("-", "neg_").replace(".", "p")
        return f"{self.board_size}x{self.board_size}_{self.ruleset_name}_komi_{komi}"


class GroupedValidationBatchSampler(Sampler[list[int]]):
    """Cover every validation cohort with homogeneous batches.

    A positive ``length`` caps the total number of batches, but it must leave at
    least one batch for every cohort. Fixed mode replays identical records at
    every evaluation; rotating mode advances independently within each cohort.
    """

    def __init__(
        self,
        dataset: PreparedDataset,
        batch_size: int,
        seed: int,
        *,
        length: int = 0,
        fixed: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if length < 0:
            raise ValueError("validation batch length must be non-negative")
        self.cohorts = dataset.validation_cohorts()
        if not self.cohorts:
            raise ValueError("validation data has no board-size/ruleset cohorts")
        if length and length < len(self.cohorts):
            raise ValueError(
                f"val_batches={length} cannot cover all {len(self.cohorts)} "
                "board-size/ruleset cohorts"
            )
        rng = random.Random(seed)
        self._batches: dict[tuple[int, int], list[list[int]]] = {}
        for cohort in self.cohorts:
            indices = list(cohort.indices)
            rng.shuffle(indices)
            self._batches[(cohort.board_size, cohort.ruleset_code)] = [
                indices[offset : offset + batch_size]
                for offset in range(0, len(indices), batch_size)
            ]
        schedule: list[tuple[int, int]] = []
        depth = 0
        while True:
            added = False
            for cohort in self.cohorts:
                key = (cohort.board_size, cohort.ruleset_code)
                if depth < len(self._batches[key]):
                    schedule.append(key)
                    added = True
            if not added:
                break
            depth += 1
        self._schedule = schedule[:length] if length else schedule
        self._fixed = fixed
        self._cursors = {key: 0 for key in self._batches}

    def __iter__(self) -> Iterator[list[int]]:
        occurrences: Counter[tuple[int, int]] = Counter()
        for key in self._schedule:
            batches = self._batches[key]
            if self._fixed:
                index = occurrences[key]
                occurrences[key] += 1
            else:
                index = self._cursors[key] % len(batches)
                self._cursors[key] += 1
            yield batches[index]

    def __len__(self) -> int:
        return len(self._schedule)


class FixedBatchSampler(Sampler[list[int]]):
    """Replays a precomputed batch list; every iteration yields identical batches.

    Used for fixed-subset validation: freeze the first `count` batches of a
    seeded SizeGroupedBatchSampler so every evaluation measures the same
    samples (smooth step-to-step deltas, at the cost of never covering the rest
    of the val set).
    """

    def __init__(self, batches: list[list[int]]) -> None:
        if not batches:
            raise ValueError("FixedBatchSampler needs at least one batch")
        self.batches = batches

    @classmethod
    def freeze(cls, sampler: SizeGroupedBatchSampler, count: int) -> "FixedBatchSampler":
        iterator = iter(sampler)
        return cls([next(iterator) for _ in range(count)])

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def batch_from_prepared_arrays(arrays: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Convert already-batched prepared arrays into the training batch contract."""

    def tensor(name: str, dtype: type | None = None) -> torch.Tensor:
        array = np.asarray(arrays[name])
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return torch.from_numpy(array)

    return {
        "board_size": tensor("board_size"),
        "ruleset_code": tensor("ruleset_code"),
        "board": tensor("board_planes", np.float32),
        "rules": tensor("rule_features", np.float32),
        "ply": tensor("ply"),
        "wdl_target": tensor("wdl", np.float32),
        "wdl_mask": tensor("wdl_mask"),
        "score_target": tensor("score", np.float32).unsqueeze(1),
        "score_mask": tensor("score_mask"),
        "policy_target": tensor("policy", np.float32),
        "policy_mask": tensor("policy_mask"),
        "budget_target": tensor("budget", np.float32),
        "budget_mask": tensor("budget_mask"),
        "legal_mask": tensor("legal_mask"),
        "legal_mask_available": tensor("legal_available"),
    }


def collate_prepared(
    samples: dict[str, torch.Tensor] | list[dict[str, np.ndarray]],
) -> dict[str, torch.Tensor]:
    """Assemble the contract batch layout from prepared samples."""
    if isinstance(samples, dict):
        return samples
    if not samples:
        raise ValueError("cannot collate an empty batch")
    board_size = int(samples[0]["board_size"])
    if any(int(sample["board_size"]) != board_size for sample in samples):
        raise ValueError("batches must contain one board size")

    def stack(name: str, dtype: type | None = None) -> np.ndarray:
        array = np.stack([sample[name] for sample in samples])
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    return batch_from_prepared_arrays(
        {
            "board_size": stack("board_size"),
            "ruleset_code": stack("ruleset_code"),
            "board_planes": stack("board_planes", np.float32),
            "rule_features": stack("rule_features", np.float32),
            "ply": stack("ply"),
            "wdl": stack("wdl", np.float32),
            "wdl_mask": stack("wdl_mask"),
            "score": stack("score", np.float32),
            "score_mask": stack("score_mask"),
            "policy": stack("policy", np.float32),
            "policy_mask": stack("policy_mask"),
            "budget": stack("budget", np.float32),
            "budget_mask": stack("budget_mask"),
            "legal_mask": stack("legal_mask"),
            "legal_available": stack("legal_available"),
        }
    )


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
