"""Generation plan: (board size × ruleset × komi) variants with exact quotas.

Ported behavior-identically from Training/generate_katago_phase1.py.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from sakigo.rulesets import RulesetSpec, ruleset_from_name

DEFAULT_SAMPLES_PER_COMBINATION = 2**13
MAX_BOARD_SIZE = 19


@dataclass(frozen=True)
class GenerationVariant:
    board_size: int
    ruleset: RulesetSpec

    def key(self) -> str:
        return f"{self.board_size}|{self.ruleset.key()}"

    def metadata(self) -> dict[str, Any]:
        return {"board_size": self.board_size, "ruleset": self.ruleset.metadata()}


@dataclass(frozen=True)
class GenerationPlan:
    variants: list[GenerationVariant]
    quotas: dict[str, int]
    quota_mode: str
    samples_per_combination: int | None

    @property
    def target_samples(self) -> int:
        return sum(self.quotas.values())


class GenerationSchedule:
    def __init__(self, plan: GenerationPlan, rng: random.Random) -> None:
        self.plan = plan
        self.rng = rng
        self.written = {variant.key(): 0 for variant in plan.variants}
        self.reserved = {variant.key(): 0 for variant in plan.variants}
        self.total_written = 0

    def can_reserve(self, variant: GenerationVariant) -> bool:
        key = variant.key()
        return self.written[key] + self.reserved[key] < self.plan.quotas[key]

    def choose_variant(self) -> GenerationVariant | None:
        available = [variant for variant in self.plan.variants if self.can_reserve(variant)]
        if not available:
            return None
        return self.rng.choice(available)

    def reserve(self, variant: GenerationVariant) -> bool:
        if not self.can_reserve(variant):
            return False
        self.reserved[variant.key()] += 1
        return True

    def complete(self, variant: GenerationVariant, *, success: bool) -> bool:
        key = variant.key()
        if self.reserved[key] <= 0:
            raise RuntimeError(f"generation schedule reservation underflow for {key}")
        self.reserved[key] -= 1
        if not success:
            return False
        if self.written[key] >= self.plan.quotas[key]:
            raise RuntimeError(f"generation schedule quota overflow for {key}")
        self.written[key] += 1
        self.total_written += 1
        return True


def parse_board_sizes(raw: str) -> list[int]:
    sizes = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--board-sizes must include at least one size")
    if any(size <= 0 or size > MAX_BOARD_SIZE for size in sizes):
        raise ValueError(f"--board-sizes entries must be in [1, {MAX_BOARD_SIZE}]")
    if len(set(sizes)) != len(sizes):
        raise ValueError("--board-sizes must not contain duplicates")
    return sizes


def parse_komis(raw: str, default: float) -> list[float]:
    if not raw.strip():
        return [default]
    komis = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not komis:
        raise ValueError("--komis must include at least one value")
    if any(not math.isfinite(komi) for komi in komis):
        raise ValueError("--komis must contain only finite values")
    if len(set(komis)) != len(komis):
        raise ValueError("--komis must not contain duplicates")
    return komis


def parse_ruleset_names(raw: str) -> list[str]:
    names = [part.strip() for part in raw.split(",") if part.strip()]
    if not names:
        raise ValueError("--rulesets must include at least one ruleset")
    canonical = [ruleset_from_name(name).name for name in names]
    if len(set(canonical)) != len(canonical):
        raise ValueError("--rulesets must not contain duplicates")
    return canonical


def build_generation_plan(
    board_sizes: list[int],
    base_rulesets: list[RulesetSpec],
    komis: list[float],
    *,
    samples: int | None,
    samples_per_combination: int | None,
) -> GenerationPlan:
    variants = [
        GenerationVariant(board_size=board_size, ruleset=ruleset.with_komi(komi))
        for board_size in board_sizes
        for ruleset in base_rulesets
        for komi in komis
    ]
    if not variants:
        raise ValueError("generation plan must include at least one combination")
    if samples is not None:
        if samples <= 0:
            raise ValueError("--samples must be positive")
        if samples_per_combination is not None:
            raise ValueError("use either --samples or --samples-per-combination, not both")
        base_quota, remainder = divmod(samples, len(variants))
        quotas = {
            variant.key(): base_quota + (1 if index < remainder else 0)
            for index, variant in enumerate(variants)
        }
        return GenerationPlan(
            variants=variants, quotas=quotas, quota_mode="total", samples_per_combination=None
        )

    per_combination = (
        DEFAULT_SAMPLES_PER_COMBINATION
        if samples_per_combination is None
        else samples_per_combination
    )
    if per_combination <= 0:
        raise ValueError("--samples-per-combination must be positive")
    return GenerationPlan(
        variants=variants,
        quotas={variant.key(): per_combination for variant in variants},
        quota_mode="per_combination",
        samples_per_combination=per_combination,
    )
