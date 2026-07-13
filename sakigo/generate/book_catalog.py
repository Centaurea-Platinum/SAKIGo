"""Catalog of the small-board KataGo books used for mixed distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from bisect import bisect_right
from collections import Counter
import random
from typing import Iterable, Mapping

from sakigo.rulesets import RulesetSpec, ruleset_from_overrides


ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_ROOT = ROOT / "Distillation" / "downloads"
DOWNLOAD_BASE_URL = "https://katagobooks.org/downloads"


@dataclass(frozen=True)
class BookSpec:
    book_id: str
    filename: str
    board_size: int
    ruleset_name: str
    komi: float
    expected_bytes: int

    @property
    def url(self) -> str:
        return f"{DOWNLOAD_BASE_URL}/{self.filename}"

    @property
    def archive(self) -> Path:
        return DOWNLOAD_ROOT / self.filename

    def ruleset(self) -> RulesetSpec:
        return ruleset_from_overrides(ruleset=self.ruleset_name, komi=self.komi)

    def metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "url": self.url,
            "archive": str(self.archive.resolve()),
            "ruleset": self.ruleset().metadata(),
        }


BOOK_SPECS = (
    BookSpec(
        "book9x9tt-20260226",
        "book9x9tt-20260226.tar.gz",
        9,
        "tromp-taylor",
        7.0,
        1_405_673_423,
    ),
    BookSpec(
        "book9x9jp-20260226",
        "book9x9jp-20260226.tar.gz",
        9,
        "japanese",
        6.0,
        1_356_442_873,
    ),
    BookSpec(
        "book8x8tt-20211114",
        "book8ttb40s9854-20211114.tar.gz",
        8,
        "tromp-taylor",
        10.0,
        1_208_394_193,
    ),
    BookSpec(
        "book8x8jp-20211114",
        "book8jpb40s9854-20211114.tar.gz",
        8,
        "japanese",
        9.0,
        1_212_339_474,
    ),
    BookSpec(
        "book7x7tt-20210806",
        "book7ttb40s9435-20210806.tar.gz",
        7,
        "tromp-taylor",
        9.0,
        259_534_246,
    ),
    BookSpec(
        "book7x7jp-20210806",
        "book7jpb40s9435-20210806.tar.gz",
        7,
        "japanese",
        8.0,
        257_585_581,
    ),
)

BOOK_BY_ID = {spec.book_id: spec for spec in BOOK_SPECS}


def resolve_books(raw: str | Iterable[str] | None = None) -> tuple[BookSpec, ...]:
    if raw is None:
        return BOOK_SPECS
    names = raw.split(",") if isinstance(raw, str) else list(raw)
    requested = [name.strip() for name in names if name.strip()]
    if not requested or requested == ["all"]:
        return BOOK_SPECS
    unknown = sorted(set(requested) - BOOK_BY_ID.keys())
    if unknown:
        available = ", ".join(BOOK_BY_ID)
        raise ValueError(f"unknown books {unknown}; available: {available}")
    if len(requested) != len(set(requested)):
        raise ValueError("book list must not contain duplicates")
    return tuple(BOOK_BY_ID[name] for name in requested)


def allocate_global_random_sample(
    *,
    train_total: int,
    validation_total: int,
    capacities: Mapping[str, int],
    specs: tuple[BookSpec, ...],
    seed: int,
) -> dict[str, dict[str, int]]:
    """Allocate balanced validation cohorts, then sample training globally.

    Each catalog book is one board-size/ruleset cohort. Validation is spread as
    evenly as capacities allow and includes every selected cohort. Training is
    still weighted only by the remaining eligible populations, with no quota.
    Exact split totals remain deterministic.
    """

    if train_total <= 0 or validation_total <= 0:
        raise ValueError("sample totals must be positive")
    if not specs:
        raise ValueError("at least one book is required")
    counts = [int(capacities.get(spec.book_id, 0)) for spec in specs]
    if any(count < 0 for count in counts):
        raise ValueError("book capacities must be non-negative")
    total_capacity = sum(counts)
    requested = train_total + validation_total
    if requested > total_capacity:
        raise ValueError(
            f"requested {requested:,} samples from {total_capacity:,} eligible nodes"
        )
    if validation_total < len(specs):
        raise ValueError(
            "validation_total must provide at least one record for every "
            "selected board-size/ruleset cohort"
        )
    if any(count == 0 for count in counts):
        empty = [spec.book_id for spec, count in zip(specs, counts) if count == 0]
        raise ValueError(f"validation cohorts have no eligible records: {empty}")

    # A seeded tie order makes remainder placement deterministic without always
    # favoring the first catalog entry. Repeatedly fill the least-populated
    # cohort, skipping any cohort that has reached its capacity.
    tie_order = list(range(len(specs)))
    random.Random(seed + 1).shuffle(tie_order)
    validation_counts = [0] * len(specs)
    for _ in range(validation_total):
        eligible = [index for index in tie_order if validation_counts[index] < counts[index]]
        if not eligible:
            raise ValueError("validation sample exceeds eligible cohort capacity")
        minimum = min(validation_counts[index] for index in eligible)
        selected_index = next(
            index for index in eligible if validation_counts[index] == minimum
        )
        validation_counts[selected_index] += 1

    remaining_counts = [
        count - validation_counts[index] for index, count in enumerate(counts)
    ]
    remaining_capacity = sum(remaining_counts)
    if train_total > remaining_capacity:
        raise ValueError(
            f"requested {train_total:,} training samples after reserving validation "
            f"from {remaining_capacity:,} remaining nodes"
        )
    cumulative: list[int] = []
    running = 0
    for count in remaining_counts:
        running += count
        cumulative.append(running)
    selected = random.Random(seed).sample(range(remaining_capacity), train_total)

    def allocation(values: list[int]) -> Counter[int]:
        return Counter(bisect_right(cumulative, value) for value in values)

    train = allocation(selected)
    return {
        spec.book_id: {
            "train": train[index],
            "validation": validation_counts[index],
            "total": train[index] + validation_counts[index],
        }
        for index, spec in enumerate(specs)
    }
