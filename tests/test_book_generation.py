from __future__ import annotations

import io
import sqlite3
import tarfile
from pathlib import Path

import pytest

from sakigo.constants import BOARD_PLANE_COUNT, DISTILLATION_SCHEMA_VERSION
from sakigo.data.records import record_from_json
from sakigo.generate.book import (
    assign_validated_histories,
    freeze_uniform_sample,
    index_book_archive,
    iter_frozen_sample,
    page_constants,
    row_actions,
)
from sakigo.generate.book_distillation import (
    DEFAULT_TRAIN_SAMPLES,
    DEFAULT_VALIDATION_SAMPLES,
    parse_args,
)
from sakigo.generate.record_builder import build_book_training_record
from sakigo.generate.targets import ConcreteBookMove, book_budget, book_policy
from sakigo.rulesets import BLACK, WHITE, ruleset_from_name, ruleset_from_overrides


def _page(
    *,
    next_player: int,
    parent: str | None,
    board: list[int],
    links: str,
    moves: str,
) -> str:
    parent_literal = "null" if parent is None else repr(parent)
    board_literal = ",".join(map(str, board))
    return f"""<script>
    const nextPla={next_player};const pLink={parent_literal};const pSym=0;
    const board=[{board_literal}];const links={links};
    const linkSyms={{{'0:0' if links != '{}' else ''}}};const moves={moves};
    </script>"""


def _synthetic_archive(path: Path) -> None:
    empty = [0] * 81
    child = [0] * 81
    child[-1] = 1  # Canonical page orientation differs from history orientation.
    pages = {
        "html/index.html": _page(
            next_player=1,
            parent=None,
            board=empty,
            links="{0:'child.html'}",
            moves="[{xy:[[0,0]],ssM:0,wl:0,av:10},{move:'other',av:3}]",
        ),
        "html/child.html": _page(
            next_player=2,
            parent="index.html",
            board=child,
            links="{}",
            moves="[{move:'pass',ssM:0,wl:0,av:8},{move:'other',av:2}]",
        ),
    }
    with tarfile.open(path, "w:gz") as handle:
        for name, text in pages.items():
            payload = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))


def test_real_page_literal_shape_and_actions() -> None:
    raw = page_constants(
        """const nextPla=1;const pLink=null;const pSym=0;
        const board=[0,0,0];const links={15:'child.html'};const linkSyms={15:1};
        const moves=[{xy:[[6,1]],ssM:2.3,wl:0.5,av:10},{move:'other',av:4}];"""
    )
    assert raw["nextPla"] == 1 and raw["links"] == {15: "child.html"}
    assert row_actions(raw["moves"][0]) == (15,)
    assert row_actions(raw["moves"][1]) == ()


def test_archive_index_replay_sample_and_book_record(tmp_path: Path) -> None:
    archive = tmp_path / "book.tar.gz"
    database = tmp_path / "book.sqlite"
    _synthetic_archive(archive)
    assert index_book_archive(archive, database) == {"parsed": 2, "rejected": 0}
    report = assign_validated_histories(database)
    assert report["valid"] == 2 and report["rejected"] == 0
    # A structurally valid terminal/other-only page has no active training target.
    with sqlite3.connect(database) as connection:
        connection.execute(
            """INSERT INTO nodes
               (node_id,next_player,parent_symmetry,board_json,moves_json,ply,
                history_json,page_to_history_symmetry,valid)
               VALUES('terminal','B',0,'[]','[{"move":"other","av":10}]',
                      0,'[]',0,1)"""
        )
    assert freeze_uniform_sample(
        database, train_count=1, validation_count=1, seed=7
    ) == {"train": 1, "validation": 1, "total": 2}
    tasks = list(iter_frozen_sample(database, split="train")) + list(
        iter_frozen_sample(database, split="validation")
    )
    assert len(tasks) == 2 and {task["node_id"] for task in tasks} == {
        "html/index.html",
        "html/child.html",
    }
    ruleset = ruleset_from_overrides(ruleset="tromp-taylor", komi=7.0)
    for task in tasks:
        record = build_book_training_record(task, ruleset=ruleset)
        assert "ownership" not in record
        assert sum(record["policy"]) == pytest.approx(1.0)
        assert sum(record["budget"]) == pytest.approx(1.0)
        record_from_json(record)


def test_other_row_is_dropped_but_page_targets_remain() -> None:
    moves = [
        ConcreteBookMove((0, 8), score_lead=1.24, a_visits=30),
        ConcreteBookMove((4,), score_lead=1.26, a_visits=10),
        ConcreteBookMove((), score_lead=99, a_visits=1000, is_other=True),
    ]
    policy, score = book_policy(moves, to_move=BLACK, action_count=10)
    assert score == 1.5 and policy[4] == 1.0
    budget = book_budget(moves, action_count=10)
    assert budget[0] == budget[8] == pytest.approx(0.375)
    assert budget[4] == pytest.approx(0.25)


def test_equal_rounded_optima_are_uniform() -> None:
    moves = [
        ConcreteBookMove((0, 8), score_lead=-1.24, a_visits=1),
        ConcreteBookMove((4,), score_lead=-1.10, a_visits=1),
    ]
    policy, score = book_policy(moves, to_move=WHITE, action_count=10)
    assert score == -1.0
    assert policy[0] == policy[4] == policy[8] == pytest.approx(1 / 3)


def test_schema_v2_accepts_soft_book_policy() -> None:
    ruleset = ruleset_from_name("tromp-taylor")
    raw = {
        "schema_version": DISTILLATION_SCHEMA_VERSION,
        "board_size": 3,
        "ply": 0,
        "position_key": "position",
        "ruleset": ruleset.metadata(),
        "board_planes": [0.0] * (BOARD_PLANE_COUNT * 9),
        "rule_features": ruleset.rule_features(to_move=BLACK, captures=(0, 0), board_area=9),
        "policy": [0.5, 0.5] + [0.0] * 8,
        "legal_mask": [True] * 10,
    }
    assert record_from_json(raw).policy[:2].tolist() == pytest.approx([0.5, 0.5])


def test_cli_defaults_to_requested_power_of_two_counts(tmp_path: Path) -> None:
    args = parse_args(["sample", "--run-dir", str(tmp_path)])
    assert args.train_samples == DEFAULT_TRAIN_SAMPLES == 2**20
    assert args.validation_samples == DEFAULT_VALIDATION_SAMPLES == 2**12
