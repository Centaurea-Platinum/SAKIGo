from __future__ import annotations

import io
import json
import hashlib
import sqlite3
import tarfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from sakigo.constants import BOARD_PLANE_COUNT, DISTILLATION_SCHEMA_VERSION
from sakigo.data.records import record_from_json
from sakigo.generate.book import (
    assign_validated_histories,
    count_book_target_eligible,
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
from sakigo.generate.book_catalog import (
    BOOK_SPECS,
    allocate_global_random_sample,
)
from sakigo.generate.multi_book_distillation import _run_parallel
import sakigo.generate.multi_book_distillation as multi_book_module
from sakigo.generate.record_builder import build_book_training_record
from sakigo.generate.targets import ConcreteBookMove, book_budget, book_policy
from sakigo.rulesets import BLACK, WHITE, ruleset_from_name, ruleset_from_overrides
from sakigo.train.auto_book_suite import _manifest_shards, _sample_command


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


def _synthetic_archive(path: Path, board_size: int = 9) -> None:
    area = board_size * board_size
    empty = [0] * area
    child = [0] * area
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


def test_variable_board_japanese_book_record(tmp_path: Path) -> None:
    archive = tmp_path / "book7jp.tar.gz"
    database = tmp_path / "book7jp.sqlite"
    _synthetic_archive(archive, board_size=7)
    ruleset = ruleset_from_overrides(ruleset="japanese", komi=8.0)
    assert index_book_archive(archive, database) == {"parsed": 2, "rejected": 0}
    report = assign_validated_histories(
        database, board_size=7, ruleset=ruleset
    )
    assert report["valid"] == 2 and report["rejected"] == 0
    assert count_book_target_eligible(database, board_size=7) == 2
    freeze_uniform_sample(
        database,
        train_count=1,
        validation_count=1,
        seed=17,
        board_size=7,
    )
    task = next(iter_frozen_sample(database, split="train"))
    record = build_book_training_record(
        task,
        ruleset=ruleset,
        board_size=7,
        book_id="book7x7jp-test",
    )
    assert record["board_size"] == 7
    assert len(record["policy"]) == len(record["budget"]) == 50
    assert record["ruleset"]["komi"] == 8.0
    assert record["source"]["book"] == "book7x7jp-test"
    record_from_json(record)


@pytest.mark.parametrize("board_size", [7, 8, 9])
def test_book_preprocessing_is_identical_across_board_sizes(board_size: int) -> None:
    ruleset = ruleset_from_overrides(ruleset="tromp-taylor", komi=7.0)
    task = {
        "node_id": f"empty-{board_size}",
        "history": [],
        "moves": [
            {"xy": [[0, 0]], "ssM": -1.24, "wl": -0.4, "av": 30},
            {"xy": [[1, 0]], "ssM": -1.20, "wl": -0.2, "av": 10},
            {"move": "other", "ssM": -99, "wl": -1, "av": 10_000},
        ],
        "page_to_history_symmetry": 0,
        "split": "train",
        "task_id": f"book-{board_size}",
        "task_index": 0,
    }
    record = build_book_training_record(
        task,
        ruleset=ruleset,
        board_size=board_size,
        book_id=f"book{board_size}test",
    )
    area = board_size * board_size
    assert record["score"] == pytest.approx(1.0 / area)
    assert record["policy"][0:2] == pytest.approx([0.5, 0.5])
    assert record["budget"][0:2] == pytest.approx([0.75, 0.25])
    assert sum(record["budget"][2:]) == pytest.approx(0.0)
    record_from_json(record)


def test_sample_allocation_balances_validation_and_keeps_training_global() -> None:
    specs = BOOK_SPECS[:2]
    capacities = {specs[0].book_id: 999, specs[1].book_id: 1}
    first = allocate_global_random_sample(
        train_total=400,
        validation_total=100,
        capacities=capacities,
        specs=specs,
        seed=29,
    )
    second = allocate_global_random_sample(
        train_total=400,
        validation_total=100,
        capacities=capacities,
        specs=specs,
        seed=29,
    )
    assert first == second
    assert sum(value["train"] for value in first.values()) == 400
    assert sum(value["validation"] for value in first.values()) == 100
    assert first[specs[1].book_id]["validation"] == 1
    assert first[specs[1].book_id]["train"] == 0
    assert first[specs[0].book_id]["total"] != first[specs[1].book_id]["total"]


def test_validation_allocation_requires_every_selected_cohort() -> None:
    specs = BOOK_SPECS[:2]
    with pytest.raises(ValueError, match="every selected"):
        allocate_global_random_sample(
            train_total=1,
            validation_total=1,
            capacities={spec.book_id: 10 for spec in specs},
            specs=specs,
            seed=7,
        )


def test_default_books_receive_nearly_equal_validation_cohorts() -> None:
    allocation = allocate_global_random_sample(
        train_total=120,
        validation_total=100,
        capacities={spec.book_id: 1_000 for spec in BOOK_SPECS},
        specs=BOOK_SPECS,
        seed=41,
    )
    validation = [allocation[spec.book_id]["validation"] for spec in BOOK_SPECS]
    assert all(count > 0 for count in validation)
    assert max(validation) - min(validation) <= 1
    assert sum(value["train"] for value in allocation.values()) == 120


def test_per_book_orchestration_actually_overlaps_workers() -> None:
    specs = BOOK_SPECS[:2]
    barrier = threading.Barrier(2, timeout=2.0)

    def operation(spec):
        barrier.wait()
        return spec.board_size

    results = _run_parallel(specs, 2, operation)
    assert results == {spec.book_id: spec.board_size for spec in specs}


def test_auto_suite_selects_matching_generation_pipeline(tmp_path: Path) -> None:
    report = tmp_path / "book_index_report.json"
    report.write_text(
        json.dumps({"books": [{"book_id": "a"}, {"book_id": "b"}]}),
        encoding="utf-8",
    )
    command = _sample_command(tmp_path, report)
    assert "sakigo.generate.multi_book_distillation" in command
    assert command[-1] == "a,b"

    report.write_text(json.dumps({"database": "book_index.sqlite"}), encoding="utf-8")
    legacy = _sample_command(tmp_path, report)
    assert "sakigo.generate.book_distillation" in legacy

    train = tmp_path / "train.jsonl.zst"
    validation = tmp_path / "validation.jsonl.zst"
    train.touch()
    validation.touch()
    manifest = {
        "train_shards": [{"path": str(train), "records": 1}],
        "validation_shards": [str(validation)],
    }
    assert _manifest_shards(
        manifest, "train_shards", require_identity=False
    ) == (train.resolve(),)
    assert _manifest_shards(
        manifest, "validation_shards", require_identity=False
    ) == (validation.resolve(),)


def test_prepare_uses_only_manifest_listed_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "run"
    train = run_dir / "dataset" / "train" / "listed.jsonl.zst"
    stale = run_dir / "dataset" / "train" / "stale.jsonl.zst"
    validation = run_dir / "dataset" / "validation" / "listed.jsonl.zst"
    for path in (train, stale, validation):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"placeholder")
    def identity(path: Path) -> dict[str, object]:
        return {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (run_dir / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "pipeline_version": multi_book_module.PIPELINE_VERSION,
                "sample_allocation_version": multi_book_module.SAMPLE_ALLOCATION_VERSION,
                "state": "complete",
                "selection_seed": 17,
                "train_records": 1,
                "validation_records": 1,
                "train_shards": [
                    {"path": str(train), "records": 1, **identity(train)}
                ],
                "validation_shards": [
                    {
                        "path": str(validation),
                        "records": 1,
                        **identity(validation),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    seen: dict[str, object] = {}

    def fake_prepare(data, out_dir, **kwargs):
        seen["data"] = data
        seen["validation_data"] = kwargs["validation_data"]
        seen["seed"] = kwargs["seed"]
        return {
            "groups": [
                {"split": "train", "count": 1},
                {"split": "val", "count": 1},
            ],
            "ruleset_keys": [],
        }

    monkeypatch.setattr(multi_book_module, "prepare_tensor_shards", fake_prepare)
    multi_book_module.stage_prepare(
        SimpleNamespace(run_dir=run_dir, force_prepare=False), ()
    )
    assert seen == {
        "data": [train.resolve()],
        "validation_data": [validation.resolve()],
        "seed": 17,
    }


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
