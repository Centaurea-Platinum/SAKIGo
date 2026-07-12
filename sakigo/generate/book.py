"""Streaming KataGo book-page parsing and a resumable SQLite index."""

from __future__ import annotations

import ast
import json
import posixpath
import re
import sqlite3
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from sakigo.generate.game import index_from_coord

_CONST = re.compile(r"\bconst\s+([A-Za-z_]\w*)\s*=\s*(.*?);", re.DOTALL)
_OBJECT_KEY = re.compile(r"([,{]\s*)([A-Za-z_]\w*)\s*:")


def _javascript_literal(text: str) -> Any:
    text = _OBJECT_KEY.sub(r"\1'\2':", text.strip())
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\b(?:null|undefined)\b", "None", text)
    return ast.literal_eval(text)


def page_constants(text: str) -> dict[str, Any]:
    wanted = {"nextPla", "pLink", "pSym", "board", "links", "linkSyms", "moves"}
    output: dict[str, Any] = {}
    for name, literal in _CONST.findall(text):
        if name in wanted:
            output[name] = _javascript_literal(literal)
    return output


@dataclass(frozen=True)
class BookPage:
    node_id: str
    next_player: str
    parent_link: str | None
    parent_symmetry: int
    board: Any
    links: dict[int, str]
    link_symmetries: dict[int, int]
    moves: tuple[dict[str, Any], ...]


def parse_book_page(path: Path, root: Path) -> BookPage:
    return parse_book_text(
        path.read_text(encoding="utf-8", errors="replace"),
        path.relative_to(root).as_posix(),
    )


def parse_book_text(text: str, node_id: str) -> BookPage:
    raw = page_constants(text)
    missing = {"nextPla", "board", "links", "moves"} - raw.keys()
    if missing:
        raise ValueError(f"{node_id} is missing book constants: {sorted(missing)}")
    raw_links = raw["links"]
    if isinstance(raw_links, dict):
        links = {int(action): str(link) for action, link in raw_links.items()}
    else:
        links = {
            action: str(link)
            for action, link in enumerate(raw_links)
            if link is not None
        }
    raw_symmetries = raw.get("linkSyms", {})
    if isinstance(raw_symmetries, dict):
        link_symmetries = {
            int(action): int(value) for action, value in raw_symmetries.items()
        }
    else:
        link_symmetries = {
            action: int(value) for action, value in enumerate(raw_symmetries)
        }
    missing_symmetries = set(links) - set(link_symmetries)
    if missing_symmetries:
        raise ValueError(
            f"{node_id} has links without linkSyms: {sorted(missing_symmetries)}"
        )
    next_player_raw = raw["nextPla"]
    if next_player_raw in (1, "1", "B", "b"):
        next_player = "B"
    elif next_player_raw in (2, "2", -1, "-1", "W", "w"):
        next_player = "W"
    else:
        raise ValueError(f"{node_id} has invalid nextPla {next_player_raw!r}")
    return BookPage(
        node_id=node_id,
        next_player=next_player,
        parent_link=None if raw.get("pLink") in (None, "") else str(raw["pLink"]),
        parent_symmetry=int(raw.get("pSym", 0)),
        board=raw["board"],
        links=links,
        link_symmetries=link_symmetries,
        moves=tuple(dict(move) for move in raw["moves"]),
    )


def _resolve_link(parent_id: str, link: str) -> str:
    base = str(PurePosixPath(parent_id).parent)
    return posixpath.normpath(posixpath.join(base, link))


def _connect(database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS nodes(
          node_id TEXT PRIMARY KEY,
          next_player TEXT NOT NULL,
          parent_link TEXT,
          parent_symmetry INTEGER NOT NULL,
          board_json TEXT NOT NULL,
          moves_json TEXT NOT NULL,
          ply INTEGER,
          history_json TEXT,
          page_to_history_symmetry INTEGER,
          valid INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS edges(
          parent_id TEXT NOT NULL,
          action INTEGER NOT NULL,
          child_id TEXT NOT NULL,
          child_symmetry INTEGER NOT NULL,
          PRIMARY KEY(parent_id, action)
        );
        CREATE INDEX IF NOT EXISTS edges_child ON edges(child_id);
        """
    )
    return connection


def index_book_pages(book_root: Path, database: Path) -> dict[str, int]:
    """Parse pages without loading the multi-gigabyte book into memory."""

    connection = _connect(database)
    parsed = 0
    rejected = 0
    try:
        for path in book_root.rglob("*.html"):
            try:
                page = parse_book_page(path, book_root)
            except ValueError:
                rejected += 1
                continue
            connection.execute(
                """INSERT OR REPLACE INTO nodes
                   (node_id,next_player,parent_link,parent_symmetry,board_json,moves_json)
                   VALUES(?,?,?,?,?,?)""",
                (
                    page.node_id,
                    page.next_player,
                    page.parent_link,
                    page.parent_symmetry,
                    json.dumps(page.board, separators=(",", ":")),
                    json.dumps(page.moves, separators=(",", ":")),
                ),
            )
            connection.execute("DELETE FROM edges WHERE parent_id=?", (page.node_id,))
            for action, link in sorted(page.links.items()):
                symmetry = page.link_symmetries[action]
                connection.execute(
                    "INSERT INTO edges VALUES(?,?,?,?)",
                    (page.node_id, action, _resolve_link(page.node_id, link), symmetry),
                )
            parsed += 1
            if parsed % 4096 == 0:
                connection.commit()
        connection.commit()
    finally:
        connection.close()
    return {"parsed": parsed, "rejected": rejected}


def index_book_archive(archive: Path, database: Path) -> dict[str, int]:
    """Stream a compressed book directly into SQLite without extracting tiny files."""

    connection = _connect(database)
    parsed = 0
    rejected = 0
    try:
        with tarfile.open(archive, "r|gz") as handle:
            for member in handle:
                if not member.isfile() or not member.name.endswith(".html"):
                    continue
                source = handle.extractfile(member)
                if source is None:
                    rejected += 1
                    continue
                try:
                    page = parse_book_text(
                        source.read().decode("utf-8", errors="replace"), member.name
                    )
                except ValueError:
                    rejected += 1
                    continue
                connection.execute(
                    """INSERT OR REPLACE INTO nodes
                       (node_id,next_player,parent_link,parent_symmetry,board_json,moves_json)
                       VALUES(?,?,?,?,?,?)""",
                    (
                        page.node_id,
                        page.next_player,
                        page.parent_link,
                        page.parent_symmetry,
                        json.dumps(page.board, separators=(",", ":")),
                        json.dumps(page.moves, separators=(",", ":")),
                    ),
                )
                connection.execute("DELETE FROM edges WHERE parent_id=?", (page.node_id,))
                for action, link in sorted(page.links.items()):
                    connection.execute(
                        "INSERT INTO edges VALUES(?,?,?,?)",
                        (
                            page.node_id,
                            action,
                            _resolve_link(page.node_id, link),
                            page.link_symmetries[action],
                        ),
                    )
                parsed += 1
                if parsed % 4096 == 0:
                    connection.commit()
        connection.commit()
    finally:
        connection.close()
    return {"parsed": parsed, "rejected": rejected}


def assign_canonical_histories(database: Path, root_id: str | None = None) -> dict[str, int]:
    """Freeze one lexicographically stable parent history for every reachable node."""

    connection = _connect(database)
    try:
        connection.execute("UPDATE nodes SET ply=NULL, history_json=NULL, valid=0")
        if root_id is None:
            row = connection.execute(
                "SELECT node_id FROM nodes WHERE parent_link IS NULL ORDER BY node_id LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError("book index has no root")
            root_id = str(row[0])
        connection.execute(
            "UPDATE nodes SET ply=0,history_json='[]',valid=1 WHERE node_id=?", (root_id,)
        )
        frontier = [root_id]
        reached = 1
        while frontier:
            next_frontier: list[str] = []
            for parent_id in sorted(frontier):
                parent = connection.execute(
                    "SELECT ply,history_json,next_player FROM nodes WHERE node_id=?",
                    (parent_id,),
                ).fetchone()
                if parent is None:
                    continue
                ply, history_json, next_player = parent
                history = json.loads(history_json)
                for action, child_id in connection.execute(
                    "SELECT action,child_id FROM edges WHERE parent_id=? ORDER BY action,child_id",
                    (parent_id,),
                ):
                    coord = "pass" if action == 81 else _coord_9x9(int(action))
                    child_history = history + [[next_player, coord]]
                    changed = connection.execute(
                        """UPDATE nodes SET ply=?,history_json=?,valid=1
                           WHERE node_id=? AND ply IS NULL""",
                        (int(ply) + 1, json.dumps(child_history, separators=(",", ":")), child_id),
                    ).rowcount
                    if changed:
                        next_frontier.append(str(child_id))
                        reached += 1
            connection.commit()
            frontier = next_frontier
        total = int(connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
        return {"root_id": root_id, "reachable": reached, "unreachable": total - reached}
    finally:
        connection.close()


def assign_validated_histories(database: Path, root_id: str | None = None) -> dict[str, int]:
    """Assign histories, align page orientation, and reject replay disagreements."""

    from sakigo.generate.d4 import transform_action, transform_spatial
    from sakigo.generate.game import GeneratorGame
    from sakigo.rulesets import BLACK, ruleset_from_overrides
    import random

    ruleset = ruleset_from_overrides(ruleset="tromp-taylor", komi=7.0)
    connection = _connect(database)

    def validate(node_id: str, history: list[list[str]]) -> int | None:
        row = connection.execute(
            "SELECT next_player,board_json,moves_json FROM nodes WHERE node_id=?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        next_player, board_json, moves_json = row
        game = GeneratorGame(0, random.Random(0), 9, ruleset)
        for color, coord in history:
            expected = "B" if game.to_move == BLACK else "W"
            if str(color).upper() != expected:
                return None
            try:
                game.play(index_from_coord(str(coord), 9))
            except (ValueError, RuntimeError):
                return None
        expected_next = "B" if game.to_move == BLACK else "W"
        if expected_next != next_player:
            return None
        page_board = [int(value) for value in json.loads(board_json)]
        engine_board = [
            2 if int(value) == -1 else int(value) for value in game.engine.board()
        ]
        symmetries = [
            symmetry
            for symmetry in range(8)
            if transform_spatial(page_board, 9, symmetry) == engine_board
        ]
        if not symmetries:
            return None
        legal = game.legal_mask()
        moves = json.loads(moves_json)
        for symmetry in symmetries:
            actions = [
                transform_action(action, 9, symmetry)
                for move in moves
                if str(move.get("move", "")).lower() != "other"
                for action in row_actions(move)
            ]
            if all(legal[action] for action in actions):
                return symmetry
        return None

    try:
        connection.execute(
            "UPDATE nodes SET ply=NULL,history_json=NULL,page_to_history_symmetry=NULL,valid=0"
        )
        if root_id is None:
            row = connection.execute(
                "SELECT node_id FROM nodes WHERE parent_link IS NULL ORDER BY node_id LIMIT 1"
            ).fetchone()
            if row is None:
                raise ValueError("book index has no root")
            root_id = str(row[0])
        root_symmetry = validate(root_id, [])
        if root_symmetry is None:
            raise ValueError(f"book root {root_id} disagrees with local Tromp-Taylor replay")
        connection.execute(
            """UPDATE nodes SET ply=0,history_json='[]',page_to_history_symmetry=?,valid=1
               WHERE node_id=?""",
            (root_symmetry, root_id),
        )
        frontier = [root_id]
        valid_count = 1
        rejected = 0
        while frontier:
            next_frontier: list[str] = []
            for parent_id in sorted(frontier):
                parent = connection.execute(
                    """SELECT ply,history_json,next_player,page_to_history_symmetry
                       FROM nodes WHERE node_id=?""",
                    (parent_id,),
                ).fetchone()
                if parent is None:
                    continue
                ply, history_json, next_player, parent_symmetry = parent
                history = json.loads(history_json)
                for page_action, child_id in connection.execute(
                    "SELECT action,child_id FROM edges WHERE parent_id=? ORDER BY action,child_id",
                    (parent_id,),
                ):
                    child_row = connection.execute(
                        "SELECT valid,parent_link FROM nodes WHERE node_id=?", (child_id,)
                    ).fetchone()
                    if child_row is None or int(child_row[0]) != 0:
                        continue
                    parent_link = child_row[1]
                    if parent_link is not None and _resolve_link(str(child_id), parent_link) != parent_id:
                        continue
                    action = transform_action(int(page_action), 9, int(parent_symmetry))
                    coord = "pass" if action == 81 else _coord_9x9(action)
                    child_history = history + [[next_player, coord]]
                    child_symmetry = validate(str(child_id), child_history)
                    if child_symmetry is None:
                        connection.execute(
                            "UPDATE nodes SET valid=-1 WHERE node_id=?", (child_id,)
                        )
                        rejected += 1
                        continue
                    connection.execute(
                        """UPDATE nodes SET ply=?,history_json=?,page_to_history_symmetry=?,valid=1
                           WHERE node_id=?""",
                        (
                            int(ply) + 1,
                            json.dumps(child_history, separators=(",", ":")),
                            child_symmetry,
                            child_id,
                        ),
                    )
                    next_frontier.append(str(child_id))
                    valid_count += 1
            connection.commit()
            frontier = next_frontier
        total = int(connection.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])
        return {
            "root_id": root_id,
            "valid": valid_count,
            "rejected": rejected,
            "unreachable": total - valid_count - rejected,
        }
    finally:
        connection.close()


def _coord_9x9(action: int) -> str:
    from sakigo.generate.game import coord_from_index

    return coord_from_index(action, 9)


def row_actions(row: dict[str, Any], board_size: int = 9) -> tuple[int, ...]:
    if str(row.get("move", "")).lower() == "pass":
        return (board_size * board_size,)
    raw = row.get("xy")
    if raw is None:
        return ()
    if not isinstance(raw, (list, tuple)):
        raw = [raw]
    actions: list[int] = []
    index = 0
    while index < len(raw):
        value = raw[index]
        if isinstance(value, str):
            actions.append(index_from_coord(value, board_size))
            index += 1
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            x, y = map(int, value)
            actions.append(y * board_size + x)
            index += 1
        elif isinstance(value, int) and 0 <= value < board_size * board_size:
            actions.append(value)
            index += 1
        else:
            raise ValueError(f"unsupported book xy value {value!r}")
    return tuple(sorted(set(actions)))


def iter_index_nodes(database: Path, where: str = "ply IS NOT NULL") -> Iterator[dict[str, Any]]:
    connection = sqlite3.connect(database)
    try:
        for row in connection.execute(
            f"SELECT node_id,next_player,ply,history_json,moves_json FROM nodes WHERE {where} ORDER BY node_id"
        ):
            yield {
                "node_id": row[0],
                "next_player": row[1],
                "ply": row[2],
                "history": json.loads(row[3]),
                "moves": json.loads(row[4]),
            }
    finally:
        connection.close()


def iter_index_candidates(
    database: Path, *, leaves_only: bool = False
) -> Iterator[dict[str, Any]]:
    connection = sqlite3.connect(database)
    try:
        leaf_clause = (
            "AND NOT EXISTS(SELECT 1 FROM edges e WHERE e.parent_id=n.node_id)"
            if leaves_only
            else ""
        )
        query = f"""SELECT n.node_id,n.next_player,n.ply,n.history_json
                    FROM nodes n WHERE n.valid=1 AND n.ply IS NOT NULL {leaf_clause}
                    ORDER BY n.node_id"""
        for node_id, next_player, ply, history_json in connection.execute(query):
            yield {
                "node_id": node_id,
                "position_id": node_id,
                "next_player": next_player,
                "ply": ply,
                "history": json.loads(history_json),
            }
    finally:
        connection.close()


def enrich_book_tasks(database: Path, tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    connection = sqlite3.connect(database)
    output: list[dict[str, Any]] = []
    try:
        for task in tasks:
            enriched = dict(task)
            row = connection.execute(
                "SELECT moves_json,page_to_history_symmetry FROM nodes WHERE node_id=?",
                (task["node_id"],),
            ).fetchone()
            if row is None:
                raise ValueError(f"book task references missing node {task['node_id']}")
            enriched["moves"] = json.loads(row[0])
            enriched["page_to_history_symmetry"] = int(row[1] or 0)
            output.append(enriched)
        return output
    finally:
        connection.close()


def freeze_uniform_sample(
    database: Path,
    *,
    train_count: int,
    validation_count: int,
    seed: int,
) -> dict[str, int]:
    """Freeze an exact, uniform hash sample of valid canonical book nodes."""

    if train_count <= 0 or validation_count <= 0:
        raise ValueError("training and validation sample counts must be positive")
    connection = _connect(database)
    total = train_count + validation_count

    def sample_hash(node_id: str) -> int:
        from hashlib import blake2b

        value = int.from_bytes(
            blake2b(f"{seed}:{node_id}".encode(), digest_size=8).digest(), "big"
        )
        return value & ((1 << 63) - 1)

    connection.create_function("sample_hash", 1, sample_hash, deterministic=True)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sample_metadata(
              singleton INTEGER PRIMARY KEY CHECK(singleton=1),
              seed INTEGER NOT NULL,
              train_count INTEGER NOT NULL,
              validation_count INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sampled_nodes(
              task_index INTEGER PRIMARY KEY,
              split TEXT NOT NULL,
              node_id TEXT NOT NULL UNIQUE
            );
            """
        )
        metadata = connection.execute(
            "SELECT seed,train_count,validation_count FROM sample_metadata WHERE singleton=1"
        ).fetchone()
        if metadata is not None:
            expected = (seed, train_count, validation_count)
            if tuple(map(int, metadata)) != expected:
                raise ValueError(
                    f"book sample is already frozen as {tuple(metadata)}, requested {expected}"
                )
            sampled = int(connection.execute("SELECT COUNT(*) FROM sampled_nodes").fetchone()[0])
            if sampled != total:
                raise ValueError("frozen book sample is incomplete")
            return {
                "train": train_count,
                "validation": validation_count,
                "total": total,
            }
        connection.execute("DELETE FROM sampled_nodes")
        available = int(
            connection.execute("SELECT COUNT(*) FROM nodes WHERE valid=1").fetchone()[0]
        )
        if available < total:
            raise ValueError(
                f"book has {available:,} valid nodes, fewer than the requested {total:,}"
            )
        cursor = connection.execute(
            """SELECT node_id FROM nodes WHERE valid=1
               ORDER BY sample_hash(node_id),node_id LIMIT ?""",
            (total,),
        )
        rows = []
        for task_index, (node_id,) in enumerate(cursor):
            split = "train" if task_index < train_count else "validation"
            rows.append((task_index, split, node_id))
            if len(rows) == 8192:
                connection.executemany("INSERT INTO sampled_nodes VALUES(?,?,?)", rows)
                rows.clear()
        if rows:
            connection.executemany("INSERT INTO sampled_nodes VALUES(?,?,?)", rows)
        connection.execute(
            "INSERT INTO sample_metadata VALUES(1,?,?,?)",
            (seed, train_count, validation_count),
        )
        connection.commit()
        return {
            "train": train_count,
            "validation": validation_count,
            "total": total,
        }
    finally:
        connection.close()


def iter_frozen_sample(
    database: Path,
    *,
    split: str,
    offset: int = 0,
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    if split not in {"train", "validation"}:
        raise ValueError(f"unknown sample split {split!r}")
    connection = sqlite3.connect(database)
    try:
        query = """SELECT s.task_index,n.node_id,n.next_player,n.ply,n.history_json,
                          n.moves_json,n.page_to_history_symmetry
                   FROM sampled_nodes s JOIN nodes n ON n.node_id=s.node_id
                   WHERE s.split=? ORDER BY s.task_index LIMIT ? OFFSET ?"""
        row_limit = -1 if limit is None else limit
        for row in connection.execute(query, (split, row_limit, offset)):
            yield {
                "task_index": int(row[0]),
                "task_id": f"book-{int(row[0]):09d}",
                "split": split,
                "source": "book",
                "node_id": row[1],
                "position_id": row[1],
                "next_player": row[2],
                "ply": int(row[3]),
                "history": json.loads(row[4]),
                "moves": json.loads(row[5]),
                "page_to_history_symmetry": int(row[6] or 0),
            }
    finally:
        connection.close()
