"""Rust engine bindings (sakigo_engine pyo3 module).

Build: with a repo-local CARGO_HOME (see Engine/.cargo/config.toml):
  uvx maturin build --manifest-path Engine/Cargo.toml --release --out dist
  uv pip install dist/sakigo_engine-*.whl
"""

from __future__ import annotations

try:
    from sakigo_engine import BOARD_PLANE_COUNT, RULE_FEATURE_COUNT, Game

    ENGINE_AVAILABLE = True
except ImportError:  # wheel not installed
    Game = None  # type: ignore[assignment]
    BOARD_PLANE_COUNT = 6
    RULE_FEATURE_COUNT = 10
    ENGINE_AVAILABLE = False

__all__ = ["BOARD_PLANE_COUNT", "ENGINE_AVAILABLE", "Game", "RULE_FEATURE_COUNT"]
