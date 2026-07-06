"""Legacy-checkpoint compatibility: real pre-rebuild checkpoints under
Training/runs must keep loading into the unified SakiGoNet (scalar runs via
the scripted remap). Output-level parity vs the legacy code was verified
before the P6 cutover; this keeps the load path itself pinned.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sakigo.model import SakiGoNet, config_from_dict, remap_legacy_scalar_state_dict

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "Training" / "runs"


def _latest_checkpoints(limit: int = 2) -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    found: dict[str, Path] = {}
    for path in sorted(RUNS_DIR.glob("*/checkpoints/step_*.pt")):
        found[str(path.parent.parent)] = path  # keep the last (highest step) per run
    return list(found.values())[:limit]


CHECKPOINTS = _latest_checkpoints()


@pytest.mark.parametrize(
    "checkpoint_path",
    CHECKPOINTS or [None],
    ids=[str(p.relative_to(RUNS_DIR)) for p in CHECKPOINTS] or ["none"],
)
def test_legacy_checkpoint_loads_and_runs(checkpoint_path: Path | None) -> None:
    if checkpoint_path is None:
        pytest.skip("no legacy checkpoints under Training/runs/")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = config_from_dict(payload["model_config"])
    state = payload.get("model_state", payload.get("model"))
    assert state is not None
    if config.group_size == 1:
        state = remap_legacy_scalar_state_dict(state)
    model = SakiGoNet(config).eval()
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing and not unexpected

    generator = torch.Generator().manual_seed(5)
    board = (torch.rand(2, 6, 9, 9, generator=generator) > 0.6).float()
    rules = torch.zeros(2, 10)
    rules[:, 0] = 1.0
    rules[:, 4] = 1.0
    rules[:, 6] = 1.0
    with torch.no_grad():
        output = model(board, rules)
    assert set(output) == {"wdl_logits", "score", "ownership_logits", "policy_logits", "budget_logits"}
    assert all(torch.isfinite(value).all() for value in output.values())
