"""Metric accumulation (ported from Training/metrics.py) + CSV mirror schema.

The wide-CSV schema is kept as a thin mirror so the HTML pipeline viewer keeps
working; TensorBoard is the primary sink (written by the Trainer).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from sakigo.constants import ACTION_HEADS, HEADS, WDL_LABELS
from sakigo.train.losses import LossWeights


def format_metric(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return ""
    return f"{value:.6f}"


def confusion_fields() -> list[str]:
    return [
        f"val_wdl_true_{true_label}_pred_{pred_label}"
        for true_label in WDL_LABELS
        for pred_label in WDL_LABELS
    ]


def metric_fields() -> list[str]:
    fields = ["step"]
    for prefix in ("train", "val"):
        fields.append(f"{prefix}_loss")
        for head in HEADS:
            fields.append(f"{prefix}_{head}_loss")
            fields.append(f"{prefix}_{head}_target_count")
        fields.append(f"{prefix}_wdl_acc")
        for head in ACTION_HEADS:
            fields.append(f"{prefix}_{head}_target_entropy")
            fields.append(f"{prefix}_{head}_excess_ce")
            fields.append(f"{prefix}_{head}_illegal_mass")
            fields.append(f"{prefix}_{head}_illegal_target_count")
        fields.append(f"{prefix}_score_mae")
        fields.append(f"{prefix}_ownership_sign_acc")
        fields.append(f"{prefix}_ownership_cell_count")
    fields.extend(confusion_fields())
    return fields


def write_metrics_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(metric_fields())


def _format_csv_field(field: str, value: float | int) -> str:
    if field.endswith("_count") or field in confusion_fields():
        numeric = float(value)
        if math.isnan(numeric):
            return ""
        return str(int(round(numeric)))
    return format_metric(value)


def append_metrics(path: Path, row: dict[str, float | int]) -> None:
    fields = metric_fields()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writerow(
            {field: _format_csv_field(field, row.get(field, float("nan"))) for field in fields}
        )


def _nan() -> float:
    return float("nan")


class MetricAccumulator:
    """Accumulates metrics as device tensors; host sync happens only in averages()."""

    def __init__(self, loss_weights: LossWeights) -> None:
        self.loss_weights = loss_weights.as_dict()
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self._device: torch.device | None = None
        self._sums: dict[str, torch.Tensor] = {}
        self.confusion = torch.zeros((len(WDL_LABELS), len(WDL_LABELS)), dtype=torch.float64)

    def _sum(self, key: str, value: torch.Tensor) -> None:
        value = value.detach().to(torch.float64)
        if key in self._sums:
            self._sums[key] += value
        else:
            self._sums[key] = value.clone()

    def add_batch(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        head_losses: dict[str, torch.Tensor],
        total_loss: torch.Tensor,
    ) -> None:
        self.steps += 1
        if self._device is None:
            self._device = total_loss.device
            self.confusion = self.confusion.to(self._device)
        self._sum("loss", total_loss)
        for head in HEADS:
            count = batch[f"{head}_mask"].sum()
            self._sum(f"{head}_count", count)
            self._sum(f"{head}_loss", head_losses[head].detach() * count)
        self._add_wdl(output, batch)
        for head in ACTION_HEADS:
            self._add_action(output, batch, head)
        self._add_score(output, batch)
        self._add_ownership(output, batch)

    def _add_wdl(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        mask = batch["wdl_mask"]
        weights = mask.to(torch.float64)
        true = batch["wdl_target"].detach().argmax(dim=-1)
        predicted = output["wdl_logits"].detach().argmax(dim=-1)
        self._sum("wdl_correct", ((predicted == true).to(torch.float64) * weights).sum())
        self._sum("wdl_total", weights.sum())
        encoded = true * len(WDL_LABELS) + predicted
        self.confusion += torch.bincount(
            encoded,
            weights=weights,
            minlength=len(WDL_LABELS) ** 2,
        ).reshape(len(WDL_LABELS), len(WDL_LABELS))

    def _add_action(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        head: str,
    ) -> None:
        mask = batch[f"{head}_mask"]
        selected = mask.float()
        logits = output[f"{head}_logits"].detach()
        target = batch[f"{head}_target"].detach()
        log_probs = F.log_softmax(logits.float(), dim=-1)
        probabilities = log_probs.exp()
        per_ce = -(target * log_probs).sum(dim=-1)
        per_entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1)
        self._sum(f"{head}_entropy", (per_entropy * selected).sum())
        self._sum(f"{head}_excess", ((per_ce - per_entropy) * selected).sum())

        legal_available = (mask & batch["legal_mask_available"]).float()
        illegal_mass = (probabilities * (~batch["legal_mask"]).float()).sum(dim=-1)
        self._sum(f"{head}_illegal", (illegal_mass * legal_available).sum())
        self._sum(f"{head}_illegal_count", legal_available.sum())

    def _add_score(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        mask = batch["score_mask"].float()
        error = (output["score"].detach().reshape_as(batch["score_target"]) - batch["score_target"]).abs()
        self._sum("score_abs", (error.squeeze(1) * mask).sum())
        self._sum("score_total", mask.sum())

    def _add_ownership(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        mask = batch["ownership_mask"].float()
        predicted = output["ownership_logits"].detach() >= 0.0
        target = batch["ownership_target"] >= 0.0
        per_record = (predicted == target).float().sum(dim=-1)
        self._sum("ownership_correct", (per_record * mask).sum())
        self._sum("ownership_cells", mask.sum() * batch["ownership_target"].shape[-1])

    def averages(self) -> dict[str, float]:
        sums = {key: float(value.cpu()) for key, value in self._sums.items()}
        out: dict[str, float] = {
            "loss": sums.get("loss", _nan()) / self.steps if self.steps else _nan(),
        }
        for head in HEADS:
            count = sums.get(f"{head}_count", 0.0)
            out[f"{head}_loss"] = sums.get(f"{head}_loss", 0.0) / count if count else _nan()
            out[f"{head}_target_count"] = count
        wdl_total = sums.get("wdl_total", 0.0)
        out["wdl_acc"] = sums.get("wdl_correct", 0.0) / wdl_total if wdl_total else _nan()
        for head in ACTION_HEADS:
            count = sums.get(f"{head}_count", 0.0)
            out[f"{head}_target_entropy"] = sums.get(f"{head}_entropy", 0.0) / count if count else _nan()
            out[f"{head}_excess_ce"] = sums.get(f"{head}_excess", 0.0) / count if count else _nan()
            illegal_count = sums.get(f"{head}_illegal_count", 0.0)
            out[f"{head}_illegal_mass"] = (
                sums.get(f"{head}_illegal", 0.0) / illegal_count if illegal_count else _nan()
            )
            out[f"{head}_illegal_target_count"] = illegal_count
        score_total = sums.get("score_total", 0.0)
        out["score_mae"] = sums.get("score_abs", 0.0) / score_total if score_total else _nan()
        cells = sums.get("ownership_cells", 0.0)
        out["ownership_sign_acc"] = sums.get("ownership_correct", 0.0) / cells if cells else _nan()
        out["ownership_cell_count"] = cells
        return out


def prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def add_val_confusion(row: dict[str, float | int], accumulator: MetricAccumulator) -> None:
    for true_index, true_label in enumerate(WDL_LABELS):
        for pred_index, pred_label in enumerate(WDL_LABELS):
            row[f"val_wdl_true_{true_label}_pred_{pred_label}"] = accumulator.confusion[
                true_index,
                pred_index,
            ].item()
