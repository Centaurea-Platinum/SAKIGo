from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from .common import ACTION_HEADS, HEADS, WDL_LABELS, format_metric
from .losses import LossWeights


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
            {
                field: _format_csv_field(field, row.get(field, float("nan")))
                for field in fields
            }
        )


def _nan() -> float:
    return float("nan")


class ProgressBar:
    """Single-line ASCII progress bar refreshed in place with carriage returns."""

    def __init__(self, total_steps: int, start_step: int, batch_size: int, enabled: bool, width: int = 28) -> None:
        self.total = max(1, total_steps)
        self.start = start_step
        self.batch = batch_size
        self.enabled = enabled
        self.width = width
        self.start_time = time.monotonic()
        self._line_len = 0

    def render(self, step: int, loss: float) -> None:
        if not self.enabled:
            return
        elapsed = max(time.monotonic() - self.start_time, 1e-9)
        done = max(step - self.start, 0)
        rate = done * self.batch / elapsed
        remaining = (self.total - step) * self.batch / rate if rate > 0 else 0.0
        filled = min(self.width, int(self.width * step / self.total))
        bar = "#" * filled + "-" * (self.width - filled)
        eta_minutes, eta_seconds = divmod(int(remaining), 60)
        loss_text = f"{loss:.3f}" if not math.isnan(loss) else "-"
        text = (
            f"[{bar}] {step}/{self.total} {rate:,.0f} samp/s "
            f"eta {eta_minutes:d}:{eta_seconds:02d} loss {loss_text}"
        )
        padding = max(self._line_len - len(text), 0)
        print("\r" + text + " " * padding, end="", flush=True)
        self._line_len = len(text)

    def clear(self) -> None:
        if not self.enabled or self._line_len == 0:
            return
        print("\r" + " " * self._line_len + "\r", end="", flush=True)
        self._line_len = 0


class MetricAccumulator:
    def __init__(self, loss_weights: LossWeights) -> None:
        self.loss_weights = loss_weights.as_dict()
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.loss_sum = 0.0
        self.last_loss = float("nan")
        self.head_loss_sums = {head: 0.0 for head in HEADS}
        self.head_counts = {head: 0.0 for head in HEADS}
        self.wdl_correct = 0.0
        self.wdl_count = 0.0
        self.confusion = torch.zeros((len(WDL_LABELS), len(WDL_LABELS)), dtype=torch.float64)
        self.action_entropy_sums = {head: 0.0 for head in ACTION_HEADS}
        self.action_excess_sums = {head: 0.0 for head in ACTION_HEADS}
        self.action_illegal_sums = {head: 0.0 for head in ACTION_HEADS}
        self.action_illegal_counts = {head: 0.0 for head in ACTION_HEADS}
        self.score_abs_sum = 0.0
        self.score_count = 0.0
        self.ownership_sign_correct = 0.0
        self.ownership_cell_count = 0.0

    def add_batch(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        head_losses: dict[str, torch.Tensor],
        total_loss: torch.Tensor,
    ) -> None:
        self.steps += 1
        loss_value = float(total_loss.detach().cpu())
        self.loss_sum += loss_value
        self.last_loss = loss_value
        for head in HEADS:
            count = float(batch[f"{head}_mask"].sum().detach().cpu())
            self.head_counts[head] += count
            if count > 0.0:
                self.head_loss_sums[head] += float(head_losses[head].detach().cpu()) * count
        self._add_wdl(output, batch)
        for head in ACTION_HEADS:
            self._add_action(output, batch, head)
        self._add_score(output, batch)
        self._add_ownership(output, batch)

    def _add_wdl(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        mask = batch["wdl_mask"]
        if not mask.any().item():
            return
        logits = output["wdl_logits"][mask].detach()
        target = batch["wdl_target"][mask].detach()
        true = target.argmax(dim=-1)
        predicted = logits.argmax(dim=-1)
        self.wdl_correct += float((predicted == true).float().sum().cpu())
        self.wdl_count += float(true.numel())
        encoded = true * len(WDL_LABELS) + predicted
        self.confusion += torch.bincount(
            encoded.cpu(),
            minlength=len(WDL_LABELS) ** 2,
        ).reshape(len(WDL_LABELS), len(WDL_LABELS)).to(dtype=torch.float64)

    def _add_action(
        self,
        output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
        head: str,
    ) -> None:
        mask = batch[f"{head}_mask"]
        if not mask.any().item():
            return
        logits = output[f"{head}_logits"].detach()
        target = batch[f"{head}_target"].detach()
        log_probs = F.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        per_ce = -(target * log_probs).sum(dim=-1)
        per_entropy = -(target * target.clamp_min(1e-12).log()).sum(dim=-1)
        selected = mask.float()
        self.action_entropy_sums[head] += float((per_entropy * selected).sum().cpu())
        self.action_excess_sums[head] += float(((per_ce - per_entropy) * selected).sum().cpu())

        legal_available = mask & batch["legal_mask_available"]
        if legal_available.any().item():
            illegal_mass = (probabilities * (~batch["legal_mask"]).float()).sum(dim=-1)
            self.action_illegal_sums[head] += float(illegal_mass[legal_available].sum().cpu())
            self.action_illegal_counts[head] += float(legal_available.sum().cpu())

    def _add_score(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        mask = batch["score_mask"]
        if not mask.any().item():
            return
        error = (output["score"].detach().reshape_as(batch["score_target"]) - batch["score_target"]).abs()
        self.score_abs_sum += float(error[mask].sum().cpu())
        self.score_count += float(mask.sum().cpu())

    def _add_ownership(self, output: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> None:
        mask = batch["ownership_mask"]
        if not mask.any().item():
            return
        predicted = output["ownership_logits"].detach()[mask] >= 0.0
        target = batch["ownership_target"][mask] >= 0.0
        self.ownership_sign_correct += float((predicted == target).float().sum().cpu())
        self.ownership_cell_count += float(target.numel())

    def averages(self) -> dict[str, float]:
        out: dict[str, float] = {
            "loss": self.loss_sum / self.steps if self.steps else _nan(),
        }
        for head in HEADS:
            count = self.head_counts[head]
            out[f"{head}_loss"] = self.head_loss_sums[head] / count if count else _nan()
            out[f"{head}_target_count"] = count
        out["wdl_acc"] = self.wdl_correct / self.wdl_count if self.wdl_count else _nan()
        for head in ACTION_HEADS:
            count = self.head_counts[head]
            out[f"{head}_target_entropy"] = (
                self.action_entropy_sums[head] / count if count else _nan()
            )
            out[f"{head}_excess_ce"] = (
                self.action_excess_sums[head] / count if count else _nan()
            )
            illegal_count = self.action_illegal_counts[head]
            out[f"{head}_illegal_mass"] = (
                self.action_illegal_sums[head] / illegal_count if illegal_count else _nan()
            )
            out[f"{head}_illegal_target_count"] = illegal_count
        out["score_mae"] = self.score_abs_sum / self.score_count if self.score_count else _nan()
        out["ownership_sign_acc"] = (
            self.ownership_sign_correct / self.ownership_cell_count
            if self.ownership_cell_count
            else _nan()
        )
        out["ownership_cell_count"] = self.ownership_cell_count
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


def progress_line(row: dict[str, float | int]) -> str:
    parts = [
        f"step={int(row['step'])}",
        f"loss={float(row['train_loss']):.4f}",
        f"val={float(row['val_loss']):.4f}",
    ]
    for field in ("val_wdl_loss", "val_policy_loss", "val_budget_loss", "val_score_loss"):
        value = float(row.get(field, float("nan")))
        if not math.isnan(value):
            parts.append(f"{field.removeprefix('val_').removesuffix('_loss')}={value:.4f}")
    acc = float(row.get("val_wdl_acc", float("nan")))
    if not math.isnan(acc):
        parts.append(f"wdl_acc={acc:.3f}")
    return " ".join(parts)

