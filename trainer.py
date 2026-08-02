"""Training orchestration for CAVYAA."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[assignment]
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from config import ExperimentConfig
from domain_adaptation import grl_schedule
from evaluation import evaluate
from losses import cavyaa_loss, kl_anneal
from utils import append_csv, assert_finite_mapping, assert_finite_tensor, ensure_dir
from utils import ensure_dir


class EarlyStopping:
    """Track validation loss and stop when improvement stalls."""

    def __init__(self, patience: int, min_delta: float) -> None:
        """Initialize early stopping state."""
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Return True when training should stop."""
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class Trainer:
    """Training loop with AMP, clipping, schedules, checkpointing, resume, and logs."""
    """Full training loop with AMP, clipping, schedules, checkpointing, and TensorBoard."""

    def __init__(self, model: torch.nn.Module, config: ExperimentConfig, device: torch.device) -> None:
        """Create optimizer, scheduler, scaler, writer, and artifact locations."""
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        warmup = LinearLR(self.optimizer, start_factor=0.2, total_iters=max(1, config.train.warmup_epochs))
        cosine = CosineAnnealingLR(self.optimizer, T_max=max(1, config.train.epochs - config.train.warmup_epochs))
        self.scheduler = SequentialLR(self.optimizer, schedulers=[warmup, cosine], milestones=[max(1, config.train.warmup_epochs)])
        self.scaler = torch.amp.GradScaler(device.type, enabled=config.train.use_mixed_precision and device.type == "cuda")
        self.writer = SummaryWriter(str(config.train.log_dir)) if SummaryWriter is not None else None
        self.checkpoint_dir = ensure_dir(config.train.checkpoint_dir)
        self.best_path = self.checkpoint_dir / "best_model.pt"
        self.last_path = self.checkpoint_dir / "last_model.pt"
        self.best_val_loss = float("inf")
        self.start_epoch = 0

    def _move(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch to the configured device."""
        return {key: value.to(self.device, non_blocking=True) for key, value in batch.items()}

    def _save_checkpoint(self, path: Path, epoch: int, history: List[Dict[str, float]]) -> None:
        """Save full training state for reproducible resume."""
        ensure_dir(path.parent)
        torch.save(
            {
                "epoch": epoch,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
                "scaler_state": self.scaler.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config.as_dict(),
                "history": history,
            },
            path,
        )

    def load_checkpoint(self, path: Path | str) -> List[Dict[str, float]]:
        """Restore model, optimizer, scheduler, scaler, and history."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            self.scaler.load_state_dict(checkpoint["scaler_state"])
        self.best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        return list(checkpoint.get("history", []))

    def train_epoch(self, loader: DataLoader, epoch: int) -> tuple[float, float]:
        """Train for one epoch and return mean loss plus gradient norm."""
        self.model.train()
        beta = kl_anneal(epoch, self.config.loss)
        self.model.set_grl_lambda(grl_schedule(epoch, self.config.loss.grl_warmup_epochs))
        total, count, grad_total, grad_steps = 0.0, 0, 0.0, 0
        for batch in loader:
            batch = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.model(batch)
                loss, parts = cavyaa_loss(outputs, batch, beta, self.config.loss)
            assert_finite_mapping(parts)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
            assert_finite_tensor("gradient_norm", grad_norm)
        self.optimizer = AdamW(model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(1, config.train.epochs))
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.train.use_mixed_precision and device.type == "cuda")
        self.writer = SummaryWriter(str(config.train.log_dir)) if SummaryWriter is not None else None
        self.checkpoint_dir = ensure_dir(config.train.checkpoint_dir)
        self.best_path = self.checkpoint_dir / "best_model.pt"

    def _move(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Move a batch to the configured device."""
        return {key: value.to(self.device) for key, value in batch.items()}

    def train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Train for one epoch and return mean loss."""
        self.model.train()
        beta = kl_anneal(epoch, self.config.loss)
        self.model.set_grl_lambda(grl_schedule(epoch, self.config.loss.grl_warmup_epochs))
        total, count = 0.0, 0
        for batch in loader:
            batch = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                outputs = self.model(batch)
                loss, parts = cavyaa_loss(outputs, batch, beta, self.config.loss)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += parts["loss"] * len(batch["label"])
            count += len(batch["label"])
            grad_total += float(grad_norm.detach().cpu())
            grad_steps += 1
        self.scheduler.step()
        return total / max(1, count), grad_total / max(1, grad_steps)
        self.scheduler.step()
        return total / max(1, count)

    @torch.no_grad()
    def validation_loss(self, loader: DataLoader, epoch: int) -> float:
        """Compute validation objective without gradient updates."""
        self.model.eval()
        beta = kl_anneal(epoch, self.config.loss)
        total, count = 0.0, 0
        for batch in loader:
            batch = self._move(batch)
            outputs = self.model(batch)
            loss, parts = cavyaa_loss(outputs, batch, beta, self.config.loss)
            assert_finite_mapping(parts)
            loss, _ = cavyaa_loss(outputs, batch, beta, self.config.loss)
            total += float(loss) * len(batch["label"])
            count += len(batch["label"])
        return total / max(1, count)

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> Tuple[List[Dict[str, float]], Path]:
        """Run the complete training loop and save best and last checkpoints."""
        history: List[Dict[str, float]] = []
        if self.config.train.resume_from is not None:
            history = self.load_checkpoint(self.config.train.resume_from)
        stopper = EarlyStopping(self.config.train.early_stopping_patience, self.config.train.min_delta)
        stopper.best = self.best_val_loss
        for epoch in range(self.start_epoch, self.config.train.epochs):
            train_loss, grad_norm = self.train_epoch(train_loader, epoch)
            val_loss = self.validation_loss(val_loader, epoch)
            metrics = evaluate(self.model, val_loader, self.device)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "grad_norm": grad_norm,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                **metrics,
            }
            assert_finite_mapping(row)
            history.append(row)
            append_csv(row, self.config.train.history_csv)
            if self.writer:
                self.writer.add_scalar("loss/train", train_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("optimization/grad_norm", grad_norm, epoch)
                self.writer.add_scalar("optimization/lr", row["learning_rate"], epoch)
                self.writer.add_scalar("metrics/roc_auc", metrics["roc_auc"], epoch)
            if val_loss < self.best_val_loss - self.config.train.min_delta:
                self.best_val_loss = val_loss
                self._save_checkpoint(self.best_path, epoch, history)
            self._save_checkpoint(self.last_path, epoch, history)
        """Run the complete training loop and save the best checkpoint."""
        stopper = EarlyStopping(self.config.train.early_stopping_patience, self.config.train.min_delta)
        history: List[Dict[str, float]] = []
        for epoch in range(self.config.train.epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validation_loss(val_loader, epoch)
            metrics = evaluate(self.model, val_loader, self.device)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **metrics}
            history.append(row)
            if self.writer:
                self.writer.add_scalar("loss/train", train_loss, epoch)
                self.writer.add_scalar("loss/val", val_loss, epoch)
                self.writer.add_scalar("metrics/roc_auc", metrics["roc_auc"], epoch)
            if val_loss <= min(item["val_loss"] for item in history):
                torch.save({"model_state": self.model.state_dict(), "config": self.config.as_dict(), "history": history}, self.best_path)
            if stopper.step(val_loss):
                break
        if self.writer:
            self.writer.flush()
            self.writer.close()
        return history, self.best_path
