"""
学習・評価パイプライン
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def smape(y_true, y_pred, eps=1.0):
    """Symmetric MAPE — ゼロ付近でも安定"""
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "MAE":   mae(y_true, y_pred),
        "MSE":   mse(y_true, y_pred),
        "RMSE":  rmse(y_true, y_pred),
        "sMAPE": smape(y_true, y_pred),
    }


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
        return self.counter >= self.patience

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_loss += criterion(pred, y).item() * x.size(0)
        preds.append(pred.cpu().numpy())
        trues.append(y.cpu().numpy())
    return (
        total_loss / len(loader.dataset),
        np.concatenate(preds),
        np.concatenate(trues),
    )


def train_model(
    model: nn.Module,
    loaders: dict,
    epochs: int = 30,
    lr: float = 1e-3,
    patience: int = 5,
    device: str = "cpu",
    verbose: bool = True,
) -> dict:
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    stopper = EarlyStopping(patience=patience)
    history = {"train_loss": [], "val_loss": []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss, _, _ = eval_epoch(model, loaders["val"], criterion, device)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if verbose and epoch % 5 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {elapsed:.1f}s")

        if stopper.step(val_loss, model):
            if verbose:
                print(f"  Early stopping at epoch {epoch}")
            break

    stopper.restore(model)
    return history


def test_model(model: nn.Module, loaders: dict, scaler, target_idx: int, device: str = "cpu") -> dict:
    criterion = nn.MSELoss()
    _, preds_scaled, trues_scaled = eval_epoch(model, loaders["test"], criterion, device)

    # スケールを元に戻す (OT列のみ)
    ot_std = scaler.scale_[target_idx]
    ot_mean = scaler.mean_[target_idx]
    preds = preds_scaled * ot_std + ot_mean
    trues = trues_scaled * ot_std + ot_mean

    metrics = evaluate_metrics(trues.flatten(), preds.flatten())
    return metrics, preds, trues
