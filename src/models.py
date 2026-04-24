"""
時系列予測モデル実装
1. Baseline (Naive / Moving Average)
2. LSTM
3. Transformer (Informer-lite)
4. PatchTST-lite
全モデルに RevIN (分布シフト対策) を適用
"""
import numpy as np
import torch
import torch.nn as nn
import math


# ─────────────────────────────────────────────
# RevIN: Reversible Instance Normalization
# Kim et al., 2022 — テスト時の分布シフトを補正
# ─────────────────────────────────────────────

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., 2022)
    入力を正規化し、予測後に元スケールへ戻す。
    分布シフト（訓練/テストの季節差）を補正する。
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.w = nn.Parameter(torch.ones(num_features))
        self.b = nn.Parameter(torch.zeros(num_features))

    def norm(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) → per-instance 正規化
        self._mean = x.mean(dim=1, keepdim=True)          # (B, 1, C)
        self._std  = x.std(dim=1, keepdim=True) + self.eps  # (B, 1, C)
        return (x - self._mean) / self._std * self.w + self.b

    def denorm(self, x: torch.Tensor, target_idx: int = -1) -> torch.Tensor:
        # x: (B, pred_len) — ターゲット列の予測値
        w = self.w[target_idx]
        b = self.b[target_idx]
        std  = self._std[:, 0, target_idx]   # (B,)
        mean = self._mean[:, 0, target_idx]  # (B,)
        return (x - b) / (w + self.eps) * std.unsqueeze(1) + mean.unsqueeze(1)


# ─────────────────────────────────────────────
# Baseline Models
# ─────────────────────────────────────────────

class NaiveBaseline:
    """直近値をそのまま繰り返す予測 (Last-Value Repeat)"""

    def predict(self, x: np.ndarray, pred_len: int) -> np.ndarray:
        # x: (N, seq_len) – target column only
        last_val = x[:, -1:]  # (N, 1)
        return np.tile(last_val, (1, pred_len))


class MovingAverageBaseline:
    """移動平均による予測"""

    def __init__(self, window: int = 24):
        self.window = window

    def predict(self, x: np.ndarray, pred_len: int) -> np.ndarray:
        ma = x[:, -self.window:].mean(axis=1, keepdims=True)
        return np.tile(ma, (1, pred_len))


# ─────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────

class LSTMForecaster(nn.Module):
    """
    Encoder-Decoder LSTM + RevIN
    - Encoder: 多変量入力 → 隠れ状態
    - Decoder: 隠れ状態 → pred_len ステップ先のOT予測
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        num_layers: int = 2,
        pred_len: int = 24,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.revin = RevIN(input_size)
        self.encoder = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_size)
        x_norm = self.revin.norm(x)
        _, (h_n, _) = self.encoder(x_norm)
        out = self.decoder(h_n[-1])  # (B, pred_len) — 正規化空間の予測
        return self.revin.denorm(out, target_idx=-1)


# ─────────────────────────────────────────────
# Transformer
# ─────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerForecaster(nn.Module):
    """
    Vanilla Transformer + RevIN による時系列予測
    - 入力プロジェクション → Positional Encoding → Transformer Encoder → 予測ヘッド
    """

    def __init__(
        self,
        input_size: int = 7,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        pred_len: int = 24,
        seq_len: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.revin = RevIN(input_size)
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN: 学習安定化
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * seq_len, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, input_size)
        x_norm = self.revin.norm(x)
        h = self.input_proj(x_norm)
        h = self.pos_enc(h)
        h = self.transformer(h)
        out = self.head(h)  # (B, pred_len)
        return self.revin.denorm(out, target_idx=-1)


# ─────────────────────────────────────────────
# PatchTST-lite (効率改善版 Transformer)
# ─────────────────────────────────────────────

class PatchTSTLite(nn.Module):
    """
    PatchTST (Nie et al., 2023) の簡易実装
    時系列をパッチに分割することで局所的パターンを捉える
    """

    def __init__(
        self,
        input_size: int = 7,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        pred_len: int = 24,
        seq_len: int = 96,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.revin = RevIN(input_size)
        self.patch_embed = nn.Linear(patch_len * input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.num_patches + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * self.num_patches, pred_len),
        )

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq_len, C) → (B, num_patches, patch_len * C)
        B, L, C = x.shape
        patches = []
        for i in range(self.num_patches):
            start = i * self.stride
            patch = x[:, start : start + self.patch_len, :]  # (B, patch_len, C)
            patches.append(patch.reshape(B, -1))
        return torch.stack(patches, dim=1)  # (B, num_patches, patch_len*C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.revin.norm(x)
        p = self._patchify(x_norm)
        p = self.patch_embed(p)
        p = self.pos_enc(p)
        p = self.transformer(p)
        out = self.head(p)  # (B, pred_len)
        return self.revin.denorm(out, target_idx=-1)


def build_model(model_name: str, cfg: dict) -> nn.Module:
    if model_name == "lstm":
        return LSTMForecaster(**cfg)
    elif model_name == "transformer":
        return TransformerForecaster(**cfg)
    elif model_name == "patchtst":
        return PatchTSTLite(**cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")
