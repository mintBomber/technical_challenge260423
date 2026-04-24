"""
ETT (Electricity Transformer Temperature) データセットのローダーと前処理モジュール
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import torch


FEATURE_COLS = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
TARGET_COL = "OT"

# ETTh: 時間単位, ETTm: 15分単位
SPLIT_HOURS = {
    "h": {"train": 8736, "val": 2880, "test": 2880},  # 12/4/4 months
    "m": {"train": 34560, "val": 11520, "test": 11520},
}


def load_ett(path: str, freq: str = "h") -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    splits = SPLIT_HOURS[freq]
    n_train = splits["train"]
    n_val = splits["val"]

    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val : n_train + n_val + splits["test"]]

    scaler = StandardScaler()
    train_vals = scaler.fit_transform(train_df[FEATURE_COLS].values)
    val_vals = scaler.transform(val_df[FEATURE_COLS].values)
    test_vals = scaler.transform(test_df[FEATURE_COLS].values)

    data = {
        "train": {"df": train_df, "scaled": train_vals},
        "val": {"df": val_df, "scaled": val_vals},
        "test": {"df": test_df, "scaled": test_vals},
        "scaler": scaler,
        "full_df": df,
    }
    return df, data


class ETTDataset(Dataset):
    def __init__(
        self,
        scaled_data: np.ndarray,
        seq_len: int = 96,
        pred_len: int = 24,
        target_idx: int = -1,
    ):
        self.data = torch.FloatTensor(scaled_data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_idx = target_idx

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, self.target_idx]
        return x, y


def make_dataloaders(
    data: dict,
    seq_len: int = 96,
    pred_len: int = 24,
    batch_size: int = 64,
    seed: int = 42,
) -> dict:
    target_idx = FEATURE_COLS.index(TARGET_COL)
    g = torch.Generator()
    g.manual_seed(seed)
    loaders = {}
    for split in ["train", "val", "test"]:
        ds = ETTDataset(data[split]["scaled"], seq_len, pred_len, target_idx)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
            drop_last=(split == "train"),
            generator=g if split == "train" else None,
        )
    return loaders
