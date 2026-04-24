"""
ETT変圧器オイル温度予測 PoC メインスクリプト
ATHENA TECHNOLOGIES INC. 選考課題

4データセット（ETTh1, ETTh2, ETTm1, ETTm2）を順次処理し、
データセット横断比較まで行う。

実行方法:
    python run_poc.py

出力:
    outputs/<DATASET_NAME>/ に各データセットの EDA・学習曲線・予測結果を保存
    outputs/all_metrics.csv と outputs/cross_dataset_comparison.png を最後に生成
"""
import os
import sys
import random
import warnings
import numpy as np
import torch
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from src.data_loader import load_ett, make_dataloaders, FEATURE_COLS, TARGET_COL
from src.eda import run_eda
from src.models import LSTMForecaster, TransformerForecaster, PatchTSTLite, NaiveBaseline, MovingAverageBaseline
from src.train import train_model, test_model, evaluate_metrics
from src.visualize import (
    plot_loss_curves, plot_predictions, plot_horizon_metrics,
    plot_error_distribution, print_metrics_table, plot_cross_dataset_comparison,
)

# ─── 共通設定 ────────────────────────────────────────────────────────
DEVICE      = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR  = "outputs"
SEQ_LEN     = 96
PRED_LENS   = [24, 48, 96, 168]
BATCH_SIZE  = 64
EPOCHS      = 30
LR          = 1e-3
PATIENCE    = 7

# ETTh: step=1時間 / ETTm: step=15分
# pred_lens は全データセットで同一ステップ数（時間スケールは異なる）
DATASETS = {
    "ETTh1": {"path": "data/ETTh1.csv", "freq": "h"},
    "ETTh2": {"path": "data/ETTh2.csv", "freq": "h"},
    "ETTm1": {"path": "data/ETTm1.csv", "freq": "m"},
    "ETTm2": {"path": "data/ETTm2.csv", "freq": "m"},
}

MODEL_CONFIGS = {
    "LSTM": {
        "cls": LSTMForecaster,
        "cfg_base": {"input_size": len(FEATURE_COLS), "hidden_size": 128, "num_layers": 2, "dropout": 0.1},
    },
    "Transformer": {
        "cls": TransformerForecaster,
        "cfg_base": {"input_size": len(FEATURE_COLS), "d_model": 128, "nhead": 4, "num_encoder_layers": 3,
                     "dim_feedforward": 256, "seq_len": SEQ_LEN, "dropout": 0.1},
    },
    "PatchTST": {
        "cls": PatchTSTLite,
        "cfg_base": {"input_size": len(FEATURE_COLS), "patch_len": 16, "stride": 8, "d_model": 128,
                     "nhead": 4, "num_layers": 3, "seq_len": SEQ_LEN, "dropout": 0.1},
    },
}

print(f"Device: {DEVICE}")
print(f"Seq len: {SEQ_LEN}, Pred horizons: {PRED_LENS}")
print(f"Datasets: {list(DATASETS.keys())}")
print("=" * 60)

os.makedirs(OUTPUT_DIR, exist_ok=True)
target_idx = FEATURE_COLS.index(TARGET_COL)

all_dataset_results = {}   # {ds_name: {model_name: {pred_len: metrics}}}

# ─── データセットループ ───────────────────────────────────────────────
for ds_name, ds_cfg in DATASETS.items():
    print(f"\n{'='*60}")
    print(f"データセット: {ds_name}  (freq={ds_cfg['freq']})")
    print("=" * 60)

    ds_output_dir = os.path.join(OUTPUT_DIR, ds_name)
    os.makedirs(ds_output_dir, exist_ok=True)

    # ── 1. EDA ──────────────────────────────────────────────────────
    print(f"\n[{ds_name}] EDA")
    full_df, data = load_ett(ds_cfg["path"], freq=ds_cfg["freq"])
    run_eda(full_df, output_dir=ds_output_dir, freq=ds_cfg["freq"])

    # ── 2. Baseline ────────────────────────────────────────────────
    print(f"\n[{ds_name}] Baseline 評価")
    all_results = {m: {} for m in ["Naive", "MovingAvg", "LSTM", "Transformer", "PatchTST"]}

    ot_test_scaled = data["test"]["scaled"][:, target_idx]
    ot_std  = data["scaler"].scale_[target_idx]
    ot_mean = data["scaler"].mean_[target_idx]
    ot_test = ot_test_scaled * ot_std + ot_mean

    for pred_len in PRED_LENS:
        n_samples = len(ot_test) - SEQ_LEN - pred_len + 1
        X_ot = np.array([ot_test[i : i + SEQ_LEN] for i in range(n_samples)])
        Y_ot = np.array([ot_test[i + SEQ_LEN : i + SEQ_LEN + pred_len] for i in range(n_samples)])

        naive = NaiveBaseline()
        ma    = MovingAverageBaseline(window=24)
        all_results["Naive"][pred_len]    = evaluate_metrics(Y_ot.flatten(), naive.predict(X_ot, pred_len).flatten())
        all_results["MovingAvg"][pred_len] = evaluate_metrics(Y_ot.flatten(), ma.predict(X_ot, pred_len).flatten())
        print(f"  h={pred_len} | Naive MAE={all_results['Naive'][pred_len]['MAE']:.3f}"
              f" | MA MAE={all_results['MovingAvg'][pred_len]['MAE']:.3f}")

    # ── 3. 深層学習モデル学習 ────────────────────────────────────────
    print(f"\n[{ds_name}] 深層学習モデル学習")
    saved_preds = {}

    for model_name, spec in MODEL_CONFIGS.items():
        print(f"\n  --- {model_name} ---")
        for pred_len in PRED_LENS:
            loaders = make_dataloaders(data, SEQ_LEN, pred_len, BATCH_SIZE, seed=SEED)
            cfg = {**spec["cfg_base"], "pred_len": pred_len}
            model = spec["cls"](**cfg)

            print(f"  h={pred_len}, params={sum(p.numel() for p in model.parameters()):,}")
            history = train_model(model, loaders, epochs=EPOCHS, lr=LR, patience=PATIENCE,
                                  device=DEVICE, verbose=True)

            plot_loss_curves(
                history, f"{ds_name}/{model_name} (h={pred_len})",
                save_path=os.path.join(ds_output_dir, f"loss_{model_name}_h{pred_len}.png"),
            )

            metrics, preds, trues = test_model(model, loaders, data["scaler"], target_idx, device=DEVICE)
            all_results[model_name][pred_len] = metrics
            print(f"  Test MAE={metrics['MAE']:.4f} | RMSE={metrics['RMSE']:.4f} | sMAPE={metrics['sMAPE']:.2f}%")

            if pred_len == PRED_LENS[0]:
                saved_preds[model_name] = (preds, trues)

    # ── 4. 結果集計・可視化 ─────────────────────────────────────────
    print(f"\n[{ds_name}] 結果集計")
    metrics_df = print_metrics_table(all_results)
    metrics_df.to_csv(os.path.join(ds_output_dir, "metrics_summary.csv"), index=False)

    trues_ref  = saved_preds[list(saved_preds.keys())[0]][1]
    preds_dict = {name: p for name, (p, _) in saved_preds.items()}
    plot_predictions(trues_ref, preds_dict, n_samples=300,
                     save_path=os.path.join(ds_output_dir, "predictions.png"))
    plot_error_distribution(trues_ref, preds_dict,
                            save_path=os.path.join(ds_output_dir, "error_dist.png"))

    dl_results = {k: v for k, v in all_results.items() if k in MODEL_CONFIGS}
    plot_horizon_metrics(dl_results, title=ds_name,
                         save_path=os.path.join(ds_output_dir, "horizon_comparison.png"))

    all_dataset_results[ds_name] = all_results
    print(f"  → '{ds_output_dir}/' に保存完了")

# ─── データセット横断比較 ────────────────────────────────────────────
print("\n\n[全体] データセット横断比較")
plot_cross_dataset_comparison(
    all_dataset_results,
    save_path=os.path.join(OUTPUT_DIR, "cross_dataset_comparison.png"),
)

# 全データセット統合 CSV
all_rows = []
for ds_name, ds_results in all_dataset_results.items():
    for model_name, horizon_results in ds_results.items():
        for horizon, metrics in horizon_results.items():
            all_rows.append({"Dataset": ds_name, "Model": model_name,
                             "Horizon": f"{horizon}steps", **metrics})
all_metrics_df = pd.DataFrame(all_rows)
all_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "all_metrics.csv"), index=False)

print(f"\n完了: 全結果を '{OUTPUT_DIR}/' に保存しました")
print(f"  ├── <DATASET>/metrics_summary.csv  (各データセット個別)")
print(f"  ├── all_metrics.csv                (4データセット統合)")
print(f"  └── cross_dataset_comparison.png   (横断比較グラフ)")
