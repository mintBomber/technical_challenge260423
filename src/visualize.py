"""
予測結果の可視化モジュール
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def plot_loss_curves(history: dict, model_name: str, save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss", color="steelblue")
    ax.plot(history["val_loss"], label="Val Loss", color="coral")
    ax.set_title(f"{model_name} - 学習曲線")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_predictions(trues: np.ndarray, preds_dict: dict, n_samples: int = 200, save_path: str = None):
    """複数モデルの予測を重ねて表示"""
    # 最初のサンプルの最初のステップを時系列としてプロット
    ot_true = trues[:n_samples, 0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(ot_true, label="実測値 (OT)", color="black", linewidth=1.5, zorder=10)

    colors = ["steelblue", "coral", "green", "purple"]
    for (name, preds), color in zip(preds_dict.items(), colors):
        ot_pred = preds[:n_samples, 0]
        ax.plot(ot_pred, label=f"{name} (予測)", color=color, alpha=0.8, linewidth=1.0)

    ax.set_title("オイル温度予測 vs 実測値 (テストセット冒頭)")
    ax.set_xlabel("時刻ステップ")
    ax.set_ylabel("Oil Temperature (°C)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_horizon_metrics(results: dict, title: str = "", save_path: str = None):
    """予測ホライズン別のMAE/RMSEをモデル間比較"""
    horizons = sorted({h for model_results in results.values() for h in model_results.keys()})
    models = list(results.keys())
    x = np.arange(len(horizons))
    width = 0.8 / len(models)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    suptitle = f"{title} — 予測ホライズン別 モデル性能比較" if title else "予測ホライズン別 モデル性能比較"
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")

    colors = plt.cm.tab10.colors
    for metric_idx, metric in enumerate(["MAE", "RMSE"]):
        ax = axes[metric_idx]
        for i, model in enumerate(models):
            vals = [results[model].get(h, {}).get(metric, np.nan) for h in horizons]
            ax.bar(x + i * width, vals, width, label=model, color=colors[i], alpha=0.8)
        ax.set_title(metric)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels([f"{h}h" for h in horizons])
        ax.set_xlabel("予測ホライズン")
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_error_distribution(trues: np.ndarray, preds_dict: dict, save_path: str = None):
    """残差分布の可視化"""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue", "coral", "green", "purple"]

    for (name, preds), color in zip(preds_dict.items(), colors):
        errors = (preds - trues).flatten()
        ax.hist(errors, bins=60, alpha=0.5, label=f"{name}", color=color, density=True)

    ax.axvline(0, color="black", linewidth=1.5, linestyle="--")
    ax.set_title("予測残差の分布")
    ax.set_xlabel("残差 (予測値 - 実測値)")
    ax.set_ylabel("密度")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_cross_dataset_comparison(all_dataset_results: dict, metric: str = "MAE", save_path: str = None):
    """4データセット横断のモデル性能比較（2×2サブプロット）"""
    ds_names = list(all_dataset_results.keys())
    n_ds = len(ds_names)
    ncols = 2
    nrows = (n_ds + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    fig.suptitle(f"データセット横断 {metric} 比較（深層学習モデル）", fontsize=14, fontweight="bold")
    axes = axes.flatten() if n_ds > 1 else [axes]

    colors = plt.cm.tab10.colors
    dl_models = ["LSTM", "Transformer", "PatchTST"]

    for ax_idx, ds_name in enumerate(ds_names):
        ax = axes[ax_idx]
        ds_results = all_dataset_results[ds_name]
        horizons = sorted({h for m in dl_models for h in ds_results.get(m, {}).keys()})
        x = np.arange(len(horizons))
        width = 0.8 / len(dl_models)

        for i, model in enumerate(dl_models):
            if model not in ds_results:
                continue
            vals = [ds_results[model].get(h, {}).get(metric, np.nan) for h in horizons]
            ax.bar(x + i * width, vals, width, label=model, color=colors[i], alpha=0.8)
            for j, v in enumerate(vals):
                if not np.isnan(v):
                    ax.text(x[j] + i * width, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

        # Naive ベースラインを折れ線で重ねる
        if "Naive" in ds_results:
            naive_vals = [ds_results["Naive"].get(h, {}).get(metric, np.nan) for h in horizons]
            ax.plot(x + width, naive_vals, color="black", linestyle="--", linewidth=1.2,
                    marker="x", label="Naive", zorder=5)

        freq_note = "(1 step=1h)" if "ETTh" in ds_name else "(1 step=15min)"
        ax.set_title(f"{ds_name} {freq_note}")
        ax.set_xticks(x + width)
        ax.set_xticklabels([f"{h} steps" for h in horizons])
        ax.set_xlabel("予測ホライズン（ステップ数）")
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    for ax_idx in range(n_ds, len(axes)):
        axes[ax_idx].set_visible(False)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def print_metrics_table(all_results: dict):
    """全モデル・全ホライズンのメトリクスを表形式で表示"""
    rows = []
    for model_name, horizon_results in all_results.items():
        for horizon, metrics in horizon_results.items():
            rows.append({
                "Model": model_name,
                "Horizon": f"{horizon}h",
                **{k: f"{v:.4f}" for k, v in metrics.items()},
            })
    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("モデル性能比較サマリー")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70)
    return df
