"""
EDA (探索的データ分析) モジュール
変圧器オイル温度データの統計的・視覚的分析
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import acf, pacf, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
warnings.filterwarnings("ignore")

FEATURE_LABELS = {
    "HUFL": "High Useful Load",
    "HULL": "High Useless Load",
    "MUFL": "Middle Useful Load",
    "MULL": "Middle Useless Load",
    "LUFL": "Low Useful Load",
    "LULL": "Low Useless Load",
    "OT":   "Oil Temperature (Target)",
}


def basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats_df = df.drop(columns=["date"]).describe().T
    stats_df["skew"] = df.drop(columns=["date"]).skew()
    stats_df["kurtosis"] = df.drop(columns=["date"]).kurtosis()
    return stats_df


def plot_time_series_overview(df: pd.DataFrame, save_path: str = None):
    fig, axes = plt.subplots(7, 1, figsize=(16, 20), sharex=True)
    fig.suptitle("ETTh1: 全特徴量の時系列推移", fontsize=14, fontweight="bold")

    colors = plt.cm.tab10.colors
    for i, (col, label) in enumerate(FEATURE_LABELS.items()):
        ax = axes[i]
        ax.plot(df["date"], df[col], color=colors[i], linewidth=0.5, alpha=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)
        if col == "OT":
            ax.set_facecolor("#fff8f0")

    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=30)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_ot_seasonality(df: pd.DataFrame, save_path: str = None):
    df = df.copy()
    df["hour"] = df["date"].dt.hour
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["season"] = pd.cut(
        df["month"], bins=[0, 3, 6, 9, 12],
        labels=["冬(1-3月)", "春(4-6月)", "夏(7-9月)", "秋(10-12月)"]
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("オイル温度の季節性分析", fontsize=13, fontweight="bold")

    df.groupby("hour")["OT"].mean().plot(ax=axes[0], marker="o", color="steelblue")
    axes[0].set_title("時間帯別 平均オイル温度")
    axes[0].set_xlabel("時刻 (h)")
    axes[0].set_ylabel("OT (標準化前)")
    axes[0].grid(alpha=0.3)

    df.groupby("month")["OT"].mean().plot(ax=axes[1], marker="s", color="coral")
    axes[1].set_title("月別 平均オイル温度")
    axes[1].set_xlabel("月")
    axes[1].grid(alpha=0.3)

    df.groupby("dayofweek")["OT"].mean().plot(ax=axes[2], marker="^", color="green")
    axes[2].set_title("曜日別 平均オイル温度")
    axes[2].set_xticklabels(["", "月", "火", "水", "木", "金", "土", "日"])
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_correlation(df: pd.DataFrame, save_path: str = None):
    corr = df[list(FEATURE_LABELS.keys())].corr()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("特徴量間の相関分析", fontsize=13, fontweight="bold")

    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, ax=axes[0], mask=mask, annot=True, fmt=".2f",
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        xticklabels=list(FEATURE_LABELS.keys()),
        yticklabels=list(FEATURE_LABELS.keys()),
    )
    axes[0].set_title("相関行列")

    ot_corr = corr["OT"].drop("OT").sort_values(ascending=True)
    ot_corr.plot(kind="barh", ax=axes[1], color=["red" if v < 0 else "steelblue" for v in ot_corr])
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_title("OTとの相関係数")
    axes[1].set_xlabel("Pearson r")
    axes[1].grid(alpha=0.3, axis="x")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_acf_pacf(series: pd.Series, lags: int = 168, save_path: str = None):
    acf_vals = acf(series.dropna(), nlags=lags)
    pacf_vals = pacf(series.dropna(), nlags=lags)
    conf = 1.96 / np.sqrt(len(series))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("オイル温度の自己相関 / 偏自己相関", fontsize=13, fontweight="bold")

    for ax, vals, title in zip(axes, [acf_vals, pacf_vals], ["ACF", "PACF"]):
        ax.bar(range(len(vals)), vals, color="steelblue", alpha=0.7)
        ax.axhline(conf, color="red", linestyle="--", linewidth=1, label=f"95% CI (±{conf:.3f})")
        ax.axhline(-conf, color="red", linestyle="--", linewidth=1)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("ラグ (h)")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def adf_test(series: pd.Series) -> dict:
    result = adfuller(series.dropna(), autolag="AIC")
    return {
        "ADF統計量": result[0],
        "p値": result[1],
        "定常性": "定常" if result[1] < 0.05 else "非定常",
        "臨界値(1%)": result[4]["1%"],
        "臨界値(5%)": result[4]["5%"],
    }


def plot_seasonal_decompose(series: pd.Series, period: int = 24, save_path: str = None):
    decomp = seasonal_decompose(series.dropna(), model="additive", period=period)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"オイル温度の時系列分解 (周期={period}h)", fontsize=13, fontweight="bold")

    components = [
        (decomp.observed, "観測値", "steelblue"),
        (decomp.trend, "トレンド", "darkorange"),
        (decomp.seasonal, "季節成分", "green"),
        (decomp.resid, "残差", "gray"),
    ]
    for ax, (comp, label, color) in zip(axes, components):
        ax.plot(comp, color=color, linewidth=0.7)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def run_eda(df: pd.DataFrame, output_dir: str = "outputs", freq: str = "h"):
    """
    freq="h": ETTh（hourly）— ACF lags=168, seasonal period=24
    freq="m": ETTm（15-min）— ACF lags=192 (=48h), seasonal period=96 (=24h)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    acf_lags       = 168 if freq == "h" else 192
    seasonal_period = 24  if freq == "h" else 96

    print("=" * 60)
    print("1. 基本統計量")
    print("=" * 60)
    print(basic_stats(df).to_string())

    print("\n" + "=" * 60)
    print("2. ADF定常性検定 (OT)")
    print("=" * 60)
    adf = adf_test(df["OT"])
    for k, v in adf.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    plot_time_series_overview(df, save_path=f"{output_dir}/01_time_series.png")
    plot_ot_seasonality(df, save_path=f"{output_dir}/02_seasonality.png")
    plot_correlation(df, save_path=f"{output_dir}/03_correlation.png")
    plot_acf_pacf(df["OT"], lags=acf_lags, save_path=f"{output_dir}/04_acf_pacf.png")
    plot_seasonal_decompose(df["OT"], period=seasonal_period, save_path=f"{output_dir}/05_decompose.png")

    print("\n" + "=" * 60)
    print("EDA 完了: outputs/ に画像を保存しました")
    print("=" * 60)
