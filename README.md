# 変圧器オイル温度予測 PoC

ETT（Electricity Transformer Temperature）データセット 4種を用いた時系列予測モデルの技術検証プロジェクト。

## プロジェクト構成

```
.
├── run_poc.py              # メイン実行スクリプト（4データセット対応）
├── requirements.txt        # 依存ライブラリ
├── data/                   # ETTh1 / ETTh2 / ETTm1 / ETTm2 CSV
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
└── src/
    ├── data_loader.py      # データ読み込み・前処理・DataLoader
    ├── eda.py              # 探索的データ分析
    ├── models.py           # LSTM / Transformer / PatchTST-Lite / Baseline
    ├── train.py            # 学習・評価パイプライン
    └── visualize.py        # 結果可視化
```

実行後、`outputs/` ディレクトリが自動生成されます（`.gitignore` で除外済み）。

## データセット

| データセット | 計測頻度 | レコード数 | 訓練 / 検証 / テスト |
|------------|---------|---------|-----------------|
| ETTh1 / ETTh2 | 1時間 | 17,420行 | 8,736 / 2,880 / 2,880 |
| ETTm1 / ETTm2 | 15分 | 69,680行 | 34,560 / 11,520 / 11,520 |

- 特徴量: 6種の電気負荷（HUFL, HULL, MUFL, MULL, LUFL, LULL）+ **OT（目標変数: 変圧器オイル温度）**
- 予測ホライズン: **24 / 48 / 96 / 168 steps**（全データセット共通）

## セットアップ

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 実行

```bash
MPLBACKEND=Agg python run_poc.py
```

> macOS / Linux 環境で GUI なし実行時に `MPLBACKEND=Agg` を付与。Windows や Jupyter 環境では不要。

## モデル

| モデル | アーキテクチャ | パラメータ数（24h） |
|--------|-------------|----------------|
| Naive | 直近値を繰り返す | 0 |
| MovingAvg | 直近 24h 移動平均 | 0 |
| **LSTM** | RevIN → LSTM(hidden=128, 2層) → Linear | 約 21 万 |
| Transformer | RevIN → Pre-LN TransformerEncoder × 3 → Linear | 約 69 万 |
| PatchTST-Lite | RevIN → Patchify(patch=16, stride=8) → TransformerEncoder × 3 | 約 64 万 |

## 実験結果（LSTM 24steps MAE）

| データセット | LSTM | Naive | 改善率 |
|------------|------|-------|-------|
| ETTh1（1step=1h） | 1.18°C | 1.25°C | −6% |
| ETTh2（1step=1h） | 2.44°C | 4.03°C | **−39%** |
| ETTm1（1step=15min） | 0.69°C | 0.71°C | −3% |
| ETTm2（1step=15min） | 1.21°C | 2.34°C | **−48%** |

> 深層学習の優位性はデータの性質に依存する。ETTh2・ETTm2 では大幅改善、ETTh1・ETTm1 ではベースラインが既に強力。

## 技術的なポイント

- **RevIN**（Kim et al., 2022）: 訓練（夏）→ テスト（冬）で生じる約 10°C の分布シフトをインスタンス単位の正規化で吸収
- **sMAPE（ε=1.0）**: OT の最小値 −4.08°C で MAPE の分母がほぼゼロになる問題を回避
- **再現性**: seed=42 を全乱数源（numpy / torch / DataLoader の `torch.Generator`）に固定

## 参考文献

- Zhou et al., 2021 — Informer (*AAAI 2021*) — ETT データセット原著
- Kim et al., 2022 — RevIN (*ICLR 2022*) — Reversible Instance Normalization
- Nie et al., 2023 — PatchTST (*ICLR 2023*) — A Time Series is Worth 64 Words
- Vaswani et al., 2017 — Attention Is All You Need (*NeurIPS 2017*)
