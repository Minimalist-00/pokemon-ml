# ポケモン画像分類

初代ポケモン150種類の画像を深層学習で分類するプロジェクトです。
転移学習（MobileNetV2）を使い、少ない画像データでも高い精度を目指します。

## セットアップ

```bash
python3 -m venv venv
source venv/bin/activate
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn kaggle
```

## 使い方

### 1. データセットのダウンロード

Kaggle の API キーを用意し、`kaggle.json` をプロジェクト直下に配置してください。

```bash
export KAGGLE_CONFIG_DIR=$(pwd)
kaggle datasets download -d kvpratama/pokemon-images-dataset -p data/ --unzip
```

### 2. データの前処理

元データは各ポケモン1枚しかないため、回転や反転などで水増しして学習用データセットを作ります。

```bash
python src/prepare_data.py
```

### 3. 学習と評価

MobileNetV2 をベースにした分類モデルを学習し、結果をグラフとして出力します。

```bash
python src/train.py
```

実行後、以下のファイルが生成されます。

- `training_history.png` — 学習曲線
- `confusion_matrix.png` — 混同行列
- `top_confusions.png` — 間違えやすいポケモンのペア
- `class_performance.png` — クラスごとの正解率
- `predictions.png` — 予測結果のサンプル
