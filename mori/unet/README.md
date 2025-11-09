### 📁 データセットの準備 (Dataset Setup)

すべての画像ファイル (`.jpg`) と、対応するランドマークファイル (`.pts`) は、**unetディレクトリ**直下に配置してください。

**例:** `train.py` や `inference.py` が存在する `code` ディレクトリの**一つ上の階層**に `cropped_dataset` を配置

### 3. 🎯 実行方法 (Usage)

`train.py` と `inference.py` の使い方

(code ディレクトリ内で実行)
python train.py --data_dir ../cropped_dataset --epochs 30 --batch_size 16 --lr 0.0001 --output_dir ./run_output_unet

python inference.py --model_path ./run_output_unet/unet_landmark_regressor_final.pth --data_dir ../cropped_dataset --samples_per_category 5 --inference_output_root ./run_output_unet/inference_results
### 📐 評価指標 (Metrics)

本プロジェクトでは、以下の指標を用いてモデルの性能を評価します。

* **損失関数 (Loss):** **MSE (Mean Squared Error)** - 予測ヒートマップと正解ヒートマップ間のピクセル単位の誤差を最小化します。
* **評価指標 (Metric):** **NME (Normalized Mean Error)** - 予測座標と正解座標間の平均ユークリッド距離を、顔のバウンディングボックス対角線長で正規化して算出します。値が小さいほど高性能です。
