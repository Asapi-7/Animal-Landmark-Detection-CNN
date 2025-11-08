### 📁 データセットの準備 (Dataset Setup)

現在のディレクトリ直下(unetディレクトリ)に `cropped_dataset` フォルダをしてください

### 3. 🎯 実行方法 (Usage)

`train.py` と `inference.py` の使い方を明確にします。

(code ディレクトリ内で実行)
python train.py --data_dir ../cropped_dataset --epochs 30 --batch_size 16 --lr 0.0001

python inference.py --model_path unet_landmark_regressor_final.pth --data_dir ../cropped_dataset --num_samples 10
    
### 📐 評価指標 (Metrics)

本プロジェクトでは、以下の指標を用いてモデルの性能を評価します。

* **損失関数 (Loss):** **MSE (Mean Squared Error)** - 予測ヒートマップと正解ヒートマップ間のピクセル単位の誤差を最小化します。
* **評価指標 (Metric):** **NME (Normalized Mean Error)** - 予測座標と正解座標間の平均ユークリッド距離を、顔のバウンディングボックス対角線長で正規化して算出します。値が小さいほど高性能です。
