import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

# 依存ファイルのインポート
from unet import UNet 
from dataset import LandmarkHeatmapDataset # Datasetクラス
from utils import extract_keypoints_from_heatmap # ヒートマップ->座標変換ヘルパー

# ----------------------------------------------------------------
# ランドマークと幾何学的特徴を描画して保存する関数
# ----------------------------------------------------------------
def save_landmark_predictions(model, data_loader, device, model_input_size, num_samples=10, save_dir="./inference_output"):
    """
    テストデータに対して推論を行い、予測されたランドマークを画像に描画して保存する。
    
    Args:
        model: 訓練済みPyTorchモデル (UNet)
        data_loader: テストデータのDataLoader
        device: デバイス ('cpu' or 'cuda')
        model_input_size (int): モデルの入力画像サイズ (例: 224)
        num_samples (int): 保存するサンプル画像の最大数
        save_dir (str): 画像の保存先ディレクトリ
    """
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_count = 0
    NUM_LANDMARKS = 9 # ランドマーク数

    print(f"\n--- 予測ランドマークの描画と保存を開始 (最大 {num_samples} 枚) ---")
    
    with torch.no_grad():
        # tqdmを使用してデータローダーの進捗を表示
        for data in tqdm(data_loader, desc="Inference"):
            # データローダーの出力は (画像, ヒートマップ, 座標, 画像パス)
            images_tensor = data[0].to(device)
            img_paths = data[3] # 画像パスを取得

            outputs = model(images_tensor).cpu() # 推論結果 (ヒートマップ) をCPUに戻す

            # ヒートマップから予測座標を抽出 (N, 9, 2)
            predicted_coords_reshaped = extract_keypoints_from_heatmap(outputs) 

            for i in range(images_tensor.size(0)):
                if saved_count >= num_samples:
                    return # 指定サンプル数に達したら終了
                    
                # 1. 元の画像の読み込み
                original_img_path = img_paths[i]
                original_image_pil = Image.open(original_img_path).convert("RGB")
                original_w, original_h = original_image_pil.size
                
                # 2. 予測されたランドマーク座標を取得 (NumPy配列)
                predicted_landmarks_np = predicted_coords_reshaped[i].numpy() # (9, 2)
                
                # 3. 予測座標を元の画像サイズにスケーリング
                scale_x = original_w / model_input_size
                scale_y = original_h / model_input_size
                
                scaled_landmarks_x = predicted_landmarks_np[:, 0] * scale_x
                scaled_landmarks_y = predicted_landmarks_np[:, 1] * scale_y
                scaled_landmarks = np.stack([scaled_landmarks_x, scaled_landmarks_y], axis=1) # (9, 2)

                # --- Matplotlibで描画 ---
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(original_image_pil)
                
                # --- A. 1, 2個目の点を直径とする円 (赤色) の描画 ---
                p1 = scaled_landmarks[0]
                p2 = scaled_landmarks[1]
                center_x12 = (p1[0] + p2[0]) / 2
                center_y12 = (p1[1] + p2[1]) / 2
                diameter12 = np.linalg.norm(p1 - p2)
                radius12 = diameter12 / 2
                circle12 = plt.Circle((center_x12, center_y12), radius12, 
                                        color='red', fill=False, linewidth=2)
                ax.add_artist(circle12)
                
                # --- B. 3, 4個目の点を直径とする円 (赤色) の描画 ---
                p3 = scaled_landmarks[2]
                p4 = scaled_landmarks[3]
                center_x34 = (p3[0] + p4[0]) / 2
                center_y34 = (p3[1] + p4[1]) / 2
                diameter34 = np.linalg.norm(p3 - p4)
                radius34 = diameter34 / 2
                circle34 = plt.Circle((center_x34, center_y34), radius34, 
                                        color='red', fill=False, linewidth=2)
                ax.add_artist(circle34)

                # --- C. 6, 8, 7, 9, 6の順に直線をつなげた線 (赤色) の描画 ---
                # インデックスは 0 から始まるため、(6, 8, 7, 9, 6) -> [5, 7, 6, 8, 5]
                indices = [5, 7, 6, 8, 5] 
                line_x = scaled_landmarks[:, 0][indices]
                line_y = scaled_landmarks[:, 1][indices]
                ax.plot(line_x, line_y, color='red', linestyle='-', linewidth=2)

                # --- ランドマーク点を描画 ---
                ax.scatter(scaled_landmarks[:, 0], scaled_landmarks[:, 1], 
                            c='red', marker='o', s=50, label=None)
                
                # --- ランドマークに番号を振る ---
                tmp = 10 # テキストオフセット
                for k_idx in range(NUM_LANDMARKS): 
                    x, y = scaled_landmarks[k_idx]
                    # 番号 (黄色)
                    ax.annotate(str(k_idx+1), (x + tmp, y + tmp), color='yellow', fontsize=16, 
                                ha='center', va='center', fontweight='bold',
                                path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground="black")])


                ax.set_title(f"Predicted Landmarks: {os.path.basename(original_img_path)}")
                ax.axis('off') 
                
                # --- ファイル保存 ---
                base_name = os.path.splitext(os.path.basename(original_img_path))[0]
                save_path = os.path.join(save_dir, f"pred_geometric_{base_name}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig) 
                
                # print(f"✅ 予測画像を保存: {save_path}")
                saved_count += 1

# ----------------------------------------------------------------
# メイン実行関数
# ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Inference Script for UNet Landmark Prediction")
    parser.add_argument('--model_path', type=str, required=True, help="訓練済みモデルのパス (.pth)")
    parser.add_argument('--data_dir', type=str, required=True, help="テストデータセットのディレクトリパス")
    parser.add_argument('--num_samples', type=int, default=10, help="推論・描画するサンプル数")
    parser.add_argument('--batch_size', type=int, default=16, help="DataLoaderのバッチサイズ")
    parser.add_argument('--img_size', type=int, default=224, help="モデルの入力画像サイズ")
    args = parser.parse_args()

    # 1. デバイスの設定
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(f"使用デバイス: {device}")

    # 2. モデルの初期化と重みのロード
    model = UNet(in_channels=3, out_channels=9).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"モデル重み '{args.model_path}' をロードしました。")
    except FileNotFoundError:
        print(f"エラー: モデルファイルが見つかりません: {args.model_path}")
        return

    # 3. テストデータの準備 (データセット全体からランダムにファイルを選ぶ)
    # 訓練済みモデルは訓練時に分割したテストデータを使うのが理想ですが、ここでは手軽に全ファイルから選択します
    all_jpg_files = glob.glob(os.path.join(args.data_dir, "*.jpg"))
    if not all_jpg_files:
        print(f"エラー: データセットディレクトリ '{args.data_dir}' 内に.jpgファイルが見つかりません。")
        return
    
    # 全ファイルリストをデータセットに渡す (推論用)
    inference_dataset = LandmarkHeatmapDataset(
        file_list=all_jpg_files, # 全ファイルをリストとして渡す
        root_dir=args.data_dir,
        image_size=args.img_size,
        sigma=3.0 # シグマ値は推論には関係ないがDatasetの初期化に必要
    )
    
    # 推論時、shuffleは不要
    inference_dataloader = DataLoader(
        inference_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() or 1
    )

    # 4. 推論と描画の実行
    save_landmark_predictions(
        model=model,
        data_loader=inference_dataloader,
        device=device,
        model_input_size=args.img_size,
        num_samples=args.num_samples,
        save_dir="./inference_output"
    )
    print("--- 予測ランドマークの描画と保存が完了しました。---")

if __name__ == '__main__':
    main()