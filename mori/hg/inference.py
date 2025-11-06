import os
import cv2
import torch
import numpy as np
import argparse
from model import HourglassNet # model.pyからHourglassNetをインポート

# --- 設定パラメータ (train.pyと一致させる) ---
IMAGE_SIZE = 256
HEATMAP_SIZE = 64
NUM_LANDMARKS = 9
NUM_STACKS = 2
FEATURE_CHANNELS = 256
MODEL_PATH = './model/animal_hourglass_model.pth' # モデルのパス

def heatmap_to_landmark(heatmap):
    """
    ヒートマップの各チャネルから、最大値の位置をランドマーク座標として抽出する。
    
    Args:
        heatmap (numpy.ndarray): 形状 (NUM_LANDMARKS, HEATMAP_SIZE, HEATMAP_SIZE) のヒートマップ。
    
    Returns:
        numpy.ndarray: 形状 (NUM_LANDMARKS, 2) のランドマーク座標 ([x, y] 形式)。
    """
    num_landmarks, h, w = heatmap.shape
    landmarks = []
    for i in range(num_landmarks):
        # 最大値のインデックスを取得
        idx = np.unravel_index(np.argmax(heatmap[i]), (h, w))
        # 座標は (y, x) で返されるため、(x, y) に並び替える
        pt_y, pt_x = idx
        landmarks.append([pt_x, pt_y])
    
    return np.array(landmarks, dtype=np.float32)

def main(image_path, model_path):
    # --- デバイス設定 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- モデルの初期化と重みのロード ---
    print("Initializing model...")
    model = HourglassNet(nstack=NUM_STACKS, nclasses=NUM_LANDMARKS, nfeats=FEATURE_CHANNELS).to(device)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # モデルの重みをロード
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 評価モードに設定
    print(f"Model loaded successfully from {model_path}")

    # --- 画像の前処理 ---
    print(f"Processing image: {image_path}")
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Error: Image not found or could not be loaded at {image_path}")
        return

    # OpenCVのBGRからRGBに変換
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_h, original_w, _ = image.shape

    # データセットと同じサイズにリサイズ
    image_resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # HWC -> CHW に変換し、正規化、テンソル化 (データセットクラスの処理を再現)
    image_tensor = torch.from_numpy(np.transpose(image_resized, (2, 0, 1))).float() / 255.0
    
    # バッチ次元を追加 (C, H, W) -> (1, C, H, W)
    input_tensor = image_tensor.unsqueeze(0).to(device)

    # --- 推論実行 ---
    print("Running inference...")
    with torch.no_grad():
        outputs = model(input_tensor)

    # HourglassNetはスタックごとの出力リストを返す。最後のスタックの予測を使用する
    predicted_heatmaps_tensor = outputs[-1].squeeze(0) # (NUM_LANDMARKS, HEATMAP_SIZE, HEATMAP_SIZE)
    predicted_heatmaps_np = predicted_heatmaps_tensor.cpu().numpy()

    # --- ランドマークの抽出とスケーリング ---
    # ヒートマップからランドマーク座標を抽出 (HEATMAP_SIZEのスケール)
    landmarks_heatmap_scale = heatmap_to_landmark(predicted_heatmaps_np)

    # 元の画像サイズにスケーリング
    # ランドマーク座標を元の画像サイズ (original_w, original_h) に戻す
    scale_factor_x = original_w / HEATMAP_SIZE
    scale_factor_y = original_h / HEATMAP_SIZE
    
    landmarks_original_scale = landmarks_heatmap_scale.copy()
    landmarks_original_scale[:, 0] *= scale_factor_x
    landmarks_original_scale[:, 1] *= scale_factor_y
    landmarks_original_scale = landmarks_original_scale.astype(int)
    
    print(f"Detected {len(landmarks_original_scale)} landmarks.")

    # --- 結果の可視化 ---
    # BGR形式に戻す（描画のため）
    display_image = original_image.copy()

    # ランドマークを描画
    for (x, y) in landmarks_original_scale:
        # 赤い丸を描画
        cv2.circle(display_image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
    
    # 結果の保存
    output_path = os.path.join(os.path.dirname(image_path), f"predicted_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, display_image)
    print(f"Result saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference on an animal image using the HourglassNet model.")
    parser.add_argument('image_path', type=str, help='Path to the input animal image.')
    args = parser.parse_args()
    
    # モデルのディレクトリが存在しない場合は作成
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    main(args.image_path, MODEL_PATH)