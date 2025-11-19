import torch
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import argparse
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

# unet2プロジェクトのモジュールをインポート
from model import UNet
import config

def decode_heatmaps_to_coordinates(heatmaps, image_size):
    """
    ヒートマップから座標をデコードする。
    train.pyから移植。
    """
    batch_size, num_keypoints, H, W = heatmaps.shape
    flat_heatmaps = heatmaps.view(batch_size, num_keypoints, -1)
    max_indices = torch.argmax(flat_heatmaps, dim=2)
    y_coords = max_indices // W
    x_coords = max_indices % W
    # 座標をピクセル中心に補正
    x_coords = (x_coords.float() + 0.5)
    y_coords = (y_coords.float() + 0.5)
    coordinates = torch.stack((x_coords, y_coords), dim=2) # [N, C, 2]
    return coordinates

def main():
    parser = argparse.ArgumentParser(description="Inference script for a single image.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth).")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    parser.add_argument('--output_dir', type=str, default='output', help="Directory to save the output image.")
    args = parser.parse_args()

    # 1. デバイス設定
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # 2. 出力ディレクトリ作成
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. モデルの初期化と重みロード
    model = UNet(in_channels=3, out_channels=config.NUM_KEYPOINTS).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"Model weights loaded from '{args.model_path}'.")
    except FileNotFoundError:
        print(f"Error: Model file not found at '{args.model_path}'")
        return
    model.eval()

    # 4. 画像の前処理
    transform = Compose([
        Resize(config.IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image_pil = Image.open(args.image_path).convert('RGB')
        original_w, original_h = image_pil.size
    except FileNotFoundError:
        print(f"Error: Input image not found at '{args.image_path}'")
        return

    # unsqueeze(0)でバッチ次元を追加
    image_tensor = transform(image_pil).unsqueeze(0).to(device)

    # 5. 推論の実行
    with torch.no_grad():
        heatmaps_pred = model(image_tensor)

    # 6. 後処理: ヒートマップから座標へ変換
    # (B, C, H, W) -> (B, C, 2)
    predicted_coords = decode_heatmaps_to_coordinates(heatmaps_pred.cpu(), config.IMAGE_SIZE)
    # バッチ次元を削除
    predicted_coords_np = predicted_coords.squeeze(0).numpy() # (9, 2)

    # 7. 座標を元の画像サイズにスケーリング
    scale_x = original_w / config.IMAGE_SIZE[1]
    scale_y = original_h / config.IMAGE_SIZE[0]
    scaled_landmarks = predicted_coords_np.copy()
    scaled_landmarks[:, 0] *= scale_x
    scaled_landmarks[:, 1] *= scale_y

    # 8. 結果の描画と保存
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image_pil)
    ax.scatter(scaled_landmarks[:, 0], scaled_landmarks[:, 1], c='red', s=50)

    # ランドマークに番号を振る
    for i in range(scaled_landmarks.shape[0]):
        ax.annotate(str(i + 1), (scaled_landmarks[i, 0] + 10, scaled_landmarks[i, 1] + 10),
                    color='yellow', fontsize=16, fontweight='bold',
                    path_effects=[pe.withStroke(linewidth=3, foreground="black")])

    ax.axis('off')
    base_name = os.path.basename(args.image_path)
    save_path = os.path.join(args.output_dir, f"inference_{base_name}")
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    print(f"Inference complete. Output saved to '{save_path}'")

if __name__ == '__main__':
    main()
