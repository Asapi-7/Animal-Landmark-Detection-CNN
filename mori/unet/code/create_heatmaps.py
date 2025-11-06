import os
import sys
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import load_landmarks_from_pts_to_tensor, generate_gaussian_heatmap
import numpy as np
from PIL import Image

IMG_SIZE = 224
SIGMA = 3.0
SAVE_DIR = "../heatmap_sample"

def process_and_save_heatmap(pts_path):
    """
    単一の.ptsファイルを処理し、ヒートマップを生成して画像ファイルとして保存する
    """
    base_path_without_ext = os.path.splitext(pts_path)[0]
    jpg_path = base_path_without_ext + ".jpg"

    base_name = os.path.basename(base_path_without_ext)
    save_filename = f"{base_name}_heatmap.png"
    save_path = os.path.join(SAVE_DIR, save_filename)

    print(f"-> 処理中: {pts_path} と {jpg_path}")

    try:
        keypoints = load_landmarks_from_pts_to_tensor(pts_path)
        heatmaps = generate_gaussian_heatmap(keypoints, IMG_SIZE, SIGMA)
        overlay_heatmap = np.max(heatmaps, axis=2)

        if not os.path.exists(jpg_path):
            raise FileNotFoundError(f"対応するJPEGファイルが見つかりません: {jpg_path}")

        image = Image.open(jpg_path).convert("RGB")
        image = image.resize((IMG_SIZE, IMG_SIZE))
        image_np = np.array(image)

        fig, axes = plt.subplots(1, 2, figsize=(12,6))

        ax_img = axes[0]
        ax_img.imshow(image_np, extent=[0, IMG_SIZE, IMG_SIZE, 0])
        ax_img.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=40, marker='o', edgecolors="black", linewidths=1.5, label='Landmarks')
        ax_img.set_title(f'Original Image with Landmarks ({base_name})')
        ax_img.axis('off')
        ax_img.set_aspect('auto')

        ax_map = axes[1]
        im = ax_map.imshow(overlay_heatmap, cmap='jet', vmin=0, vmax=1)
        ax_map.scatter(keypoints[:, 0], keypoints[:, 1], c='yellow', s=20, marker='*')
        ax_map.set_title(f'Overlay Gaussian Heatmap ($\sigma={SIGMA}$)')
        ax_map.axis('off')
        ax_map.set_aspect('auto')

       

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)

        print(f"保存完了！ : {save_path}")

        return True

    except Exception as e:
        print(f"処理エラー ({pts_path}): {e}")
        return False

def main():
    if len(sys.argv) != 2:
        print("使用方法： ↓ ")
        print(f"   python {sys.argv[0]} <ptsファイルパス>")
        sys.exit(1)

    pts_file_path = sys.argv[1]

    os.makedirs(SAVE_DIR, exist_ok=True)

    process_and_save_heatmap(pts_file_path)

if __name__ == "__main__":
    main()