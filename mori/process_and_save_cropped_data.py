import torch
from torchvision import transforms
from PIL import Image
from PIL.Image import LANCZOS
import numpy as np
#import matplotlib.pyplot as plt
import os
import pathlib
from tqdm import tqdm

DATASET_ROOT = pathlib.Path('../raw/animal_dataset_v1_clean_check')
OUTPUT_DIR = 'cropped_dataset'
PADDING_RATIO = 0.3
TARGET_SIZE = 224

def load_pts_file(pts_filepath):
    points = []
    with open(pts_filepath, 'r') as f:
        lines = f.readlines()

    start_index = -1
    for i, line in enumerate(lines):
        if line.strip() == '{':
            start_index = i + 1
            break

    if start_index == -1:
        raise ValueError("Invalid .pts format: Cannot find starting bracket '{'.")

    for line in lines[start_index : start_index + 9]:
        try:
            x, y = map(float, line.strip().split())
            points.append([x, y])
        except ValueError:
            continue

    if len(points) != 9:
        raise ValueError(f"Expected 9 points, but found {len(points)} in {pts_filepath}")

    return np.array(points)

def save_pts_file(pts_filepath: str, points: np.ndarray):
    if points.shape != (9, 2):
        raise ValueError("ランドマーク座標は(9, 2)のNumpy配列である必要があります。")

    lines = ["version: 1\n", "n_points: 9\n", "{\n"]

    for x, y in points:
        lines.append(f"  {x:.6f} {y:.6f}\n")
    lines.append("}")

    with open(pts_filepath, 'w') as f:
        f.writelines(lines)
    print(f"新しいランドマークファイル '{pts_filepath}'を保存しました。")

def crop_and_save_data(image_path: str, pts_path: str, output_dir: str,  padding_ratio: float = 0.3):
    try:
        image_orig = Image.open(image_path).convert('RGB')
        points = load_pts_file(pts_path)
    except FileNotFoundError as e:
        print(f"エラー：必要なファイルが見つかりません。{e}")
        return 

    W_orig, H_orig = image_orig.size

    x_min = np.min(points[:, 0])
    x_max = np.max(points[:, 0])
    y_min = np.min(points[:, 1])
    y_max = np.max(points[:, 1])

    face_width = x_max - x_min
    face_height = y_max - y_min

    pad_x = face_width * padding_ratio
    pad_y = face_height * padding_ratio

    #pad_x = max(30, pad_x)
    #pad_y = max(30, pad_y)

    crop_x_start = x_min - pad_x
    crop_y_start = y_min - pad_y

    crop_x_end = x_max + pad_x
    crop_y_end = y_max + pad_y

    final_box = [
            int(max(0, crop_x_start)),
            int(max(0, crop_y_start)),
            int(min(W_orig, crop_x_end)),
            int(min(H_orig, crop_y_end))
            ]

    crop_W = final_box[2] - final_box[0]
    crop_H = final_box[3] - final_box[1]

    if crop_W <= 0 or crop_H <= 0:
        print("警告：トリミング領域が無効")
        return

    image_cropped = image_orig.crop(final_box)


    #ランドマーク座標

    offset_x = final_box[0]
    offset_y = final_box[1]

    new_points = np.copy(points)
    new_points[:, 0] -= offset_x
    new_points[:, 1] -= offset_y

    #スケーリング処理
    image_cropped = image_cropped.resize((TARGET_SIZE, TARGET_SIZE),Image.Resampling.LANCZOS )
    scale_x = TARGET_SIZE / crop_W
    scale_y = TARGET_SIZE /crop_H
    new_points[:, 0] *= scale_x
    new_points[:, 1] *= scale_y

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    #output_dir = "cropped_output"
    #os.makedirs(output_dir, exist_ok=True)

    output_img_path = os.path.join(output_dir, f"{base_name}.jpg")
    image_cropped.save(output_img_path)
    #print(f"トリミング画像を'{output_img_path}'に保存しました。")

    output_pts_path = os.path.join(output_dir, f"{base_name}.pts")
    save_pts_file(output_pts_path, new_points)


    #画像を確認する
    #plt.figure(figsize=(6,6))
    #plt.imshow(image_cropped)

    #plt.scatter(new_points[:, 0], new_points[:, 1], c='red', marker='x', s=50)
    #plt.title(f"Cropped Image ({crop_W}x{crop_H})")
    #plt.axis('off')
    #plt.show()


def main():
    if not DATASET_ROOT.is_dir():
        print(f"エラー: データセットディレクトリが見つかりません: {DATASET_ROOT}")
        return

    # 出力ディレクトリを作成
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # .jpgファイルのリストを取得し、ベース名でソート
    all_jpg_files = sorted(DATASET_ROOT.glob('*.jpg'))
    
    print(f"合計 {len(all_jpg_files)} 個の画像を処理します。")

    # tqdm で進捗バーを表示
    for jpg_path in tqdm(all_jpg_files, desc="Processing Images"):
        
        # 対応する .pts ファイルのパスを構築
        base_name = jpg_path.stem
        pts_path = DATASET_ROOT / f"{base_name}.pts"
        
        # ptsファイルが存在するかチェック
        if not pts_path.exists():
            print(f"警告: {base_name}.pts が見つかりません。スキップします。")
            continue
        
        try:
            # トリミングと保存を実行
            crop_and_save_data(
                str(jpg_path), 
                str(pts_path), 
                OUTPUT_DIR, 
                PADDING_RATIO
            )
        except Exception as e:
            print(f"\n致命的なエラー: {base_name} の処理中にエラーが発生しました: {e}")
            # エラーが発生しても、次のファイルに進む
            continue

    print("\n✅ 全ての画像処理が完了しました。")
    print(f"新しいデータセットは '{OUTPUT_DIR}' ディレクトリに保存されました。")

if __name__ == '__main__':
    main()
