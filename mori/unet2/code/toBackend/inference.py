import os
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from detect import load_ml_model, detect_face_and_lndmk 

# --- 設定 ---
TEST_IMAGE_PATH = "R.jpg" 
OUTPUT_IMAGE_PATH = "./output_inference.jpg"
SCORE_THRESHOLD = 0.3
LANDMARK_COLOR = 'red'
BBOX_COLOR = 'green'
POINT_SIZE = 10
LINE_WIDTH = 3
# ---

def visualize_and_save(image_path: str, results, output_path: str):
    try:
        img_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"❌ 画像ファイルが見つかりません: {image_path}")
        return

    # 2. BBoxとランドマーク座標の取得
    # results[0] = [xmin, ymin], results[1] = [xmax, ymax]
    bbox_min = results[0]
    bbox_max = results[1]
    landmarks = np.array(results[2:]) # 9点のランドマーク座標

    # 3. Matplotlibを使ってプロット
    plt.figure(figsize=(12, 12))
    plt.imshow(img_pil)
    
    # BBoxのプロット
    xmin, ymin = bbox_min
    xmax, ymax = bbox_max
    
    # BBoxを描画
    rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                         fill=False, edgecolor=BBOX_COLOR, linewidth=LINE_WIDTH)
    plt.gca().add_patch(rect)
    
    # ランドマークのプロット (赤い点)
    # X座標 (landmarks[:, 0]) と Y座標 (landmarks[:, 1])
    plt.scatter(landmarks[:, 0], landmarks[:, 1], c=LANDMARK_COLOR, s=POINT_SIZE, zorder=10)
    
    plt.title(f"Animal Landmark Detection Result")
    plt.axis('off') # 軸を非表示
    
    # 4. 画像を保存
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"\n結果画像が '{output_path}' に保存されました。")


def main():
    print("--- モデルロード中 ---")
    load_ml_model() # detect.py のモデルロード関数を呼び出し

    print(f"\n--- 推論開始: {TEST_IMAGE_PATH} ---")
    
    # 推論実行
    results = detect_face_and_lndmk(TEST_IMAGE_PATH, score_threshold=SCORE_THRESHOLD)
    
    if results is not None:
        visualize_and_save(TEST_IMAGE_PATH, results, OUTPUT_IMAGE_PATH)
        
    else:
        print("❌ 処理失敗、または閾値未満の検出結果でした。画像パスを確認してください。")


if __name__ == "__main__":
    main()