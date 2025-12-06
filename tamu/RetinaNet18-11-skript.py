
import argparse
import os
import glob
import time
import torch
import torchvision.transforms as T
from PIL import Image

# --------------------------------------------
# ここが重要：学習の時のモデル定義を読み込む
# --------------------------------------------
from RetinaNet18 import RetinaNet18   # ←あなたの学習モデルのファイル名に合わせて変更

def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = T.ToTensor()
    return transform(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)

    # --------------------------------------------
    # 1. モデルロード（state_dict を読み込む正しい方法）
    # --------------------------------------------
    model = RetinaNet18(num_classes=2)  # ←学習と同じクラス数にする
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 画像一覧取得
    image_paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
    if len(image_paths) == 0:
        print("No images found.")
        return

    print(f"{len(image_paths)} 枚の画像で推論時間を測定します（読み込み時間は除外）")

    # 画像読み込み（計測外）
    images = [load_image(p) for p in image_paths]

    # バッチ分割
    batch_size = args.batch
    batches = [images[i:i+batch_size] for i in range(0, len(images), batch_size)]

    # 推論時間測定
    torch.cuda.synchronize()
    total_time = 0.0
    total_images = 0

    with torch.no_grad():
        for batch in batches:
            batch_tensor = torch.stack(batch).to(device)

            start = time.time()
            _ = model(batch_tensor)
            torch.cuda.synchronize()
            end = time.time()

            total_time += (end - start)
            total_images += len(batch)

    avg = (total_time / total_images) * 1000
    print(f"\n平均推論時間（読み込み除外）: {avg:.2f} ms/枚")

if __name__ == "__main__":
    main()
