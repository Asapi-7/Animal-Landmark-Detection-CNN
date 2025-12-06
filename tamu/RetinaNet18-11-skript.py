import argparse
import os
import glob
import time
import torch
import torchvision.transforms as T
from PIL import Image

def load_image(path):
    img = Image.open(path).convert("RGB")
    transform = T.ToTensor()
    return transform(img)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ------------------------
    # 1. モデルロード
    # ------------------------
    model = torch.load(args.weights, map_location=device)
    model.to(device)
    model.eval()

    # ------------------------
    # 2. 画像ファイル一覧
    # ------------------------
    image_paths = sorted(glob.glob(os.path.join(args.dir, "*.jpg")))
    if len(image_paths) == 0:
        print("No images found.")
        return

    print(f"{len(image_paths)} 枚で平均推論時間を計測します（読み込み時間は除外）")

    # ------------------------
    # 3. 画像読み込み（ここは時間に含めない）
    # ------------------------
    images = [load_image(p) for p in image_paths]

    # ------------------------
    # 4. バッチ分割
    # ------------------------
    batch_size = args.batch
    batches = [
        images[i:i + batch_size]
        for i in range(0, len(images), batch_size)
    ]

    # ------------------------
    # 5. 推論時間計測（GPU演算のみ）
    # ------------------------
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

    avg_time_ms = (total_time / total_images) * 1000
    print(f"\n平均推論時間（画像読み込み除く）: {avg_time_ms:.2f} ms/枚")

if __name__ == "__main__":
    main()
