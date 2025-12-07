import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

# ============================
# PIL単体で画像に描画する関数
# ============================
def draw_landmarks_pil(image, landmarks, color='red', point_size=5):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except IOError:
        font = ImageFont.load_default()

    # tensor→numpyに変換
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()

    # (N,2) 形式でない場合 → reshape
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(-1, 2)

    for i, (x, y) in enumerate(landmarks, start=1):
        x, y = int(x), int(y)
        draw.ellipse([x-point_size, y-point_size, x+point_size, y+point_size], fill=color)
        draw.text((x+point_size+3, y+point_size+3), str(i), fill="yellow", font=font)

    return image


# ============================
# ランドマーク予測を画像に描画して保存
# ============================
def save_landmark_predictions(model, data_loader, device, num_samples=5, save_dir="./predictions_output",
                              IMG_SIZE=224, NUM_LANDMARKS=9):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    saved_count = 0

    with torch.no_grad():
        for images, targets, img_paths in data_loader:
            outputs = model(images.to(device)).cpu()  # (B, N*2)

            for idx in range(images.size(0)):
                if saved_count >= num_samples: return

                img = Image.open(img_paths[idx]).convert("RGB")
                W, H = img.size

                lm = outputs[idx].numpy().reshape(-1, 2)  # (9,2)

                # 元解像度へ変換
                lm[:, 0] *= (W / IMG_SIZE)
                lm[:, 1] *= (H / IMG_SIZE)

                # ------- Matplotlib描画 -------
                fig, ax = plt.subplots(figsize=(8,8))
                ax.imshow(img)
                ax.scatter(lm[:,0], lm[:,1], color='red', s=50)

                # 番号
                for i in range(NUM_LANDMARKS):
                    ax.text(lm[i,0]+8, lm[i,1]+8, str(i+1), color='yellow', fontsize=18)

                # 幾何処理（円2つ＋ライン）
                p12 = (lm[0], lm[1])
                p34 = (lm[2], lm[3])

                for pA, pB in [p12, p34]:
                    cx = (pA[0]+pB[0])/2; cy = (pA[1]+pB[1])/2
                    r = np.linalg.norm(pA-pB)/2
                    ax.add_patch(plt.Circle((cx,cy), r, fill=False, color='red', linewidth=2))

                idxs=[5,7,6,8,5]
                ax.plot(lm[idxs,0], lm[idxs,1], color='red', linewidth=3)

                ax.axis("off")
                out=f"{save_dir}/pred_{saved_count+1:03}.jpg"
                plt.savefig(out,bbox_inches='tight',pad_inches=0)
                plt.close(); saved_count+=1

                print(f"保存：{out}")
