import os
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T

from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from resnet18_backbone import resnet18  # ← あなたのresnet18_backbone.pyを使用

# ==========================
# モデル構築関数
# ==========================
def build_model(num_classes=2, weight_path=None, device="cpu"):
    backbone = resnet18(pretrained=False)
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),),
    )

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator
    )

    if weight_path is not None:
        print(f"✅ Loading model weights: {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location=device))

    model.to(device)
    model.eval()
    return model


# ==========================
# 画像の推論関数
# ==========================
def run_inference(model, image_path, device="cpu", score_thresh=0.5, save_result=True):
    # 画像読み込み
    img = Image.open(image_path).convert("RGB")

    # 前処理
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).to(device)

    # 推論
    with torch.no_grad():
        outputs = model([img_tensor])

    # 出力抽出
    boxes = outputs[0]['boxes'].cpu().numpy()
    scores = outputs[0]['scores'].cpu().numpy()

    if len(scores) == 0:
        print("⚠️ No detections found.")
        return

    # 最もスコアの高いBBoxを1つだけ選ぶ
    best_idx = np.argmax(scores)
    best_box = boxes[best_idx]
    best_score = scores[best_idx]

    # 閾値未満ならスキップ
    if best_score < score_thresh:
        print(f"⚠️ No box above threshold ({best_score:.2f} < {score_thresh})")
        return

    print(f"🔍 Best detection: score = {best_score:.2f}")

    # 結果の描画
    img_cv = np.array(img)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = best_box.astype(int)
    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_cv, f"{best_score:.2f}", (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 保存 or 表示
    if save_result:
        out_path = os.path.splitext(image_path)[0] + "_best_result.jpg"
        cv2.imwrite(out_path, img_cv)

    # 表示
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


# ==========================
# 実行部
# ==========================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === モデルの読み込み ===
    weight_path = "fasterrcnn3_resnet18_20.pth"  # ← 学習済みモデルのパスに変更
    model = build_model(num_classes=2, weight_path=weight_path, device=device)

    # === 推論する画像 ===
    test_image = "dog.jpg"  # ← 検出したい画像パスに変更

    run_inference(model, test_image, device=device, score_thresh=0.3)
