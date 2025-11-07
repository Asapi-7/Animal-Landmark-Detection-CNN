import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from PIL import Image
from torchvision import transforms as T
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from resnet18_backbone import resnet18
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7 
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import RetinaNet 

# ==========================
# モデル再構築
# ==========================
custom_backbone = resnet18(pretrained=False)
out_channels = 256
backbone_fpn = _resnet_fpn_extractor(
    custom_backbone, 
    trainable_layers=5, 
    extra_blocks=LastLevelP6P7(out_channels, out_channels)
)

base_sizes = [8, 16, 32, 64, 128, 256]
sizes_for_anchor = tuple((s,) for s in base_sizes[:6])
anchor_generator = AnchorGenerator(
    sizes=sizes_for_anchor,
    aspect_ratios=((0.5, 1.0, 2.0),) * 6
)

NUM_CLASSES = 2
model = RetinaNet(
    backbone=backbone_fpn,
    num_classes=NUM_CLASSES,
    anchor_generator=anchor_generator
)

weights_path = "retinanet_epoch20.pth"
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print(f"✅ モデルロード完了: {weights_path}")

# ==========================
# 推論関数（元解像度維持・単一顔用）
# ==========================
def run_inference(model, image_path, threshold=0.05, resize_to=None):
    model.eval()
    img = Image.open(image_path).convert("RGB")
    
    # resize_to=Noneなら元解像度のまま
    if resize_to is not None:
        img = img.resize(resize_to)

    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)

    output = outputs[0]
    boxes = output['boxes'].cpu()
    scores = output['scores'].cpu()
    labels = output['labels'].cpu()

    # スコア閾値でフィルタリング
    keep = scores >= threshold
    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # ✅ 最もスコアが高いボックスだけに絞る
    if len(boxes) > 0:
        best_idx = scores.argmax()
        boxes = boxes[best_idx:best_idx+1]
        scores = scores[best_idx:best_idx+1]
        labels = labels[best_idx:best_idx+1]

    print(f"推論結果: {len(boxes)} 個のボックス, スコア範囲 [{scores.min() if len(scores)>0 else 0:.3f}, {scores.max() if len(scores)>0 else 0:.3f}]")

    return img, boxes, scores, labels


# ==========================
# 可視化関数（スコアで色分け）
# ==========================
def visualize_predictions(img, boxes, scores, labels):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)

    for (box, score, label) in zip(boxes, scores, labels):
        xmin, ymin, xmax, ymax = box
        color = 'lime' if score >= 0.5 else 'red'
        rect = patches.Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(xmin, ymin - 5, f"{score:.2f}", color='yellow', fontsize=10, weight='bold')

    plt.axis('off')
    plt.show()

# ==========================
# 実行例
# ==========================
test_image_path = "./dataset/kirinn.jpg"
img, boxes, scores, labels = run_inference(model, test_image_path, threshold=0.1, resize_to=None)
visualize_predictions(img, boxes, scores, labels)