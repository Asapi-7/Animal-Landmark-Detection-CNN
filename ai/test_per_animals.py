import os
import torch
import numpy as np
from PIL import Image
from torchvision.ops import box_iou
from torchvision import transforms as T
from tqdm import tqdm
from resnet18_backbone import resnet18
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops.feature_pyramid_network import LastLevelP6P7 
from torchvision.models.detection import RetinaNet
from your_dataset_file import CustomObjectDetectionDataset, custom_collate_fn  # 👈 あなたの定義を使う

# ==========================================================
# 準備
# ==========================================================

DATA_ROOT = '/workspace/dataset'
MODEL_PATH = 'retinanet_custom_weights_final.pth'  # 学習済みモデルのパス

# 画像一覧の取得
import glob
all_imgs = sorted(glob.glob(os.path.join(DATA_ROOT, "*.jpg")))

# テストデータの再現（train_test_splitでrandom_state固定してたので同じ分割に）
from sklearn.model_selection import train_test_split
_, test_imgs = train_test_split(all_imgs, test_size=0.2, random_state=42)

# Dataset / DataLoader
test_dataset = CustomObjectDetectionDataset(test_imgs, DATA_ROOT, transforms=T.ToTensor())
from torch.utils.data import DataLoader
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# ==========================================================
# モデル構築（学習時と同じ設定）
# ==========================================================
custom_backbone = resnet18(pretrained=False)
out_channels = 256
backbone_fpn = _resnet_fpn_extractor(
    custom_backbone, trainable_layers=5, extra_blocks=LastLevelP6P7(out_channels, out_channels)
)

# AnchorGenerator設定（学習時と同じ）
base_sizes = [8, 16, 32, 64, 128, 256]
sizes_for_anchor = tuple((s,) for s in base_sizes[:5])
anchor_generator = AnchorGenerator(
    sizes=sizes_for_anchor,
    aspect_ratios=((0.5, 1.0, 2.0),) * len(sizes_for_anchor)
)

model = RetinaNet(backbone=backbone_fpn, num_classes=2, anchor_generator=anchor_generator)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ==========================================================
# 動物ごとの集計
# ==========================================================

# 集計用辞書
stats = {}  # { 'dog': {'correct': 0, 'total': 0}, ... }

def extract_animal_name(path):
    """ファイル名から動物名を抽出 (例: cat_001.jpg → 'cat')"""
    base = os.path.basename(path)
    name = base.split("_")[0]  # アンダースコア区切りを想定
    return name.lower()

iou_threshold = 0.5

with torch.no_grad():
    for (images, targets), img_path in tqdm(zip(test_loader, test_imgs), total=len(test_imgs)):
        animal = extract_animal_name(img_path)
        stats.setdefault(animal, {"correct": 0, "total": 0})

        images = [img.to(device).to(torch.float32) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images)
        output = outputs[0]
        target = targets[0]

        pred_boxes = output['boxes']
        scores = output['scores']
        true_boxes = target['boxes']

        stats[animal]["total"] += 1

        if pred_boxes.size(0) > 0:
            # 一番スコアの高い予測を採用
            max_idx = scores.argmax()
            pred_boxes = pred_boxes[max_idx].unsqueeze(0)
        else:
            continue  # 予測が無い → 誤り扱い

        if true_boxes.size(0) == 0:
            continue  # 正解ボックス無し → スキップ

        iou = box_iou(pred_boxes, true_boxes).max().item()
        if iou >= iou_threshold:
            stats[animal]["correct"] += 1

# ==========================================================
# 結果表示
# ==========================================================
print("\n=== 動物ごとの正答率 ===")
for animal, v in stats.items():
    acc = v["correct"] / v["total"] * 100 if v["total"] > 0 else 0.0
    print(f"{animal:10s}  正解数: {v['correct']:3d} / {v['total']:3d}  ({acc:.2f}%)")

# 全体平均
total_correct = sum(v["correct"] for v in stats.values())
total_total = sum(v["total"] for v in stats.values())
overall_acc = total_correct / total_total * 100 if total_total > 0 else 0.0
print(f"\n全体正答率: {overall_acc:.2f}%")
