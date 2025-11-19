import os
import torch
import numpy as np
from PIL import Image
import glob
from tqdm import tqdm

# -------------- あなたの backbone を import --------------
from resnet18_backbone import resnet18

# -------------- PyTorch detection --------------
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision.ops import box_iou

# ==========================
# Dataset（学習コードと完全互換）
# ==========================
class CustomObjectDetectionDataset(Dataset):
    def __init__(self, img_list, root):
        self.root = root
        self.imgs = img_list

    def _parse_pts(self, pts_path):
        xs, ys = [], []

        if not os.path.exists(pts_path):
            return np.empty((0, 4), np.float32), np.empty((0,), np.int64)

        with open(pts_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("version") or line in ["{", "}"]:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    x, y = float(parts[0]), float(parts[1])
                    xs.append(x)
                    ys.append(y)
                except ValueError:
                    continue

        if len(xs) >= 2 and len(ys) >= 2:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            if xmax - xmin > 0 and ymax - ymin > 0:
                boxes = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32)
                labels = np.array([1], dtype=np.int64)
                return boxes, labels
        
        return np.empty((0,4),np.float32), np.empty((0,),np.int64)

    def __getitem__(self, idx):
        img_path_full = self.imgs[idx]
        img = Image.open(img_path_full).convert("RGB")

        pts_path = os.path.join(
            self.root, os.path.splitext(os.path.basename(img_path_full))[0] + ".pts"
        )
        boxes_np, labels_np = self._parse_pts(pts_path)

        boxes = torch.as_tensor(boxes_np, dtype=torch.float32) if boxes_np.size > 0 else torch.empty((0, 4))
        labels = torch.as_tensor(labels_np, dtype=torch.int64) if labels_np.size > 0 else torch.empty((0,), dtype=torch.int64)

        img = F.to_tensor(img)
        
        # 画像パスも target に含める
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx]), "img_path": img_path_full}
        return img, target

    def __len__(self):
        return len(self.imgs)

def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# ==========================
# 評価関数（動物ごとの正答率）
# ==========================
def evaluate_fasterRCNN_per_animal(model, dataloader, device, iou_threshold=0.5):
    model.eval()
    animal_stats = {}  # {animal_name: {"total": int, "correct": int, "iou_sum": float}}

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device).float() for img in images]
            targets = [
                {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()}
                 for t in targets
            ]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                # target から画像パスを取得
                img_path = target["img_path"]
                animal_name = os.path.basename(img_path).split("_")[0]

                if animal_name not in animal_stats:
                    animal_stats[animal_name] = {"total": 0, "correct": 0, "iou_sum": 0.0}

                animal_stats[animal_name]["total"] += 1

                pred_boxes = output['boxes']
                scores = output['scores']
                true_boxes = target['boxes']

                if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                    continue

                max_idx = scores.argmax()
                pred_box = pred_boxes[max_idx].unsqueeze(0)

                iou = box_iou(pred_box, true_boxes)[0, 0].item()
                animal_stats[animal_name]["iou_sum"] += iou

                if iou >= iou_threshold:
                    animal_stats[animal_name]["correct"] += 1

    # 結果表示
    print(f"\n--- 動物ごとの評価結果 ---")
    for animal, stats in animal_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        avg_iou = stats["iou_sum"] / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        print(f"{animal:20} 正解数: {correct:3d} / {total:3d} ({accuracy*100:5.2f}%), 平均IoU: {avg_iou:.4f}")

    return animal_stats

# ==========================
# メイン処理
# ==========================
if __name__ == "__main__":

    MODEL_PATH = "fasterrcnn_resnet18_adam_20.pth"
    DATA_ROOT = "./daset_test/dataset_test"  # 評価したい画像フォルダ

    img_list = sorted(glob.glob(os.path.join(DATA_ROOT, "*.jpg")))
    print(f"評価対象の画像: {len(img_list)} 枚")

    dataset = CustomObjectDetectionDataset(img_list, DATA_ROOT)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=custom_collate_fn)

    # モデル構築（ResNet18 backbone）
    backbone = resnet18(pretrained=False)
    backbone.out_channels = 512

    anchor_generator = AnchorGenerator(
        sizes=((16, 32, 64, 128),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    model = FasterRCNN(
        backbone,
        num_classes=2,
        rpn_anchor_generator=anchor_generator
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)

    # 評価開始
    evaluate_fasterRCNN_per_animal(model, dataloader, device)
