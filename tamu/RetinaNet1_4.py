import os
import torch
import numpy as np
import time
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader
import glob # 👈 追加: ファイルパスのリスト取得用
from sklearn.model_selection import train_test_split # 👈 追加: データ分割用

# モデル構築用
from resnet18_backbone import resnet18
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7 
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection import RetinaNet
from tqdm import tqdm

import torch.optim as optim
from torchvision.ops import box_iou

import random
import cv2

# データセット
class CustomObjectDetectionDataset(Dataset):
    def __init__(self, img_list, root, transforms=None, augment=False):
        """
        img_list: 画像ファイルのリスト
        root: .ptsファイルのあるルートディレクトリ
        augment: データ拡張を行うかどうか
        """
        self.root = root
        self.imgs = img_list
        self.transforms = transforms
        self.augment = augment

        # カラージッター設定（BBoxに影響しない）
        self.color_transform = T.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2
        )
        self.to_tensor = T.ToTensor()

    def _parse_pts(self, pts_path):
        boxes = []
        labels = []

        if not os.path.exists(pts_path):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

        xs, ys = [], []
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
            boxes = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32)
            labels = np.array([1], dtype=np.int64)
        else:
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        return boxes, labels

    def __getitem__(self, idx):
        img_path_full = self.imgs[idx]
        img_filename = os.path.basename(img_path_full)
        base_name = os.path.splitext(img_filename)[0]
        pts_path = os.path.join(self.root, base_name + ".pts")

        # --- 画像読み込み (OpenCVでBGR→RGB変換)
        img = cv2.imread(img_path_full)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape

        # --- BBox取得
        boxes_np, labels_np = self._parse_pts(pts_path)

        # --- データ拡張 ---
        if self.augment and boxes_np.size > 0:
            x1, y1, x2, y2 = boxes_np[0]

            # 1. 左右反転（確率50%）
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
                x1_new = W - x2
                x2_new = W - x1
                x1, x2 = x1_new, x2_new

            boxes_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)

            # 2. 色変換（BBoxに影響しない）
            pil_img = T.functional.to_pil_image(img)
            img = self.color_transform(pil_img)
            img = T.functional.to_tensor(img)  # PIL→Tensor
        else:
            img = self.to_tensor(img)

        # --- ターゲット作成 ---
        if boxes_np.size == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(labels_np, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return img, target

    def __len__(self):
        return len(self.imgs)

# 画像変換（トランスフォーム）を返す関数
def get_transform(train=False):
    transforms = []

    # PIL → Tensor 変換
    transforms.append(T.ToTensor())

    return T.Compose(transforms)


# Collate Functionの定義
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# =========================================================
# データの読み込みと分割 (この部分がデータセット分割の核心です)
# =========================================================

# データのルートディレクトリを指定（画像と.ptsファイルがある場所）
DATA_ROOT = '/workspace/dataset'

# 1. 全ての画像ファイルパスを取得
# os.path.join(DATA_ROOT, "*.jpg") は、例: /workspace/dataset/*.jpg になります
all_imgs = sorted(glob.glob(os.path.join(DATA_ROOT, "*.jpg")))

print(f"発見した全サンプル数: {len(all_imgs)}")

# 2. 学習用 (80%) とテスト用 (20%) に分割
# test_size=0.2 で 20% をテスト用に割り当てる
train_imgs, test_imgs = train_test_split(
    all_imgs, 
    test_size=0.2, 
    random_state=42 # シード固定で再現性を確保
)

print(f"学習用サンプル数 (80%): {len(train_imgs)}, テスト用サンプル数 (20%): {len(test_imgs)}")


# 3. Datasetのインスタンス作成（分割したリストを渡す）
train_dataset = CustomObjectDetectionDataset(train_imgs, DATA_ROOT, get_transform(train=True))
test_dataset = CustomObjectDetectionDataset(test_imgs, DATA_ROOT, get_transform(train=False))


# 4. DataLoaderの作成
train_loader = DataLoader(
    train_dataset,
    batch_size=16, 
    shuffle=True,
    num_workers=2, 
    collate_fn=custom_collate_fn 
)

# ⚠️ テストローダーも作成
test_loader = DataLoader(
    test_dataset,
    batch_size=16, 
    shuffle=False, # 評価時はシャッフル不要
    num_workers=2, 
    collate_fn=custom_collate_fn 
)

# =========================================================
# モデルの構築と学習ループ (変更なし)
# =========================================================

# ResNet18を使えるようにする
custom_backbone = resnet18(pretrained=False) 

# FPNを構築するための設定
out_channels = 256

backbone_fpn = _resnet_fpn_extractor(
    custom_backbone, 
    trainable_layers=5, 
    extra_blocks=LastLevelP6P7(out_channels, out_channels)
)


# =========================================================
# 1️⃣ FPN 出力層数を確認して AnchorGenerator を自動設定
# =========================================================

# ダミー画像をFPNに通して出力層の構造を確認
with torch.no_grad():
    dummy_image = torch.rand(1, 3, 224, 224)  # バッチサイズ1 RGBの3
    features = backbone_fpn(dummy_image)
    print("FPN 出力層のキー:", list(features.keys()))
    print("各層の出力形状:")
    for k, v in features.items():
        print(f"  {k}: {tuple(v.shape)}")

num_feature_maps = len(features)
print("FPN 出力層数:", num_feature_maps)

# AnchorGenerator を出力層数に合わせて作成
base_sizes = [8, 16, 32, 64, 128, 224]
sizes_for_anchor = tuple((s,) for s in base_sizes[:num_feature_maps])

anchor_generator = AnchorGenerator(
    sizes=sizes_for_anchor,
    aspect_ratios=((0.5, 1.0, 2.0),) * num_feature_maps
)

print("AnchorGenerator 設定:", anchor_generator)


# RetinaNetモデルの構築
NUM_CLASSES = 2

model = RetinaNet(
    backbone=backbone_fpn,
    num_classes=NUM_CLASSES,
    anchor_generator=anchor_generator
)

# ==========================================================
# 学習・評価ループ（RetinaNet + IoU/精度統合）
# ==========================================================

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# オプティマイザ
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.001,
    momentum=0.9,
    weight_decay=0.001
)

# スケジューラー
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=5,
    gamma=0.1
)

# 評価関数
def evaluate_retinanet(model, dataloader, device, iou_threshold=0.5):
    """
    1画像につき予測を1つだけに制限して評価
    正解ボックスも1つだけの想定
    """
    model.eval()
    
    total_ground_truth_boxes = 0
    total_pred_boxes = 0
    total_correct_detections_for_recall = 0
    total_correct_detections_for_precision = 0
    total_iou_sum = 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device).to(torch.float32) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                scores = output['scores']  # スコアも取得
                true_boxes = target['boxes']

                # --- 予測を1つだけに制限 ---
                if pred_boxes.size(0) > 0:
                    max_idx = scores.argmax()
                    pred_boxes = pred_boxes[max_idx].unsqueeze(0)  # [1,4]

                total_pred_boxes += pred_boxes.size(0)

                if true_boxes.size(0) == 0:
                    continue  # 正解BOXがない場合はスキップ

                total_ground_truth_boxes += true_boxes.size(0)

                if pred_boxes.size(0) == 0:
                    continue  # 予測BOXがない場合はスキップ

                ious = box_iou(pred_boxes, true_boxes)  # [1,1] の想定

                # Recall (正解BOX基準)
                if ious.max() >= iou_threshold:
                    total_correct_detections_for_recall += 1
                    total_iou_sum += ious.max().item()

                # Precision (予測BOX基準)
                if ious.max() >= iou_threshold:
                    total_correct_detections_for_precision += 1

    # 指標計算
    recall = (total_correct_detections_for_recall / total_ground_truth_boxes * 100.0
              if total_ground_truth_boxes > 0 else 0.0)
    precision = (total_correct_detections_for_precision / total_pred_boxes * 100.0
                 if total_pred_boxes > 0 else 0.0)
    avg_iou = (total_iou_sum / total_correct_detections_for_recall
               if total_correct_detections_for_recall > 0 else 0.0)

    print(f"\n--- 評価結果 (1予測/画像) ---")
    print(f"Recall (IoU > {iou_threshold}): {recall:.2f}% ({total_correct_detections_for_recall}/{total_ground_truth_boxes})")
    print(f"Precision (IoU > {iou_threshold}): {precision:.2f}% ({total_correct_detections_for_precision}/{total_pred_boxes})")
    print(f"Average IoU: {avg_iou:.4f}")

    return avg_iou, recall, precision

# ==========================================================
# 学習ループ
# ==========================================================
num_epochs = 20

for epoch in range(num_epochs):
    print(f"\n=== Epoch {epoch+1}/{num_epochs} ===")
    model.train()
    total_epoch_loss = 0.0

    for step, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
        images = [img.to(device).to(torch.float32) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_epoch_loss += losses.item()

        # NaNチェック
        if torch.isnan(losses):
            print(f"⚠️ NaN detected at step {step}, skipping this batch.")
            continue

        losses.backward()
        optimizer.step()

        # ログ
        if step % 50 == 0:
            print(f"Step {step}, Total Loss: {losses.item():.4f}, "
                  f"Cls Loss: {loss_dict['classification'].item():.4f}, "
                  f"Box Loss: {loss_dict['bbox_regression'].item():.4f}")
            
    # 学習率の出力
    current_lr = optimizer.param_groups[0]["lr"]
    tqdm.write(f"LR: {current_lr:.6f}")
    
    # スケジューラーステップ：学習率を調整
    scheduler.step()

    print(f"--- Epoch {epoch+1} 完了: 平均損失 {total_epoch_loss/len(train_loader):.4f} ---")

    # 10エポックごとに評価
    if (epoch + 1) % 5 == 0:
        print(f"\n--- 評価 (Epoch {epoch+1}) ---")
        evaluate_retinanet(model, test_loader, device, iou_threshold=0.5)
        torch.save(model.state_dict(), f"retinanet_epoch{epoch+1}.pth")
        print(f"Checkpoint saved: retinanet_epoch{epoch+1}.pth")

# ==========================================================
# 学習完了後の最終評価
# ==========================================================
torch.save(model.state_dict(), 'retinanet_custom_weights_final.pth')
print("\n--- 最終評価 ---")
evaluate_retinanet(model, test_loader, device, iou_threshold=0.5)