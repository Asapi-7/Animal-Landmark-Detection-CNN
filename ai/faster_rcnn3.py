import os
import torch
import numpy as np
from PIL import Image
import glob # 👈 追加: ファイルパスのリスト取得用
from sklearn.model_selection import train_test_split # 👈 追加: データ分割用
import random

import torchvision
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader

# モデル構築用
from resnet18_backbone import resnet18
from torchvision.models.detection.faster_rcnn import FasterRCNN, FastRCNNPredictor
from tqdm import tqdm
from torchvision.models.detection.rpn import AnchorGenerator

import torch.optim as optim
from torchvision.ops import box_iou

# ==========================
# カスタムDatasetの定義
# ==========================
class CustomObjectDetectionDataset(Dataset):
    def __init__(self, img_list, root, train=True):
        self.root = root
        self.imgs = img_list
        self.train = train

        # 物体検出向けに実用的なカラー拡張だけ残す
        self.color_transform = T.Compose([
            T.RandomGrayscale(p=0.2),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
            T.RandomAutocontrast(p=0.2),
        ])

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
            if xmax - xmin > 0 and ymax - ymin > 0:
                boxes = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32)
                labels = np.array([1], dtype=np.int64)
                return boxes, labels

        return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

            
    def __getitem__(self, idx):
        img_path_full = self.imgs[idx]
        img = Image.open(img_path_full).convert("RGB")
        W, H = img.size

        # バウンディングボックス取得
        boxes_np, labels_np = self._parse_pts(os.path.join(
            self.root, os.path.splitext(os.path.basename(img_path_full))[0] + ".pts"
        ))
        boxes = torch.as_tensor(boxes_np, dtype=torch.float32) if boxes_np.size > 0 else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels_np, dtype=torch.int64) if labels_np.size > 0 else torch.empty((0,), dtype=torch.int64)

        # ===== データ拡張 =====
        if self.train:
            # 1. 左右反転 50%
            if random.random() > 0.5:
                img = F.hflip(img)
                if boxes.numel() > 0:
                    boxes[:, [0, 2]] = W - boxes[:, [2, 0]]

            # 2. カラー拡張
            img = self.color_transform(img)

        # Tensor化
        img = F.to_tensor(img)

        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        return img, target

    
    def __len__(self):
        return len(self.imgs)


# Collate Functionの定義
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

#===========================
# データ準備
#===========================

DATA_ROOT = 'dataset'
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
train_dataset = CustomObjectDetectionDataset(train_imgs, DATA_ROOT, train=True)
test_dataset = CustomObjectDetectionDataset(test_imgs, DATA_ROOT, train=False)


# 4. DataLoaderの作成
train_loader = DataLoader(
    train_dataset,
    batch_size=16, 
    shuffle=True,
    num_workers=4, 
    collate_fn=custom_collate_fn 
)

# ⚠️ テストローダーも作成
test_loader = DataLoader(
    test_dataset,
    batch_size=16, 
    shuffle=False, # 評価時はシャッフル不要
    num_workers=4, 
    collate_fn=custom_collate_fn 
)

# ==========================
# モデル構築
# ==========================
backbone = resnet18(pretrained=False)
backbone.out_channels = 512
anchor_generator = AnchorGenerator(sizes = ((16,32,64,128),),
                                   aspect_ratios = ((0.5,1.0,2.0),))

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)

model = FasterRCNN(backbone,num_classes=2,
                   rpn_anchor_generator=anchor_generator)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)     

optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    patience=2,
    verbose=True,
    min_lr=1e-5)

# ==========================
# 評価関数
# ==========================
def evaluate_fasterRCNN(model,dataloader,device,iou_threshold=0.5):
    model.eval()
    total_images = 0
    correct_detections = 0
    total_iou_sum = 0.0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device).float() for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                total_images += 1
                pred_boxes = output['boxes']
                scores = output['scores']
                true_boxes = target['boxes']

                if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                    continue

                max_idx = scores.argmax()
                pred_box = pred_boxes[max_idx].unsqueeze(0)
                iou = box_iou(pred_box, true_boxes)[0, 0].item()
                total_iou_sum += iou

                if iou >= iou_threshold:
                    correct_detections += 1

                
    accuracy = correct_detections / total_images if total_images > 0 else 0
    avg_iou = total_iou_sum / total_images if total_images > 0 else 0.0

    print(f"\n--- 評価結果 ---")
    print(f"Accuracy (IoU > {iou_threshold}): {accuracy:.2f} ({correct_detections}/{total_images})")
    print(f"Average IoU: {avg_iou:.4f}")

    return avg_iou, accuracy


# ==========================
# 学習
# ==========================                            
import matplotlib.pyplot as plt

num_epochs = 20
train_loss_list = []
test_loss_list = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

    # ======== 学習ループ ========
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
        images = [img.to(device).to(torch.float32) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()

    avg_train_loss = train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)


    # ======== テストループ ========
    model.train()
    test_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]"):
            images = [img.to(device).to(torch.float32) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            test_loss += losses.item()

    avg_test_loss = test_loss / len(test_loader)
    test_loss_list.append(avg_test_loss)

    scheduler.step(avg_test_loss)

    # ======== ログ出力 ========
    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Test  Loss: {avg_test_loss:.4f}")

    # ======== チェックポイント保存 ========
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': avg_train_loss,
        'test_loss': avg_test_loss,
    }, f"checkpoint_epoch_{epoch+1}.pth")

    # ============================
    if(epoch + 1) %5 == 0 or (epoch +1) == num_epochs:
        evaluate_fasterRCNN(model, test_loader, device)
        save_path = f"fasterrcnn3_resnet18_{epoch+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"model save: {save_path}")
        
print("Training complete.")


# ==========================
# 📈 学習曲線のプロット
# ==========================
plt.figure(figsize=(8,5))
plt.plot(range(1, num_epochs+1), train_loss_list, marker='o', label='Train Loss')
plt.plot(range(1, num_epochs+1), test_loss_list, marker='s', label='Test Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Faster R-CNN: Train vs Test Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ==========================    
# 最終評価
# ==========================
evaluate_fasterRCNN(model, test_loader, device)
