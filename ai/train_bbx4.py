# ==========================================================
# ライブラリのインポート
# ==========================================================
import os
import glob
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations.pytorch import ToTensorV2
import cv2

# ==========================================================
# Dataset クラス (.jpg + .pts 用)
# ==========================================================
class AnimalWebDataset(Dataset):
    def __init__(self, img_files, augment=False):
        self.img_files = img_files
        self.augment = augment

        self.normalization = albu.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )

        bbox_params = albu.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.1,  # 見えなくなる bbox を削除しすぎない
            check_each_transform=True
        )

        if self.augment:
            self.transform = albu.Compose([
                albu.HorizontalFlip(p=0.5),
                albu.Rotate(limit=10, p=0.5),
                albu.RandomBrightnessContrast(p=0.5),
                albu.GaussNoise(var_limit=(5.0, 20.0), p=0.5),
                self.normalization,
                ToTensorV2()
            ], bbox_params=bbox_params)
        else:
            self.transform = albu.Compose([
                self.normalization,
                ToTensorV2()
            ], bbox_params=bbox_params)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        name = os.path.splitext(os.path.basename(img_path))[0]
        pts_path = os.path.join(os.path.dirname(img_path), f"{name}.pts")

        # --- 画像読み込み ---
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"画像が読み込めません: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # --- バウンディングボックス読み込み ---
        with open(pts_path, "r") as f:
            lines = f.readlines()
        coords = []
        read = False
        for line in lines:
            line = line.strip()
            if line == "{":
                read = True
                continue
            if line == "}":
                break
            if read:
                x, y = map(float, line.split())
                coords.append([x, y])

        if len(coords) != 2:
            raise ValueError(f"PTSファイルの形式が不正: {pts_path}")

        bb = [coords[0][0], coords[0][1], coords[1][0], coords[1][1]]
        labels = [0]  # ダミーラベル

        # 変換適用
        transformed = self.transform(image=img, bboxes=[bb], labels=labels)
        img_tensor = transformed['image']
        H_new, W_new = img_tensor.shape[1:]

        # --- bbox 消失時の補完・クリップ ---
        if len(transformed['bboxes']) == 0:
            # 拡張でbboxが消えた場合、無効な [0,0,0,0] を返す
            bb_tensor = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        else:
            # 変換後のbboxピクセル座標を取得
            bb_new_pixels = torch.tensor(transformed['bboxes'][0], dtype=torch.float32)
            
            # 変換後のサイズ (H_new, W_new) で正規化 (0-1の範囲に)
            bb_tensor = bb_new_pixels / torch.tensor([W_new, H_new, W_new, H_new], dtype=torch.float32)
            
            # 念のため 0.0-1.0 の範囲にクリップする
            bb_tensor = torch.clamp(bb_tensor, min=0.0, max=1.0)

        return img_tensor, bb_tensor


# ==========================================================
# ResNet18 定義（変更なし）
# ==========================================================
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18Custom(nn.Module):
    def __init__(self, num_output=4):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_output)

    def _make_layer(self, out_channels, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(self.fc(x)) # 0-1に正規化

        return x


# ==========================================================
# データロード (.jpg + .pts)
# ==========================================================
DATA_DIR = os.path.join(os.getcwd(), "jpg_pts_dataset")
all_imgs = sorted(glob.glob(os.path.join(DATA_DIR, "*.jpg")))
print(f"発見したサンプル数: {len(all_imgs)}")

train_imgs, test_imgs = train_test_split(all_imgs, test_size=0.2, random_state=42)
print(f"Train: {len(train_imgs)}, Test: {len(test_imgs)}")

train_dataset = AnimalWebDataset(train_imgs, augment=True)
test_dataset = AnimalWebDataset(test_imgs, augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# IOU計算関数
def calculate_iou(box1, box2, eps=1e-7):
    xi1 = torch.max(box1[..., 0], box2[..., 0])
    yi1 = torch.max(box1[..., 1], box2[..., 1])
    xi2 = torch.min(box1[..., 2], box2[..., 2])
    yi2 = torch.min(box1[..., 3], box2[..., 3])

    inter_width = (xi2 - xi1).clamp(0)
    inter_height = (yi2 - yi1).clamp(0)
    inter_area = inter_width * inter_height

    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / (union_area + eps)
    iou = torch.clamp(iou, 0.0, 1.0)  # 安全化

    return iou

# ==========================================================
# 評価関数（可視化なし）
# ==========================================================
def evaluate_model(model, test_loader, criterion, device, iou_threshold=0.5):
    """
    モデルを評価し、ロスとIoUベースの精度（%）を返す。
    """
    model.eval()
    total_loss = 0
    count = 0
    correct_iou_count = 0  # IoUがしきい値を超えた数をカウント

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # モデルによる予測
            outputs = model(imgs)
            
            # ロスの計算
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

            # IoU精度の計算
            iou_scores = calculate_iou(outputs, labels)
            correct_iou_count += (iou_scores >= iou_threshold).sum().item()

    avg_loss = total_loss / count
    accuracy_percent = (correct_iou_count / count) * 100.0
    
    return avg_loss, accuracy_percent


# ==========================================================
# 学習ループ（可視化なし）
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet18Custom(num_output=4).to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_losses = []
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f"resnet18_animalweb_epoch{epoch+1}.pth")
        print(f"Checkpoint saved at epoch {epoch+1}")
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# ==========================================================
# 学習済みモデル評価
# ==========================================================
WEIGHT_FILE = "resnet18_animalweb_epoch50.pth"

if os.path.exists(WEIGHT_FILE):
    model_eval = ResNet18Custom(num_output=4).to(device)
    model_eval.load_state_dict(torch.load(WEIGHT_FILE, map_location=device))
    print(f"Loaded weights from: {WEIGHT_FILE}")

    final_test_loss, final_test_accuracy = evaluate_model(
        model_eval, 
        test_loader, 
        criterion, 
        device, 
        iou_threshold=0.5
    )

    print("\n--- Final Evaluation Results ---")
    print(f"Test Loss: {final_test_loss:.4f}")
    print(f"Test Accuracy (IoU > 0.5): {final_test_accuracy:.2f}%")
else:
    print(f"Error: Weight file not found at {WEIGHT_FILE}")