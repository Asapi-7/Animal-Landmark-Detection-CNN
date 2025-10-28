# ==========================================================
# ライブラリのインポート
# ==========================================================
import os
import glob
import random
from tqdm import tqdm
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ==========================================================
# Dataset クラス (.jpg + .pts 用)
# ==========================================================
class AnimalWebDataset(Dataset):
    def __init__(self, img_files, augment=False):
        self.img_files = img_files
        self.augment = augment

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
        H, W = img.shape[:2]

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

        # --- Tensor変換 ---
        bb = torch.tensor([coords[0][0], coords[0][1], coords[1][0], coords[1][1]], dtype=torch.float32)

        # --- 正規化 ---
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        bb = bb / torch.tensor([W, H, W, H], dtype=torch.float32)

        # --- データ拡張（左右反転） ---
        if self.augment and random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            x1, y1, x2, y2 = bb
            bb = torch.tensor([1 - x2, y1, 1 - x1, y2], dtype=torch.float32)

        return img, bb


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


# ==========================================================
# 評価関数
# ==========================================================
def evaluate_model(model, test_loader, criterion, device, visualize=False, max_visualize=5):
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

            if visualize:
                for i in range(min(imgs.size(0), max_visualize)):
                    img_t = imgs[i].cpu()
                    bb = outputs[i].cpu()
                    H, W = img_t.shape[1:]
                    x1, y1, x2, y2 = (bb * torch.tensor([W, H, W, H])).int()
                    img_vis = img_t.permute(1, 2, 0).clone()
                    img_vis = img_vis.numpy()  # 可視化時のみnumpyへ
                    import cv2
                    cv2.rectangle(
                        img_vis,
                        (x1.item(), y1.item()),
                        (x2.item(), y2.item()),
                        color=(1, 0, 0),
                        thickness=2
                    )
                    plt.figure()
                    plt.imshow(img_vis)
                    plt.title("Predicted BB")
                    plt.axis('off')
                    plt.show()

    avg_loss = total_loss / count
    return avg_loss


# ==========================================================
# 学習ループ
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
        test_loss = evaluate_model(model, test_loader, criterion, device, visualize=True, max_visualize=3)
        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}")

final_test_loss = evaluate_model(model, test_loader, criterion, device, visualize=True, max_visualize=5)
print(f"Final Test Loss: {final_test_loss:.4f}")

plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve")
plt.show()

SAVE_PATH = os.path.join(os.getcwd(), "resnet18_animalweb.pth")
torch.save(model.state_dict(), SAVE_PATH)
print(f"モデルを保存しました: {SAVE_PATH}")
