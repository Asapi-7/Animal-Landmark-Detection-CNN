# ==========================================================
# ライブラリのインポート
# ==========================================================
import os
import glob
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# Dataset クラス
# ==========================================================
class AnimalWebDataset(Dataset):
    def __init__(self, file_paths, augment=False):
        self.file_paths = file_paths
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True).item()
        img = data["img"]
        bb = data["bb"].copy()

        # img を [C,H,W] に変換＆0~1正規化
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        bb = torch.from_numpy(bb).float()

        # データ拡張: 左右反転
        if self.augment and random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            x1, y1, x2, y2 = bb
            img_width = img.shape[2]
            bb[0] = img_width - x2
            bb[2] = img_width - x1

        return img, bb

# ==========================================================
# ResNet18 定義
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
        strides = [stride] + [1]*(blocks-1)
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
# データロード
# ==========================================================
DATA_DIR = os.path.join(os.getcwd(), "npy_dataset")  # 作業ディレクトリ直下
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
print(f"発見したサンプル数: {len(all_files)}")

from sklearn.model_selection import train_test_split
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
print(f"Train: {len(train_files)}, Test: {len(test_files)}")

train_dataset = AnimalWebDataset(train_files, augment=True)
test_dataset  = AnimalWebDataset(test_files, augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# ==========================================================
# モデル・最適化
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet18Custom(num_output=4).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================================
# 学習ループ
# ==========================================================
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {running_loss/len(train_loader):.4f}")

# ==========================================================
# モデル保存
# ==========================================================
SAVE_PATH = os.path.join(os.getcwd(), "resnet18_animalweb.pth")
torch.save(model.state_dict(), SAVE_PATH)
print(f"モデルを保存しました: {SAVE_PATH}")
# ==========================================================
# ライブラリのインポート
# ==========================================================
import os
import glob
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# Dataset クラス
# ==========================================================
class AnimalWebDataset(Dataset):
    def __init__(self, file_paths, augment=False):
        self.file_paths = file_paths
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True).item()
        img = data["img"]
        bb = data["bb"].copy()

        # img を [C,H,W] に変換＆0~1正規化
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        bb = torch.from_numpy(bb).float()

        # データ拡張: 左右反転
        if self.augment and random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            x1, y1, x2, y2 = bb
            img_width = img.shape[2]
            bb[0] = img_width - x2
            bb[2] = img_width - x1

        return img, bb

# ==========================================================
# ResNet18 定義
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
        strides = [stride] + [1]*(blocks-1)
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
# データロード
# ==========================================================
DATA_DIR = os.path.join(os.getcwd(), "npy_dataset")  # 作業ディレクトリ直下
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
print(f"発見したサンプル数: {len(all_files)}")

from sklearn.model_selection import train_test_split
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
print(f"Train: {len(train_files)}, Test: {len(test_files)}")

train_dataset = AnimalWebDataset(train_files, augment=True)
test_dataset  = AnimalWebDataset(test_files, augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# ==========================================================
# モデル・最適化
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet18Custom(num_output=4).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================================
# 学習ループ
# ==========================================================
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {running_loss/len(train_loader):.4f}")

# ==========================================================
# モデル保存
# ==========================================================
SAVE_PATH = os.path.join(os.getcwd(), "resnet18_animalweb.pth")
torch.save(model.state_dict(), SAVE_PATH)
print(f"モデルを保存しました: {SAVE_PATH}")
# ==========================================================
# ライブラリのインポート
# ==========================================================
import os
import glob
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================================
# Dataset クラス
# ==========================================================
class AnimalWebDataset(Dataset):
    def __init__(self, file_paths, augment=False):
        self.file_paths = file_paths
        self.augment = augment

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        data = np.load(self.file_paths[idx], allow_pickle=True).item()
        img = data["img"]
        bb = data["bb"].copy()

        # img を [C,H,W] に変換＆0~1正規化
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        bb = torch.from_numpy(bb).float()

        # データ拡張: 左右反転
        if self.augment and random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            x1, y1, x2, y2 = bb
            img_width = img.shape[2]
            bb[0] = img_width - x2
            bb[2] = img_width - x1

        return img, bb

# ==========================================================
# ResNet18 定義
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
        strides = [stride] + [1]*(blocks-1)
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
# データロード
# ==========================================================
DATA_DIR = os.path.join(os.getcwd(), "npy_dataset")  # 作業ディレクトリ直下
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
print(f"発見したサンプル数: {len(all_files)}")

from sklearn.model_selection import train_test_split
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
print(f"Train: {len(train_files)}, Test: {len(test_files)}")

train_dataset = AnimalWebDataset(train_files, augment=True)
test_dataset  = AnimalWebDataset(test_files, augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# ==========================================================
# モデル・最適化
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNet18Custom(num_output=4).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================================
# 学習ループ
# ==========================================================
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (imgs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]")):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {running_loss/len(train_loader):.4f}")

# ==========================================================
# モデル保存
# ==========================================================
SAVE_PATH = os.path.join(os.getcwd(), "resnet18_animalweb.pth")
torch.save(model.state_dict(), SAVE_PATH)
print(f"モデルを保存しました: {SAVE_PATH}")
