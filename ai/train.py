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
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.models as models

# ==========================================================
# Dataset は ResNet18版と同じ
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
        H, W = img.shape[0], img.shape[1]
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        bb = torch.from_numpy(bb).float() / torch.tensor([W,H,W,H], dtype=torch.float)
        if self.augment and random.random() > 0.5:
            img = torch.flip(img, dims=[2])
            x1,y1,x2,y2 = bb
            bb[0] = 1.0 - x2
            bb[2] = 1.0 - x1
        return img, bb

# ==========================================================
# MobileNetV2 定義
# ==========================================================
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_output=4, pretrained=True):
        super().__init__()
        self.model = models.mobilenet_v2(pretrained=pretrained)
        # classifierの最後を顔検出用に置き換え
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_output)

    def forward(self, x):
        return self.model(x)

# ==========================================================
# データロード（ResNet18版と同じ）
# ==========================================================
DATA_DIR = os.path.join(os.getcwd(), "npy_dataset")
all_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.npy")))
train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)

train_dataset = AnimalWebDataset(train_files, augment=True)
test_dataset  = AnimalWebDataset(test_files, augment=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

# ==========================================================
# モデル・最適化
# ==========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
model = MobileNetV2Custom(num_output=4).to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ==========================================================
# 評価関数（ResNet18版と同じ）
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
                    img_np = imgs[i].cpu().permute(1,2,0).numpy()
