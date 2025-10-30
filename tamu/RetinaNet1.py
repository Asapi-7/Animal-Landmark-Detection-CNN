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
from resnet50_backbone import resnet50 
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7 # 👈 以前のModuleNotFoundErrorの修正
from torchvision.models.detection import RetinaNet

import torch.optim as optim

# データセット
class CustomObjectDetectionDataset(Dataset):
    # ⚠️ __init__を修正: rootではなく、画像パスのリストを受け取る
    def __init__(self, img_list, root, transforms=None):
        self.root = root # .ptsファイルを見つけるためにrootを保持
        self.transforms = transforms
        self.imgs = img_list # 👈 既に分割された画像パスのリストを使用
        
    def _parse_pts(self, pts_path):
        """
        .ptsファイルからバウンディングボックスとラベルをパースする関数
        """
        boxes = []
        labels = []

        # ファイルが存在するか確認
        if not os.path.exists(pts_path):
             # .ptsファイルがない場合は空リストを返す（物体なし）
             return np.array([], dtype=np.float32).reshape(0, 4), np.array([], dtype=np.int64)
        
        with open(pts_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # 形式: class_id x_min y_min x_max y_max
                class_id = int(parts[0])
                # 座標は整数に変換
                coords = [int(p) for p in parts[1:5]]
                
                boxes.append(coords)
                labels.append(class_id)
        
        # NumPy配列に変換
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        return boxes, labels
        
    def __getitem__(self, idx):
        # 1. 画像とPTSファイルのパス
        # self.imgs には 'dataset/img001.jpg' のような相対パスが入っていることを想定
        img_path_full = self.imgs[idx]
        
        # rootからファイル名を抽出（img_listが絶対パスの場合、ここではファイル名だけ抽出する）
        img_filename = os.path.basename(img_path_full)
        base_name = os.path.splitext(img_filename)[0]
        pts_filename = base_name + ".pts"
        
        # .ptsファイルのパスを作成
        pts_path = os.path.join(self.root, pts_filename)

        # 2. データ読み込み
        img = Image.open(img_path_full).convert("RGB") # 👈 修正: img_path_fullを使用
        boxes_np, labels_np = self._parse_pts(pts_path)

        # 3. ターゲット辞書の作成（RetinaNetの要求形式）
        if boxes_np.size == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(labels_np, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # 4. 変換（transforms）の適用
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Transformsの定義
def get_transform(train):
    t = [T.ToTensor()] 
    if train:
        # t.append(T.RandomHorizontalFlip(0.5))
        pass 
    return T.Compose(t)


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
    batch_size=4, 
    shuffle=True,
    num_workers=2, 
    collate_fn=custom_collate_fn 
)

# ⚠️ テストローダーも作成
test_loader = DataLoader(
    test_dataset,
    batch_size=4, 
    shuffle=False, # 評価時はシャッフル不要
    num_workers=2, 
    collate_fn=custom_collate_fn 
)

# =========================================================
# モデルの構築と学習ループ (変更なし)
# =========================================================

# ResNet50を使えるようにする
custom_backbone = resnet50(pretrained=False) 

# FPNを構築するための設定
out_channels = 256

backbone_fpn = _resnet_fpn_extractor(
    custom_backbone, 
    trainable_layers=5, 
    extra_blocks=LastLevelP6P7(out_channels, out_channels)
)

# RetinaNetモデルの構築
NUM_CLASSES = 10 

model = RetinaNet(
    backbone=backbone_fpn,
    num_classes=NUM_CLASSES,
    weights=None 
)

# デバイス設定
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# オプティマイザの定義
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0001
)

# 学習するエポック数
num_epochs = 10 

print(f"学習を {device} で開始します...")

model.train() # モデルをトレーニングモードに設定

for epoch in range(num_epochs):
    start_time = time.time()
    total_epoch_loss = 0
    
    for step, (images, targets) in enumerate(train_loader):
        # 1. データとターゲットをGPUに移動
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 2. 勾配をゼロクリア
        optimizer.zero_grad()

        # 3. フォワードパス: 損失を計算
        loss_dict = model(images, targets) 
        
        # 損失の合計
        losses = sum(loss for loss in loss_dict.values())
        total_epoch_loss += losses.item()

        # 4. バックワードパス: 勾配を計算
        losses.backward()

        # 5. オプティマイザのステップ: 重みを更新
        optimizer.step()
        
        # ログ出力
        if step % 50 == 0:
            print(f"  Epoch: {epoch+1}/{num_epochs}, Step: {step}, Total Loss: {losses.item():.4f}, Cls Loss: {loss_dict['classification'].item():.4f}")
    
    end_time = time.time()
    print(f"\n--- Epoch {epoch+1} 完了。 平均損失: {total_epoch_loss / len(train_loader):.4f}, 処理時間: {(end_time - start_time):.2f}s ---\n")

print("全学習プロセスが完了しました。")

# モデルの重みを保存
torch.save(model.state_dict(), 'retinanet_custom_weights_final.pth')
