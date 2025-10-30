import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader

# モデル構築用
from resnet50_backbone import resnet50 
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.models.detection import RetinaNet

import torch.optim as optim
import time


# データセット
class CustomObjectDetectionDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted([f for f in os.listdir(root) if f.endswith(".jpg")])
        
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
        img_filename = self.imgs[idx]
        base_name = os.path.splitext(img_filename)[0]
        pts_filename = base_name + ".pts"
        
        img_path = os.path.join(self.root, img_filename)
        pts_path = os.path.join(self.root, pts_filename)

        # 2. データ読み込み
        img = Image.open(img_path).convert("RGB")
        boxes_np, labels_np = self._parse_pts(pts_path)

        # 3. ターゲット辞書の作成（RetinaNetの要求形式）
        # 物体がない場合（boxes_npが空の場合）に対応
        if boxes_np.size == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(labels_np, dtype=torch.int64)
        
        # その他の必須ではないキーを追加
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # 4. 変換（transforms）の適用
        if self.transforms is not None:
            # PyTorch Object Detectionモデルでは、画像とターゲットの両方に変換を適用します
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

# Transformsの定義(画像の前処理・データ拡張)
def get_transform(train):
    """トレーニング（学習）用と評価用の変換パイプラインを定義"""
    
    # 標準的な前処理
    t = [T.ToTensor()] 
    
    if train:
        # 学習時のみ行うデータ拡張（オプション）
        # t.append(T.RandomHorizontalFlip(0.5))
        pass 
        
    return T.Compose(t)


# Collate Functionの定義(データを1つにまとめるバッチ形式に変換する)
def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets

# DataLoaderの構築
# データのルートディレクトリを指定 
DATA_ROOT = r'\\Users\st6323079@isws-dnnserver02\dataset' 

# Datasetのインスタンス作成
dataset = CustomObjectDetectionDataset(DATA_ROOT, get_transform(train=True))

# DataLoaderの作成
train_loader = DataLoader(
    dataset,
    batch_size=4, 
    shuffle=True,
    num_workers=2, # 環境に合わせて調整
    collate_fn=custom_collate_fn 
)



# ResNet50を使えるようにする
custom_backbone = resnet50(pretrained=False) # ResNetインスタンスを作成

# FPNを構築するための設定
in_channels_list = [512, 1024, 2048]
out_channels = 256

backbone_fpn = _resnet_fpn_extractor(
    custom_backbone, 
    return_layers={"layer2": "0", "layer3": "1", "layer4": "2"}, # FPNに渡す層
    in_channels_list=in_channels_list, 
    out_channels=out_channels, 
    extra_blocks=LastLevelP6P7(out_channels, out_channels), 
)

# RetinaNetモデルの構築
NUM_CLASSES = 10 

model = RetinaNet(
    backbone=backbone_fpn,
    num_classes=NUM_CLASSES,
    # weights=None で事前学習済みの重みロードをスキップ
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
        # imagesはリスト、targetsは辞書を含むリスト
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
            print(f"  Epoch: {epoch+1}/{num_epochs}, Step: {step}, Total Loss: {losses.item():.4f}, Cls Loss: {loss_dict['classification'].item():.4f}")
    
    end_time = time.time()
    print(f"\n--- Epoch {epoch+1} 完了。 平均損失: {total_epoch_loss / len(train_loader):.4f}, 処理時間: {(end_time - start_time):.2f}s ---\n")

print("全学習プロセスが完了しました。")

# モデルの重みを保存
torch.save(model.state_dict(), 'retinanet_custom_weights_final.pth')