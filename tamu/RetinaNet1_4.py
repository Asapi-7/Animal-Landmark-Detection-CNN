import os
import torch
import numpy as np
import time
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T_v2
from torchvision.tv_tensors import BoundingBoxes, Mask, Image as TVImage
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

# データセット
class CustomObjectDetectionDataset(Dataset):
    # ⚠️ __init__を修正: rootではなく、画像パスのリストを受け取る
    def __init__(self, img_list, root, transforms=None):
        self.root = root # .ptsファイルを見つけるためにrootを保持
        self.transforms = transforms
        self.imgs = img_list # 👈 既に分割された画像パスのリストを使用
        
    def _parse_pts(self, pts_path):
    
         #.ptsファイルから2点 (左上と右下など) を読み取り、
    
        boxes = []
        labels = []

        if not os.path.exists(pts_path):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)

        xs, ys = [], []
        with open(pts_path, 'r') as f:
            for line in f:
                line = line.strip()
            # 空行やヘッダー、波括弧をスキップ
                if not line or line.startswith("version") or line in ["{", "}"]:
                    continue

            # "129 100" のような座標ペアを読む
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
            labels = np.array([1], dtype=np.int64)  # ← 全て同じクラス扱い
        else:
            # 点が足りない場合は空にしておく
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

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

        """"
        # 2. データ読み込み
        img = Image.open(img_path_full).convert("RGB") # 👈 修正: img_path_fullを使用
        boxes_np, labels_np = self._parse_pts(pts_path)

        # 3. ターゲット辞書の作成（RetinaNetの要求形式）
        if boxes_np.size == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(labels_np, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # 4. 変換（transforms）の適用
        if self.transforms is not None:
            img = self.transforms(img)

        return img, target
        """
        # 2. データ読み込み
        img = Image.open(img_path_full).convert("RGB")
        boxes_np, labels_np = self._parse_pts(pts_path)

        # 3. ターゲット辞書の作成と v2 形式への変換 👈 ここを修正

        # 3-1. 画像のサイズを取得 (H, W) 224×224
        W, H = img.size # PIL Imageのサイズは (W, H)

        if boxes_np.size == 0:
            # BBOXがない場合は空のテンソルを作成
            boxes_tensor = torch.empty((0, 4), dtype=torch.float32)
        else:
            boxes_tensor = torch.as_tensor(boxes_np, dtype=torch.float32)

        labels_tensor = torch.as_tensor(labels_np, dtype=torch.int64)

        # 3-2. v2 形式の BoundingBoxes に変換
        boxes_v2 = BoundingBoxes(
            boxes_tensor, 
            format="XYXY",  # あなたのデータ形式に合わせる
            canvas_size=(H, W)
        )
        
        target = {}
        target["boxes"] = boxes_v2 # 👈 v2形式のBBOXを格納
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])
        
        # 4. 変換（transforms）の適用 👈 ターゲットも一緒に渡す
        if self.transforms is not None:
            # v2では、Transformsに画像とターゲットの両方を渡す
            img, target = self.transforms(img, target) 

        # 変換後、target["boxes"] は BoundingBoxes オブジェクトのままなので、
        # そのままRetinaNetに渡すことができます。

        return img, target

    def __len__(self):
        return len(self.imgs)

# Transformsの定義 データ拡張
"""""
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    return T.Compose(transforms)
"""

# Transformsの定義 データ拡張をv2に置き換える
def get_transform(train):
    transforms = []
    # v2のToTensor()を使用: PIL Image/NumPy array -> Tensorに変換
    transforms.append(T_v2.ToTensor()) 
    
    if train:
        # v2のRandomHorizontalFlipを使用: BBOXも自動でフリップされる
        transforms.append(T_v2.RandomHorizontalFlip(0.5))
        # v2のColorJitterを使用
        transforms.append(T_v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
        
    # T.ComposeではなくT_v2.Composeを使用
    return T_v2.Compose(transforms)


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
    dummy_image = torch.rand(1, 3, 224, 224)  # バッチサイズ1
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