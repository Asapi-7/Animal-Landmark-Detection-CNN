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

    def __len__(self):
        return len(self.imgs)

# Transformsの定義
def get_transform(train):
    t = [T.ToTensor()] 
    if train:
        # t.append(T.RandomHorizontalFlip(0.5))
        pass 
    return T.ToTensor()


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
    batch_size=8, 
    shuffle=True,
    num_workers=2, 
    collate_fn=custom_collate_fn 
)

# ⚠️ テストローダーも作成
test_loader = DataLoader(
    test_dataset,
    batch_size=8, 
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


# ダミー画像を作成（バッチサイズ1、3チャンネル、224x224）
dummy_images = [torch.rand(3, 224, 224)]

# FPN に通して出力を確認
features = backbone_fpn(dummy_images)
print("FPN 出力層のキー:", list(features.keys()))
num_feature_maps = len(features)
print("FPN 出力層数:", num_feature_maps)

# AnchorGenerator を出力層数に合わせて作成
# サイズは小さい順に適当に設定（必要に応じて調整可）
base_sizes = [32, 64, 128, 256, 512]  # 最大5層まで
# 実際の層数に合わせてスライス
sizes_for_anchor = tuple((s,) for s in base_sizes[:num_feature_maps])

anchor_generator = AnchorGenerator(
    sizes=sizes_for_anchor,
    aspect_ratios=((0.5, 1.0, 2.0),) * num_feature_maps
)

print("AnchorGenerator 設定:", anchor_generator)



# RetinaNetモデルの構築
NUM_CLASSES = 1

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
    lr=0.0001,
    momentum=0.9,
    weight_decay=0.001
)

# 評価関数
def evaluate_retinanet(model, dataloader, device, iou_threshold=0.5):
    model.eval()
    
    total_ground_truth_boxes = 0 # 全正解BOXの総数
    total_pred_boxes = 0         # 全予測BOXの総数
    
    # Recall用: IoU>th を満たした「正解BOX」の数
    total_correct_detections_for_recall = 0
    
    # Precision用: IoU>th を満たした「予測BOX」の数
    total_correct_detections_for_precision = 0 
    
    total_iou_sum = 0.0          # 検出成功したBOXのIoU合計

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"):
            images = [img.to(device).to(torch.float32) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                true_boxes = target['boxes']

                total_pred_boxes += pred_boxes.size(0)
                
                if true_boxes.size(0) == 0:
                    continue # 正解がない画像はスキップ

                total_ground_truth_boxes += true_boxes.size(0)

                if pred_boxes.size(0) == 0:
                    continue # 予測がない場合はスキップ

                ious = box_iou(pred_boxes, true_boxes) # [N_pred, N_true]

                # --- Recall 計算 (正解BOX基準) ---
                max_iou_per_true_box, _ = ious.max(dim=0) # 各「正解BOX」に対する最大IoU
                correct_recall = (max_iou_per_true_box >= iou_threshold).sum().item()
                total_correct_detections_for_recall += correct_recall

                # --- Precision 計算 (予測BOX基準) ---
                max_iou_per_pred_box, _ = ious.max(dim=1) # 各「予測BOX」に対する最大IoU
                correct_precision = (max_iou_per_pred_box >= iou_threshold).sum().item()
                total_correct_detections_for_precision += correct_precision

                # --- 平均IoU (Recallが成功したものを基準) ---
                if correct_recall > 0:
                    total_iou_sum += max_iou_per_true_box[max_iou_per_true_box >= iou_threshold].mean().item()


    # --- 最終的な指標の計算 ---
    if total_ground_truth_boxes == 0:
        print("⚠️ 評価可能な正解BOXがありませんでした")
        return 0.0, 0.0, 0.0
    
    recall = total_correct_detections_for_recall / total_ground_truth_boxes * 100.0
    
    precision = 0.0
    if total_pred_boxes > 0:
        precision = total_correct_detections_for_precision / total_pred_boxes * 100.0
    
    avg_iou = 0.0
    if total_correct_detections_for_recall > 0:
        # 平均IoUは、検出できた「正解BOX」の数で割る（Recallの分母）
        avg_iou = total_iou_sum / total_correct_detections_for_recall

    print(f"\n--- 評価結果 ---")
    print(f"Recall (IoU > {iou_threshold}): {recall:.2f}%  ({total_correct_detections_for_recall} / {total_ground_truth_boxes} boxes)")
    print(f"Precision (IoU > {iou_threshold}): {precision:.2f}%  ({total_correct_detections_for_precision} / {total_pred_boxes} boxes)")
    print(f"Average IoU (of correct detections): {avg_iou:.4f}")
    
    return avg_iou, recall, precision
# ==========================================================
# 学習ループ
# ==========================================================
num_epochs = 10

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