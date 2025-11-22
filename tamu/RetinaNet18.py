# 基本ライブラリ
import os # ファイルパスの操作
import time # 学習時間の計測
import glob # ファイルパスのリスト取得用
import numpy as np # 数値計算
from tqdm import tqdm # 進捗表示

# Pytorch関連
import torch # pytorchの基本機能
import torch.optim as optim # 最適化アルゴリズム(SGD)
from torch.utils.data import Dataset # データセットの定義と使用
from torch.utils.data import DataLoader # データローダーの定義と使用
from torchvision import transforms as T # 画像変換(Tensorに)
from torchvision.ops import box_iou # IoUの計算(IoU：)
from torchvision import transforms
from torch.optim.lr_scheduler import MultiStepLR

# モデル構築用
from resnet18_backbone import resnet18 # ResNet18のバックボーン
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor # ResNetからFPNを構築
from torchvision.ops.feature_pyramid_network import LastLevelP6P7 # FPNの最終レベル(P6,P7)を追加する
from torchvision.models.detection.anchor_utils import AnchorGenerator # RetinaNetのアンカー生成器
from torchvision.models.detection import RetinaNet # RetinaNetモデル
import torch.nn as nn
import torch.nn.functional as F

# データ
from sklearn.model_selection import train_test_split # データ分割用
from PIL import Image # 画像ファイルの読み込みとRBG変換
import random # データ拡張
import albumentations as A
from albumentations.pytorch import ToTensorV2

#----------------------------------------------------------------------------------
# データセットを整えるクラス
class CustomObjectDetectionDataset(Dataset): # DAtasetクラスを継承
    # 初期化処理
    def __init__(self, img_list, root, transforms=None, augment=False): 
        self.root = root # .ptsファイルを保存するrootを保持
        self.transforms = transforms # 画像に適応する前処理(今回はなし)
        self.imgs = img_list # 画像パスのリストを保持する
        self.augment = augment # データ拡張用
        self.color_transform = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02 ) # 色変換用
        self.augment_transform = A.Compose([ # データ拡張
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03, p=0.7),
            A.Perspective(scale=(0.05, 0.1), keep_size=True, p=0.4),
            A.Rotate(limit=10, border_mode=0, p=0.5),
            A.RandomScale(scale_limit=0.2, p=0.4),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.0, rotate_limit=0, border_mode=0, p=0.5),
            ToTensorV2()
        ], 
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.5,
        ))

    # バウンディングボックスの情報を抽出する    
    def _parse_pts(self, pts_path):
        
        # 初期化
        boxes = [] # バウンディングボックス
        labels = [] # ラベル

        # パスが存在しない場合、空のバウンディングボックスとラベルを返す
        if not os.path.exists(pts_path):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.int64)
        
        # .ptsファイルを読み込む
        xs, ys = [], []
        with open(pts_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("version") or line in ["{", "}"]: # 空行やヘッダー、波括弧をスキップ
                    continue

                parts = line.split()
                if len(parts) != 2: # 座標が二つ(x,y)のみ通す
                    continue

                try:
                    x, y = float(parts[0]), float(parts[1])
                    xs.append(x)
                    ys.append(y) # 座標が二つあったらそれぞれ保存
                except ValueError:
                    continue

        # バウンディングボックスの作成
        if len(xs) >= 2 and len(ys) >= 2:
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            boxes = np.array([[xmin, ymin, xmax, ymax]], dtype=np.float32)
            labels = np.array([1], dtype=np.int64)  # 全て1にして単一クラス扱い
        else:
            # 点が足りない場合は空にしておく
            boxes = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int64)

        return boxes, labels

        
    def __getitem__(self, idx): # 指定されたインデックス(番号)の画像とｱﾉﾃｰｼｮﾝを返す
        # ファイルパスの構築
        img_path_full = self.imgs[idx] # 画像ファイルのパスを取得
        img_filename = os.path.basename(img_path_full) # 画像のファイル名を得る
        base_name = os.path.splitext(img_filename)[0] # 画像のファイル名から拡張子を除いた名前を得る
        pts_filename = base_name + ".pts" # .ptsファイル名を作成
        pts_path = os.path.join(self.root, pts_filename) # .ptsファイルのパスを作成

        # データ読み込み
        img = Image.open(img_path_full).convert("RGB") # 画像をRGB形式で読み込む
        #W, H = self._parse_pts(pts_path) # .ptsファイルからバウンディングボックスとラベルをNumpy配列で取得

        # BBox読み込み
        boxes_np, labels_np = self._parse_pts(pts_path)

        if boxes_np.size > 0:
            x1, y1, x2, y2 = boxes_np[0]
            if x2 - x1 < 1 or y2 - y1 < 1:
                boxes_np = np.empty((0, 4), dtype=np.float32)
                labels_np = np.empty((0,), dtype=np.int64)

        # --- Albumentations augment ---
        if self.augment and boxes_np.size > 0:
            augmented = self.augment_transform(
                image=np.array(img),
                bboxes=boxes_np.tolist(),
                labels=labels_np.tolist()
            )
            img = augmented['image']
            boxes_np = np.array(augmented['bboxes'], dtype=np.float32)
            labels_np = np.array(augmented['labels'], dtype=np.int64)

        else:
            img = T.functional.to_tensor(img)

        if boxes_np.size == 0:
            boxes = torch.empty((0,4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(labels_np, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return img, target

    def __len__(self):
        return len(self.imgs)


    """
        # データ拡張
        if self.augment and boxes_np.size > 0:


            x1, y1, x2, y2 = boxes_np[0]
            
            # 左右反転
            if random.random() > 0.5:
                img = T.functional.hflip(img)  # PIL の左右反転
                width, height = img.size

                # BBox も左右反転
                x1_new = width - x2
                x2_new = width - x1
                
                x1 = min(x1_new, x2_new)
                x2 = max(x1_new, x2_new)

            boxes_np = np.array([[x1, y1, x2, y2]], dtype=np.float32)

            # 2色変換
            if self.color_transform is not None:
                img = self.color_transform(img)

            # Tensor に変換
            img = T.functional.to_tensor(img)

        else:
            # augment が無い場合画像変換の適用
            if self.transforms is not None:
                img = self.transforms(img) # 前処理を適用
            else:
                img = T.functional.to_tensor(img)


        # ターゲット辞書の作成
        if boxes_np.size == 0: # バウンディングボックスが空なら空のテンソルを作成
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else: # NumPy配列をPytorchテンソルに変換
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(labels_np, dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        } # ターゲット辞書の構築

        return img, target
    
    # データセットのサイズを返す
    def __len__(self):
        return len(self.imgs)

    """

# 前処理(Transforms)の定義
def get_transform(train):
    t = [T.ToTensor()] # PIL画像をテンソル形式に変換
    if train: # データ拡張
        # t.append(T.RandomHorizontalFlip(0.5)) # 50%の確率で左右反転
        pass 
    return T.ToTensor()

# コレート関数(Collate Function)の定義 (RetinaNetにはリスト形式で渡すため)
def custom_collate_fn(batch): # batch：(img,target)
    images = [item[0] for item in batch] # 画像のみのリストを作成
    targets = [item[1] for item in batch] # ｱﾉﾃｰｼｮﾝのみのリストを作成
    return images, targets

# データの読み込みと分割
DATA_ROOT = '/workspace/dataset' # データのルートディレクトリを指定

# 全ての画像ファイルパスを取得
all_imgs = sorted(glob.glob(os.path.join(DATA_ROOT, "*.jpg"))) # shorted()でファイル名順に並び替えられる
print(f"全画像数: {len(all_imgs)}")

# =====================================================================================================
# 8:1:1にする
#=======================================================================================================
# 学習用 (80%) とテスト用 (20%) に分割
train_imgs, test_imgs = train_test_split( 
    all_imgs, 
    test_size=0.2, 
    random_state=42 # シード固定で再現性を確保(同じようにデータセットを分けれるようにする)
)
print(f"学習用サンプル数 (80%): {len(train_imgs)}, テスト用サンプル数 (20%): {len(test_imgs)}")

# Datasetのインスタンス作成　それぞれのデータセットを作成
train_dataset = CustomObjectDetectionDataset(train_imgs, DATA_ROOT, get_transform(train=True), augment=True) # 拡張可能
test_dataset = CustomObjectDetectionDataset(test_imgs, DATA_ROOT, get_transform(train=False), augment=False)

# DataLoaderの作成
train_loader = DataLoader(
    train_dataset,
    batch_size=16, 
    shuffle=True, # シャッフルあり
    num_workers=2, # データ読み込みの並列処理
    collate_fn=custom_collate_fn # 画像とｱﾉﾃｰｼｮﾝをそれぞれリスト形式でまとめる
)

# TestLoaderの作成
test_loader = DataLoader(
    test_dataset,
    batch_size=16, 
    shuffle=False, # シャッフルなし
    num_workers=2, 
    collate_fn=custom_collate_fn 
)

#-----------------------------------------------------------------------------------------
# バックボーンとアンカー生成器の構築
custom_backbone = resnet18(pretrained=False) # ResNet18を使えるようにする (重みなし)

# FPNを構築するための設定
out_channels = 256 # FPNの各出力マップのチャンネル数

# 特徴ピラミッドネットワーク(FPN)の作成
class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels): # チャネル＝経路
        super().__init__()
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, kernel_size=1) # カーネル1の畳み込み
            for in_ch in in_channels_list
        ])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) # カーネル３の畳み込み
            for _ in in_channels_list
        ])
        self.p6 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) # カーネル3、ストライド2の畳み込み
        self.p7 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1) # カーネル3、ストライド2の畳み込み

    def forward(self, inputs):  # inputs = [C3, C4, C5]
        lateral_feats = [lateral(x) for lateral, x in zip(self.lateral_convs, inputs)]
        results = []
        x = lateral_feats[-1]
        results.append(self.output_convs[-1](x))  # P5

        for i in reversed(range(len(lateral_feats) - 1)):
            x = F.interpolate(x, scale_factor=2, mode='nearest') + lateral_feats[i]
            results.insert(0, self.output_convs[i](x))  # P4, P3

        p6 = self.p6(results[-1]) # P6
        p7 = self.p7(F.relu(p6)) # P7
        results.extend([p6, p7])

        return {str(i): f for i, f in enumerate(results)} # 辞書型で返す
    
# ResNetとFPN両方適用
class BackboneWithFPN(nn.Module):
    def __init__(self, resnet, fpn,out_channels):
        super().__init__()
        self.body = resnet
        self.fpn = fpn
        self.out_channels = out_channels

    def forward(self, x):
        c3, c4, c5 = self.body(x)  # 自作ResNetが中間特徴を返すように設計
        return self.fpn([c3, c4, c5])

fpn = FeaturePyramidNetwork(in_channels_list=[128, 256, 512], out_channels=256) #自作FPNを適用する

backbone = BackboneWithFPN(custom_backbone, fpn,out_channels=256) # ResNet + FPN を統合

# ダミー画像をFPNに通して出力層の構造を確認
with torch.no_grad():
    dummy_image = torch.rand(1, 3, 224, 224)  # バッチサイズ1 RGBの3
    features = backbone(dummy_image)
    print("FPN 出力層のキー:", list(features.keys()))
    print("各層の出力形状:")
    for k, v in features.items():
        print(f"  {k}: {tuple(v.shape)}")

num_feature_maps = len(features)
print("FPN 出力層数:", num_feature_maps)

# アンカー生成器の定義 (候補領域の作成)
sizes=[8, 16, 32, 64, 128, 224] # アンカーのサイズ
sizes_for_anchor = tuple((s,) for s in sizes[:num_feature_maps]) 

anchor_generator = AnchorGenerator(
    sizes=sizes_for_anchor,
    aspect_ratios=((0.5, 1.0, 2.0),) * num_feature_maps
)


# RetinaNetモデルの構築
NUM_CLASSES = 2 # 検出対象(背景を除く)

model = RetinaNet(
    backbone=backbone,
    num_classes=NUM_CLASSES,
    anchor_generator=anchor_generator
)

# デバイス設定
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# オプティマイザの定義 (SGD：確率的勾配降下法) ハイパーパラメータ
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.01, # 学習率
    momentum=0.9,
    weight_decay=0.0005 # 過学習防止
)

# 学習率を下げる
scheduler = MultiStepLR(
    optimizer,
    milestones=[10, 15],   # 10 epoch で lr を下げ、15 epoch でさらに下げる
    gamma=0.1              # 1/10 に減衰
)

#---------------------------------------------------------------------------
# 評価関数
def evaluate_retinanet(model, dataloader, device, iou_threshold=0.5):
    """
    1画像につき予測を1つだけに制限して評価
    正解ボックスも1つだけの想定
    """
    model.eval() # 推論モードに切り替え
    
    total_images = 0 
    total_iou_sum = 0.0
    total_preds = 0  # 予測が存在した画像数（=pred_boxがあった画像数）
    correct_detections = 0    # 正解ボックス総数＝画像数

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating"): # images:画像テンソルのリスト、targets:ｱﾉﾃｰｼｮﾝ
            images = [img.to(device).float() for img in images]
            outputs = model(images)  # 予測結果(boxes,scores,labels)

            # cpuに変える
            outputs = [{k: v.cpu() for k, v in out.items()} for out in outputs]
            targets = [{k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in t.items()} for t in targets]

            # 評価
            for output, target in zip(outputs, targets): # output:予測結果、target:正解ｱﾉﾃｰｼｮﾝ
                total_images += 1 # 全画像数に+1
                true_boxes = target["boxes"] # 正解ボックスがなければ、空テンソル

                # スコアが最も高い予測ボックスを1つ選ぶ
                pred_boxes = output["boxes"] # モデルが予測したリスト
                scores = output["scores"] # 予測が顔である確信度

                if pred_boxes.numel() == 0 or true_boxes.size(0) == 0:# 予測ゼロもしくは正解がない
                    continue

                top_idx = scores.argmax() # 一番スコアが高い予測
                pred_box = pred_boxes[top_idx].unsqueeze(0)  # 結果となる予測の形を変える[1,4]
                total_preds += 1 # 予測が存在する枚数+1

                iou = box_iou(pred_box, true_boxes)[0,0].item()  # IoUの計算

                total_iou_sum += iou # IoUの合計

                if iou >= iou_threshold: # 閾値ごとに成功判定
                    correct_detections += 1

    # 指標計算
    avg_iou = (total_iou_sum / total_preds) if total_images > 0 else 0.0
    accuracy = correct_detections / total_images if total_images > 0 else 0

    # pretty print
    print(f"\n--- 評価結果 ---")
    print(f"Accuracy (IoU > {iou_threshold}): {accuracy:.2f} ({correct_detections}/{total_images})")
    print(f"Average IoU: {avg_iou:.4f}")

    return avg_iou, accuracy

# ------------------------------------------------------------------------------
# 学習するエポック数
num_epochs = 20 

# 学習
for epoch in range(num_epochs):
    model.train()
    total_epoch_loss = 0.0
    
    for step, (images, targets) in enumerate(tqdm(train_loader, desc="Training")):
        # データとターゲットをGPUに移動
        images = [image.to(device).to(torch.float32) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 勾配を初期化
        optimizer.zero_grad()

        # 損失計算
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        total_epoch_loss += losses.item()

        # NaNチェック
        if torch.isnan(losses):
            print(f"NaN detected at step {step}, skipping this batch.")
            continue

        # バックワードパス: 勾配を計算
        losses.backward()

        # オプティマイザのステップ: 重みを更新
        optimizer.step() 
        
    tqdm.write(f"--- Epoch [{epoch}/{num_epochs}] 完了。 平均損失: {total_epoch_loss / len(train_loader):.4f}s ---")
    avg_train_loss = total_epoch_loss / len(train_loader)

    scheduler.step()

    ### --- テストロス計算ループ ---###
    model.train()   
    test_loss = 0.0

    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc=f"Testing {epoch+1}/{num_epochs}"):
            images = [img.to(device).float() for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # 学習時と同じように loss を計算
            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            test_loss += losses.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Epoch {epoch+1} Test Loss: {avg_test_loss:.4f}")

#------------------------------------------------------------------------------------

# モデルの重みを保存
torch.save(model.state_dict(), 'retinanet_custom_weights_final.pth')

# 学習後にIoUを評価
evaluate_retinanet(model, test_loader, device, iou_threshold=0.5)
