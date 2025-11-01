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

# モデル構築用
from resnet18_backbone import resnet18 # ResNet18のバックボーン
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor # ResNetからFPNを構築
from torchvision.ops.feature_pyramid_network import LastLevelP6P7 # FPNの最終レベル(P6,P7)を追加する
from torchvision.models.detection.anchor_utils import AnchorGenerator # RetinaNetのアンカー生成器
from torchvision.models.detection import RetinaNet # RetinaNetモデル

# データ
from sklearn.model_selection import train_test_split # データ分割用
from PIL import Image # 画像ファイルの読み込みとRBG変換


# データセットを整えるクラス
class CustomObjectDetectionDataset(Dataset): # DAtasetクラスを継承
    # 初期化処理
    def __init__(self, img_list, root, transforms=None): 
        self.root = root # .ptsファイルを保存するrootを保持
        self.transforms = transforms # 画像に適応する前処理(今回はなし)
        self.imgs = img_list # 画像パスのリストを保持する

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
        boxes_np, labels_np = self._parse_pts(pts_path) # .ptsファイルからバウンディングボックスとラベルをNumpy配列で取得

        # ターゲット辞書の作成
        if boxes_np.size == 0: # バウンディングボックスが空なら空のテンソルを作成
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)
        else: # NumPy配列をPytorchテンソルに変換
            boxes = torch.as_tensor(boxes_np, dtype=torch.float32)
            labels = torch.as_tensor(labels_np, dtype=torch.int64)
        
        target = {} # ターゲット辞書の構築
        target["boxes"] = boxes 
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        
        # 画像変換の適用
        if self.transforms is not None:
            img = self.transforms(img) # 前処理を適用

        return img, target
    
    # データセットのサイズを返す
    def __len__(self):
        return len(self.imgs)

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

# 学習用 (80%) とテスト用 (20%) に分割
train_imgs, test_imgs = train_test_split( 
    all_imgs, 
    test_size=0.2, 
    random_state=42 # シード固定で再現性を確保
)
print(f"学習用サンプル数 (80%): {len(train_imgs)}, テスト用サンプル数 (20%): {len(test_imgs)}")

# Datasetのインスタンス作成　それぞれのデータセットを作成
train_dataset = CustomObjectDetectionDataset(train_imgs, DATA_ROOT, get_transform(train=True)) # 拡張可能
test_dataset = CustomObjectDetectionDataset(test_imgs, DATA_ROOT, get_transform(train=False))

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
    batch_size=2, 
    shuffle=False, # シャッフルなし
    num_workers=2, 
    collate_fn=custom_collate_fn 
)


# バックボーンとアンカー生成器の構築
custom_backbone = resnet18(pretrained=False) # ResNet18を使えるようにする (重みなし)

# FPNを構築するための設定
out_channels = 256 # FPNの各出力マップのチャンネル数

backbone_fpn = _resnet_fpn_extractor(
    custom_backbone, 
    trainable_layers=5, # ResNetのすべての層を学習可能に
    extra_blocks=LastLevelP6P7(out_channels, out_channels) # さらに高レベルの特徴マップ(P6,P7)を追加
)

# アンカー生成器の定義 (候補領域の作成)
anchor_generator = AnchorGenerator(
    sizes=((32,), (64,), (128,), (256,), (512,), (1024,)), # アンカーのサイズ
    aspect_ratios=((0.5, 1.0, 2.0),) * 6 # 縦横比
)


# RetinaNetモデルの構築
NUM_CLASSES = 1 # 検出対象(背景を除く)

model = RetinaNet(
    backbone=backbone_fpn,
    num_classes=NUM_CLASSES,
    anchor_generator=anchor_generator
)

# デバイス設定
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# オプティマイザの定義 (SGD：確率的勾配降下法)
optimizer = optim.SGD(
    model.parameters(), 
    lr=0.0001, # 学習率
    momentum=0.9,
    weight_decay=0.0001 # 過学習防止
)

# 学習するエポック数
num_epochs = 30 

# 物体検出精度をIoUで評価する
def evaluate_iou(model, dataloader, device):
    model.eval() # 評価モード
    total_iou = 0.0
    total_images = 0

    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating IoU"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images) # モデルで予測

            for output, target in zip(outputs, targets):
                pred_boxes = output['boxes']
                true_boxes = target['boxes']

                if pred_boxes.size(0) == 0 or true_boxes.size(0) == 0:
                    continue  # 空ならスキップ

                # IoUの計算
                ious = box_iou(pred_boxes, true_boxes)  # [N_pred, N_true] のIoU行列
                max_ious, _ = ious.max(dim=1)  # 各予測に対して最大IoUを取得

                total_iou += max_ious.mean().item()
                total_images += 1

    if total_images == 0:
        print("⚠️ IoU評価できる画像がありませんでした。")
    else: # 平均IoUの出力
        avg_iou = total_iou / total_images
        print(f"\n📊 平均IoU: {avg_iou:.4f}（{total_images}枚の画像で評価）\n")

# モデルを学習させる
model.train() # トレーニングモード

for epoch in range(1, num_epochs + 1):
    start_time = time.time()
    total_epoch_loss = 0
    
    for step, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch}/{num_epochs}]")):
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
        
    end_time = time.time()
    tqdm.write(f"--- Epoch [{epoch}/{num_epochs}] 完了。 平均損失: {total_epoch_loss / len(train_loader):.4f}, 処理時間: {(end_time - start_time):.2f}s ---")

print("全学習プロセスが完了しました。")

# モデルの重みを保存
torch.save(model.state_dict(), 'retinanet_custom_weights_final.pth')

# 学習後にIoUを評価
evaluate_iou(model, test_loader, device)
