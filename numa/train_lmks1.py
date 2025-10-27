import numpy as np
import os
import torch

def load_landmarks_from_pts(pts_path):
    """
    .ptsファイルからランドマーク座標を読み込み、フラットなテンソルで返す。
    """
    try:
        with open(pts_path, 'r') as f:
            lines = f.readlines()

        # 座標データが始まる行を見つける
        start_index = lines.index('{\n') + 1
        end_index = lines.index('}\n')

        # 座標部分だけを抽出し、NumPy配列に変換
        coords = []
        for line in lines[start_index:end_index]:
            x, y = map(float, line.strip().split())
            coords.extend([x, y]) # [x1, y1, x2, y2, ...] の順で平坦化

        # [18] の形状のPyTorch Tensorとして返す
        return torch.tensor(coords, dtype=torch.float32)

    except Exception as e:
        print(f"Error loading {pts_path}: {e}")
        # エラー時は0埋めされたダミーデータを返すなどの例外処理が必要
        return torch.zeros(18, dtype=torch.float32)
        import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ターゲットサイズをあなたのデータセット作成コードに合わせる
IMG_SIZE = 224

# =================================================================
# 1. ランドマーク座標 (.pts) の読み込み関数
# (あなたの load_pts_file ロジックを PyTorch で利用)
# =================================================================
def load_landmarks_from_pts_to_tensor(pts_path):
    """
    .ptsファイルからランドマーク座標を読み込み、平坦化されたテンソルで返す。
    注意: この関数は、データセット作成時に既にスケーリングされた座標を読み込む。
    """
    points = []
    with open(pts_path, 'r') as f:
        lines = f.readlines()

    start_index = -1
    for i, line in enumerate(lines):
        if line.strip() == '{':
            start_index = i + 1
            break

    # .ptsファイルの行数が多いため、正確に9点だけを読む
    for line in lines[start_index : start_index + 9]:
        try:
            x, y = map(float, line.strip().split())
            points.extend([x, y]) # [x1, y1, x2, y2, ...] の順で平坦化 (18要素)
        except ValueError:
            # データセット作成コードの挙動に合わせ、不正な行はスキップ
            continue

    # データセット作成コードの検証ステップに合わせる
    if len(points) != 18:
         raise ValueError(f"Expected 18 coordinates (9 points), but found {len(points)} in {pts_path}")

    # [18] の形状のPyTorch Tensorとして返す
    return torch.tensor(points, dtype=torch.float32)

# =================================================================
# 2. PyTorch Dataset クラス
# =================================================================
class LandmarkDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = []
        self.landmark_files = []

        # ファイルペアのリスト作成
        for filename in os.listdir(data_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(data_dir, filename)
                pts_path = os.path.join(data_dir, filename.replace(".jpg", ".pts"))

                if os.path.exists(pts_path):
                    self.image_files.append(img_path)
                    self.landmark_files.append(pts_path)

        # モデルに合わせた最終的な画像変換を定義
        self.transform = transforms.Compose([
            # データセット作成コードで既に224x224にリサイズされているが、
            # 念のためResizeを入れ、ToTensorで[C, H, W]に変換
            transforms.ToTensor(),
            # 標準化 (ImageNetの統計値を使用)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 画像の読み込み (PIL)
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")

        # データセット作成コードでIMG_SIZEにリサイズされているので、ここではテンソル変換と正規化のみ
        image = self.transform(image)

        # 座標の読み込み (スケーリング済み相対座標 [18] テンソル)
        pts_path = self.landmark_files[idx]
        landmarks = load_landmarks_from_pts_to_tensor(pts_path)

        return image, landmarks

# =================================================================
# 3. DataLoader の準備と確認
# =================================================================
DATA_DIR = "./cropped_dataset"
BATCH_SIZE = 32

dataset = LandmarkDataset(data_dir=DATA_DIR)
print(f"データセットの画像数: {len(dataset)}")

data_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4
)

# 例として最初のバッチを確認
# for images, targets in data_loader:
#     print(f"入力画像テンソルの形状: {images.shape}")
#     print(f"ターゲット座標テンソルの形状: {targets.shape}")
#     break