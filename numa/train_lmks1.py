# =================================================================
# 0. 必要なライブラリのインポート
# =================================================================
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# ターゲットサイズ (データセット作成コードと合わせる)
IMG_SIZE = 224

# =================================================================
# 1. ランドマーク座標 (.pts) の読み込み関数
# =================================================================
def load_landmarks_from_pts_to_tensor(pts_path):
    """ .ptsファイルから9点のランドマーク座標を読み込み、平坦化されたTensor [18] で返す """
    points = []
    with open(pts_path, 'r') as f:
        lines = f.readlines()

    start_index = -1
    for i, line in enumerate(lines):
        if line.strip() == '{':
            start_index = i + 1
            break
            
    # 9点分を抽出
    for line in lines[start_index : start_index + 9]:
        try:
            x, y = map(float, line.strip().split())
            points.extend([x, y]) # [x1, y1, x2, y2, ...] の順 (18要素)
        except ValueError:
             # 不正な行は無視
            continue

    if len(points) != 18:
          raise ValueError(f"Expected 18 coordinates (9 points), but found {len(points)} in {pts_path}")

    return torch.tensor(points, dtype=torch.float32)

# =================================================================
# 2. PyTorch Dataset クラス
# =================================================================
class LandmarkDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.image_files = []
        self.landmark_files = []

        # ファイルペアのリスト作成 (cropped_dataset内の.jpgと.ptsのペアを探す)
        for filename in os.listdir(data_dir):
            if filename.endswith(".jpg"):
                img_path = os.path.join(data_dir, filename)
                pts_path = os.path.join(data_dir, filename.replace(".jpg", ".pts"))

                if os.path.exists(pts_path):
                    self.image_files.append(img_path)
                    self.landmark_files.append(pts_path)

        # モデルへの入力に合わせた最終的な画像変換 (正規化)
        self.transform = transforms.Compose([
            transforms.ToTensor(), # HWC -> CHW, 0-255 -> 0-1
            # ImageNetの統計値で標準化
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB") # 画像をRGBで読み込み
        image = self.transform(image) # 変換と正規化を実行

        pts_path = self.landmark_files[idx]
        landmarks = load_landmarks_from_pts_to_tensor(pts_path) # 座標 [18] を読み込み

        return image, landmarks

# =================================================================
# 3. モデル定義 (ResNet18 + GAP + Dense層)
# =================================================================
class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks=9):
        super(LandmarkRegressor, self).__init__()
        
        # 1. Backbone: ResNet18 (学習済み重みを使用)
        self.backbone = resnet18(weights='IMAGENET1K_V1') 
        
        # 2. Head: Dense層 (最終層) の変更
        num_features = self.backbone.fc.in_features # ResNet18の最終層の入力サイズ (512)
        
        # 3. 出力層をランドマークの数 (18) に置き換え (GAPはResNet内部に含まれる)
        self.backbone.fc = nn.Linear(num_features, num_landmarks * 2)

    def forward(self, x):
        # 画像 x を入力として ResNet に流し込み、[batch_size, 18] の出力
        return self.backbone(x)

# =================================================================
# 4. メインの訓練関数
# =================================================================
def train_model():
    # --- パラメータ設定 ---
    DATA_DIR = "./cropped_dataset" # 訓練データセットのパス
    BATCH_SIZE = 32
    NUM_LANDMARKS = 9
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # --- デバイス設定 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"訓練デバイス: {device}")
    
    # --- データローダーの準備 ---
    try:
        dataset = LandmarkDataset(data_dir=DATA_DIR)
        print(f"データセットの画像数: {len(dataset)}")
    except Exception as e:
        print(f"データセットの初期化エラー: {e}")
        print("cropped_dataset フォルダが./ (カレントディレクトリ) に存在し、データが揃っているか確認してください。")
        return
        
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )
    
    # --- モデル、損失関数、最適化手法の設定 ---
    model = LandmarkRegressor(num_landmarks=NUM_LANDMARKS)
    model.to(device)
    
    # 損失関数: 平均二乗誤差 (回帰タスク用)
    criterion = nn.MSELoss() 
    # 最適化手法: Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 訓練ループ ---
    print("\n--- 訓練開始 ---")
    for epoch in range(NUM_EPOCHS):
        model.train() # 訓練モード
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(data_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad() # 勾配をゼロクリア
            
            outputs = model(images) # 順伝播
            loss = criterion(outputs, targets) # 損失計算
            
            loss.backward() # 逆伝播
            optimizer.step() # パラメータ更新
            
            running_loss += loss.item()
            
            if (i + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {running_loss / (i+1):.4f}")
                
        avg_epoch_loss = running_loss / len(data_loader)
        print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] 完了. 平均損失: {avg_epoch_loss:.4f} ---")

    # --- モデルの保存 ---
    MODEL_PATH = 'landmark_regressor_final.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nモデルが '{MODEL_PATH}' として保存されました。")

# =================================================================
# 5. スクリプト実行
# =================================================================
if __name__ == '__main__':
    train_model()