# =================================================================
# 0. 必要なライブラリのインポート aaaaaa
# =================================================================
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt # <--- 追加: 学習曲線プロット用
from tqdm import tqdm # <--- 後のtrain_modelで使われていないため追加

# ターゲットサイズ (データセット作成コードと合わせる)
IMG_SIZE = 224

# =================================================================
# 1. ランドマーク座標 (.pts) の読み込み関数
# =================================================================
def load_landmarks_from_pts_to_tensor(pts_path):
    # .ptsファイルを読み込みテンソル形式に変換
    
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
    # 訓練に合わせたデータセットを供給する
    # ファイルパスのリストを外部から受け取るよう修正 (train/test分割に必要)
    def __init__(self, file_paths):
        self.image_files = file_paths

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
        
        # .pts ファイルパスを .jpg パスから構築
        pts_path = img_path.replace(".jpg", ".pts") 

        image = Image.open(img_path).convert("RGB") # 画像をRGBで読み込み
        image = self.transform(image) # 変換と正規化を実行

        landmarks = load_landmarks_from_pts_to_tensor(pts_path) # 座標 [18] を読み込み

        return image, landmarks

# =================================================================
# 3. モデル定義 (カスタムResNet18を使用)
# =================================================================

class BasicBlock(nn.Module):
    # ... (BasicBlock定義は変更なし)
    '''
    ResNet18における残差ブロック
    in_channels : 入力チャネル数
    out_channels: 出力チャネル数
    stride      : 畳み込み層のストライド
    '''
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int=1):
        super().__init__()

        # 残差接続
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # スキップ接続のダウンサンプリング (寸法合わせ)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
             self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    '''
    順伝播関数
    x: 入力, [バッチサイズ, 入力チャネル数, 高さ, 幅]
    '''
    def forward(self, x: torch.Tensor):
        identity = x # 恒等写像 (スキップ接続) を保存

        # 残差写像
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # ダウンサンプリング処理
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 残差写像と恒等写像の要素毎の和を計算
        out += identity

        out = self.relu(out)

        return out

class ResNet18(nn.Module):
    # ... (ResNet18定義は変更なし)
    '''
    ResNet18モデル
    num_classes: ランドマーク回帰用
    '''
    def __init__(self, num_classes: int):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(kernel_size=3,
                                     stride=2, padding=1)

        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64),
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256),
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.linear = nn.Linear(512, num_classes) # num_classesはここでは仮の値

    '''
    順伝播関数
    x           : 入力, [バッチサイズ, 入力チャネル数, 高さ, 幅]
    return_embed: 特徴量を返すかロジットを返すかを選択する真偽値
    '''
    def forward(self, x: torch.Tensor, return_embed: bool=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.flatten(1)

        if return_embed:
            return x

        x = self.linear(x)

        return x

class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks=9):
        super(LandmarkRegressor, self).__init__()
        
        # 1. Backbone: カスタムResNet18を使用
        self.backbone = ResNet18(num_classes=1000)
        
        # 2. Head: Dense層 (最終層) の変更
        num_features = self.backbone.linear.in_features
        
        # 3. 出力層をランドマークの数 (18) に置き換え 
        self.backbone.linear = nn.Linear(num_features, num_landmarks * 2)

    def forward(self, x):
        return self.backbone(x)

# =================================================================
# 4. 評価関数 (evaluate_model) <--- train/test分割に必要
# =================================================================
def evaluate_model(model, data_loader, criterion, device):
    """テストデータセットに対する損失を計算し、モデルの精度を確認する"""
    model.eval()
    total_loss = 0
    count = 0

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)
            count += imgs.size(0)

    avg_loss = total_loss / count
    return avg_loss

# =================================================================
# 5. 損失曲線プロット関数 <--- 追加
# =================================================================
def plot_loss_curve(train_losses, test_losses, num_epochs):
    """訓練損失とテスト損失の推移をプロットする"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss (MSE)', marker='o')
    if test_losses:
        plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss (MSE)', marker='s')
    
    plt.title('Training and Test Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error (MSE) Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    # Colabで画像をファイルとして保存したい場合は以下の行をコメント解除
    # plt.savefig('loss_curve.png') 
    # plt.close()

# =================================================================
# 6. メインの訓練関数 (train/test分割を含む)
# =================================================================
def train_model():
    # --- パラメータ設定 ---
    DATA_DIR = "./cropped_dataset" # 訓練データセットのパス
    TEST_SIZE = 0.2 # テストデータの割合 (20%)
    BATCH_SIZE = 32
    NUM_LANDMARKS = 9
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # --- デバイス設定 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"訓練デバイス: {device}")
    
    # --- ファイルリストの取得と分割 (train/test分割) ---
    try:
        all_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
        if not all_files:
             raise FileNotFoundError("データセット内に.jpgファイルが見つかりません。")
        
        train_files, test_files = train_test_split(
            all_files, test_size=TEST_SIZE, random_state=42
        )
        print(f"全画像ファイル数: {len(all_files)}")
        print(f"Train: {len(train_files)} files, Test: {len(test_files)} files")
        
    except Exception as e:
        print(f"データセットの準備エラー: {e}")
        print("cropped_dataset フォルダが./ (カレントディレクトリ) に存在し、データが揃っているか確認してください。")
        return
    
    # --- データローダーの準備 ---
    train_dataset = LandmarkDataset(train_files)
    test_dataset = LandmarkDataset(test_files)
        
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # --- モデル、損失関数、最適化手法の設定 ---
    model = LandmarkRegressor(num_landmarks=NUM_LANDMARKS)
    # ここで学習済み重みをロードする処理を入れることも可能
    model.to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 損失記録用のリスト --- <--- 追加
    train_losses = []
    test_losses = []
    
    # --- 訓練ループ ---
    print("\n--- 訓練開始 ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train() # 訓練モード
        running_loss = 0.0
        
        # tqdm を使用して進捗バーを表示
        for i, (images, targets) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
                
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss) # <--- 訓練損失を記録
        
        # --- テストデータでの評価 ---
        test_loss = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss) # <--- テスト損失を記録
        
        print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] 完了. Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f} ---")

    # --- 最終評価とモデルの保存 ---
    final_test_loss = evaluate_model(model, test_loader, criterion, device)
    print(f"\n✅ Final Test Loss: {final_test_loss:.4f}")

    MODEL_PATH = 'landmark_regressor_final.pth'
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"モデルが '{MODEL_PATH}' として保存されました。")
    
    # --- 学習曲線のプロット --- <--- 追加
    print("\n--- 学習曲線を表示 ---")
    plot_loss_curve(train_losses, test_losses, NUM_EPOCHS)

# =================================================================
# 7. スクリプト実行
# =================================================================
if __name__ == '__main__':
    train_model()