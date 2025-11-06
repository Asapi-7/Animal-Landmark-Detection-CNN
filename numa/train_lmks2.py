# =================================================================
# 0. 必要なライブラリのインポート
# =================================================================
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont 
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from tqdm import tqdm
import numpy as np 

# ターゲットサイズ (データセット作成コードと合わせる)
IMG_SIZE = 224
NUM_LANDMARKS = 9

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

        # 訓練用画像 (正規化済み)
        image = Image.open(img_path).convert("RGB") # 画像をRGBで読み込み
        transformed_image = self.transform(image) # 変換と正規化を実行

        landmarks = load_landmarks_from_pts_to_tensor(pts_path) # 座標 [18] を読み込み

        # 推論・描画のために元の画像パスも返す
        return transformed_image, landmarks, img_path # <--- 変更: img_path を追加

# =================================================================
# 3. モデル定義 (カスタムResNet18を使用)
# =================================================================

class BasicBlock(nn.Module):
    '''
    ResNet18における残差ブロック
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
    '''
    ResNet18モデル
    num_classes: 分類対象の物体クラス数 (ランドマーク回帰用に置き換えられる)
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

        # 回帰ヘッドの多層化に備えて、暫定的な線形層を定義
        self.linear = nn.Linear(512, num_classes) 

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

        # LandmarkRegressorで置き換えられる線形層
        x = self.linear(x) 

        return x

class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks=9):
        super(LandmarkRegressor, self).__init__()
        
        # 1. Backbone: カスタムResNet18を使用
        self.backbone = ResNet18(num_classes=1000)
        
        # 2. Head: Dense層 (最終層) の変更
        num_features = self.backbone.linear.in_features  # 512
        output_dim = num_landmarks * 2                   # 9 * 2 = 18

        # --- 💥 回帰ヘッドの強化 💥 ---
        HIDDEN_DIM = 256
        DROPOUT_RATE = 0.5
        
        # 多層回帰ヘッドの定義 (512 -> 256 -> 18)
        self.regressor_head = nn.Sequential(
            nn.Linear(num_features, HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(p=DROPOUT_RATE),
            nn.Linear(HIDDEN_DIM, output_dim)
        )
        
        # 3. ResNet18の線形層を、強化した回帰ヘッドに置き換え
        self.backbone.linear = self.regressor_head

    def forward(self, x):
        return self.backbone(x)

# =================================================================
# 4. 評価関数 (evaluate_model)
# =================================================================
def evaluate_model(model, data_loader, criterion, device):
    """テストデータセットに対する損失を計算し、モデルの精度を確認する"""
    model.eval()
    total_loss = 0
    total_nme = 0
    count = 0

    with torch.no_grad():
        # データローダーのイテレーションを imgs, labels のみに変更 (img_path は評価に使わないため)
        for data in data_loader:
            imgs, labels = data[0].to(device), data[1].to(device)
            outputs = model(imgs)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * imgs.size(0)

            # --- NMEの計算と集計 ---
            nme_batch = calculate_nme(outputs, labels, device)
            total_nme += nme_batch.item() * imgs.size(0)

            count += imgs.size(0)

    avg_loss = total_loss / count
    avg_nme = total_nme / count
    return avg_loss, avg_nme

# =================================================================
# 4.1. NME (Normalized Mean Error) 計算関数
# =================================================================
def calculate_normalization_factor(landmarks):
    """ 
    ランドマーク [N, 18] から、バウンディングボックスの対角線長を計算する。
    """
    # 座標を (N, 9, 2) に整形: (x1, y1, x2, y2, ...) -> ((x1, y1), (x2, y2), ...)
    coords = landmarks.reshape(-1, 9, 2)
    
    # バウンディングボックスの計算 (全点の min/max を使用)
    x_min = coords[..., 0].min(dim=1).values
    x_max = coords[..., 0].max(dim=1).values
    y_min = coords[..., 1].min(dim=1).values
    y_max = coords[..., 1].max(dim=1).values
    
    # 対角線長の計算: sqrt((x_max - x_min)^2 + (y_max - y_min)^2)
    width = x_max - x_min
    height = y_max - y_min
    
    # 対角線長 (NMEの正規化基準)
    diagonal = torch.sqrt(width**2 + height**2)
    
    # 対角線長がゼロになるのを防ぐため、小さな値を加える
    return diagonal + 1e-6


def calculate_nme(outputs, labels, device):
    """ NME (Normalized Mean Error) を計算する """
    num_landmarks = 9
    
    # 出力と正解を (N, 9, 2) に整形
    outputs_reshaped = outputs.reshape(-1, num_landmarks, 2)
    labels_reshaped = labels.reshape(-1, num_landmarks, 2)

    # 1. 予測座標と正解座標間のユークリッド距離を計算 (各ランドマークごと)
    distances = torch.linalg.norm(outputs_reshaped - labels_reshaped, dim=2) # [N, 9]

    # 2. 正規化ファクター（バウンディングボックスの対角線長）を計算
    normalization_factors = calculate_normalization_factor(labels).to(device) # [N]

    # 3. 各ランドマークの距離を正規化ファクターで割る
    # unsqueeze(1) で [N] -> [N, 1] にしてブロードキャストを可能にする
    normalized_distances = distances / normalization_factors.unsqueeze(1) # [N, 9]

    # 4. 全ての正規化距離の平均を取る (これが NME)
    nme = normalized_distances.mean()

    return nme

# =================================================================
# 5. 損失曲線プロット関数
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

# =================================================================
# 6. ランドマーク描画ヘルパー関数 (PIL用、現在は未使用)
# =================================================================
def draw_landmarks_pil(image, landmarks, color='red', point_size=5):
    """
    PIL Image にランドマークを描画し、インデックス (1-9) を付与する
    """
    draw = ImageDraw.Draw(image)
    
    try:
        font_size = 18
        # システムに arial.ttf が存在しない場合があるため注意
        font = ImageFont.truetype("arial.ttf", size=font_size) 
    except IOError:
        font = ImageFont.load_default() 
    
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()
        
    for k in range(0, len(landmarks) // 2):
        i = k * 2
        x = int(landmarks[i])
        y = int(landmarks[i+1])
        
        landmark_index = k + 1 
        
        bbox = [x - point_size, y - point_size, x + point_size, y + point_size]
        draw.ellipse(bbox, fill=color)
        
        text_position = (x + point_size + 2, y + point_size + 2) 
        draw.text(text_position, str(landmark_index), fill="yellow", font=font)
        
    return image

# =================================================================
# 7. 予測結果を画像に描画して保存する関数 
# =================================================================
def save_landmark_predictions(model, data_loader, device, num_samples=5, save_dir="./predictions_output"):
    """
    テストデータに対して推論を行い、予測されたランドマークを画像に描画して保存する
    """
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_count = 0
    
    with torch.no_grad():
        for images, targets, img_paths in data_loader:
            if saved_count >= num_samples:
                break
                
            images_tensor = images.to(device)
            outputs = model(images_tensor).cpu() # 推論結果をCPUに戻す

            for i in range(images_tensor.size(0)):
                if saved_count >= num_samples:
                    break
                    
                # 1. 元の画像の読み込み
                original_img_path = img_paths[i]
                original_image_pil = Image.open(original_img_path).convert("RGB")
                
                # 2. 予測されたランドマーク座標を取得 (NumPy配列)
                predicted_landmarks_flat = outputs[i].numpy() # [x1,y1,x2,y2,...]
                predicted_landmarks_reshaped = predicted_landmarks_flat.reshape(-1, 2) # (9, 2)

                # --- Matplotlibで描画 ---
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(original_image_pil)
                
                # 予測座標を元の画像サイズにスケーリング
                original_w, original_h = original_image_pil.size
                
                # IMG_SIZE はモデルの入力サイズ（224）
                scaled_landmarks_x = predicted_landmarks_reshaped[:, 0] * (original_w / IMG_SIZE)
                scaled_landmarks_y = predicted_landmarks_reshaped[:, 1] * (original_h / IMG_SIZE)
                
                scaled_landmarks = np.stack([scaled_landmarks_x, scaled_landmarks_y], axis=1) # (9, 2) の形式に再構成

                # --- A. 1, 2個目の点を直径とする円 (赤色) の描画 ---
                p1 = scaled_landmarks[0]
                p2 = scaled_landmarks[1]
                center_x12 = (p1[0] + p2[0]) / 2
                center_y12 = (p1[1] + p2[1]) / 2
                diameter12 = np.linalg.norm(p1 - p2)
                radius12 = diameter12 / 2
                circle12 = plt.Circle((center_x12, center_y12), radius12, 
                                     color='red', fill=False, linewidth=2)
                ax.add_artist(circle12)
                
                # --- B. 3, 4個目の点を直径とする円 (赤色) の描画 ---
                p3 = scaled_landmarks[2]
                p4 = scaled_landmarks[3]
                center_x34 = (p3[0] + p4[0]) / 2
                center_y34 = (p3[1] + p4[1]) / 2
                diameter34 = np.linalg.norm(p3 - p4)
                radius34 = diameter34 / 2
                circle34 = plt.Circle((center_x34, center_y34), radius34, 
                                     color='red', fill=False, linewidth=2)
                ax.add_artist(circle34)

                # --- C. 6, 8, 7, 9, 6の順に直線をつなげた線 (赤色) の描画 ---
                # インデックスは 0 から始まるため、(6, 8, 7, 9, 6) -> [5, 7, 6, 8, 5]
                indices = [5, 7, 6, 8, 5] 
                line_x = scaled_landmarks_x[indices]
                line_y = scaled_landmarks_y[indices]
                ax.plot(line_x, line_y, color='red', linestyle='-', linewidth=2)

                # --- ランドマーク点を描画 ---
                ax.scatter(scaled_landmarks_x, scaled_landmarks_y, 
                           c='red', marker='o', s=50, label=None)
                
                # --- ランドマークに番号を振る ---
                # 注釈描画のロジックは簡略化しました (文字の影付けを省略)
                tmp = 10
                for k_idx in range(NUM_LANDMARKS): 
                     ax.annotate(str(k_idx+1), (scaled_landmarks_x[k_idx] + tmp + 2, scaled_landmarks_y[k_idx] + tmp + 2), color='yellow', fontsize=20, ha='center', va='center',
                                 path_effects=[plt.matplotlib.patheffects.withStroke(linewidth=3, foreground='black')])


                ax.set_title(f"Predicted Landmarks (Sample {saved_count+1})")
                ax.axis('off') # 軸を非表示に
                
                # --- ファイル保存 ---
                base_name = os.path.basename(original_img_path)
                save_path = os.path.join(save_dir, f"pred_geometric_{base_name}")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0) # 余白なしで保存
                plt.close(fig) # 現在の図を閉じてメモリを解放
                
                print(f"✅ 予測画像を保存: {save_path}")
                saved_count += 1

# =================================================================
# 8. メインの訓練関数 (train/test分割を含む)
# =================================================================
def train_model():
    # --- パラメータ設定 ---
    DATA_DIR = "./cropped_dataset" # 訓練データセットのパス
    TEST_SIZE = 0.2 # テストデータの割合 (20%)
    BATCH_SIZE = 64 # 11/6 1: 32とする
    NUM_LANDMARKS = 9
    NUM_EPOCHS = 32 # 11/6 1: 20とする
    LEARNING_RATE = 0.001
    
    # --- デバイス設定 ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"訓練デバイス: {device}")
    
    # --- ファイルリストの取得と分割 (train/test分割) ---
    try:
        all_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
        if not all_files:
            raise FileNotFoundError("データセット内に.jpgファイルが見つかりません。")
        
        # train/test分割
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
    # テスト時の画像パスの対応のため shuffle=False にしておく
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 
    )
    
    # --- モデル、損失関数、最適化手法の設定 ---
    model = LandmarkRegressor(num_landmarks=NUM_LANDMARKS)
    
    # スクラッチ学習のため、重みはランダム初期化のまま
    # 転移学習を避けるため、事前学習済みの重みはロードしない
    print("\nモデルはランダムな重みで初期化されました (転移学習回避)。")

    model.to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- 損失記録用のリスト ---
    train_losses = []
    test_losses = []
    
    # --- 訓練ループ ---
    print("\n--- 訓練開始 ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train() # 訓練モード
        running_loss = 0.0
        
        # tqdm を使用して進捗バーを表示
        # データローダーから images, targets, img_paths を受け取る
        for i, (images, targets, _) in enumerate(tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]")):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # --- テストデータでの評価 ---
        test_loss, test_nme = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        
        print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] 完了. Train Loss: {avg_train_loss:.4f}, Test Loss: {test_loss:.4f}, Test NME: {test_nme:.4f} ---")

    # --- 最終評価とモデルの保存 ---
    final_test_loss , final_test_nme = evaluate_model(model, test_loader, criterion, device)
    print(f"\n✅ Final Test Loss: {final_test_loss:.4f}, Final Test NME: {final_test_nme:.4f}")

    MODEL_PATH_SAVE = 'landmark_regressor_scratch_improved.pth' # モデル保存パス
    torch.save(model.state_dict(), MODEL_PATH_SAVE)
    print(f"モデルが '{MODEL_PATH_SAVE}' として保存されました。")
    
    # --- 学習曲線のプロット ---
    print("\n--- 学習曲線を表示 ---")
    plot_loss_curve(train_losses, test_losses, NUM_EPOCHS)
    
    # モデルとテストローダーを返して、後続の処理で利用できるようにする
    return model, test_loader, device 

# =================================================================
# 9. スクリプト実行 
# =================================================================
if __name__ == '__main__':
    # 訓練を実行し、訓練済みモデルとテストローダーを取得
    trained_model, test_loader, device = train_model()

    # 予測結果の描画と保存を実行
    print("\n--- 予測ランドマークの描画と保存を開始 ---")
    save_landmark_predictions(
        model=trained_model, 
        data_loader=test_loader, 
        device=device, 
        num_samples=5, 
        save_dir="./predictions_output" # 保存先ディレクトリ
    )
    print("--- 予測ランドマークの描画と保存が完了しました。---")