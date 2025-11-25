import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F 
from PIL import Image, ImageDraw, ImageFont 
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from tqdm import tqdm
import numpy as np 


IMG_SIZE = 224
NUM_LANDMARKS = 9


# ランドマーク座標 (.pts) の読み込み関数
def load_landmarks_from_pts_to_tensor(pts_path):
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
            points.extend([x, y]) 
        except ValueError:
            continue

    if len(points) != 18:
          raise ValueError(f"Expected 18 coordinates (9 points), but found {len(points)} in {pts_path}")

    return torch.tensor(points, dtype=torch.float32)


#  幾何学的拡張クラス (座標変換を伴う) の定義
class RandomAffineLandmarks:
    def __init__(self, degrees, translate, scale_ranges, shear, W=IMG_SIZE):
        self.W = W #画像の幅
        self.degrees = degrees #degrees:回転角度
        self.translate = translate  #translate:平行移動の比率
        self.scale_ranges = scale_ranges #scale_ranges:拡大縮小率
        self.shear = shear #せん断の角度
        self.center = (W / 2, W / 2) # 画像の中心 

    def get_params(self): #ランダムパラメータ生成
        # 回転角度
        angle = float(torch.empty(1).uniform_(float(self.degrees[0]), float(self.degrees[1])).item())
        
        # 平行移動のピクセル
        max_dx = self.translate[0] * self.W
        max_dy = self.translate[1] * self.W
        translate = [
            float(torch.empty(1).uniform_(-max_dx, max_dx).item()),
            float(torch.empty(1).uniform_(-max_dy, max_dy).item())
        ]

        # 拡大縮小率
        scale_factor = float(torch.empty(1).uniform_(self.scale_ranges[0], self.scale_ranges[1]).item())
        
        # せん断の角度
        shear = [0.0, 0.0]
        if self.shear is not None:
            shear_x = float(torch.empty(1).uniform_(self.shear[0], self.shear[1]).item())
            shear_y = float(torch.empty(1).uniform_(self.shear[0], self.shear[1]).item())
            shear = [shear_x, shear_y]

        return angle, translate, scale_factor, shear


    def __call__(self, img, landmarks):
        # パラメータ取得
        angle, translate, scale_factor, shear = self.get_params()

        # 画像変換
        img_transformed = F.affine(
            img, angle=angle, translate=translate, scale=scale_factor, shear=shear, 
            interpolation=Image.BICUBIC, fill=0
        )

        # ランドマーク変換 
        pts = landmarks.reshape(-1, 2).double()

        # 中心を(0, 0)に移動
        center_tensor = torch.tensor(self.center, dtype=torch.float64)
        pts = pts - center_tensor

        # 回転を適用
        angle_rad = np.deg2rad(angle)
        R = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ], dtype=torch.float64)
        pts = pts @ R.T

        # 拡大縮小を適用
        pts = pts * scale_factor

        # シアー
        shear_x_rad = np.deg2rad(shear[0])
        shear_y_rad = np.deg2rad(shear[1])
        S = torch.tensor([
            [1.0, np.tan(shear_x_rad)],
            [np.tan(shear_y_rad), 1.0]
        ], dtype=torch.float64)
        pts = pts @ S.T

        # 中心を元に戻す
        pts = pts + center_tensor

        # 平行移動を適用
        translate_tensor = torch.tensor(translate, dtype=torch.float64)
        pts = pts + translate_tensor

        return img_transformed, pts.flatten().float()


class RandomHorizontalFlipWithLandmarks:
    #水平反転とランドマーク座標変換
    def __init__(self, p=0.5, W=IMG_SIZE):
        self.p = p #水平反転を行う確率
        self.W = W #画像の幅
        # ランドマーク順序交換
        self.point_swap_map = {0: 3, 3: 0, 1: 2, 2: 1, 5: 6, 6: 5}

    def __call__(self, img, landmarks):
        if torch.rand(1) < self.p:
            # 水平反転
            img = transforms.functional.hflip(img)
            
            # ランドマーク座標を反転
            flipped = landmarks.clone()
            flipped[::2] = (self.W - 1) - flipped[::2]

            coords = flipped.reshape(-1, 2).clone()
            
            # ランドマーク順序交換
            new_coords = coords.clone()
            for src, dst in self.point_swap_map.items():
                new_coords[dst] = coords[src]
            
            return img, new_coords.flatten().float()
        
        return img, landmarks.clone().float()

#データ拡張
class LandmarkDataset(Dataset):
    def __init__(self, file_paths, is_train=False): 
        self.image_files = file_paths 
        self.is_train = is_train #データセットが訓練用で使用されているか
        #正規化
        final_transforms = [
            transforms.ToTensor(), 
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if is_train:
            # 画素値拡張 
            self.transform_tensor = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 色ジッター
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),              # ガウシアンブラー
            ] + final_transforms)
            
            # 幾何学的拡張
            self.affine_transform = RandomAffineLandmarks(
                degrees=(-10, 10), translate=(0.05, 0.05), # ランダム回転と平行移動
                scale_ranges=(0.95, 1.05), shear=(-5, 5)   # 拡大縮小とシアー
            )
            self.hflip_transform = RandomHorizontalFlipWithLandmarks(p=0.5, W=IMG_SIZE)
            
        else: #テスト時は拡張なし
            self.transform_tensor = transforms.Compose(final_transforms)
            self.affine_transform = None
            self.hflip_transform = None


    def __len__(self): 
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        pts_path = img_path.replace(".jpg", ".pts") 

        image = Image.open(img_path).convert("RGB") 
        landmarks = load_landmarks_from_pts_to_tensor(pts_path) 

        # 幾何学的拡張の適用 
        if self.is_train:
            # ランダム水平反転
            image, landmarks = self.hflip_transform(image, landmarks) 
            # ランダムアフィン変換 (回転、平行移動、スケール、シアー)
            image, landmarks = self.affine_transform(image, landmarks) 
        
        # 画素値拡張の適用
        transformed_image = self.transform_tensor(image) 

        return transformed_image, landmarks, img_path 

#Resnet18
class BasicBlock(nn.Module): #残差ブロック
    def __init__(self, in_channels: int, out_channels: int,
                 stride: int=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        # スキップ接続
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
             self.downsample = nn.Sequential(
                 nn.Conv2d(in_channels, out_channels, kernel_size=1,
                           stride=stride, bias=False),
                 nn.BatchNorm2d(out_channels)
             )

    #順伝播関数
    def forward(self, x: torch.Tensor):
        identity = x 

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
             identity = self.downsample(identity)

        # 残差写像と恒等写像の要素毎の和を計算
        out += identity

        out = self.relu(out)

        return out

class ResNet18(nn.Module):
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

        x = self.linear(x)

        return x

class LandmarkRegressor(nn.Module):
    def __init__(self, num_landmarks=9):
        super(LandmarkRegressor, self).__init__()
        
        # 1. Backbone:ResNet18を使用
        self.backbone = ResNet18(num_classes=1000)
        
        # 2. Head: Dense層 (最終層) の変更
        num_features = self.backbone.linear.in_features
        
        # 3. 出力層をランドマークの数 (18) に置き換え 
        self.backbone.linear = nn.Linear(num_features, num_landmarks * 2)

    def forward(self, x):
        return self.backbone(x)

#評価関数
def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_nme = 0
    count = 0

    with torch.no_grad():
        for data in data_loader:
             imgs, labels = data[0].to(device), data[1].to(device)
             outputs = model(imgs)

             loss = criterion(outputs, labels)
             total_loss += loss.item() * imgs.size(0)

             nme_batch = calculate_nme(outputs, labels, device)
             total_nme += nme_batch.item() * imgs.size(0)


             count += imgs.size(0)

    avg_loss = total_loss / count
    avg_nme = total_nme / count
    return avg_loss, avg_nme

#NME
def calculate_normalization_factor(landmarks): 
    #ランドマーク [N, 18] から、バウンディングボックスの対角線長を計算する。
    
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
    # NME (Normalized Mean Error) を計算する 
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

    # 4. 全ての正規化距離の平均を取る 
    nme = normalized_distances.mean()

    return nme

#損失曲線
def plot_loss_curve(train_losses, test_losses, num_epochs):
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

#ランドマーク描画
def draw_landmarks_pil(image, landmarks, color='red', point_size=5):
    draw = ImageDraw.Draw(image)
    
    try:
        font_size = 18
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
        draw.text(text_position, str(landmark_index), fill="black", font=font, stroke_width=1, stroke_fill="black")
        draw.text(text_position, str(landmark_index), fill="yellow", font=font)
        
    return image

#予測結果を画像に描画
def save_landmark_predictions(model, data_loader, device, num_samples=5, save_dir="./predictions_output"):
    model.eval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_count = 0
    
    with torch.no_grad():
        for images, targets, img_paths in data_loader:
            if saved_count >= num_samples:
                break
                
            images_tensor = images.to(device)
            outputs = model(images_tensor).cpu() 

            for i in range(images_tensor.size(0)):
                if saved_count >= num_samples:
                    break
                    
                # 元の画像の読み込み
                original_img_path = img_paths[i]
                original_image_pil = Image.open(original_img_path).convert("RGB")
                
                predicted_landmarks_flat = outputs[i].numpy() 
                predicted_landmarks_reshaped = predicted_landmarks_flat.reshape(-1, 2) 

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(original_image_pil)
                
                # 予測座標を元の画像サイズにスケーリング
                original_w, original_h = original_image_pil.size
                
                scaled_landmarks_x = predicted_landmarks_reshaped[:, 0] * (original_w / IMG_SIZE)
                scaled_landmarks_y = predicted_landmarks_reshaped[:, 1] * (original_h / IMG_SIZE)
                
                scaled_landmarks = np.stack([scaled_landmarks_x, scaled_landmarks_y], axis=1) 

                # 1, 2個目の点を直径とする円 (赤色) の描画
                p1 = scaled_landmarks[0]
                p2 = scaled_landmarks[1]
                center_x12 = (p1[0] + p2[0]) / 2
                center_y12 = (p1[1] + p2[1]) / 2
                diameter12 = np.linalg.norm(p1 - p2)
                radius12 = diameter12 / 2
                circle12 = plt.Circle((center_x12, center_y12), radius12, 
                                     color='red', fill=False, linewidth=2)
                ax.add_artist(circle12)
                
                # 3, 4個目の点を直径とする円 (赤色) の描画 
                p3 = scaled_landmarks[2]
                p4 = scaled_landmarks[3]
                center_x34 = (p3[0] + p4[0]) / 2
                center_y34 = (p3[1] + p4[1]) / 2
                diameter34 = np.linalg.norm(p3 - p4)
                radius34 = diameter34 / 2
                circle34 = plt.Circle((center_x34, center_y34), radius34, 
                                     color='red', fill=False, linewidth=2)
                ax.add_artist(circle34)

                
                indices = [5, 7, 6, 8, 5] 
                line_x = scaled_landmarks_x[indices]
                line_y = scaled_landmarks_y[indices]
                ax.plot(line_x, line_y, color='red', linestyle='-', linewidth=2)

                # ランドマーク点を描画 
                ax.scatter(scaled_landmarks_x, scaled_landmarks_y, 
                           c='red', marker='o', s=50, label=None)
                
                # ランドマークに番号表示
                for k_idx in range(NUM_LANDMARKS): 
                    tmp = 10
                    ax.annotate(str(k_idx+1), (scaled_landmarks_x[k_idx] + tmp + 1, scaled_landmarks_y[k_idx] + tmp + 1), color='black', fontsize=20, ha='center', va='center')
                    ax.annotate(str(k_idx+1), (scaled_landmarks_x[k_idx] + tmp + 3, scaled_landmarks_y[k_idx] + tmp + 1), color='black', fontsize=20, ha='center', va='center')
                    ax.annotate(str(k_idx+1), (scaled_landmarks_x[k_idx] + tmp + 1, scaled_landmarks_y[k_idx] + tmp + 3), color='black', fontsize=20, ha='center', va='center')
                    ax.annotate(str(k_idx+1), (scaled_landmarks_x[k_idx] + tmp + 3, scaled_landmarks_y[k_idx] + tmp + 3), color='black', fontsize=20, ha='center', va='center')
                    ax.annotate(str(k_idx+1), (scaled_landmarks_x[k_idx] + tmp + 2, scaled_landmarks_y[k_idx] + tmp + 2), color='yellow', fontsize=20, ha='center', va='center')


                ax.set_title(f"Predicted Landmarks (Sample {saved_count+1})")
                ax.axis('off')
                
               
                base_name = os.path.basename(original_img_path)
                save_path = os.path.join(save_dir, f"pred_geometric_{base_name}")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0) 
                plt.close(fig) 
                
                print(f"予測画像を保存: {save_path}")
                saved_count += 1


def train_model():
    DATA_DIR = "./cropped_dataset" # 訓練データセットのパス
    TEST_SIZE = 0.2 # テストデータの割合 (20%)
    BATCH_SIZE = 32
    NUM_LANDMARKS = 9
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練デバイス: {device}")
    
    
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
    
   
    train_dataset = LandmarkDataset(train_files, is_train=True) 
  
    test_dataset = LandmarkDataset(test_files, is_train=False) 
        
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4 
    )
    
    model = LandmarkRegressor(num_landmarks=NUM_LANDMARKS)
    model.to(device)
    
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 記録用リスト 
    train_losses = []
    test_losses = []

    train_nmes = []
    test_nmes = [] 
    
    # 4つの値を保存するためのログファイル
    TRAIN_LOG_FILE = os.path.join(os.getcwd(), 'training_evaluation_log.txt')
    
    # 古いログをクリアし、ヘッダーを書き込む
    if os.path.exists(TRAIN_LOG_FILE):
        os.remove(TRAIN_LOG_FILE)
    
    with open(TRAIN_LOG_FILE, 'w') as f:
        # ヘッダー行を記述
         f.write("Epoch,Train_Loss(MSE),Train_NME,Test_Loss(MSE),Test_NME\n")
        
    print(f"訓練・評価ログファイルを準備中: {TRAIN_LOG_FILE}")

    # 訓練ループ 
    print("\n--- 訓練開始 ---")
    
    for epoch in range(NUM_EPOCHS):
        model.train() 
        running_loss = 0.0
        
        # 進捗表示
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
        
        # 訓練データでの評価（NMEを計算）
        train_loss_epoch, train_nme_epoch = evaluate_model(model, train_loader, criterion, device)
        train_nmes.append(train_nme_epoch)
        
        #テストデータでの評価 
        test_loss, test_nme = evaluate_model(model, test_loader, criterion, device)
        test_losses.append(test_loss)
        test_nmes.append(test_nme) 

        with open(TRAIN_LOG_FILE, 'a') as f:
            f.write(
            f"{epoch+1},{avg_train_loss:.6f},{train_nme_epoch:.6f},"
            f"{test_loss:.6f},{test_nme:.6f}\n"
        )
        print(f"--- Epoch [{epoch+1}/{NUM_EPOCHS}] 完了. Train Loss: {avg_train_loss:.4f}, Train NME: {train_nme_epoch:.4f}, Test Loss: {test_loss:.4f}, Test NME: {test_nme:.4f} ---")

    # 最終評価とモデルの保存 
    final_test_loss , final_test_nme = evaluate_model(model, test_loader, criterion, device)
    print(f"\n Final Test Loss: {final_test_loss:.4f}, Final Test NME: {final_test_nme:.4f}")

    MODEL_PATH_SAVE = 'landmark_regressor_final_2.pth' 
    torch.save(model.state_dict(), MODEL_PATH_SAVE)
    print(f"モデルが '{MODEL_PATH_SAVE}' として保存されました。")
    
    print("\n--- 学習曲線を表示 ---")
    plot_loss_curve(train_losses, test_losses, NUM_EPOCHS)
    
    return model, test_loader, device


# スクリプト実行
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