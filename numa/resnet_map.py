import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F 
from PIL import Image, ImageDraw, ImageFont 
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from tqdm import tqdm
import numpy as np 
from torchvision.models import resnet18, ResNet18_Weights


IMG_SIZE = 224
HEATMAP_SIZE = 56  # 出力ヒートマップサイズ
NUM_LANDMARKS = 9
SIGMA = 1.5  # ガウシアンの標準偏差


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


def generate_gaussian_heatmap(landmarks, heatmap_size=HEATMAP_SIZE, sigma=SIGMA):
    """
    landmarks: tensor [18] -> (x1, y1, x2, y2, ...)
    """
    num_landmarks = len(landmarks) // 2
    heatmaps = np.zeros((num_landmarks, heatmap_size, heatmap_size), dtype=np.float32)
    
    for i in range(num_landmarks):
        x, y = landmarks[2*i].item(), landmarks[2*i+1].item()
        # 元画像サイズからヒートマップサイズにスケール
        x = x * heatmap_size / IMG_SIZE
        y = y * heatmap_size / IMG_SIZE

        xx = np.arange(heatmap_size)
        yy = np.arange(heatmap_size)
        yy, xx = np.meshgrid(yy, xx)
        heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2*sigma**2))
    
    return torch.tensor(heatmaps, dtype=torch.float32)

class LandmarkDataset(Dataset):
    def __init__(self, file_paths, is_train=False):
        self.image_files = file_paths
        self.is_train = is_train
        self.W = IMG_SIZE

        # Tensor化・正規化
        self.transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        if is_train:
            self.affine_transform = RandomRotation(degrees=(-10, 10), W=self.W)
            self.hflip_transform = RandomHorizontalFlip(p=0.5, W=self.W)
        else:
            self.affine_transform = None
            self.hflip_transform = None

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        pts_path = img_path.replace(".jpg", ".pts")

        image = Image.open(img_path).convert("RGB")
        landmarks = load_landmarks_from_pts_to_tensor(pts_path)

        if self.is_train:
            image, landmarks = self.affine_transform(image, landmarks)
            image, landmarks = self.hflip_transform(image, landmarks)

        image = self.transform_tensor(image)
        heatmaps = generate_gaussian_heatmap(landmarks)

        return image, heatmaps, img_path


class LandmarkHeatmapRegressor(nn.Module):
    def __init__(self, num_landmarks=9, output_size=56, pretrained=True):
        super().__init__()
        # ResNet18 backbone
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        
        # 最後のFCをIdentityにして特徴マップを保持
        self.backbone = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, num_landmarks, kernel_size=4, stride=2, padding=1)
        )
        self.output_size = output_size

    def forward(self, x):
        x = self.backbone(x)  # (B, 512, H/32, W/32)
        heatmaps = self.deconv_layers(x)  # (B, num_landmarks, output_size, output_size)
        return heatmaps

def heatmap_to_coord(heatmaps):
    """
    heatmaps: (B, num_landmarks, H, W)
    return: coords (B, num_landmarks, 2)
    """
    B, L, H, W = heatmaps.shape
    coords = torch.zeros(B, L, 2, device=heatmaps.device)

    for b in range(B):
        for l in range(L):
            hmap = heatmaps[b, l]
            idx = torch.argmax(hmap)
            y, x = divmod(idx.item(), W)
            coords[b, l] = torch.tensor([x, y], device=heatmaps.device)
    return coords

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_nme = 0
    count = 0

    with torch.no_grad():
        for data in data_loader:
            imgs, labels = data[0].to(device), data[1].to(device)
            
            # モデル出力はヒートマップ (B, num_landmarks, H, W)
            outputs = model(imgs)
            
            # labels をヒートマップに変換して MSE 損失
            target_heatmaps = generate_gaussian_heatmap(labels_scaled, heatmap_size=outputs.shape[2])
            target_heatmaps = target_heatmaps.to(device)
            
            loss = criterion(outputs, target_heatmaps)
            total_loss += loss.item() * imgs.size(0)
            
            # NME 計算のため座標に変換
            pred_coords = heatmap_to_coord(outputs)
            nme_batch = calculate_nme(pred_coords, labels, device)
            total_nme += nme_batch.item() * imgs.size(0)
            
            count += imgs.size(0)

    avg_loss = total_loss / count
    avg_nme = total_nme / count
    return avg_loss, avg_nme


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
    # unsqueeze(1) で [N] -> [N, 1] にしてブローgenerateドキャストを可能にする
    normalized_distances = distances / normalization_factors.unsqueeze(1) # [N, 9]

    # 4. 全ての正規化距離の平均を取る 
    nme = normalized_distances.mean()

    return nme


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

def save_landmark_predictions(model, data_loader, device, num_samples=5, save_dir="./predictions_map"):
    modelneval()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    saved_count = 0
    with torch.no_grad():
        for images, targets, img_paths in data_loader:
            if saved_count >= num_samples:
                break

            images_tensor = images.to(device)
            outputs = model(images_tensor)  # (B, num_landmarks, H, W)

            # ヒートマップ → 座標に変換
            pred_coords_batch = heatmap_to_coord(outputs)

            for i in range(images_tensor.size(0)):
                if saved_count >= num_samples:
                    break

                # 元の画像読み込み
                original_img_path = img_paths[i]
                original_image_pil = Image.open(original_img_path).convert("RGB")
                original_w, original_h = original_image_pil.size

                # ヒートマップ座標を元画像サイズにスケーリング
                H_hm, W_hm = outputs.shape[2], outputs.shape[3]
                pred_coords = pred_coords_batch[i].cpu().numpy()
                scaled_x = pred_coords[:, 0] * (original_w / W_hm)
                scaled_y = pred_coords[:, 1] * (original_h / H_hm)
                scaled_landmarks = np.stack([scaled_x, scaled_y], axis=1)

                # 描画
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(original_image_pil)

                # 例：ランドマーク線や円も描画したい場合はここで scaled_landmarks を使用
                ax.scatter(scaled_landmarks[:, 0], scaled_landmarks[:, 1],
                           c='red', marker='o', s=50)

                # ランドマーク番号表示
                for k_idx in range(scaled_landmarks.shape[0]):
                    ax.annotate(str(k_idx+1),
                                (scaled_landmarks[k_idx, 0]+10, scaled_landmarks[k_idx, 1]+10),
                                color='yellow', fontsize=15, ha='center', va='center')

                ax.set_title(f"Predicted Landmarks (Sample {saved_count+1})")
                ax.axis('off')

                # 保存
                base_name = os.path.basename(original_img_path)
                save_path = os.path.join(save_dir, f"pred_{base_name}")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close(fig)
                print(f"予測画像を保存: {save_path}")
                saved_count += 1

def train_model():
    DATA_DIR = "./cropped_dataset"
    BATCH_SIZE = 32
    NUM_LANDMARKS = 9
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練デバイス: {device}")

    all_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    train_files, temp_files = train_test_split(all_files, test_size=0.2, random_state=42)
    valid_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    train_dataset = LandmarkDataset(train_files, is_train=True)
    valid_dataset = LandmarkDataset(valid_files)
    test_dataset  = LandmarkDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset , batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = LandmarkRegressor(num_landmarks=NUM_LANDMARKS, img_size=IMG_SIZE).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    LOG_FILE = "log_resnet_map.txt"
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)
    with open(LOG_FILE, "w") as f:
        f.write("Epoch,Train_MSE,Train_NME,Valid_MSE,Valid_NME\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_nme  = 0.0
        for images, targets, _ in tqdm(train_loader, desc=f"Train {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(device)
            targets = targets.to(device)
            target_heatmaps = generate_gaussian_heatmap(targets, heatmap_size=IMG_SIZE).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target_heatmaps)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred_coords = heatmap_to_coord(outputs)
            train_nme += calculate_nme(pred_coords, targets, device).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_train_nme  = train_nme / len(train_loader)
        valid_loss, valid_nme = evaluate_model(model, valid_loader, criterion, device)

        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.6f},{avg_train_nme:.6f},{valid_loss:.6f},{valid_nme:.6f}\n")

        print(f"[Epoch {epoch+1}] Train MSE: {avg_train_loss:.4f}, Train NME: {avg_train_nme:.4f}, "
              f"Valid MSE: {valid_loss:.4f}, Valid NME: {valid_nme:.4f}")

    # テスト評価
    test_loss, test_nme = evaluate_model(model, test_loader, criterion, device)
    print(f"Test MSE: {test_loss:.4f}, Test NME: {test_nme:.4f}")

    torch.save(model.state_dict(), "model_map.pth")
    print("モデル保存完了: model_map.pth")

    return model, test_loader, device


if __name__ == "__main__":
    # 訓練を実行
    trained_model, test_loader, device = train_model()

    print("\n予測ランドマークの描画と保存を開始 ")
    save_landmark_predictions(
        model=trained_model,
        data_loader=test_loader,
        device=device,
        num_samples=5,
        save_dir="./predictions_map"  # 保存フォルダは自由に変更
    )
    print("--- 予測ランドマークの描画と保存が完了しました ---")
