import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from torchvision.transforms import functional as F  
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

IMG_SIZE = 224
HEATMAP_SIZE = 56  # 出力ヒートマップサイズ
NUM_LANDMARKS = 9
SIGMA = 5  # ガウシアンの標準偏差


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


class LandmarkDataset(Dataset):
    def __init__(self, file_paths):
        self.image_files = file_paths
        self.transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        pts_path = img_path.replace(".jpg", ".pts")

        image = Image.open(img_path).convert("RGB")
        landmarks = load_landmarks_from_pts_to_tensor(pts_path)

        image = self.transform_tensor(image)
        return image, landmarks, img_path


def generate_gaussian_heatmap_batch(landmarks_batch, heatmap_size=HEATMAP_SIZE, sigma=SIGMA, device='cuda'):
    B = landmarks_batch.shape[0]
    num_landmarks = landmarks_batch.shape[1] // 2
    heatmaps = torch.zeros((B, num_landmarks, heatmap_size, heatmap_size), device=device)

    xx = torch.arange(heatmap_size, device=device).view(1, 1, heatmap_size).float()
    yy = torch.arange(heatmap_size, device=device).view(1, heatmap_size, 1).float()

    for b in range(B):
        for i in range(num_landmarks):
            x = landmarks_batch[b, 2*i] * heatmap_size / IMG_SIZE
            y = landmarks_batch[b, 2*i+1] * heatmap_size / IMG_SIZE
            heatmaps[b, i] = torch.exp(-((xx - x)**2 + (yy - y)**2) / (2*sigma**2))

    return heatmaps



class LandmarkHeatmapRegressor(nn.Module):
    def __init__(self, num_landmarks=NUM_LANDMARKS, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)

        self.conv1 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1 
        self.layer2 = backbone.layer2 
        self.layer3 = backbone.layer3 
        self.layer4 = backbone.layer4 

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
     
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
      
        self.deconv3 = nn.ConvTranspose2d(128, num_landmarks, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x_c1 = self.conv1(x)
        x_c2 = self.layer1(x_c1)
        x_c3 = self.layer2(x_c2) 
        x_c4 = self.layer3(x_c3) 
        x_c5 = self.layer4(x_c4) 

        x_d1 = self.deconv1(x_c5)    
        x_d2 = self.deconv2(x_d1)
   
        heatmaps = self.deconv3(x_d2)
        return heatmaps



def heatmap_to_coord(heatmaps):
    B, N, H, W = heatmaps.shape
    coords = torch.zeros((B, N*2), device=heatmaps.device)
    for b in range(B):
        for i in range(N):
            hm = heatmaps[b, i]
            idx = torch.argmax(hm)
            y, x = divmod(idx.item(), W)
            coords[b, 2*i] = x * IMG_SIZE / W
            coords[b, 2*i+1] = y * IMG_SIZE / H
    return coords


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
    # unsqueeze(1) で [N] -> [N, 1] にしてブローgenerateドキャストを可能にする
    normalized_distances = distances / normalization_factors.unsqueeze(1) # [N, 9]

    # 4. 全ての正規化距離の平均を取る 
    nme = normalized_distances.mean()

    return nme

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_nme = 0.0
    with torch.no_grad():
        for images, targets, _ in data_loader:
            try:
                images = images.to(device)
                targets = targets.to(device)
                target_heatmaps = generate_gaussian_heatmap_batch(targets, device=device)
                outputs = model(images)
                loss = criterion(outputs, target_heatmaps)
                total_loss += loss.item()
                pred_coords = heatmap_to_coord(outputs)
                total_nme += calculate_nme(pred_coords, targets, device).item()
            except Exception as e:
                print("Error in evaluate_model:", e)
                break
    return total_loss / len(data_loader), total_nme / len(data_loader)



def draw_landmarks_pil(image, landmarks, color='red', point_size=5):
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("arial.ttf", size=18)
    except IOError:
        font = ImageFont.load_default()

    # tensor→numpyに変換
    if isinstance(landmarks, torch.Tensor):
        landmarks = landmarks.cpu().numpy()

    # (N,2) 形式でない場合 → reshape
    if landmarks.ndim == 1:
        landmarks = landmarks.reshape(-1, 2)

    for i, (x, y) in enumerate(landmarks, start=1):
        x, y = int(x), int(y)
        draw.ellipse([x-point_size, y-point_size, x+point_size, y+point_size], fill=color)
        draw.text((x+point_size+3, y+point_size+3), str(i), fill="yellow", font=font)

    return image



def save_landmark_predictions(model, data_loader, device, num_samples=5, save_dir="./predictions_output",
                              IMG_SIZE=224, NUM_LANDMARKS=9):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    saved_count = 0

    with torch.no_grad():
        for images, targets, img_paths in data_loader:
            outputs = model(images.to(device)).cpu()  # (B, N*2)

            for idx in range(images.size(0)):
                if saved_count >= num_samples: return

                img = Image.open(img_paths[idx]).convert("RGB")
                W, H = img.size

                lm = outputs[idx].numpy().reshape(-1, 2)  # (9,2)

                # 元解像度へ変換
                lm[:, 0] *= (W / IMG_SIZE)
                lm[:, 1] *= (H / IMG_SIZE)

                # ------- Matplotlib描画 -------
                fig, ax = plt.subplots(figsize=(8,8))
                ax.imshow(img)
                ax.scatter(lm[:,0], lm[:,1], color='red', s=50)

                # 番号
                for i in range(NUM_LANDMARKS):
                    ax.text(lm[i,0]+8, lm[i,1]+8, str(i+1), color='yellow', fontsize=18)

                # 幾何処理（円2つ＋ライン）
                p12 = (lm[0], lm[1])
                p34 = (lm[2], lm[3])

                for pA, pB in [p12, p34]:
                    cx = (pA[0]+pB[0])/2; cy = (pA[1]+pB[1])/2
                    r = np.linalg.norm(pA-pB)/2
                    ax.add_patch(plt.Circle((cx,cy), r, fill=False, color='red', linewidth=2))

                idxs=[5,7,6,8,5]
                ax.plot(lm[idxs,0], lm[idxs,1], color='red', linewidth=3)

                ax.axis("off")
                out=f"{save_dir}/pred_{saved_count+1:03}.jpg"
                plt.savefig(out,bbox_inches='tight',pad_inches=0)
                plt.close(); saved_count+=1

                print(f"保存：{out}")


def train_model():
    DATA_DIR = "./cropped_dataset"
    BATCH_SIZE = 32
    NUM_EPOCHS = 9
    LEARNING_RATE = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"訓練デバイス: {device}")

    all_files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
    train_files, temp_files = train_test_split(all_files, test_size=0.2, random_state=42)
    valid_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    print("Train files:", len(train_files))
    print("Valid files:", len(valid_files))
    print("Test files:", len(test_files))

    train_dataset = LandmarkDataset(train_files)
    valid_dataset = LandmarkDataset(valid_files)
    test_dataset  = LandmarkDataset(test_files)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_dataset , batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = LandmarkHeatmapRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    LOG_FILE = "log_resnet_map.txt"
    with open(LOG_FILE, "w") as f:
        f.write("Epoch,Train_MSE,Train_NME,Valid_MSE,Valid_NME\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        for images, targets, _ in tqdm(train_loader, desc=f"Train {epoch+1}/{NUM_EPOCHS}"):
            images = images.to(device)
            targets = targets.to(device)
            target_heatmaps = generate_gaussian_heatmap_batch(targets, device=device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, target_heatmaps)
            loss.backward()
            optimizer.step()

        # 評価
        train_loss, train_nme = evaluate_model(model, train_loader, criterion, device)
        valid_loss, valid_nme = evaluate_model(model, valid_loader, criterion, device)
        print(f"[Epoch {epoch+1}] Train MSE: {train_loss:.4f}, Train NME: {train_nme:.4f}, "
              f"Valid MSE: {valid_loss:.4f}, Valid NME: {valid_nme:.4f}")
        with open(LOG_FILE, "a") as f:
            f.write(f"{epoch+1},{train_loss:.6f},{train_nme:.6f},{valid_loss:.6f},{valid_nme:.6f}\n")

    torch.save(model.state_dict(), "model_map.pth")
    print("モデル保存完了: model_map.pth")
    return model, test_loader, device


if __name__ == "__main__":
    # 訓練を実行
    trained_model, test_loader, device = train_model()

    print("\n最終テスト評価")
    criterion = torch.nn.MSELoss()
    test_loss, test_nme = evaluate_model(trained_model, test_loader, criterion, device)
    print(f"Test MSE: {test_loss:.4f}")
    print(f"Test NME: {test_nme:.4f}")

    print("\n予測ランドマークの描画と保存を開始 ")
    save_landmark_predictions(
        model=trained_model,
        data_loader=test_loader,
        device=device,
        num_samples=5,
        save_dir="./predictions_map"  
    )
    print("予測ランドマークの描画と保存が完了しました")
