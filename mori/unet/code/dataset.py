import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
from torchvision import transforms

# utils.py からインポート
from utils import load_landmarks_from_pts_to_tensor, generate_gaussian_heatmap 


class LandmarkHeatmapDataset(Dataset):
    def __init__(self, file_list, root_dir, image_size=224, sigma=3.0, transform=None):
        """
        Args:
            root_dir (str): データセットのルートディレクトリ。
            image_size (int): 画像とヒートマップの出力サイズ (H, W)。
            sigma (float): ガウシアンカーネルのシグマ値。
            transform (callable, optional): 画像に適用されるオプションの変換。
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.sigma = sigma
        self.file_list = []
        
        for jpg_path in file_list:
            base_name = os.path.splitext(jpg_path)[0] 
            pts_path = base_name + ".pts"

            if os.path.exists(pts_path):
                self.file_list.append((jpg_path, pts_path))
            else:
                print(f"Warning: Correspoinding .pts file not found for {jpg_path}")
        

        # 画像の前処理 (Tensor化と正規化)
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(), # HxWxC (0-255) を CxHxW (0.0-1.0) に変換
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNetの統計量で正規化
            ])
        else:
            self.transform = transform
            
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        jpg_path, pts_path = self.file_list[idx]
        
        # 1. 画像の読み込みと変換
        image = Image.open(jpg_path).convert("RGB")
        image_tensor = self.transform(image)
        
        # 2. ランドマークの読み込み
        keypoints_np = load_landmarks_from_pts_to_tensor(pts_path)
        
        # 3. ガウシアンヒートマップの生成 (ターゲット)
        # keypoints_npの座標は、Resize後の画像サイズにスケールされている必要があります。
        # 今回はResizeで固定サイズ(224x224)にしているため、座標もその範囲内にあると仮定します。
        heatmap_tensor = generate_gaussian_heatmap(keypoints_np, self.image_size, self.sigma)

        coords_tensor = torch.from_numpy(keypoints_np.flatten()).float()
        
        return image_tensor, heatmap_tensor, coords_tensor, jpg_path