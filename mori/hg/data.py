
import os
import cv2
import torch
from torch.utils.data import Dataset
from utils import parse_pts
import numpy as np

class AnimalDataset(Dataset):
    """
    動物の顔のデータセットのためのカスタムDatasetクラス。
    画像を読み込み、ランドマークからヒートマップを生成する。
    """
    def __init__(self, root_dir, image_size=256, heatmap_size=64):
        self.root_dir = root_dir
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        pts_path = os.path.join(self.root_dir, img_name.replace('.jpg', '.pts'))

        # 画像の読み込みと前処理
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, _ = image.shape
        
        # リサイズ
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        # HWC -> CHW and normalize
        image_tensor = torch.from_numpy(np.transpose(image_resized, (2, 0, 1))).float() / 255.0

        # ランドマークの読み込みと座標のスケーリング
        landmarks = parse_pts(pts_path)
        landmarks_scaled = landmarks.copy()
        landmarks_scaled[:, 0] *= self.image_size / w
        landmarks_scaled[:, 1] *= self.image_size / h

        # ヒートマップの生成
        heatmaps = self._generate_heatmaps(landmarks_scaled)

        return {
            'image': image_tensor,
            'heatmaps': torch.from_numpy(heatmaps).float()
        }

    def _generate_heatmaps(self, landmarks, sigma=1.5):
        """
        ランドマーク座標からガウシアンヒートマップを生成する。
        """
        num_landmarks = landmarks.shape[0]
        heatmaps = np.zeros((num_landmarks, self.heatmap_size, self.heatmap_size), dtype=np.float32)

        for i in range(num_landmarks):
            pt = landmarks[i]
            # ヒートマップの解像度に座標を変換
            mu_x = int(pt[0] * (self.heatmap_size / self.image_size))
            mu_y = int(pt[1] * (self.heatmap_size / self.image_size))

            # 座標が範囲外の場合はスキップ
            if not (0 <= mu_x < self.heatmap_size and 0 <= mu_y < self.heatmap_size):
                continue

            # ガウシアンカーネルの生成
            x = np.arange(0, self.heatmap_size, 1, np.float32)
            y = np.arange(0, self.heatmap_size, 1, np.float32)
            y = y[:, np.newaxis] # broadcastのため

            # 2Dガウシアン分布
            heatmap = np.exp(-((x - mu_x)**2 + (y - mu_y)**2) / (2 * sigma**2))
            heatmaps[i] = heatmap

        return heatmaps
