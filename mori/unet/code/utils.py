#utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# ランドマーク座標(.pts) の読み込み関数
def load_landmarks_from_pts_to_tensor(pts_path):
    """.ptsファイルから9点のランドマーク座標を読み込み、平坦化されたTensor [18] で返す """
    points = []
    with open(pts_path, 'r') as f:
        lines = f.readlines()

    start_index = -1
    for i, line in enumerate(lines):
        if line.strip() == '{':
            start_index = i + 1
            break

    end_index = start_index + 9

    if start_index == -1:
        raise ValueError(f"'{{' marker not found in {pts_path}")

    for line in lines[start_index : end_index]:
        try:
            x, y = map(float, line.strip().split())
            points.extend([x, y])
        except ValueError:
            continue

    if len(points) != 18:
        raise ValueError(f"Expected 18 coordinates (9 points), but found {len(points)} in {pts_path}")

    return np.array(points, dtype=np.float32).reshape(9,2)

#ガウシアンヒートマップ生成関数
def generate_gaussian_heatmap(keypoints, output_size, sigma=3.0):
    """
    指定された座標に基づいてガウシアンヒートマップを生成します。
    keypoints: (N, 2) のキーポイント座標 (x, y)。
    """
    NUM_POINTS = keypoints.shape[0]
    heatmap = torch.zeros((output_size, output_size, NUM_POINTS), dtype=torch.float32)

    X_grid, Y_grid = torch.meshgrid(torch.arange(output_size), torch.arange(output_size), indexing='ij')
    
    two_sigma_sq = 2 * sigma * sigma

    for i in range(NUM_POINTS):
        # 座標を整数に変換して中心ピクセルを決定
        x, y = keypoints[i].astype(np.int32)
        
        # 座標が画像範囲内にあるかチェック
        if x < 0 or x >= output_size or y < 0 or y >= output_size:
            continue

        distance_sq = (X_grid - x)**2 + (Y_grid - y)**2
        
        # ガウス関数を適用
        gaussian_map = torch.exp(-distance_sq / two_sigma_sq)
        
        heatmap[:, :, i] = gaussian_map

    return heatmap.permute(2, 0, 1)


#ヒートマップから、最大の座標を抽出する (ヒートマップ->座標)
# utils.py / extract_keypoints_from_heatmap 関数内

def extract_keypoints_from_heatmap(heatmaps):
    N, C, H, W = heatmaps.shape
    device = heatmaps.device

    flat_heatmaps = heatmaps.view(N, C, -1)

    # ... (Softmax 適用部分) ...
    probs = F.softmax(flat_heatmaps, dim=2) # [N, C, H*W]

    # 2. 座標テンソルの作成

    # x座標グリッド [W]
    x_coords = torch.arange(W, dtype=torch.float32, device=device) # [W]
    # y座標グリッド [H]
    y_coords = torch.arange(H, dtype=torch.float32, device=device) # [H]

    # 3. 期待値 (E[x] と E[y]) の計算

    # E[x] の計算
    # (x, y) 座標の全組み合わせを持つグリッドを作成 (H*W)
    # x_grid_map の形状: [H, W]
    x_grid_map, y_grid_map = torch.meshgrid(x_coords, y_coords, indexing='xy')

    # [H, W] -> [1, 1, H*W] に変換 (N, C の次元にブロードキャストできるように準備)
    x_grid_flat = x_grid_map.contiguous().view(1, 1, H * W)
    y_grid_flat = y_grid_map.contiguous().view(1, 1, H * W)

    # E[x] の計算: 確率 * x座標の和 (形状: [N, C])
    x_keypoints = torch.sum(probs * x_grid_flat, dim=2)

    # E[y] の計算: 確率 * y座標の和 (形状: [N, C])
    y_keypoints = torch.sum(probs * y_grid_flat, dim=2)

    # 4. 座標を結合
    # 形状: [N, C, 2] (x, y)
    predicted_coords = torch.stack((x_keypoints, y_keypoints), dim=2)

    return predicted_coords


def calculate_normalization_factor(landmarks):
    coords = landmarks
    x_min = coords[..., 0].min(dim=1).values
    x_max = coords[..., 0].max(dim=1).values
    y_min = coords[..., 1].min(dim=1).values
    y_max = coords[..., 1].max(dim=1).values
    width = x_max - x_min
    height = y_max - y_min
    diagonal = torch.sqrt(width**2 + height**2)
    return diagonal + 1e-6    
