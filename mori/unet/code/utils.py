#utils.py

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

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
def extract_keypoints_from_heatmap(heatmaps):
    N, C, H, W = heatmaps.shape

    flat_heatmaps = heatmaps.view(N, C, -1)
    max_values, max_indices = torch.max(flat_heatmaps, dim=2)

    y_coords = max_indices // W
    x_coords = max_indices % W

    predicted_coords = torch.stack((x_coords, y_coords), dim=2).float()

    return predicted_coords

def calculate_normalization_factor(landmarks):
    coords = landmarks.reshape(-1, 9, 2)
    x_min = coords[..., 0].min(dim=1).values
    x_max = coords[..., 0].max(dim=1).values
    y_min = coords[..., 1].min(dim=1).values
    y_max = coords[..., 1].max(dim=1).values
    width = x_max - x_min
    height = y_max - y_min
    diagonal = torch.sqrt(width**2 + height**2)
    return diagonal + 1e-6    