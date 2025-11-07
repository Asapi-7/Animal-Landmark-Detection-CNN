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
    heatmap = np.zeros((output_size, output_size, NUM_POINTS), dtype=np.float32)

    X_grid, Y_grid = np.meshgrid(np.arange(output_size), np.arange(output_size))
    
    two_sigma_sq = 2 * sigma * sigma

    for i in range(NUM_POINTS):
        # 座標を整数に変換して中心ピクセルを決定
        x, y = keypoints[i].astype(np.int32)
        
        # 座標が画像範囲内にあるかチェック
        if x < 0 or x >= output_size or y < 0 or y >= output_size:
            continue

        distance_sq = (X_grid - x)**2 + (Y_grid - y)**2
        
        # ガウス関数を適用
        gaussian_map = np.exp(-distance_sq / two_sigma_sq)
        
        heatmap[:, :, i] = gaussian_map

    return heatmap