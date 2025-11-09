import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # 追加
from tqdm import tqdm
import os
import argparse
import torch.nn.functional as F
import csv
import matplotlib.pyplot as plt # 追加
import numpy as np
import glob
from PIL import Image
from unet import UNet
from dataset import LandmarkHeatmapDataset

from unet import UNet
from dataset import LandmarkHeatmapDataset
from utils import calculate_normalization_factor, extract_keypoints_from_heatmap

# NME (Normalized Mean Error) 計算関数 (ヒートマップ回帰用に修正)
def calculate_nme(heatmap_outputs, label_coords, device):
    """ 
    ヒートマップ出力と正解座標を使用して NME (Normalized Mean Error) を計算する 
    heatmap_outputs: モデル出力 [N, 9, H, W]
    label_coords: 正解ランドマーク座標 [N, 18]
    """
    num_landmarks = 9
    
    # 1. ヒートマップから予測座標を抽出 (N, 9, 2)
    predicted_coords_reshaped = extract_keypoints_from_heatmap(heatmap_outputs).to(device)
    
    # 2. 正解座標を (N, 9, 2) に整形
    labels_reshaped = label_coords.reshape(-1, num_landmarks, 2)

    # 3. 予測座標と正解座標間のユークリッド距離を計算 (各ランドマークごと)
    distances = torch.linalg.norm(predicted_coords_reshaped - labels_reshaped, dim=2) # [N, 9]

    # 4. 正規化ファクター（バウンディングボックスの対角線長）を計算
    # 既存の関数は [N, 18] を期待
    normalization_factors = calculate_normalization_factor(label_coords).to(device) # [N]

    # 5. 各ランドマークの距離を正規化ファクターで割る
    # unsqueeze(1) で [N] -> [N, 1] にしてブロードキャストを可能にする
    normalized_distances = distances / normalization_factors.unsqueeze(1) # [N, 9]

    # 6. 全ての正規化距離の平均を取る (これが NME)
    nme = normalized_distances.mean()

    return nme

# 評価関数 (evaluate_model) はそのまま使用可能

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs, output_dir):
    # ... (初期化部分)
    
    # --- 損失とNMEの記録用のリスト ---
    train_losses = []
    test_losses = []
    train_nmes = [] # 追加
    test_nmes = []  # 追加
    
    # --- 訓練ループ ---
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_nme = 0.0 # 訓練NME用
        count = 0
        
        # for i, (inputs, targets, label_coords, _) in enumerate(tqdm(train_dataloader, ...)):
        for data in tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train"):
            inputs = data[0].to(device)       # 画像 [N, 3, H, W]
            targets = data[1].to(device)      # ヒートマップ [N, 9, H, W]
            label_coords = data[2].to(device) # 座標 [N, 18]
            
            optimizer.zero_grad()
            outputs = model(inputs) # ヒートマップ出力 [N, 9, H, W]
            
            # 損失計算 (ヒートマップ vs. ヒートマップ)
            loss = criterion(outputs, targets) 
            
            # 訓練NME計算 (ヒートマップ出力 vs. 座標)
            train_nme_batch = calculate_nme(outputs, label_coords, device)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_nme += train_nme_batch.item() * inputs.size(0) # NMEを加算
            count += inputs.size(0)

        avg_train_loss = running_loss / count
        avg_train_nme = running_nme / count # 訓練NMEを計算
        train_losses.append(avg_train_loss)
        train_nmes.append(avg_train_nme)
        
        # --- テストデータでの評価 ---
        test_loss, test_nme = evaluate_model_nme(model, test_dataloader, criterion, device)
        test_losses.append(test_loss)
        test_nmes.append(test_nme)

        CSV_PATH = os.path.join(output_dir, 'training_log.csv')
        log_results_to_csv('training_log.csv', epoch + 1, avg_train_loss, test_loss, avg_train_nme, test_nme)
        
        print(f"\n--- Epoch [{epoch+1}/{num_epochs}] 完了 ---")
        print(f"  Train Loss: {avg_train_loss:.6f} | Train NME: {avg_train_nme:.4f}")
        print(f"  Test Loss: {test_loss:.6f} | Test NME: {test_nme:.4f}")

    # --- 最終評価とモデルの保存 ---
    final_test_loss, final_test_nme = evaluate_model_nme(model, test_dataloader, criterion, device)
    print(f"\n✅ Final Test Loss: {final_test_loss:.4f}, Final Test NME: {final_test_nme:.4f}")

    MODEL_FILENAME = 'unet_landmark_regressor_final.pth'
    MODEL_PATH_SAVE = os.path.join(output_dir, MODEL_FILENAME)
    torch.save(model.state_dict(), MODEL_PATH_SAVE)
    print(f"モデルが '{MODEL_PATH_SAVE}' として保存されました。")
    
    # --- 学習曲線のプロット ---
    print("\n--- 学習曲線を表示 ---")
    plot_metrics(train_losses, test_losses, train_nmes, test_nmes, num_epochs)

    # ... (最終評価、保存、プロット)
    return model, test_dataloader, device, train_losses, test_losses, train_nmes, test_nmes


# 評価関数をNMEを返すように修正 (evaluate_model_nmeとして再定義)
def evaluate_model_nme(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_nme = 0
    count = 0

    with torch.no_grad():
        # データローダーから imgs, targets (ヒートマップ), label_coords (座標) を取得
        for data in data_loader:
            imgs, targets, label_coords = data[0].to(device), data[1].to(device), data[2].to(device)
            outputs = model(imgs)

            loss = criterion(outputs, targets)
            total_loss += loss.item() * imgs.size(0)

            # --- NMEの計算と集計 ---
            # label_coords (座標) を使用
            nme_batch = calculate_nme(outputs, label_coords, device)
            total_nme += nme_batch.item() * imgs.size(0)
            count += imgs.size(0)

    avg_loss = total_loss / count
    avg_nme = total_nme / count
    return avg_loss, avg_nme



def log_results_to_csv(filename, epoch, train_loss, test_loss, train_nme, test_nme):
    """エポックごとの結果をCSVファイルに追記する"""
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train_Loss', 'Test_Loss', 'Train_NME', 'Test_NME']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()
            
        writer.writerow({
            'Epoch': epoch,
            'Train_Loss': f"{train_loss:.6f}",
            'Test_Loss': f"{test_loss:.6f}",
            'Train_NME': f"{train_nme:.6f}",
            'Test_NME': f"{test_nme:.6f}",
        })

# プロット関数の修正 (NMEもプロットに含める)
def plot_metrics(train_losses, test_losses, train_nmes, test_nmes, num_epochs):
    epochs = range(1, num_epochs + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Loss Plot
    axes[0].plot(epochs, train_losses, 'b-o', label='Training Loss (MSE)')
    axes[0].plot(epochs, test_losses, 'r-o', label='Testing Loss (MSE)')
    axes[0].set_title('Training and Testing Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. NME Plot
    axes[1].plot(epochs, train_nmes, 'g-o', label='Training NME')
    axes[1].plot(epochs, test_nmes, 'm-o', label='Testing NME')
    axes[1].set_title('Training and Testing NME')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('NME')
    axes[1].legend()
    axes[1].grid(True)
    
    PLOT_PATH = os.path.join(output_dir, 'training_metrics_plot.png')
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close(fig)
    print("✅ 学習曲線 (Loss & NME) を '{PLOT_PATH}' に保存しました。")


def main():
    parser = argparse.ArgumentParser(description="UNet for Landmark Heatmap Regression")
    parser.add_argument('--data_dir', type=str, required=True, help="データセット (.jpg, .pts) のディレクトリパス")
    parser.add_argument('--epochs', type=int, default=30, help="訓練エポック数")
    parser.add_argument('--batch_size', type=int, default=8, help="バッチサイズ")
    parser.add_argument('--lr', type=float, default=1e-4, help="学習率")
    parser.add_argument('--img_size', type=int, default=224, help="画像サイズ (H=W)")
    parser.add_argument('--sigma', type=float, default=3.0, help="ガウシアンヒートマップのシグマ")
    parser.add_argument('--test_size', type=float, default=0.2, help="テストデータの割合")
    parser.add_argument('--output_dir', type=str, default='./run_output', help="全ての出力ファイルを保存するディレクトリパス") # ★ 追加 ★
    args = parser.parse_args()

    #出力ディレクトリを作成
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"出力をディレクトリ：{OUTPUT_DIR}")

    # 1. デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    print(f"使用デバイス: {device}")
    
    # 2. ファイルリストの取得と分割
    all_jpg_files = glob.glob(os.path.join(args.data_dir, "*.jpg"))
    if not all_jpg_files:
        raise FileNotFoundError(f"データセットディレクトリ '{args.data_dir}' 内に.jpgファイルが見つかりません。")
        
    # train/test分割
    train_files, test_files = train_test_split(
        all_jpg_files, test_size=args.test_size, random_state=42
    )
    print(f"全画像ファイル数: {len(all_jpg_files)}. Train: {len(train_files)}, Test: {len(test_files)}")

    # 3. データローダーの準備
    # LandmarkHeatmapDataset はファイルパスリストを受け取るように調整が必要です (LandmarkDatasetのロジックに合わせる)
    # または、root_dirとファイルリストを渡す形にする必要があります。
    
    # ここでは、LandmarkHeatmapDatasetがファイルパスのリストを処理できると仮定します。
    # **注意:** LandmarkHeatmapDatasetの __init__ を調整してください。
    train_dataset = LandmarkHeatmapDataset(
        file_list=train_files, # file_listを渡すよう仮定
        root_dir=args.data_dir, # ptsファイルを探すためのroot_dir
        image_size=args.img_size, 
        sigma=args.sigma
    )
    test_dataset = LandmarkHeatmapDataset(
        file_list=test_files, # file_listを渡すよう仮定
        root_dir=args.data_dir, # ptsファイルを探すためのroot_dir
        image_size=args.img_size, 
        sigma=args.sigma
    )
    
    # DataLoaderの作成
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count() or 1
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count() or 1
    )
    
    # 4. モデル、損失関数、最適化手法の設定
    model = UNet(in_channels=3, out_channels=9).to(device) # 9点のランドマーク
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 5. 訓練の実行
    trained_model, test_loader, device, _, _, _, _ = train_model(
        model, train_dataloader, test_dataloader, criterion, optimizer, device, args.epochs, OUTPUT_DIR
    )
    
    # 6. 予測結果の描画と保存 (ご提示のコードのsave_landmark_predictions関数が必要です)
    # save_landmark_predictions(
    #     model=trained_model, data_loader=test_loader, device=device, num_samples=5, save_dir="./predictions_output"
    # )

if __name__ == '__main__':
    main()
