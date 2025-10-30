from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data import AnimalDataset
from model import HourglassNet
import os
from torch.utils.data import random_split

# --- ハイパーパラメータと設定 ---
EPOCHS = 50
BATCH_SIZE = 16 # メモリに応じて調整してください
LEARNING_RATE = 0.001
IMAGE_SIZE = 256
HEATMAP_SIZE = 64
NUM_LANDMARKS = 9 # 9点のランドマーク
NUM_STACKS = 2    # Hourglassのスタック数
FEATURE_CHANNELS = 256

# --- パス設定 ---
# Dockerコンテナ内からアクセスするパスを想定
# 必要に応じてホストの絶対パスに変更してください
DATASET_PATH = './cropped_dataset/'
SAVE_DIR = './model/'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'animal_hourglass_model.pth')

def main():
    # --- デバイス設定 ---
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        # GPUが使える場合は、何番目のGPUか、その名前を表示（0番目のGPUを想定）
        print(f'Using device: {device} ({torch.cuda.get_device_name(0)})')
    else:
        print(f'Using device: {device}')

    # --- データセットとデータローダー ---
    print("Loading dataset...")
    full_dataset = AnimalDataset(root_dir=DATASET_PATH, image_size=IMAGE_SIZE, heatmap_size=HEATMAP_SIZE)

    #データ分割処理
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Total:{len(full_dataset)}, Training: {train_size}, Validation: {val_size}")

    
    # --- モデルの初期化 ---
    print("Initializing model...")
    model = HourglassNet(nstack=NUM_STACKS, nclasses=NUM_LANDMARKS, nfeats=FEATURE_CHANNELS).to(device)

    # --- 損失関数とオプティマイザ ---
    # ヒートマップの回帰なのでMSELossを使用
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 訓練ループ ---
    print("Starting training...")
    model.train()
    for epoch in range(EPOCHS):
        epoch_loss = 0.0

        #tqdm()でtrain_laoderをラップし、説明ラベルを付ける
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} (Train)', leave=False)

        for i, batch in enumerate(train_loop):
            images = batch['image'].to(device)
            heatmaps_gt = batch['heatmaps'].to(device)

            optimizer.zero_grad()

            # モデルの出力はスタックごとのヒートマップのリスト
            outputs = model(images)
            
            # 各スタックの出力に対して損失を計算し、合計する
            total_loss = 0
            for out in outputs:
                total_loss += criterion(out, heatmaps_gt)
            
            total_loss.backward()
            optimizer.step()

            current_loss = total_loss.item()
            epoch_loss += current_loss

            train_loop.set_postfix(loss=f'{current_loss:.6f}')

            #if (i + 1) % 10 == 0:
            #    print(f'Epoch [{epoch+1}/{EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {total_loss.item():.6f}')

        avg_epoch_loss = epoch_loss / len(train_loader)
        tqdm.write(f'--- Epoch [{epoch+1}/{EPOCHS}] Average Loss: {avg_epoch_loss:.6f} ---')

    # --- モデルの保存 ---
    print(f"Training finished. Saving model to {MODEL_SAVE_PATH}")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    #検証ロジックの追加
    print("\nStarting Validation...")

    #1. モデルを評価モードに切り替える
    #DropoutaやBatchNorm等の動作を評価用に変更
    model.eval()
    val_loss = 0.0

    #2. 勾配計算を無効にする
    #メモリを節約し、計算を高速化するため
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            heatmaps_gt = batch['heatmaps'].to(device)

            outputs = model(images)

            total_loss = 0
            for out in outputs:
                total_loss += criterion(out, heatmaps_gt)

            val_loss += total_loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f'*** Validation Finished *** Average Validation Loss: {avg_val_loss:.6f}')

    #訓練モードに戻す
    model.train()

if __name__ == '__main__':
    main()
