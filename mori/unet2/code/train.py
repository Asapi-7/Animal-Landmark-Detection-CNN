import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import os
import numpy as np

import config
from dataset import AnimalKeypointDataset
from model import UNet
import pandas as pd

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


def calculate_nme(outputs, labels, device):
    num_landmarks = config.NUM_KEYPOINTS
    outputs_reshaped = outputs.reshape(-1, num_landmarks, 2)
    labels_reshaped = labels.reshape(-1, num_landmarks, 2)
    distances = torch.linalg.norm(outputs_reshaped - labels_reshaped, dim=2) # [N, 9]
    normalization_factors = calculate_normalization_factor(labels).to(device) # [N]
    normalized_distances = distances / normalization_factors.unsqueeze(1) # [N, 9]
    nme = normalized_distances.mean()
    return nme

def decode_heatmaps_to_coordinates(heatmaps, image_size):
    batch_size, num_keypoints, H, W = heatmaps.shape
    flat_heatmaps = heatmaps.view(batch_size, num_keypoints, -1)
    max_indices = torch.argmax(flat_heatmaps, dim=2) 
    y_coords = max_indices // W
    x_coords = max_indices % W
    x_coords = (x_coords.float() + 0.5)
    y_coords = (y_coords.float() + 0.5)
    coordinates = torch.stack((x_coords, y_coords), dim=2) # [N, C, 2]
    return coordinates.view(batch_size, num_keypoints * 2)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_nme = 0.0
    
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for i, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        heatmaps_true = batch['heatmaps'].to(device)
        keypoints_true = batch['keypoints'].to(device)
        
        optimizer.zero_grad()
        heatmaps_pred = model(images)
        loss = criterion(heatmaps_pred, heatmaps_true)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)

        with torch.no_grad():
            keypoints_pred = decode_heatmaps_to_coordinates(heatmaps_pred, config.IMAGE_SIZE)
            batch_nme = calculate_nme(keypoints_pred, keypoints_true, device)
            running_nme += batch_nme.item() * images.size(0)
        
        progress_bar.set_postfix(loss=f"{loss.item():.6f}", nme=f"{batch_nme.item():.4f}")
        
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_nme = running_nme / len(dataloader.dataset)

    return epoch_loss, epoch_nme

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_nme = 0.0
    
    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for batch in progress_bar:
            images = batch['image'].to(device)
            heatmaps_true = batch['heatmaps'].to(device)
            keypoints_true = batch['keypoints'].to(device)
            
            heatmaps_pred = model(images)
            loss = criterion(heatmaps_pred, heatmaps_true)

            running_loss += loss.item() * images.size(0)

            keypoints_pred = decode_heatmaps_to_coordinates(heatmaps_pred, config.IMAGE_SIZE)
            batch_nme = calculate_nme(keypoints_pred, keypoints_true, device)
            running_nme += batch_nme.item() * images.size(0)
            
    avg_loss = running_loss / len(dataloader.dataset) # データセット全体で割る
    avg_nme = running_nme / len(dataloader.dataset)
    return avg_loss, avg_nme


def main():
    print(f"Using device: {config.DEVICE}")

    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        print(f"Created directory: {config.CHECKPOINT_DIR}")

    OUTPUT_DIR = 'output'
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    resize_transform = Resize(config.IMAGE_SIZE)

    final_image_transform = Compose([
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    from dataset import RandomHorizontalFlipWithKeypoints, RandomRotationWithKeypoints, AnimalKeypointDataset

    train_augmentation_pipeline = Compose([
        lambda x: {'image': resize_transform(x['image']), 'keypoints': x['keypoints']},
        RandomHorizontalFlipWithKeypoints(p=0.5),
        RandomRotationWithKeypoints(degrees=(-10, 10))
    ])

    temp_dataset = AnimalKeypointDataset(root_dir=config.ROOT_DIR)

    full_indices = list(range(len(temp_dataset)))
    train_size = int(0.8 * len(temp_dataset))

    train_indices, test_indices = random_split(full_indices, [train_size, len(temp_dataset) - train_size])

    train_full_dataset = AnimalKeypointDataset(
        root_dir=config.ROOT_DIR,
        image_transform=final_image_transform,
        data_augmentation=train_augmentation_pipeline
    )

    test_full_dataset = AnimalKeypointDataset(
        root_dir=config.ROOT_DIR,
        image_transform=final_image_transform,
        data_augmentation=None
    )

    train_dataset = torch.utils.data.Subset(train_full_dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_full_dataset, test_indices)

    train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,shuffle=False, num_workers=4)

    print(f"Full Dataset loaded with {len(temp_dataset)} images.")
    print(f"Train Dataset size: {len(train_dataset)}")
    print(f"Test Dataset size: {len(test_dataset)}")

    model = UNet(
        in_channels=3,
        out_channels=config.NUM_KEYPOINTS
    ).to(config.DEVICE)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Starting training...")

    best_test_loss = float('inf')

    history = []

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        train_loss, train_nme = train_one_epoch(model, train_dataloader, criterion, optimizer, config.DEVICE)
        test_loss, test_nme = evaluate(model, test_dataloader, criterion, config.DEVICE)
        
        print(f"Epoch {epoch + 1} finished")
        print(f"-> Train Loss: {train_loss:.6f} | Train NME: {train_nme:.4f}")
        print(f"-> Test Loss: {test_loss:.6f} | Test NME: {test_nme:.4f}")

        history.append({
            'epoch' : epoch + 1, 
            'train_loss' : train_loss,
            'test_loss' : test_loss,
            'train_nme' : train_nme,
            'test_nme' : test_nme
        })

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'unet_best_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved! Best Test Loss: {best_test_loss:.6f}")
        
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'unet_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    df_history = pd.DataFrame(history)
    csv_path = os.path.join(OUTPUT_DIR, 'training_history.csv')
    df_history.to_csv(csv_path, index=False)
    print(f"\nTraining history saved to {csv_path}")

    print("\nTraining finished!")

if __name__ == '__main__':
    main()
