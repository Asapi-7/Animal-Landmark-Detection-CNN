import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from tqdm import tqdm
import os

import config
from dataset import AnimalKeypointDataset
from model import UNet

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """Trains the model for one epoch."""
    model.train()
    running_loss = 0.0
    
    # Use tqdm for a progress bar
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    
    for i, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        heatmaps_true = batch['heatmaps'].to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        heatmaps_pred = model(images)
        
        # Calculate loss
        loss = criterion(heatmaps_pred, heatmaps_true)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Update progress bar
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    epoch_loss = running_loss / len(dataloader)
    return epoch_loss

def main():
    """Main training loop."""
    print(f"Using device: {config.DEVICE}")

    # Create checkpoint directory if it doesn't exist
    if not os.path.exists(config.CHECKPOINT_DIR):
        os.makedirs(config.CHECKPOINT_DIR)
        print(f"Created directory: {config.CHECKPOINT_DIR}")

    # Define transforms
    transform = Compose([
        Resize(config.IMAGE_SIZE),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    # Note: For a real project, you should split this into train and validation sets.
    dataset = AnimalKeypointDataset(root_dir=config.ROOT_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4)
    
    print(f"Dataset loaded with {len(dataset)} images.")

    # Initialize model, criterion, and optimizer
    model = UNet(
        in_channels=3,
        out_channels=config.NUM_KEYPOINTS
    ).to(config.DEVICE)
    
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    print("Starting training...")

    for epoch in range(config.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{config.EPOCHS} ---")
        
        epoch_loss = train_one_epoch(model, dataloader, criterion, optimizer, config.DEVICE)
        
        print(f"Epoch {epoch + 1} finished. Average Loss: {epoch_loss:.4f}")
        
        # Save the model checkpoint
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f'unet_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    print("\nTraining finished!")

if __name__ == '__main__':
    main()
