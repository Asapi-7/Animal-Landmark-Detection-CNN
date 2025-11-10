import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import config

def parse_pts_file(pts_path):
    with open(pts_path, 'r') as f:
        lines = f.readlines()

    points_str = lines[3:-1]
    keypoints = []
    for point in points_str:
        x, y = point.strip().split()
        keypoints.append([float(x), float(y)])
    return np.array(keypoints, dtype=np.float32)

def generate_heatmaps(keypoints, output_size, sigma):
    num_keypoints = len(keypoints)
    heatmaps = np.zeros((num_keypoints, output_size[0], output_size[1]), dtype=np.float32)

    for i in range(num_keypoints):
        x, y = keypoints[i]

        xx, yy = np.meshgrid(np.arange(output_size[1]), np.arange(output_size[0]))

        heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    
    return heatmaps

class AnimalKeypointDataset(Dataset):
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, '*.jpg')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        pts_path = os.path.splitext(img_path)[0] + '.pts'

        image = Image.open(img_path).convert('RGB')
        original_size = image.size

        keypoints = parse_pts_file(pts_path)

        scaled_keypoints = keypoints.copy()
        scaled_keypoints[:, 0] *= config.IMAGE_SIZE[1] / original_size[0]
        scaled_keypoints[:, 1] *= config.IMAGE_SIZE[0] / original_size[1]

        heatmaps = generate_heatmaps(scaled_keypoints, config.IMAGE_SIZE, config.SIGMA)

        sample = {'image':image, 'label_coords':keypoints, 'heatmaps':heatmaps}

        if self.transform:
            sample['image'] = self.transform(sample['image'])
            sample['heatmaps'] = torch.from_numpy(sample['heatmaps'])
            sample['label_coords'] = torch.from_numpy(sample['label_coords'])

        return sample


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = AnimalKeyPointDataset(root_dir=config.ROOT_DIR, transform=transform)

    print(f"Data size: {len(dataset)}")

    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")