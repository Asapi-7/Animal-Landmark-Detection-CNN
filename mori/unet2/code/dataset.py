import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import config
import math
import random
import matplotlib.pyplot as plt

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
    H, W = output_size
    heatmaps = np.zeros((num_keypoints, H, W), dtype=np.float32)

    for i in range(num_keypoints):
        x, y = keypoints[i]

        xx, yy = np.meshgrid(np.arange(W), np.arange(H))

        heatmaps[i] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    
    return heatmaps

class RandomHorizontalFlipWithKeypoints:
    def __init__(self, p=0.5):
        self.p = p
        self.flip_pairs = [(1,4),(2,3),(6,7)]

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']
        if random.random() < self.p:
            W, H = image.size
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            keypoints[:, 0] = W - 1 - keypoints[:, 0]

            new_keypoints = keypoints.copy()
            for i, j in self.flip_pairs:
                new_keypoints[i], new_keypoints[j] = keypoints[j].copy(), keypoints[i].copy()

            keypoints = new_keypoints

        sample['image'] = image
        sample['keypoints'] = keypoints
        return sample

class RandomRotationWithKeypoints:
    def __init__(self, degrees):
        self.degrees = degrees
    
    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        angle = random.uniform(self.degrees[0], self.degrees[1])
        W, H = image.size

        image = image.rotate(angle, resample=Image.BILINEAR, expand=False)

        center_x, center_y = W / 2, H / 2

        theta = -angle * (math.pi / 180)

        new_keypoints = keypoints.copy()
        for i in range(len(keypoints)):
            x, y = keypoints[i]
            x_centered = x - center_x
            y_centered = y - center_y

            x_rotated = x_centered * math.cos(theta) - y_centered * math.sin(theta)
            y_rotated = x_centered * math.sin(theta) + y_centered * math.cos(theta)

            new_keypoints[i, 0] = x_rotated + center_x
            new_keypoints[i, 1] = y_rotated + center_y

        sample['image'] = image
        sample['keypoints'] = new_keypoints

        return sample

class AnimalKeypointDataset(Dataset):
    
    def __init__(self, root_dir, image_transform=None, data_augmentation=None, transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.data_augmentation = data_augmentation
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(self.root_dir, '*.jpg')))
        self.resize_transform = transforms.Resize(config.IMAGE_SIZE)

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
        target_W, target_H = config.IMAGE_SIZE[1], config.IMAGE_SIZE[0]

        scaled_keypoints[:, 0] *= target_W / original_size[0]
        scaled_keypoints[:, 1] *= target_H / original_size[1]

        sample = {'image':image, 'keypoints':scaled_keypoints}

        if self.data_augmentation:
            sample = self.data_augmentation(sample)

        if self.image_transform:
            sample['image'] = self.image_transform(sample['image'])

        heatmaps = generate_heatmaps(sample['keypoints'], config.IMAGE_SIZE, config.SIGMA)

        sample['heatmaps'] = torch.from_numpy(heatmaps)
        sample['keypoints'] = torch.from_numpy(sample['keypoints'])

        return sample

def visualize_and_save_sample(sample, save_path="transformed_sample.png"):
    img_tensor = sample['image']
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    
    img = img_tensor.permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    keypoints = sample['keypoints'].cpu().numpy()
    
    H, W, _ = img.shape
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.scatter(keypoints[:, 0], keypoints[:, 1], c='red', s=15) # X座標が横、Y座標が縦
    
    plt.title(f"Transformed Image with {len(keypoints)} Keypoints")
    plt.xlabel(f"Image Width: {W}")
    plt.ylabel(f"Image Height: {H}")
    plt.axis('off') 
    
    plt.savefig(save_path)
    print(f"\n✅ 変換後の画像が '{save_path}' に保存されました。")
    plt.close()

if __name__ == '__main__':
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    resize_transform = transforms.Resize(config.IMAGE_SIZE)

    data_augmentation_transform = transforms.Compose([
        lambda x: {'image': resize_transform(x['image']), 'keypoints': x['keypoints']},
        RandomHorizontalFlipWithKeypoints(p=0.5),
        RandomRotationWithKeypoints(degrees=(-10, 10))
    ])

    dataset = AnimalKeypointDataset(root_dir=config.ROOT_DIR, image_transform=image_transform, data_augmentation=data_augmentation_transform)

    print(f"Data size: {len(dataset)}")

    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Heatmaps shape: {sample['heatmaps'].shape}")
    print(f"Keypoints shape: {sample['keypoints'].shape}")

    visualize_and_save_sample(sample, save_path="transformed_sample_0.png")

    sample_rand = dataset[random.randint(0, len(dataset) - 1)]
    visualize_and_save_sample(sample_rand, save_path="transformed_sample_rand.png")
