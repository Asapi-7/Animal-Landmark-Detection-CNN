import torch

ROOT_DIR = '../../unet/cropped_dataset/'

IMAGE_SIZE = (256, 256)

NUM_KEYPOINTS = 9

SIGMA = 5

#DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'

EPOCHS = 30

BATCH_SIZE = 16

LEARNING_RATE = 1e-4

CHECKPOINT_DIR = 'checkpoints/'