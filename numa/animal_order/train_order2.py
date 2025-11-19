import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# GPU 情報表示
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name())
else:
    print("→ 現在 CPU で実行中です")

# データセット準備
DATA_ROOT = "dataset_animal"
classes = ["Carnivora","Crocodylia","Marsupialia","Pinnipedia",
           "Primates","Rodentia","Sphenisciformes","Ungulata"]

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
folder_names = [d.name for d in os.scandir(DATA_ROOT) if d.is_dir()]
print("フォルダ名:", folder_names)

train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size

train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 高速化：num_workers=4, pin_memory=True
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False,
                         num_workers=4, pin_memory=True)

# モデル構築
model_main = models.resnet18(weights=None)
model_main.fc = nn.Linear(model_main.fc.in_features, 8)

model_penguin = models.resnet18(weights=None)
model_penguin.fc = nn.Linear(model_penguin.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_main = model_main.to(device)
model_penguin = model_penguin.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_main = torch.optim.Adam(model_main.parameters(), lr=1e-4)
optimizer_penguin = torch.optim.Adam(model_penguin.parameters(), lr=1e-4)

softmax = nn.Softmax(dim=1)  # 高速化：毎回作らない

log_path = "two_stage_training_log.txt"
if os.path.exists(log_path):
    os.remove(log_path)

num_epochs = 20
threshold = 0.6

# トレーニング開始
for epoch in range(num_epochs):

    print(f"\n===== Epoch {epoch+1}/{num_epochs} =====")
    print("Using device:", device)

    # メインモデル学習
    print("Main model training:")
    model_main.train()
    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    for imgs, labels in tqdm(train_loader, desc="Main Train"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer_main.zero_grad()
        outputs = model_main(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_main.step()

        train_loss_sum += loss.item()
        _, pred = torch.max(outputs, 1)
        train_correct += (pred == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / len(train_loader)
    train_acc = train_correct / train_total

    # Penguinモデル学習
    print("Penguin model training:")
    model_penguin.train()
    penguin_loss_sum = 0
    penguin_correct = 0
    penguin_total = 0

    for imgs, labels in tqdm(train_loader, desc="Penguin Train"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.no_grad():
            outputs_main = model_main(imgs)
            probs = softmax(outputs_main)
            max_probs, pred_main = torch.max(probs, 1)

        mask = max_probs < threshold
        if mask.sum() == 0:
            continue

        imgs_masked = imgs[mask]
        labels_masked = labels[mask]

        penguin_labels = torch.zeros_like(labels_masked)
        penguin_labels[labels_masked == classes.index("Sphenisciformes")] = 1

        optimizer_penguin.zero_grad()
        outputs_penguin = model_penguin(imgs_masked)
        loss_penguin = criterion(outputs_penguin, penguin_labels)
        loss_penguin.backward()
        optimizer_penguin.step()

        penguin_loss_sum += loss_penguin.item()
        _, pred_p = torch.max(outputs_penguin, 1)
        penguin_correct += (pred_p == penguin_labels).sum().item()
        penguin_total += penguin_labels.size(0)

    penguin_loss = penguin_loss_sum / len(train_loader)
    penguin_acc = penguin_correct / penguin_total if penguin_total > 0 else 0.0

    # 評価
    print("Testing:")
    model_main.eval()
    model_penguin.eval()

    test_loss_sum = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Testing"):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs_main = model_main(imgs)
            probs = softmax(outputs_main)
            max_probs, pred_main = torch.max(probs, 1)

            pred_final = pred_main.clone()

            mask = max_probs < threshold
            if mask.sum() > 0:
                imgs_masked = imgs[mask]
                outputs_penguin = model_penguin(imgs_masked)
                pred_p = torch.argmax(softmax(outputs_penguin), 1)

                sp_idx = classes.index("Sphenisciformes")
                pred_final[mask] = torch.where(
                    pred_p == 1,
                    torch.tensor(sp_idx).to(device),
                    pred_main[mask]
                )

            test_correct += (pred_final == labels).sum().item()
            test_total += labels.size(0)
            test_loss_sum += criterion(outputs_main, labels).item()

    test_loss = test_loss_sum / len(test_loader)
    test_acc = test_correct / test_total

    # ログ
    line = (
        f"Epoch [{epoch+1}/{num_epochs}]  "
        f"Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  "
        f"Penguin Loss: {penguin_loss:.4f}  Penguin Acc: {penguin_acc:.4f}  "
        f"Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}\n"
    )

    print(line)
    with open(log_path, "a") as f:
        f.write(line)

print("訓練完了：ログは two_stage_training_log.txt に保存されました")
