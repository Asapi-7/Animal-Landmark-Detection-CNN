import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


DATA_ROOT = "dataset_animal"

classes = [
    "Carnivora", "Crocodylia", "Marsupialia", "Pinnipedia",
    "Primates", "Rodentia", "Sphenisciformes", "Ungulata"
]

transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
folder_names = [d.name for d in os.scandir(DATA_ROOT) if d.is_dir()]
print("フォルダ名:", folder_names)

# Train/Test 分割
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# モデル構築
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ログ保存
log_path = "training_log.txt"
if os.path.exists(log_path):
    os.remove(log_path)


num_epochs = 20

for epoch in range(num_epochs):
    # Train
    model.train()
    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item()
        _, pred = torch.max(outputs, 1)
        train_correct += (pred == labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum / len(train_loader)
    train_acc = train_correct / train_total

    # Test
    model.eval()
    test_loss_sum = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            test_loss_sum += loss.item()
            _, pred = torch.max(outputs, 1)
            test_correct += (pred == labels).sum().item()
            test_total += labels.size(0)

    test_loss = test_loss_sum / len(test_loader)
    test_acc = test_correct / test_total

    # ログ出力
    line = f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}\n"
    print(line)

    with open(log_path, "a") as f:
        f.write(line)

print("訓練完了：ログは training_log.txt に保存されました")
