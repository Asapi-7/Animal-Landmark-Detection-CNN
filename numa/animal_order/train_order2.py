import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


DATA_ROOT = "dataset_animal"
classes = ["Carnivora","Crocodylia","Marsupialia","Pinnipedia",
           "Primates","Rodentia","Sphenisciformes","Ungulata"]
transform = transforms.Compose([transforms.ToTensor()])
dataset = datasets.ImageFolder(DATA_ROOT, transform=transform)
folder_names = [d.name for d in os.scandir(DATA_ROOT) if d.is_dir()]
print("フォルダ名:", folder_names)


train_size = int(len(dataset)*0.8)
test_size = len(dataset)-train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset,[train_size,test_size])
train_loader = DataLoader(train_dataset,batch_size=16,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=16,shuffle=False)


# 本体モデル
model_main = models.resnet18(weights=None)
model_main.fc = nn.Linear(model_main.fc.in_features, 8)
# Penguin補助モデル
model_penguin = models.resnet18(weights=None)
model_penguin.fc = nn.Linear(model_penguin.fc.in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_main = model_main.to(device)
model_penguin = model_penguin.to(device)

criterion = nn.CrossEntropyLoss()  # 損失関数
optimizer_main = torch.optim.Adam(model_main.parameters(), lr=1e-4)
optimizer_penguin = torch.optim.Adam(model_penguin.parameters(), lr=1e-4)

log_path = "two_stage_training_log.txt"
if os.path.exists(log_path):
    os.remove(log_path)

num_epochs = 20
threshold = 0.6  # 自信判定の閾値

for epoch in range(num_epochs):
    
    model_main.train()
    train_loss_sum = 0
    train_correct = 0
    train_total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer_main.zero_grad()
        outputs = model_main(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_main.step()

        train_loss_sum += loss.item()
        _, pred = torch.max(outputs, 1)
        train_correct += (pred==labels).sum().item()
        train_total += labels.size(0)

    train_loss = train_loss_sum/len(train_loader)
    train_acc = train_correct/train_total

    # Penguinモデル学習 
    model_penguin.train()
    penguin_loss_sum = 0
    penguin_correct = 0
    penguin_total = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        # 本体モデルで推論
        with torch.no_grad():
            outputs_main = model_main(imgs)
            probs = nn.Softmax(dim=1)(outputs_main)
            max_probs, pred_main = torch.max(probs,1)
        # 閾値未満のサンプルのみ Penguin モデルで学習
        mask = max_probs<threshold
        if mask.sum()==0:
            continue
        imgs_masked = imgs[mask]
        labels_masked = labels[mask]
        # Penguinラベル変換
        penguin_labels = torch.zeros_like(labels_masked)
        penguin_labels[labels_masked==classes.index("Sphenisciformes")] = 1

        optimizer_penguin.zero_grad()
        outputs_penguin = model_penguin(imgs_masked)
        loss_penguin = criterion(outputs_penguin,penguin_labels)
        loss_penguin.backward()
        optimizer_penguin.step()

        penguin_loss_sum += loss_penguin.item()
        _, pred_p = torch.max(outputs_penguin,1)
        penguin_correct += (pred_p==penguin_labels).sum().item()
        penguin_total += penguin_labels.size(0)

    penguin_loss = penguin_loss_sum/len(train_loader)
    penguin_acc = penguin_correct/penguin_total if penguin_total>0 else 0.0

    # テスト評価 
    model_main.eval()
    model_penguin.eval()
    test_loss_sum = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs_main = model_main(imgs)
            probs = nn.Softmax(dim=1)(outputs_main)
            max_probs, pred_main = torch.max(probs,1)
            # threshold判定
            mask = max_probs<threshold
            pred_final = pred_main.clone()
            if mask.sum()>0:
                imgs_masked = imgs[mask]
                outputs_penguin = model_penguin(imgs_masked)
                probs_p = nn.Softmax(dim=1)(outputs_penguin)
                pred_p = torch.argmax(probs_p,1)
                # Penguinなら Sphenisciformes に置き換え
                pred_final[mask] = torch.where(pred_p==1,
                                               torch.tensor(classes.index("Sphenisciformes")).to(device),
                                               pred_main[mask])
            test_correct += (pred_final==labels).sum().item()
            test_total += labels.size(0)
            test_loss_sum += criterion(outputs_main,labels).item()

    test_loss = test_loss_sum/len(test_loader)
    test_acc = test_correct/test_total

    # ログ出力 
    line = f"Epoch [{epoch+1}/{num_epochs}]  Train Loss: {train_loss:.4f}  Train Acc: {train_acc:.4f}  Penguin Loss: {penguin_loss:.4f}  Penguin Acc: {penguin_acc:.4f}  Test Loss: {test_loss:.4f}  Test Acc: {test_acc:.4f}\n"
    print(line)
    with open(log_path,"a") as f:
        f.write(line)

print("訓練完了：ログは two_stage_training_log.txt に保存されました")
