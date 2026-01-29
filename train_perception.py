import os
import random
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import csv
import numpy as np

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DATA_DIR = "perception_data"
MODEL_DIR = "models"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "perception_net_best.pth")
NUM_CLASSES = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_CLASS_WEIGHTS = True
PRINT_EVERY = 1

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)

class PerceptionDataset(Dataset):
    def __init__(self, data_dir=DATA_DIR, transform=None):
        self.data_dir = data_dir
        labels_file = os.path.join(data_dir, "labels.csv")
        if not os.path.isfile(labels_file):
            raise FileNotFoundError(f"labels.csv not found in {data_dir}. Run generator first.")
        self.rows = []
        with open(labels_file, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.rows.append((r["filename"], int(r["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        fn, label = self.rows[idx]
        img_path = os.path.join(self.data_dir, "images", fn)
        img = Image.open(img_path).convert("L")  # grayscale
        if self.transform:
            img = self.transform(img)
        else:
            t = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
            img = t(img)
        return img, torch.tensor(label, dtype=torch.long)

class PerceptionNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=NUM_CLASSES):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def compute_class_weights(labels_list, num_classes=NUM_CLASSES):
    cnt = Counter(labels_list)
    weights = []
    for i in range(num_classes):
        if cnt[i] > 0:
            weights.append(1.0 / cnt[i])
        else:
            weights.append(0.0)
    s = sum(weights)
    if s > 0:
        weights = [w / s * num_classes for w in weights]
    return torch.tensor(weights, dtype=torch.float32)


def confusion_counts(y_true, y_pred, num_classes=NUM_CLASSES):
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm


def train_model(data_dir=DATA_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, device=DEVICE):
    train_transform = transforms.Compose([
        transforms.RandomRotation(12),
        transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
        transforms.RandomResizedCrop(16, scale=(0.9, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    full_ds = PerceptionDataset(data_dir=data_dir, transform=None)
    total = len(full_ds)
    if total == 0:
        raise RuntimeError("Dataset is empty. Generate perception_data first.")

    labels_list = [lab for _, lab in full_ds.rows]
    print("Dataset class counts:", Counter(labels_list))

    n_train = int(0.8 * total)
    n_val = total - n_train
    indices = list(range(total))
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    class IndexedDataset(Dataset):
        def __init__(self, base_ds, indices, transform):
            self.base = base_ds
            self.indices = indices
            self.transform = transform
        def __len__(self):
            return len(self.indices)
        def __getitem__(self, i):
            fn, label = self.base.rows[self.indices[i]]
            img = Image.open(os.path.join(self.base.data_dir, "images", fn)).convert("L")
            if self.transform:
                img = self.transform(img)
            else:
                img = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])(img)
            return img, torch.tensor(label, dtype=torch.long)

    train_ds = IndexedDataset(full_ds, train_idx, train_transform)
    val_ds = IndexedDataset(full_ds, val_idx, val_transform)

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    if USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(labels_list, NUM_CLASSES).to(device)
        print("Using class weights:", class_weights.cpu().numpy())
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    else:
        loss_fn = nn.CrossEntropyLoss()

    model = PerceptionNet(in_ch=1, num_classes=NUM_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)

    os.makedirs(MODEL_DIR, exist_ok=True)

    best_val_acc = 0.0
    best_epoch = -1

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        total_tr = 0
        correct_tr = 0

        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=-1)
            correct_tr += (preds == yb).sum().item()
            total_tr += xb.size(0)

        train_loss = running_loss / total_tr if total_tr > 0 else float("nan")
        train_acc = correct_tr / total_tr if total_tr > 0 else 0.0

        model.eval()
        total_va = 0
        correct_va = 0
        all_true = []
        all_pred = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                preds = logits.argmax(dim=-1)
                correct_va += (preds == yb).sum().item()
                total_va += xb.size(0)
                all_true.extend(yb.cpu().numpy().tolist())
                all_pred.extend(preds.cpu().numpy().tolist())

        val_acc = correct_va / total_va if total_va > 0 else 0.0
        cm = confusion_counts(all_true, all_pred, NUM_CLASSES)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            torch.save(model.state_dict(), BEST_MODEL_PATH)

        scheduler.step()

        if ep % PRINT_EVERY == 0:
            print(f"Epoch {ep}/{epochs} train_loss={train_loss:.4f} train_acc={train_acc:.3f} val_acc={val_acc:.3f}")
            print(" Confusion counts (gt -> pred):")
            for gt in range(NUM_CLASSES):
                print(f"  GT {gt} ->", cm[gt])

    print(f"\nTraining finished. Best val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    print(f"Best model saved to: {BEST_MODEL_PATH}")
    return model


if __name__ == "__main__":
    model = train_model()
