"""
CNN Gesture Classifier Training.

Trains a custom CNN on hand crop images (64x64 px).
Data is loaded from data/<class_name>/ folders.

Usage:
    python src/train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

DATA_DIR   = Path("data")
MODEL_PATH = Path("models/gesture_cnn.pth")

GESTURES = ["fist", "one_finger", "open_hand", "peace", "thumbs_up"]

IMG_SIZE     = 64
NUM_CLASSES  = len(GESTURES)
EPOCHS       = 30
BATCH_SIZE   = 32
LEARNING_RATE = 0.001
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


class GestureCNN(nn.Module):
    """
    Small CNN for 64x64 hand crop classification.

    Architecture:
        Conv(3→32) → ReLU → MaxPool       # 64x64 → 32x32
        Conv(32→64) → ReLU → MaxPool      # 32x32 → 16x16
        Conv(64→128) → ReLU → MaxPool     # 16x16 → 8x8
        Flatten → Linear(8192→256) → ReLU → Dropout → Linear(256→5)
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return train_tf, val_tf

def train():
    print(f"Device: {DEVICE}")

    train_tf, val_tf = get_transforms()

    full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    train_size   = int(0.8 * len(full_dataset))
    val_size     = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = GestureCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(X), y).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                correct += (model(X).argmax(1) == y).sum().item()
                total   += len(y)

        acc = correct / total
        print(f"Epoch {epoch:3d}/{EPOCHS} | Val Acc: {acc:.3f}")

        if acc > best_acc:
            best_acc = acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest Val Acc: {best_acc:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()