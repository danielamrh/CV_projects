import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import csv

DATA_PATH  = Path("data/gestures.csv")
MODEL_PATH = Path("models/gesture_classifier.pth")

GESTURES = {
    0: "fist",
    1: "one_finger",
    2: "peace",
    3: "open_hand",
    4: "thumbs_up",
}

NUM_CLASSES  = len(GESTURES)
INPUT_SIZE   = 63    # 21 landmarks × 3 (x, y, z)
HIDDEN_SIZE  = 128
EPOCHS       = 50
BATCH_SIZE   = 32
LEARNING_RATE = 0.001

class GestureDataset(Dataset):
    """Loads landmark CSV into a PyTorch Dataset."""

    def __init__(self, path: Path):
        self.X = []
        self.y = []

        with open(path, newline="") as f:
            for row in csv.reader(f):
                self.y.append(int(row[0]))
                self.X.append([float(v) for v in row[1:]])

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class GestureClassifier(nn.Module):
    """Simple 3-layer MLP for gesture classification."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)

def train():
    dataset = GestureDataset(DATA_PATH)

    # 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = GestureClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Training ──
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()

        # ── Validation ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds    = model(X_batch).argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total   += len(y_batch)

        val_acc = correct / total
        print(f"Epoch {epoch:3d}/{EPOCHS} | Val Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    print(f"\nBest Val Acc: {best_val_acc:.3f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    train()
    