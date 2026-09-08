import os
os.makedirs("models", exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FusionDataset, LABELS
from model import FusionModel

BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4
TRAIN_LIMIT = 5000
VAL_LIMIT = 500

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_ds = FusionDataset(split="train", limit=TRAIN_LIMIT)
    val_ds = FusionDataset(split="test", limit=VAL_LIMIT)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    model = FusionModel(num_classes=len(LABELS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        for batch_idx, (sar, optical, labels) in enumerate(train_loader):
            sar, optical, labels = sar.to(device), optical.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(sar, optical)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            if batch_idx % 20 == 0:
                print(f"  Epoch {epoch+1} | Batch {batch_idx} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / batch_count

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for sar, optical, labels in val_loader:
                sar, optical, labels = sar.to(device), optical.to(device), labels.to(device)
                outputs = model(sar, optical)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), "models/fusion_baseline.pth")
    print("Model saved to models/fusion_baseline.pth")

if __name__ == "__main__":
    train()