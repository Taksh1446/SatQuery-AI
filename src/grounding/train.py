import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import GroundingDataset
from model import GroundingModel

DATA_DIR = "data/vrsbench"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

def collate_fn(batch):
    images, questions, coords = zip(*batch)
    images = torch.stack(images)
    coords = torch.stack(coords)
    return images, list(questions), coords

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    full_ds = GroundingDataset(
        f"{DATA_DIR}/VRSBench_EVAL_referring.json",
        f"{DATA_DIR}/Images_val/Images_val"
    )

    # split into train/val since this file doesn't have separate splits
    val_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = GroundingModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, questions, coords in train_loader:
            images, coords = images.to(device), coords.to(device)
            optimizer.zero_grad()
            outputs = model(images, questions, device)
            loss = criterion(outputs, coords)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, questions, coords in val_loader:
                images, coords = images.to(device), coords.to(device)
                outputs = model(images, questions, device)
                val_loss += criterion(outputs, coords).item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), "models/grounding_baseline.pth")
    print("Model saved to models/grounding_baseline.pth")

if __name__ == "__main__":
    train()