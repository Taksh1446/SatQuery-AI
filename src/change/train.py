import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import ChangeDataset
from model import ChangeModel

JSON_PATH = "data/cdvqa/eval/eval/CDVQA.json"
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

def collate_fn(batch):
    img1, img2, questions, answers = zip(*batch)
    img1 = torch.stack(img1)
    img2 = torch.stack(img2)
    answers = torch.tensor(answers)
    return img1, img2, list(questions), answers

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    full_ds = ChangeDataset(JSON_PATH)
    vocab_size = len(full_ds.answer_to_idx)

    val_size = int(0.1 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = ChangeModel(vocab_size=vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for img1, img2, questions, answers in train_loader:
            img1, img2, answers = img1.to(device), img2.to(device), answers.to(device)
            optimizer.zero_grad()
            outputs = model(img1, img2, questions, device)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for img1, img2, questions, answers in val_loader:
                img1, img2, answers = img1.to(device), img2.to(device), answers.to(device)
                outputs = model(img1, img2, questions, device)
                preds = outputs.argmax(dim=1)
                correct += (preds == answers).sum().item()
                total += answers.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), "models/change_baseline.pth")
    print("Model saved to models/change_baseline.pth")

if __name__ == "__main__":
    train()