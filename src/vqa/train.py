import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import RSVQADataset, build_vocab
from model import VQAModel

DATA_DIR = "data/rsvqa_lr"
BATCH_SIZE = 32
EPOCHS = 3
LR = 1e-4

def collate_fn(batch):
    images, questions, answers = zip(*batch)
    images = torch.stack(images)
    answers = torch.tensor(answers)
    return images, list(questions), answers

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    vocab = build_vocab(f"{DATA_DIR}/clean_train.json")
    train_ds = RSVQADataset(f"{DATA_DIR}/clean_train.json", vocab)
    val_ds = RSVQADataset(f"{DATA_DIR}/clean_val.json", vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = VQAModel(vocab_size=len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for images, questions, answers in train_loader:
            images, answers = images.to(device), answers.to(device)
            optimizer.zero_grad()
            outputs = model(images, questions, device)
            loss = criterion(outputs, answers)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Quick validation accuracy check
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, questions, answers in val_loader:
                images, answers = images.to(device), answers.to(device)
                outputs = model(images, questions, device)
                preds = outputs.argmax(dim=1)
                correct += (preds == answers).sum().item()
                total += answers.size(0)

        val_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_loss:.4f} | Val Accuracy: {val_acc:.4f}")

    torch.save(model.state_dict(), "models/vqa_baseline.pth")
    print("Model saved to models/vqa_baseline.pth")

if __name__ == "__main__":
    train()