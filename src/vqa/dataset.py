import json
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = "data/rsvqa_lr"
IMAGE_DIR = f"{DATA_DIR}/Images_LR/Images_LR"

def build_vocab(train_json_path):
    with open(train_json_path) as f:
        data = json.load(f)
    unique_answers = sorted(set(d["answer"] for d in data))
    answer_to_idx = {ans: i for i, ans in enumerate(unique_answers)}
    return answer_to_idx

class RSVQADataset(Dataset):
    def __init__(self, json_path, answer_to_idx):
        with open(json_path) as f:
            self.data = json.load(f)
        self.answer_to_idx = answer_to_idx
        # keep only examples whose answer is in our vocab
        self.data = [d for d in self.data if d["answer"] in answer_to_idx]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = f"{IMAGE_DIR}/{item['image_id']}.tif"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        answer_idx = self.answer_to_idx[item["answer"]]
        return image, item["question"], answer_idx

if __name__ == "__main__":
    vocab = build_vocab(f"{DATA_DIR}/clean_train.json")
    print(f"Vocabulary size: {len(vocab)}")
    print("Sample answers:", list(vocab.items())[:10])

    dataset = RSVQADataset(f"{DATA_DIR}/clean_train.json", vocab)
    print(f"Dataset size: {len(dataset)}")
    image, question, answer_idx = dataset[0]
    print("Image tensor shape:", image.shape)
    print("Question:", question)
    print("Answer index:", answer_idx)