import json
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = "data/cdvqa/eval"
IMAGE_DIR = f"{DATA_DIR}/External_images/CDVQA"

class ChangeDataset(Dataset):
    def __init__(self, json_path):
        with open(json_path) as f:
            raw = json.load(f)

        self.data = []
        for item in raw:
            # question is embedded in the conversation text after "<video>\n"
            question_text = item["conversations"][0]["value"]
            question = question_text.split("<video>\n")[-1].strip()
            answer = item["conversations"][1]["value"].strip()

            img1_name = item["video"][0].split("/")[-1]
            img2_name = item["video"][1].split("/")[-1]

            self.data.append({
                "image1": img1_name,
                "image2": img2_name,
                "question": question,
                "answer": answer
            })

        self.answer_to_idx = self._build_vocab()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def _build_vocab(self):
        unique_answers = sorted(set(d["answer"] for d in self.data))
        return {ans: i for i, ans in enumerate(unique_answers)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img1 = Image.open(f"{IMAGE_DIR}/{item['image1']}").convert("RGB")
        img2 = Image.open(f"{IMAGE_DIR}/{item['image2']}").convert("RGB")
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        answer_idx = self.answer_to_idx[item["answer"]]
        return img1, img2, item["question"], answer_idx

if __name__ == "__main__":
    ds = ChangeDataset(f"{DATA_DIR}/CDVQA.json")
    print("Dataset size:", len(ds))
    print("Vocabulary size:", len(ds.answer_to_idx))
    print("Sample answers:", list(ds.answer_to_idx.items())[:10])
    img1, img2, question, answer_idx = ds[0]
    print("Image1 shape:", img1.shape)
    print("Image2 shape:", img2.shape)
    print("Question:", question)
    print("Answer index:", answer_idx)