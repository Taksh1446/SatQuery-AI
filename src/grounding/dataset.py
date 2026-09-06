import json
import re
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATA_DIR = "data/vrsbench"

class GroundingDataset(Dataset):
    def __init__(self, json_path, image_folder):
        with open(json_path) as f:
            self.data = json.load(f)
        self.image_folder = image_folder
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = f"{self.image_folder}/{item['image_id']}"
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        coords = [int(x) for x in re.findall(r'\d+', item["ground_truth"])]
        coords = torch.tensor(coords, dtype=torch.float32) / 100.0  # normalize 0-1

        return image, item["question"], coords

if __name__ == "__main__":
    ds = GroundingDataset(
        f"{DATA_DIR}/VRSBench_EVAL_referring.json",
        f"{DATA_DIR}/Images_val/Images_val"
    )
    print("Dataset size:", len(ds))
    image, question, coords = ds[0]
    print("Image shape:", image.shape)
    print("Question:", question)
    print("Coords (0-1 scale):", coords)