import numpy as np
from datasets import load_dataset
import torch
from torch.utils.data import IterableDataset
from torchvision import transforms

LABELS = ["water", "built_up", "vegetation", "other"]

def compute_pseudo_label(pil_image):
    """Simple RGB heuristic to generate a weak label from the optical image."""
    img = np.array(pil_image.convert("RGB").resize((64, 64))).astype(float) / 255.0
    r, g, b = img[:, :, 0].mean(), img[:, :, 1].mean(), img[:, :, 2].mean()
    brightness = (r + g + b) / 3

    if b > r and b > g and brightness < 0.4:
        return LABELS.index("water")
    elif g > r and g > b:
        return LABELS.index("vegetation")
    elif brightness > 0.5 and abs(r - g) < 0.1 and abs(g - b) < 0.1:
        return LABELS.index("built_up")
    else:
        return LABELS.index("other")

class FusionDataset(IterableDataset):
    def __init__(self, split="train", limit=5000):
        self.split = split
        self.limit = limit
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __iter__(self):
        ds = load_dataset("mespinosami/sen12mscr", split=self.split, streaming=True)
        for i, example in enumerate(ds):
            if i >= self.limit:
                break
            s1_img = example["s1"].convert("RGB")
            s2_img = example["s2"].convert("RGB")
            label = compute_pseudo_label(s2_img)

            s1_tensor = self.transform(s1_img)
            s2_tensor = self.transform(s2_img)
            yield s1_tensor, s2_tensor, label

if __name__ == "__main__":
    ds = FusionDataset(limit=5)
    for s1, s2, label in ds:
        print("S1 shape:", s1.shape, "S2 shape:", s2.shape, "Label:", LABELS[label])