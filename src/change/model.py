import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel

class ChangeModel(nn.Module):
    def __init__(self, vocab_size, text_model_name="distilbert-base-uncased"):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feat_dim = 512

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_feat_dim = self.text_encoder.config.hidden_size

        # combine both images' features + question features
        self.classifier = nn.Sequential(
            nn.Linear(self.image_feat_dim * 2 + self.text_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size)
        )

    def forward(self, images1, images2, questions, device):
        feat1 = self.image_encoder(images1).squeeze(-1).squeeze(-1)
        feat2 = self.image_encoder(images2).squeeze(-1).squeeze(-1)

        tokens = self.tokenizer(list(questions), padding=True, truncation=True, return_tensors="pt").to(device)
        text_out = self.text_encoder(**tokens)
        text_feat = text_out.last_hidden_state[:, 0, :]

        combined = torch.cat([feat1, feat2, text_feat], dim=1)
        return self.classifier(combined)

if __name__ == "__main__":
    from dataset import ChangeDataset
    ds = ChangeDataset("data/cdvqa/eval/CDVQA.json")
    model = ChangeModel(vocab_size=len(ds.answer_to_idx))
    print("Model created successfully")
    print("Total parameters:", sum(p.numel() for p in model.parameters()))