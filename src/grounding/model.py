import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel

class GroundingModel(nn.Module):
    def __init__(self, text_model_name="distilbert-base-uncased"):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feat_dim = 512

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_feat_dim = self.text_encoder.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(self.image_feat_dim + self.text_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 4),
            nn.Sigmoid()  # keeps output between 0-1, matching our normalized coords
        )

    def forward(self, images, questions, device):
        img_feat = self.image_encoder(images).squeeze(-1).squeeze(-1)

        tokens = self.tokenizer(list(questions), padding=True, truncation=True, return_tensors="pt").to(device)
        text_out = self.text_encoder(**tokens)
        text_feat = text_out.last_hidden_state[:, 0, :]

        combined = torch.cat([img_feat, text_feat], dim=1)
        return self.regressor(combined)

if __name__ == "__main__":
    model = GroundingModel()
    print("Model created successfully")
    print("Total parameters:", sum(p.numel() for p in model.parameters()))