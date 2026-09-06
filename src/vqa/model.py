import torch
import torch.nn as nn
from torchvision import models
from transformers import AutoTokenizer, AutoModel

class VQAModel(nn.Module):
    def __init__(self, vocab_size, text_model_name="distilbert-base-uncased"):
        super().__init__()
        # Image encoder: pretrained ResNet-18, remove its final classification layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_feat_dim = 512

        # Question encoder: tiny pretrained BERT (small, fast, good enough for short questions)
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_feat_dim = self.text_encoder.config.hidden_size

        # Fusion + classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.image_feat_dim + self.text_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, vocab_size)
        )

    def forward(self, images, questions, device):
        img_feat = self.image_encoder(images).squeeze(-1).squeeze(-1)  # [batch, 512]

        tokens = self.tokenizer(list(questions), padding=True, truncation=True, return_tensors="pt").to(device)
        text_out = self.text_encoder(**tokens)
        text_feat = text_out.last_hidden_state[:, 0, :]  # [batch, hidden]

        combined = torch.cat([img_feat, text_feat], dim=1)
        return self.classifier(combined)

if __name__ == "__main__":
    from dataset import build_vocab
    vocab = build_vocab("data/rsvqa_lr/clean_train.json")
    model = VQAModel(vocab_size=len(vocab))
    print("Model created successfully")
    print("Total parameters:", sum(p.numel() for p in model.parameters()))