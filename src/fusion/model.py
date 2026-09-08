import torch
import torch.nn as nn
from torchvision import models

class FusionModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Two separate ResNet encoders - one for SAR, one for optical
        resnet_sar = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.sar_encoder = nn.Sequential(*list(resnet_sar.children())[:-1])

        resnet_optical = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.optical_encoder = nn.Sequential(*list(resnet_optical.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, sar_img, optical_img):
        sar_feat = self.sar_encoder(sar_img).squeeze(-1).squeeze(-1)
        optical_feat = self.optical_encoder(optical_img).squeeze(-1).squeeze(-1)
        combined = torch.cat([sar_feat, optical_feat], dim=1)
        return self.classifier(combined)

if __name__ == "__main__":
    model = FusionModel(num_classes=4)
    print("Model created successfully")
    print("Total parameters:", sum(p.numel() for p in model.parameters()))