# multimodal_resnet50.py
import torch
import torch.nn as nn
from torchvision.models import resnet50

class MultimodalResNet50(nn.Module):
    def __init__(self, clinical_input_dim=3, num_classes=2):
        super().__init__()
        self.image_model = resnet50(pretrained=True)
        self.image_model.fc = nn.Identity()

        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2048 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, clinical):
        img_feat = self.image_model(image)
        clin_feat = self.clinical_fc(clinical)
        combined = torch.cat([img_feat, clin_feat], dim=1)
        return self.classifier(combined)