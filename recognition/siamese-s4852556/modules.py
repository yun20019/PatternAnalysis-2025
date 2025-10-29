import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=256, freeze_backbone=False):
        super(SiameseNetwork, self).__init__()
        
        # EfficientNet-B0 backbone
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1) 
        
        # B0 outputs a 1280-dimensional feature vector
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # freeze backbone parameters during training
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # [batch, 1280]
        x = self.fc(x)
        # ✅ L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2
