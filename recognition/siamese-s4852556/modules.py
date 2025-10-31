import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class SiameseNetwork(nn.Module):
    """
    Siamese Network with EfficientNet-B0 backbone.
    """

    def __init__(self, embedding_dim=512, freeze_backbone=False, dropout=0.4):
        super(SiameseNetwork, self).__init__()   
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1) 
   
        # EfficientNet-B0 → 1280D features
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

   
    def forward_once(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # flatten [batch, 1280]
        x = self.fc(x)
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x

    # Pairwise forward 
    def forward(self, x1, x2):
        emb1 = self.forward_once(x1)
        emb2 = self.forward_once(x2)
        return emb1, emb2

    # Triplet forward 
    def forward_triplet(self, anchor, positive, negative):
        ea = self.forward_once(anchor)
        ep = self.forward_once(positive)
        en = self.forward_once(negative)
        return ea, ep, en
