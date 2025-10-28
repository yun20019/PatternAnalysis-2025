import torch
import torch.nn as nn
import torchvision.models as models

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128, freeze_backbone=False):
        super(SiameseNetwork, self).__init__()
        
        # Using resnet50 as a backbone
        backbone = models.resnet50(pretrained=True)
        modules = list(backbone.children())[:-1] 
        self.feature_extractor = nn.Sequential(*modules)
        
        # Fully connected layer to project 2048-d features from ResNet into a lower-dimensional embedding space
        self.fc = nn.Sequential(
        nn.Linear(2048, embedding_dim),
        nn.ReLU(inplace=True)
        )

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward_once(self, x):
        x = self.feature_extractor(x)          # (B, 512, 1, 1)
        x = x.view(x.size(0), -1)              # (B, 512)
        x = self.fc(x)                         # (B, embedding_dim)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

