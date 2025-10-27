import torch
from torch import nn
import torchvision.models as models
import timm

class MultiHeadResNet(nn.Module):
    def __init__(self, backbone_name="resnet50", pretrained=False, num_crops=5, num_stages=3, num_severity=4):
        super().__init__()
        if backbone_name.startswith("resnet"):
            model = getattr(models, backbone_name)(pretrained=pretrained)
            in_features = model.fc.in_features
            model.fc = nn.Identity()
            self.backbone = model
            feat_dim = in_features
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool='avg')
            feat_dim = self.backbone.num_features

        embed_dim = 1024
        self.project = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        self.crop_head = nn.Linear(embed_dim, num_crops)
        self.damaged_head = nn.Linear(embed_dim, 1)
        self.growth_head = nn.Linear(embed_dim, num_stages)
        self.severity_head = nn.Linear(embed_dim, num_severity)
        self.health_head = nn.Linear(embed_dim, 1)

    def forward(self, x):
        feats = self.backbone(x)
        if feats.ndim == 4:
            feats = feats.flatten(1)
        proj = self.project(feats)
        return {
            "crop": self.crop_head(proj),
            "is_damaged": self.damaged_head(proj).squeeze(1),
            "growth": self.growth_head(proj),
            "severity": self.severity_head(proj),
            "health": self.health_head(proj).squeeze(1),
        }
