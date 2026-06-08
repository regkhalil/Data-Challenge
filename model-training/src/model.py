import torch
import torch.nn as nn
import timm


class OcclusionModel(nn.Module):
    def __init__(self, backbone: str = "efficientnet_b0", pretrained: bool = True,
                 dropout: float = 0.3, head_dims: list = None):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features

        if head_dims is None:
            head_dims = [512, 128]

        layers = []
        in_dim = feat_dim
        for out_dim in head_dims:
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())

        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features).squeeze(1)
