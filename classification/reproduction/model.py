"""Classification models and feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


def torchvision_weights(name: str, initialization: str):
    if initialization == "randinit":
        return None
    mapping = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "densenet161": models.DenseNet161_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b3": models.EfficientNet_B3_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
    }
    return mapping[name]


def adapt_first_conv(conv: nn.Conv2d, in_channels: int, pretrained: bool) -> nn.Conv2d:
    replacement = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
    )
    with torch.no_grad():
        if pretrained and conv.in_channels == 3:
            collapsed = conv.weight.sum(dim=1, keepdim=True)
            replacement.weight.copy_(
                collapsed if in_channels == 1 else collapsed.repeat(1, in_channels, 1, 1) / in_channels
            )
        else:
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
        if replacement.bias is not None:
            if conv.bias is not None:
                replacement.bias.copy_(conv.bias)
            else:
                replacement.bias.zero_()
    return replacement


def make_2d_encoder(name: str, in_channels: int, initialization: str) -> tuple[nn.Module, int]:
    pretrained = initialization == "imagenet"
    weights = torchvision_weights(name, initialization)
    if name.startswith("resnet"):
        model = getattr(models, name)(weights=weights)
        dim = model.fc.in_features
        model.fc = nn.Identity()
        model.conv1 = adapt_first_conv(model.conv1, in_channels, pretrained)
    elif name.startswith("densenet"):
        model = getattr(models, name)(weights=weights)
        dim = model.classifier.in_features
        model.classifier = nn.Identity()
        model.features.conv0 = adapt_first_conv(model.features.conv0, in_channels, pretrained)
    elif name.startswith("efficientnet"):
        model = getattr(models, name)(weights=weights)
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels, pretrained)
    elif name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels, pretrained)
    elif name == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=weights)
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels, pretrained)
    else:
        raise ValueError(name)
    return model, dim


class EDES2DClassifier(nn.Module):
    """Stacked ED+ES or a shared ED/ES encoder for SCAG-friendly features."""

    def __init__(self, backbone: str, formulation: str, initialization: str, n_classes: int):
        super().__init__()
        self.formulation = formulation
        self.initialization = initialization
        if formulation == "stacked":
            self.encoder, dim = make_2d_encoder(backbone, 2, initialization)
            self.classifier = nn.Linear(dim, n_classes)
        elif formulation == "shared":
            self.encoder, dim = make_2d_encoder(backbone, 1, initialization)
            self.classifier = nn.Linear(dim * 3, n_classes)
        else:
            raise ValueError(formulation)
        self.feature_dim = dim

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        if self.initialization != "imagenet":
            return image
        # Grayscale equivalent of ImageNet normalization for adapted weights.
        return (image - 0.449) / 0.226

    def forward_features(self, image: torch.Tensor):
        image = self.normalize(image)
        if self.formulation == "stacked":
            return self.encoder(image)
        ed_feature = self.encoder(image[:, 0:1])
        es_feature = self.encoder(image[:, 1:2])
        return {"ed": ed_feature, "es": es_feature, "delta": es_feature - ed_feature}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(image)
        if isinstance(features, dict):
            features = torch.cat([features["ed"], features["es"], features["delta"]], dim=1)
        return self.classifier(features)
