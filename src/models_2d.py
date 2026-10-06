"""2-D CNN backbones and shared ED/ES classification heads."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .protocol import protocols


def torchvision_weights(name: str, initialization: str):
    if initialization not in ("randinit", "imagenet"):
        raise ValueError(f"Unknown initialization: {initialization}")
    return (
        None if initialization == "randinit" else models.get_model_weights(name).DEFAULT
    )


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
            grayscale = conv.weight.sum(dim=1, keepdim=True)
            if in_channels == 1:
                replacement.weight.copy_(grayscale)
            else:
                replacement.weight.copy_(
                    grayscale.repeat(1, in_channels, 1, 1) / in_channels
                )
        else:
            nn.init.kaiming_normal_(
                replacement.weight, mode="fan_out", nonlinearity="relu"
            )
        if replacement.bias is not None:
            replacement.bias.zero_() if conv.bias is None else replacement.bias.copy_(
                conv.bias
            )
    return replacement


def make_encoder(
    name: str, in_channels: int, initialization: str
) -> tuple[nn.Module, int]:
    pretrained = initialization == "imagenet"
    weights = torchvision_weights(name, initialization)
    model = getattr(models, name)(weights=weights)
    if name.startswith(("resnet", "resnext", "wide_resnet")):
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        model.conv1 = adapt_first_conv(model.conv1, in_channels, pretrained)
    elif name.startswith("densenet"):
        feature_dim = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        model.features.conv0 = adapt_first_conv(
            model.features.conv0, in_channels, pretrained
        )
    elif name.startswith("efficientnet"):
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(
            model.features[0][0], in_channels, pretrained
        )
    elif name.startswith("convnext"):
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(
            model.features[0][0], in_channels, pretrained
        )
    elif name.startswith("mobilenet_v3"):
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(
            model.features[0][0], in_channels, pretrained
        )
    else:
        raise ValueError(name)
    return model, feature_dim


def feature_layer_candidates(model: nn.Module, backbone: str) -> list[str]:
    if backbone.startswith("resnet"):
        requested = [f"encoder.layer{index}" for index in range(1, 5)]
    elif backbone.startswith("densenet"):
        requested = [f"encoder.features.denseblock{index}" for index in range(1, 5)]
    elif backbone.startswith("efficientnet"):
        requested = [
            "encoder.features.2",
            "encoder.features.3",
            "encoder.features.5",
            "encoder.features.7",
        ]
    elif backbone.startswith("convnext"):
        requested = [
            "encoder.features.1",
            "encoder.features.3",
            "encoder.features.5",
            "encoder.features.7",
        ]
    elif backbone == "mobilenet_v3_small":
        requested = [
            "encoder.features.2",
            "encoder.features.4",
            "encoder.features.9",
            "encoder.features.12",
        ]
    elif backbone == "mobilenet_v3_large":
        requested = [
            "encoder.features.3",
            "encoder.features.6",
            "encoder.features.12",
            "encoder.features.16",
        ]
    else:
        requested = []
    available = set(dict(model.named_modules()))
    return [name for name in requested if name in available]


class SAX2DClassifier(nn.Module):
    """Stacked ED+ES or a shared ED/ES encoder for SCAG-friendly features."""

    def __init__(
        self, backbone: str, formulation: str, initialization: str, n_classes: int
    ):
        super().__init__()
        self.formulation = formulation
        self.initialization = initialization
        if formulation == "stacked":
            self.encoder, dim = make_encoder(backbone, 2, initialization)
            self.classifier = nn.Linear(dim, n_classes)
        elif formulation == "shared":
            self.encoder, dim = make_encoder(backbone, 1, initialization)
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
            features = torch.cat(
                [features["ed"], features["es"], features["delta"]], dim=1
            )
        return self.classifier(features)


class ACDC2DClassifier(SAX2DClassifier):
    def __init__(self, backbone: str, formulation: str, initialization: str):
        super().__init__(
            backbone, formulation, initialization, len(protocols["acdc"]["classes"])
        )


class MnMs2SAXMid2DClassifier(SAX2DClassifier):
    def __init__(self, backbone: str, formulation: str, initialization: str):
        super().__init__(
            backbone,
            formulation,
            initialization,
            len(protocols["mnms2_sax_2d"]["classes"]),
        )


ArchitectureBank2DClassifier = SAX2DClassifier
Reproduction2DClassifier = SAX2DClassifier
