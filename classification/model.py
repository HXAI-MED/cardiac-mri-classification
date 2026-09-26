"""Classification models and feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import StochasticDepth


def torchvision_weights(name: str, initialization: str):
    if initialization == "randinit":
        return None
    mapping = {
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "mobilenet_v3_large": models.MobileNet_V3_Large_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "efficientnet_b3": models.EfficientNet_B3_Weights.DEFAULT,
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "densenet161": models.DenseNet161_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
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
            grayscale = conv.weight.sum(dim=1, keepdim=True)
            if in_channels == 1:
                replacement.weight.copy_(grayscale)
            else:
                replacement.weight.copy_(grayscale.repeat(1, in_channels, 1, 1) / in_channels)
        else:
            nn.init.kaiming_normal_(replacement.weight, mode="fan_out", nonlinearity="relu")
        if replacement.bias is not None:
            replacement.bias.zero_() if conv.bias is None else replacement.bias.copy_(conv.bias)
    return replacement


def make_encoder(name: str, in_channels: int, initialization: str) -> tuple[nn.Module, int]:
    pretrained = initialization == "imagenet"
    weights = torchvision_weights(name, initialization)
    model = getattr(models, name)(weights=weights)
    if name.startswith("resnet"):
        feature_dim = int(model.fc.in_features)
        model.fc = nn.Identity()
        model.conv1 = adapt_first_conv(model.conv1, in_channels, pretrained)
    elif name.startswith("densenet"):
        feature_dim = int(model.classifier.in_features)
        model.classifier = nn.Identity()
        model.features.conv0 = adapt_first_conv(model.features.conv0, in_channels, pretrained)
    elif name.startswith("efficientnet"):
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels, pretrained)
    elif name.startswith("convnext"):
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels, pretrained)
    elif name.startswith("mobilenet_v3"):
        feature_dim = int(model.classifier[-1].in_features)
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels, pretrained)
    else:
        raise ValueError(name)
    return model, feature_dim


def feature_layer_candidates(model: nn.Module, backbone: str) -> list[str]:
    if backbone.startswith("resnet"):
        requested = [f"encoder.layer{index}" for index in range(1, 5)]
    elif backbone.startswith("densenet"):
        requested = [f"encoder.features.denseblock{index}" for index in range(1, 5)]
    elif backbone.startswith("efficientnet"):
        requested = ["encoder.features.2", "encoder.features.3", "encoder.features.5", "encoder.features.7"]
    elif backbone.startswith("convnext"):
        requested = ["encoder.features.1", "encoder.features.3", "encoder.features.5", "encoder.features.7"]
    elif backbone == "mobilenet_v3_small":
        requested = ["encoder.features.2", "encoder.features.4", "encoder.features.9", "encoder.features.12"]
    elif backbone == "mobilenet_v3_large":
        requested = ["encoder.features.3", "encoder.features.6", "encoder.features.12", "encoder.features.16"]
    else:
        requested = []
    available = set(dict(model.named_modules()))
    return [name for name in requested if name in available]


def replace_last_linear_with_identity(model: nn.Module) -> int:
    linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise RuntimeError(f"Could not locate a linear head in {type(model).__name__}")
    name, layer = linears[-1]
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = nn.Identity()
    else:
        setattr(parent, child_name, nn.Identity())
    return int(layer.in_features)


class ChannelLayerNorm3D(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.permute(0, 2, 3, 4, 1)
        image = self.norm(image)
        return image.permute(0, 4, 1, 2, 3)


class ConvNeXtBlock3D(nn.Module):
    """An anisotropic 3-D adaptation of a ConvNeXt block."""

    def __init__(self, channels: int, drop_path: float, layer_scale: float = 1e-6):
        super().__init__()
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel_size=(7, 7, 3),
            padding=(3, 3, 1),
            groups=channels,
        )
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.expand = nn.Linear(channels, 4 * channels)
        self.activation = nn.GELU()
        self.project = nn.Linear(4 * channels, channels)
        self.gamma = nn.Parameter(layer_scale * torch.ones(channels))
        self.drop_path = StochasticDepth(drop_path, mode="row")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        residual = image
        image = self.depthwise(image).permute(0, 2, 3, 4, 1)
        image = self.project(self.activation(self.expand(self.norm(image))))
        image = (self.gamma * image).permute(0, 4, 1, 2, 3)
        return residual + self.drop_path(image)


class ConvNeXt3DEncoder(nn.Module):
    """ConvNeXt-Tiny adapted to anisotropic CineMA SAX volumes."""

    def __init__(self, in_channels: int):
        super().__init__()
        depths = (3, 3, 9, 3)
        dimensions = (96, 192, 384, 768)
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        dimensions[0],
                        kernel_size=(4, 4, 2),
                        stride=(4, 4, 2),
                    ),
                    ChannelLayerNorm3D(dimensions[0]),
                ),
                nn.Sequential(
                    ChannelLayerNorm3D(dimensions[0]),
                    nn.Conv3d(dimensions[0], dimensions[1], kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    ChannelLayerNorm3D(dimensions[1]),
                    nn.Conv3d(dimensions[1], dimensions[2], kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    ChannelLayerNorm3D(dimensions[2]),
                    nn.Conv3d(
                        dimensions[2],
                        dimensions[3],
                        kernel_size=(2, 2, 1),
                        stride=(2, 2, 1),
                    ),
                ),
            ]
        )
        rates = torch.linspace(0, 0.1, sum(depths)).tolist()
        offset = 0
        stages: list[nn.Module] = []
        for stage_index, number_of_blocks in enumerate(depths):
            stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtBlock3D(
                            dimensions[stage_index],
                            rates[offset + block_index],
                        )
                        for block_index in range(number_of_blocks)
                    ]
                )
            )
            offset += number_of_blocks
        self.stages = nn.ModuleList(stages)
        self.norm = nn.LayerNorm(dimensions[-1], eps=1e-6)
        self.feature_dim = dimensions[-1]
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        for downsample, stage in zip(self.downsample_layers, self.stages, strict=True):
            image = stage(downsample(image))
        return self.norm(image.mean(dim=(2, 3, 4)))
