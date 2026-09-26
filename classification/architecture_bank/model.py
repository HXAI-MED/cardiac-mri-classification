"""Classification models and feature extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import StochasticDepth


def torchvision_weights(name: str, initialization: str):
    if initialization == "randinit":
        return None
    mapping = {
        "resnet18": models.ResNet18_Weights.DEFAULT,
        "resnet34": models.ResNet34_Weights.DEFAULT,
        "resnet50": models.ResNet50_Weights.DEFAULT,
        "resnet101": models.ResNet101_Weights.DEFAULT,
        "resnext50_32x4d": models.ResNeXt50_32X4D_Weights.DEFAULT,
        "wide_resnet50_2": models.Wide_ResNet50_2_Weights.DEFAULT,
        "densenet121": models.DenseNet121_Weights.DEFAULT,
        "densenet169": models.DenseNet169_Weights.DEFAULT,
        "densenet161": models.DenseNet161_Weights.DEFAULT,
        "densenet201": models.DenseNet201_Weights.DEFAULT,
        "efficientnet_b0": models.EfficientNet_B0_Weights.DEFAULT,
        "efficientnet_b1": models.EfficientNet_B1_Weights.DEFAULT,
        "efficientnet_b2": models.EfficientNet_B2_Weights.DEFAULT,
        "efficientnet_b3": models.EfficientNet_B3_Weights.DEFAULT,
        "convnext_tiny": models.ConvNeXt_Tiny_Weights.DEFAULT,
        "convnext_small": models.ConvNeXt_Small_Weights.DEFAULT,
        "mobilenet_v3_small": models.MobileNet_V3_Small_Weights.DEFAULT,
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
    if name.startswith("resnet") or name.startswith("resnext") or name.startswith("wide_resnet"):
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
    elif name.startswith("convnext"):
        model = getattr(models, name)(weights=weights)
        dim = model.classifier[-1].in_features
        model.classifier[-1] = nn.Identity()
        model.features[0][0] = adapt_first_conv(model.features[0][0], in_channels, pretrained)
    elif name.startswith("mobilenet_v3"):
        model = getattr(models, name)(weights=weights)
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


def replace_last_linear_with_identity(model: nn.Module) -> int:
    """Remove a classification head and return its input feature dimension."""
    linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if not linears:
        raise RuntimeError(f"Could not locate a linear classification head in {type(model).__name__}")
    name, layer = linears[-1]
    parent_name, _, child_name = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = nn.Identity()
    else:
        setattr(parent, child_name, nn.Identity())
    return int(layer.in_features)


class ChannelLayerNorm3D(nn.Module):
    """LayerNorm over channels for a channels-first 3-D tensor."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = image.permute(0, 2, 3, 4, 1)
        image = self.norm(image)
        return image.permute(0, 4, 1, 2, 3)


class ConvNeXtBlock3D(nn.Module):
    """An anisotropic-friendly 3-D adaptation of a ConvNeXt block."""

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
    """ConvNeXt encoder adapted to CineMA's anisotropic SAX volumes."""

    def __init__(
        self,
        in_channels: int,
        depths: tuple[int, int, int, int],
        dims: tuple[int, int, int, int],
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.downsample_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(
                        in_channels,
                        dims[0],
                        kernel_size=(4, 4, 2),
                        stride=(4, 4, 2),
                    ),
                    ChannelLayerNorm3D(dims[0]),
                ),
                nn.Sequential(
                    ChannelLayerNorm3D(dims[0]),
                    nn.Conv3d(dims[0], dims[1], kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    ChannelLayerNorm3D(dims[1]),
                    nn.Conv3d(dims[1], dims[2], kernel_size=2, stride=2),
                ),
                nn.Sequential(
                    ChannelLayerNorm3D(dims[2]),
                    nn.Conv3d(
                        dims[2],
                        dims[3],
                        kernel_size=(2, 2, 1),
                        stride=(2, 2, 1),
                    ),
                ),
            ]
        )
        rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        offset = 0
        stages = []
        for stage_index, n_blocks in enumerate(depths):
            stages.append(
                nn.Sequential(
                    *[
                        ConvNeXtBlock3D(dims[stage_index], rates[offset + block_index])
                        for block_index in range(n_blocks)
                    ]
                )
            )
            offset += n_blocks
        self.stages = nn.ModuleList(stages)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.feature_dim = dims[-1]
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        for downsample, stage in zip(self.downsample_layers, self.stages):
            image = stage(downsample(image))
        return self.norm(image.mean(dim=(2, 3, 4)))


def make_3d_encoder(name: str, in_channels: int) -> tuple[nn.Module, int]:
    if name == "convnext3d_micro":
        model = ConvNeXt3DEncoder(
            in_channels=in_channels,
            depths=(2, 2, 4, 2),
            dims=(32, 64, 128, 256),
            drop_path_rate=0.05,
        )
        return model, model.feature_dim
    if name == "convnext3d_tiny":
        model = ConvNeXt3DEncoder(
            in_channels=in_channels,
            depths=(3, 3, 9, 3),
            dims=(96, 192, 384, 768),
            drop_path_rate=0.1,
        )
        return model, model.feature_dim

    from monai.networks import nets as monai_nets

    base_name = name.removesuffix("_3d")
    if base_name.startswith("resnet"):
        factory = getattr(monai_nets, base_name)
        model = factory(spatial_dims=3, n_input_channels=in_channels, num_classes=1000)
    elif base_name.startswith("densenet"):
        class_name = {
            "densenet121": "DenseNet121",
            "densenet169": "DenseNet169",
            "densenet201": "DenseNet201",
        }[base_name]
        factory = getattr(monai_nets, class_name)
        model = factory(spatial_dims=3, in_channels=in_channels, out_channels=1000)
    elif base_name.startswith("efficientnet_b"):
        variant = base_name.replace("efficientnet_", "efficientnet-")
        model = monai_nets.EfficientNetBN(
            model_name=variant,
            pretrained=False,
            spatial_dims=3,
            in_channels=in_channels,
            num_classes=1000,
        )
    else:
        raise ValueError(name)
    feature_dim = replace_last_linear_with_identity(model)
    return model, feature_dim


class EDES3DClassifier(nn.Module):
    """Native volumetric ED+ES classifier with explicit reusable features."""

    def __init__(self, backbone: str, formulation: str, n_classes: int):
        super().__init__()
        self.formulation = formulation
        if formulation == "stacked":
            self.encoder, dim = make_3d_encoder(backbone, 2)
            self.classifier = nn.Linear(dim, n_classes)
        elif formulation == "shared":
            self.encoder, dim = make_3d_encoder(backbone, 1)
            self.classifier = nn.Linear(dim * 3, n_classes)
        else:
            raise ValueError(formulation)
        self.feature_dim = dim

    def forward_features(self, image: torch.Tensor):
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


def feature_layer_candidates(model: nn.Module, backbone: str, dimensionality: int) -> list[str]:
    """Return stable module paths suitable for later hooks/dissection."""
    if "resnet" in backbone or "resnext" in backbone:
        requested = [f"encoder.layer{index}" for index in range(1, 5)]
    elif backbone.startswith("densenet"):
        requested = [f"encoder.features.denseblock{index}" for index in range(1, 5)]
    elif backbone.startswith("efficientnet") and dimensionality == 2:
        requested = ["encoder.features.2", "encoder.features.3", "encoder.features.5", "encoder.features.7"]
    elif backbone.startswith("convnext3d"):
        requested = [f"encoder.stages.{index}" for index in range(4)]
    elif backbone.startswith("convnext"):
        requested = ["encoder.features.1", "encoder.features.3", "encoder.features.5", "encoder.features.7"]
    elif backbone == "mobilenet_v3_small":
        requested = ["encoder.features.2", "encoder.features.4", "encoder.features.9", "encoder.features.12"]
    elif backbone == "mobilenet_v3_large":
        requested = ["encoder.features.3", "encoder.features.6", "encoder.features.12", "encoder.features.16"]
    else:
        requested = []

    available = set(dict(model.named_modules()))
    if backbone.startswith("efficientnet") and dimensionality == 3:
        block_indices = sorted(
            {
                int(name.split(".")[2])
                for name in available
                if name.startswith("encoder._blocks.") and len(name.split(".")) > 2 and name.split(".")[2].isdigit()
            }
        )
        if block_indices:
            positions = np.linspace(0, len(block_indices) - 1, 4).round().astype(int)
            requested = [f"encoder._blocks.{block_indices[position]}" for position in positions]
    return [name for name in requested if name in available]
