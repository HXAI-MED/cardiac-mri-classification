"""Classification models and feature extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cinema.resnet import get_resnet3d
from training_code.classification.model import ConvNeXt3DEncoder, replace_last_linear_with_identity
from training_code.classification.protocol import protocols

VIEW = protocols["shared"]["view"]


def preserve_efficientnet_slice_axis(model: nn.Module) -> None:
    """Keep MONAI EfficientNet valid for CineMA's anisotropic 16-slice input."""

    for name, convolution in model.named_modules():
        if not isinstance(convolution, nn.Conv3d) or not name.endswith(("_conv_stem", "_depthwise_conv")):
            continue
        convolution.stride = (*convolution.stride[:2], 1)
        padding = model.get_submodule(f"{name}_padding")
        if isinstance(padding, nn.ConstantPad3d):
            slice_padding = convolution.kernel_size[-1] // 2
            padding.padding = (slice_padding, slice_padding, *padding.padding[2:])


def make_divisible(value: float, divisor: int = 8) -> int:
    """TorchVision-compatible channel rounding used by MobileNetV3."""

    rounded = max(divisor, int(value + divisor / 2) // divisor * divisor)
    if rounded < 0.9 * value:
        rounded += divisor
    return rounded


class ConvNormActivation3D(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: type[nn.Module] | None = nn.ReLU,
    ):
        padding = (kernel_size - 1) // 2
        layers: list[nn.Module] = [
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels, eps=0.001, momentum=0.01),
        ]
        if activation is not None:
            layers.append(activation(inplace=True))
        super().__init__(*layers)


class SqueezeExcitation3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        squeezed_channels = make_divisible(channels // 4, 8)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.reduce = nn.Conv3d(channels, squeezed_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.expand = nn.Conv3d(squeezed_channels, channels, kernel_size=1)
        self.scale_activation = nn.Hardsigmoid(inplace=True)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        scale = self.expand(self.activation(self.reduce(self.pool(image))))
        return image * self.scale_activation(scale)


class MobileNetV3InvertedResidual3D(nn.Module):
    """MobileNetV3 inverted residual block inflated from 2-D to 3-D."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        use_hardswish: bool,
        stride: int,
    ):
        super().__init__()
        activation = nn.Hardswish if use_hardswish else nn.ReLU
        layers: list[nn.Module] = []
        if expanded_channels != in_channels:
            layers.append(
                ConvNormActivation3D(
                    in_channels,
                    expanded_channels,
                    kernel_size=1,
                    activation=activation,
                )
            )
        layers.append(
            ConvNormActivation3D(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=expanded_channels,
                activation=activation,
            )
        )
        if use_se:
            layers.append(SqueezeExcitation3D(expanded_channels))
        layers.append(
            ConvNormActivation3D(
                expanded_channels,
                out_channels,
                kernel_size=1,
                activation=None,
            )
        )
        self.block = nn.Sequential(*layers)
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        output = self.block(image)
        return output + image if self.use_residual else output


class MobileNetV3DEncoder(nn.Module):
    """TorchVision MobileNetV3 configuration with all convolutions inflated to 3-D."""

    CONFIGS: dict[str, tuple[tuple[int, int, int, int, bool, bool, int], ...]] = {
        "large": (
            (16, 3, 16, 16, False, False, 1),
            (16, 3, 64, 24, False, False, 2),
            (24, 3, 72, 24, False, False, 1),
            (24, 5, 72, 40, True, False, 2),
            (40, 5, 120, 40, True, False, 1),
            (40, 5, 120, 40, True, False, 1),
            (40, 3, 240, 80, False, True, 2),
            (80, 3, 200, 80, False, True, 1),
            (80, 3, 184, 80, False, True, 1),
            (80, 3, 184, 80, False, True, 1),
            (80, 3, 480, 112, True, True, 1),
            (112, 3, 672, 112, True, True, 1),
            (112, 5, 672, 160, True, True, 2),
            (160, 5, 960, 160, True, True, 1),
            (160, 5, 960, 160, True, True, 1),
        ),
        "small": (
            (16, 3, 16, 16, True, False, 2),
            (16, 3, 72, 24, False, False, 2),
            (24, 3, 88, 24, False, False, 1),
            (24, 5, 96, 40, True, True, 2),
            (40, 5, 240, 40, True, True, 1),
            (40, 5, 240, 40, True, True, 1),
            (40, 5, 120, 48, True, True, 1),
            (48, 5, 144, 48, True, True, 1),
            (48, 5, 288, 96, True, True, 2),
            (96, 5, 576, 96, True, True, 1),
            (96, 5, 576, 96, True, True, 1),
        ),
    }

    def __init__(self, in_channels: int, variant: str):
        super().__init__()
        if variant not in self.CONFIGS:
            raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")
        configs = self.CONFIGS[variant]
        layers: list[nn.Module] = [
            ConvNormActivation3D(
                in_channels,
                configs[0][0],
                kernel_size=3,
                stride=2,
                activation=nn.Hardswish,
            )
        ]
        layers.extend(MobileNetV3InvertedResidual3D(*config) for config in configs)
        last_channels = configs[-1][3]
        final_channels = 6 * last_channels
        layers.append(
            ConvNormActivation3D(
                last_channels,
                final_channels,
                kernel_size=1,
                activation=nn.Hardswish,
            )
        )
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.feature_dim = 1280 if variant == "large" else 1024
        self.pre_classifier = nn.Sequential(
            nn.Linear(final_channels, self.feature_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2, inplace=True),
        )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Conv3d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm3d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0, 0.01)
            nn.init.zeros_(module.bias)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.pool(self.features(image)).flatten(1)
        return self.pre_classifier(image)


def make_3d_encoder(name: str, in_channels: int) -> tuple[nn.Module, int]:
    if name == "convnext3d_tiny":
        model = ConvNeXt3DEncoder(in_channels=in_channels)
        return model, model.feature_dim

    base_name = name.removesuffix("_3d")
    if base_name.startswith("mobilenet_v3_"):
        variant = base_name.removeprefix("mobilenet_v3_")
        model = MobileNetV3DEncoder(in_channels=in_channels, variant=variant)
        return model, model.feature_dim

    from monai.networks import nets as monai_nets

    if base_name.startswith("resnet"):
        model = get_resnet3d(
            depth=int(base_name.removeprefix("resnet")),
            in_chans=in_channels,
            out_chans=1000,
            layer_inplanes=[64, 128, 256, 512],
        )
    elif base_name.startswith("densenet"):
        if base_name == "densenet161":
            model = monai_nets.DenseNet(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=1000,
                init_features=96,
                growth_rate=48,
                block_config=(6, 12, 36, 24),
                bn_size=4,
            )
        else:
            class_name = {
                "densenet121": "DenseNet121",
                "densenet169": "DenseNet169",
                "densenet201": "DenseNet201",
            }[base_name]
            factory = getattr(monai_nets, class_name)
            model = factory(
                spatial_dims=3,
                in_channels=in_channels,
                out_channels=1000,
            )
        # A 16-slice CineMA SAX patch becomes too shallow for DenseNet's third
        # isotropic transition pool. Preserve depth at that final transition
        # while retaining the published in-plane DenseNet downsampling pattern.
        transition3 = getattr(getattr(model, "features"), "transition3")
        transition3.pool = nn.AvgPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
    elif base_name.startswith("efficientnet_b"):
        variant = base_name.replace("efficientnet_", "efficientnet-")
        model = monai_nets.EfficientNetBN(
            model_name=variant,
            pretrained=False,
            spatial_dims=3,
            in_channels=in_channels,
            num_classes=1000,
        )
        preserve_efficientnet_slice_axis(model)
    else:
        raise ValueError(f"Unsupported 3-D architecture: {name}")
    feature_dim = replace_last_linear_with_identity(model)
    return model, feature_dim


class EDES3DClassifier(nn.Module):
    """Full-volume ED+ES classifier with reusable phase-specific features."""

    def __init__(self, backbone: str, formulation: str, number_of_classes: int):
        super().__init__()
        self.backbone = backbone
        self.formulation = formulation
        if formulation == "stacked":
            self.encoder, feature_dim = make_3d_encoder(backbone, in_channels=2)
            classifier_dim = feature_dim
        elif formulation == "shared":
            self.encoder, feature_dim = make_3d_encoder(backbone, in_channels=1)
            classifier_dim = feature_dim * 3
        else:
            raise ValueError(formulation)
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(classifier_dim, number_of_classes)

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        if self.backbone.startswith("resnet"):
            return self.encoder({VIEW: image})
        return self.encoder(image)

    def forward_features(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.formulation == "stacked":
            return self._encode(image)
        ed_feature = self._encode(image[:, 0:1])
        es_feature = self._encode(image[:, 1:2])
        return {
            "ed": ed_feature,
            "es": es_feature,
            "delta": es_feature - ed_feature,
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(image)
        if isinstance(features, dict):
            features = torch.cat([features["ed"], features["es"], features["delta"]], dim=1)
        return self.classifier(features)


def feature_layer_candidates(model: nn.Module, backbone: str) -> list[str]:
    if backbone.startswith("resnet"):
        requested = [f"encoder.layer{index}" for index in range(1, 5)]
    elif backbone.startswith("densenet"):
        requested = [f"encoder.features.denseblock{index}" for index in range(1, 5)]
    elif backbone.startswith("convnext3d"):
        requested = [f"encoder.stages.{index}" for index in range(4)]
    elif backbone.startswith("mobilenet_v3"):
        available = set(dict(model.named_modules()))
        block_indices = sorted(
            {
                int(name.split(".")[2])
                for name in available
                if name.startswith("encoder.features.") and len(name.split(".")) > 2 and name.split(".")[2].isdigit()
            }
        )
        internal_blocks = block_indices[1:-1]
        if internal_blocks:
            positions = np.linspace(0, len(internal_blocks) - 1, 4).round().astype(int)
            requested = [f"encoder.features.{internal_blocks[position]}" for position in positions]
    else:
        requested = []

    available = set(dict(model.named_modules()))
    if backbone.startswith("efficientnet"):
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
