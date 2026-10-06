"""Volumetric CNNs, preserving each experiment's spatial adaptations."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cinema.resnet import get_resnet3d
from torchvision.ops import StochasticDepth

from .protocol import protocols

ACDC_3D_VIEW = protocols["shared"]["view"]


def pad_last_spatial_axis(image: torch.Tensor, minimum_size: int) -> torch.Tensor:
    if image.shape[-1] >= minimum_size:
        return image
    return F.pad(
        image, (0, minimum_size - image.shape[-1], 0, 0, 0, 0), mode="replicate"
    )


def replace_last_linear_with_identity(model: nn.Module) -> int:
    linears = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
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
    """ConvNeXt encoder adapted to CineMA's anisotropic SAX volumes."""

    def __init__(
        self,
        in_channels: int,
        depths: tuple[int, int, int, int] = (3, 3, 9, 3),
        dims: tuple[int, int, int, int] = (96, 192, 384, 768),
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
        for downsample, stage in zip(self.downsample_layers, self.stages, strict=True):
            image = stage(downsample(image))
        return self.norm(image.mean(dim=(2, 3, 4)))


def preserve_efficientnet_slice_axis(model: nn.Module) -> None:
    """Keep MONAI EfficientNet valid for CineMA's anisotropic 16-slice input."""

    for name, convolution in model.named_modules():
        if not isinstance(convolution, nn.Conv3d) or not name.endswith(
            ("_conv_stem", "_depthwise_conv")
        ):
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


def acdc_3d_make_3d_encoder(name: str, in_channels: int) -> tuple[nn.Module, int]:
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


class ACDC3DClassifier(nn.Module):
    """Full-volume ED+ES classifier with reusable phase-specific features."""

    def __init__(self, backbone: str, formulation: str, number_of_classes: int):
        super().__init__()
        self.backbone = backbone
        self.formulation = formulation
        if formulation == "stacked":
            self.encoder, feature_dim = acdc_3d_make_3d_encoder(backbone, in_channels=2)
            classifier_dim = feature_dim
        elif formulation == "shared":
            self.encoder, feature_dim = acdc_3d_make_3d_encoder(backbone, in_channels=1)
            classifier_dim = feature_dim * 3
        else:
            raise ValueError(formulation)
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(classifier_dim, number_of_classes)

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        if self.backbone.startswith("resnet"):
            return self.encoder({ACDC_3D_VIEW: image})
        return self.encoder(image)

    def forward_features(
        self, image: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
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
            features = torch.cat(
                [features["ed"], features["es"], features["delta"]], dim=1
            )
        return self.classifier(features)


def acdc_3d_feature_layer_candidates(model: nn.Module, backbone: str) -> list[str]:
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
                if name.startswith("encoder.features.")
                and len(name.split(".")) > 2
                and name.split(".")[2].isdigit()
            }
        )
        internal_blocks = block_indices[1:-1]
        if internal_blocks:
            positions = np.linspace(0, len(internal_blocks) - 1, 4).round().astype(int)
            requested = [
                f"encoder.features.{internal_blocks[position]}"
                for position in positions
            ]
    else:
        requested = []

    available = set(dict(model.named_modules()))
    if backbone.startswith("efficientnet"):
        block_indices = sorted(
            {
                int(name.split(".")[2])
                for name in available
                if name.startswith("encoder._blocks.")
                and len(name.split(".")) > 2
                and name.split(".")[2].isdigit()
            }
        )
        if block_indices:
            positions = np.linspace(0, len(block_indices) - 1, 4).round().astype(int)
            requested = [
                f"encoder._blocks.{block_indices[position]}" for position in positions
            ]
    return [name for name in requested if name in available]


def architecture_bank_make_3d_encoder(
    name: str, in_channels: int
) -> tuple[nn.Module, int]:
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


class ArchitectureBank3DClassifier(nn.Module):
    """Native volumetric ED+ES classifier with explicit reusable features."""

    def __init__(self, backbone: str, formulation: str, n_classes: int):
        super().__init__()
        self.formulation = formulation
        if formulation == "stacked":
            self.encoder, dim = architecture_bank_make_3d_encoder(backbone, 2)
            self.classifier = nn.Linear(dim, n_classes)
        elif formulation == "shared":
            self.encoder, dim = architecture_bank_make_3d_encoder(backbone, 1)
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
            features = torch.cat(
                [features["ed"], features["es"], features["delta"]], dim=1
            )
        return self.classifier(features)


def architecture_bank_feature_layer_candidates(
    model: nn.Module, backbone: str, dimensionality: int
) -> list[str]:
    """Return stable module paths suitable for later hooks/dissection."""
    if "resnet" in backbone or "resnext" in backbone:
        requested = [f"encoder.layer{index}" for index in range(1, 5)]
    elif backbone.startswith("densenet"):
        requested = [f"encoder.features.denseblock{index}" for index in range(1, 5)]
    elif backbone.startswith("efficientnet") and dimensionality == 2:
        requested = [
            "encoder.features.2",
            "encoder.features.3",
            "encoder.features.5",
            "encoder.features.7",
        ]
    elif backbone.startswith("convnext3d"):
        requested = [f"encoder.stages.{index}" for index in range(4)]
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
    if backbone.startswith("efficientnet") and dimensionality == 3:
        block_indices = sorted(
            {
                int(name.split(".")[2])
                for name in available
                if name.startswith("encoder._blocks.")
                and len(name.split(".")) > 2
                and name.split(".")[2].isdigit()
            }
        )
        if block_indices:
            positions = np.linspace(0, len(block_indices) - 1, 4).round().astype(int)
            requested = [
                f"encoder._blocks.{block_indices[position]}" for position in positions
            ]
    return [name for name in requested if name in available]


def mnms2_3d_make_3d_encoder(name: str, in_channels: int) -> tuple[nn.Module, int]:
    if name == "convnext3d_tiny":
        model = ConvNeXt3DEncoder(in_channels=in_channels)
        return model, model.feature_dim

    from monai.networks import nets as monai_nets

    base_name = name.removesuffix("_3d")
    if base_name.startswith("resnet"):
        factory = getattr(monai_nets, base_name)
        model = factory(spatial_dims=3, n_input_channels=in_channels, num_classes=1000)
    elif base_name.startswith("densenet"):
        class_name = {"densenet121": "DenseNet121", "densenet169": "DenseNet169"}[
            base_name
        ]
        factory = getattr(monai_nets, class_name)
        model = factory(spatial_dims=3, in_channels=in_channels, out_channels=1000)
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
    else:
        raise ValueError(f"Unsupported 3-D architecture: {name}")
    feature_dim = replace_last_linear_with_identity(model)
    return model, feature_dim


class MnMs2SAX3DClassifier(nn.Module):
    """Full-volume ED+ES classifier with reusable phase-specific features."""

    def __init__(self, backbone: str, formulation: str, number_of_classes: int):
        super().__init__()
        self.backbone = backbone
        self.formulation = formulation
        self.minimum_efficientnet_axis_size = (
            32 if backbone.startswith("efficientnet") else None
        )
        if formulation == "stacked":
            self.encoder, feature_dim = mnms2_3d_make_3d_encoder(
                backbone, in_channels=2
            )
            classifier_dim = feature_dim
        elif formulation == "shared":
            self.encoder, feature_dim = mnms2_3d_make_3d_encoder(
                backbone, in_channels=1
            )
            classifier_dim = feature_dim * 3
        else:
            raise ValueError(formulation)
        self.feature_dim = feature_dim
        self.classifier = nn.Linear(classifier_dim, number_of_classes)

    def _maybe_pad_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.minimum_efficientnet_axis_size is None:
            return image
        return pad_last_spatial_axis(image, self.minimum_efficientnet_axis_size)

    def train(self, mode: bool = True) -> MnMs2SAX3DClassifier:
        super().train(mode)
        if mode and self.minimum_efficientnet_axis_size is not None:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward_features(
        self, image: torch.Tensor
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.formulation == "stacked":
            return self.encoder(self._maybe_pad_image(image))
        ed_feature = self.encoder(self._maybe_pad_image(image[:, 0:1]))
        es_feature = self.encoder(self._maybe_pad_image(image[:, 1:2]))
        return {
            "ed": ed_feature,
            "es": es_feature,
            "delta": es_feature - ed_feature,
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(image)
        if isinstance(features, dict):
            features = torch.cat(
                [features["ed"], features["es"], features["delta"]], dim=1
            )
        return self.classifier(features)


def mnms2_3d_feature_layer_candidates(model: nn.Module, backbone: str) -> list[str]:
    if backbone.startswith("resnet"):
        requested = [f"encoder.layer{index}" for index in range(1, 5)]
    elif backbone.startswith("densenet"):
        requested = [f"encoder.features.denseblock{index}" for index in range(1, 5)]
    elif backbone.startswith("convnext3d"):
        requested = [f"encoder.stages.{index}" for index in range(4)]
    else:
        requested = []

    available = set(dict(model.named_modules()))
    if backbone.startswith("efficientnet"):
        block_indices = sorted(
            {
                int(name.split(".")[2])
                for name in available
                if name.startswith("encoder._blocks.")
                and len(name.split(".")) > 2
                and name.split(".")[2].isdigit()
            }
        )
        if block_indices:
            positions = np.linspace(0, len(block_indices) - 1, 4).round().astype(int)
            requested = [
                f"encoder._blocks.{block_indices[position]}" for position in positions
            ]
    return [name for name in requested if name in available]
