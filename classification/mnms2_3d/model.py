"""Classification models and feature extraction."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from training_code.classification.mnms2_3d.dataset import pad_last_spatial_axis
from training_code.classification.model import ConvNeXt3DEncoder, replace_last_linear_with_identity


def make_3d_encoder(name: str, in_channels: int) -> tuple[nn.Module, int]:
    if name == "convnext3d_tiny":
        model = ConvNeXt3DEncoder(in_channels=in_channels)
        return model, model.feature_dim

    from monai.networks import nets as monai_nets

    base_name = name.removesuffix("_3d")
    if base_name.startswith("resnet"):
        factory = getattr(monai_nets, base_name)
        model = factory(spatial_dims=3, n_input_channels=in_channels, num_classes=1000)
    elif base_name.startswith("densenet"):
        class_name = {"densenet121": "DenseNet121", "densenet169": "DenseNet169"}[base_name]
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


class EDES3DClassifier(nn.Module):
    """Full-volume ED+ES classifier with reusable phase-specific features."""

    def __init__(self, backbone: str, formulation: str, number_of_classes: int):
        super().__init__()
        self.backbone = backbone
        self.formulation = formulation
        self.minimum_efficientnet_axis_size = 32 if backbone.startswith("efficientnet") else None
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

    def _maybe_pad_image(self, image: torch.Tensor) -> torch.Tensor:
        if self.minimum_efficientnet_axis_size is None:
            return image
        return pad_last_spatial_axis(image, self.minimum_efficientnet_axis_size)

    def train(self, mode: bool = True) -> EDES3DClassifier:
        super().train(mode)
        if mode and self.minimum_efficientnet_axis_size is not None:
            for module in self.modules():
                if isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self

    def forward_features(self, image: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
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
            features = torch.cat([features["ed"], features["es"], features["delta"]], dim=1)
        return self.classifier(features)


def feature_layer_candidates(model: nn.Module, backbone: str) -> list[str]:
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
                if name.startswith("encoder._blocks.") and len(name.split(".")) > 2 and name.split(".")[2].isdigit()
            }
        )
        if block_indices:
            positions = np.linspace(0, len(block_indices) - 1, 4).round().astype(int)
            requested = [f"encoder._blocks.{block_indices[position]}" for position in positions]
    return [name for name in requested if name in available]
