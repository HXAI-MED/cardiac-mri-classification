"""Classification models and feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn

from training_code.classification.model import make_encoder
from training_code.classification.protocol import protocols

CLASSES = protocols["mnms2_sax_2d"]["classes"]


class MnMs2SAXMid2DClassifier(nn.Module):
    """Stacked ED+ES or shared ED/ES encoder with reusable features."""

    def __init__(self, backbone: str, formulation: str, initialization: str) -> None:
        super().__init__()
        self.formulation = formulation
        self.initialization = initialization
        if formulation == "stacked":
            self.encoder, feature_dim = make_encoder(backbone, 2, initialization)
            classifier_dim = feature_dim
        elif formulation == "shared":
            self.encoder, feature_dim = make_encoder(backbone, 1, initialization)
            classifier_dim = 3 * feature_dim
        else:
            raise ValueError(formulation)
        self.classifier = nn.Linear(classifier_dim, len(CLASSES))
        self.feature_dim = feature_dim

    def normalize(self, image: torch.Tensor) -> torch.Tensor:
        if self.initialization != "imagenet":
            return image
        # Grayscale equivalent of ImageNet channel normalization.
        return (image - 0.449) / 0.226

    def forward_features(self, image: torch.Tensor):
        image = self.normalize(image)
        if self.formulation == "stacked":
            return self.encoder(image)
        ed = self.encoder(image[:, 0:1])
        es = self.encoder(image[:, 1:2])
        return {"ed": ed, "es": es, "delta": es - ed}

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(image)
        if isinstance(features, dict):
            features = torch.cat([features["ed"], features["es"], features["delta"]], dim=1)
        return self.classifier(features)
