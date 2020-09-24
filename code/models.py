from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
from revgrad import RevGrad


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class LabelClassifier(nn.Module):
    def __init__(self, num_ftrs, num_classes):
        """
        num_ftrs: number of features passed to this classifier (from feature extractor)
        num_classes: number of classes for classification
        
        """
        super().__init__()
        self.num_ftrs = num_ftrs
        self.num_classes = num_classes
        self.label_classifier = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc1", nn.Linear(self.num_ftrs, 100)),
                    ("c_bn1", nn.BatchNorm1d(100)),
                    ("c_relu1", nn.ReLU()),
                    ("c_drop1", nn.Dropout2d()),
                    ("c_fc2", nn.Linear(100, 100)),
                    ("c_bn2", nn.BatchNorm1d(100)),
                    ("c_relu2", nn.ReLU()),
                    ("c_fc3", nn.Linear(100, self.num_classes)),
                ]
            )
        )

    def forward(self, x):
        return self.label_classifier(x)


class DomainClassifier(nn.Module):
    def __init__(self, num_ftrs):
        """
        num_ftrs: number of features passed to this classifier (from feature extractor)
        
        """
        super().__init__()
        self.num_ftrs = num_ftrs
        self.domain_classifier = nn.Sequential(
            OrderedDict(
                [
                    ("d_revgrad", RevGrad()),
                    ("d_fc1", nn.Linear(self.num_ftrs, 100)),
                    ("d_bn1", nn.BatchNorm1d(100)),
                    ("d_relu1", nn.ReLU()),
                    ("d_fc2", nn.Linear(100, 2)),
                ]
            )
        )

    def forward(self, x):
        return self.domain_classifier(x)


class FeatureExtractor(nn.Module):
    def __init__(self, pretrained, freeze_feature_extractor):
        """
        pretrained: use pretrained weights (trained on ImageNet)
        freeze_feature_extractor: True: feature extractor is frozen - weights remain the same,
            False: fine tune feature extractor weights
        
        """
        super().__init__()
        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.freeze_feature_extractor = freeze_feature_extractor
        if self.freeze_feature_extractor:
            self._freeze_feature_extractor()

        self.num_features = self.feature_extractor.fc.in_features

        # Disable last fc layer, use ResNet only as feature extractor
        self.feature_extractor.fc = Identity()

    def _freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.feature_extractor(x)


class ImageClassifier(nn.Module):
    def __init__(
        self, num_classes=10, pretrained=True, freeze_feature_extractor=False, dann=True
    ):
        """
        num_classes: number of classes for classification
        pretrained: use pretrained weights (trained on ImageNet)
        freeze_feature_extractor: True: feature extractor is frozen - weights remain the same,
            False: fine tune feature extractor weights
        dann: True: use DANN architecture with domain and label classifier to enable training on source and target domain,
            False: simple classifier for training on only one domain

        """
        super().__init__()
        self.dann = dann
        self.feature_extractor = FeatureExtractor(
            pretrained=pretrained, freeze_feature_extractor=freeze_feature_extractor
        )
        self.num_ftrs = self.feature_extractor.num_features
        self.num_classes = num_classes

        # Use DANN architecture for training on images from the source and target domain
        if self.dann:
            self.label_classifier = LabelClassifier(self.num_ftrs, self.num_classes)
            self.domain_classifier = DomainClassifier(self.num_ftrs)

        # Use a classifier for training on only one (source) domain
        else:
            self.label_classifier = LabelClassifier(self.num_ftrs, self.num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)

        # When training on two domains with DANN, predict label and domain
        if self.dann:
            label_output = self.label_classifier(x)
            domain_output = self.domain_classifier(x)
            return label_output, domain_output

        # When training on only one (source) domain, only predict label
        else:
            label_output = self.label_classifier(x)
            return label_output
