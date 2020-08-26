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


class ResNet18(nn.Module):
    def __init__(
        self, num_classes=10, pretrained=True, freeze_feature_extractor=False, dann=True
    ):
        super().__init__()
        self.dann = dann
        self.feature_extractor = models.resnet18(pretrained=pretrained)
        self.freeze_feature_extractor = freeze_feature_extractor
        if self.freeze_feature_extractor:
            self._freeze_feature_extractor()
        self.num_ftrs = self.feature_extractor.fc.in_features
        self.num_classes = num_classes

        # Use Resnet only as feature extractor, disable last fc layer
        self.feature_extractor.fc = Identity()

        if dann:
            # Label classifier
            self.classifier = nn.Sequential(
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

            # Domain discriminator
            self.discriminator = nn.Sequential(
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
        else:
            self.classifier = nn.Linear(self.num_ftrs, self.num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        if self.dann:
            label_output = self.classifier(x)
            domain_output = self.discriminator(x)
            return label_output, domain_output
        else:
            label_output = self.classifier(x)
            return label_output

    def _freeze_feature_extractor(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False


if __name__ == "__main__":
    model = ResNet18(dann=True, freeze_feature_extractor=False)
    summary(model, input_size=(3, 224, 224))
