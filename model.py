import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# CNN MODEL
# =========================
class ChordCNN(nn.Module):

    def __init__(self, num_classes):

        super(ChordCNN, self).__init__()

        # Input shape:
        # [batch, 1, 140, 216]

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(16)

        self.pool1 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(32)

        self.pool2 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.conv3 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(64)

        self.pool3 = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # Feature map size calculation:
        # Input: [1, 140, 216]
        # After pool1: [16, 70, 108]
        # After pool2: [32, 35, 54]
        # After pool3: [64, 17, 27]

        self.flatten_size = 64 * 17 * 27

        self.fc1 = nn.Linear(
            self.flatten_size,
            256
        )

        self.dropout = nn.Dropout(0.3)

        self.fc2 = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        x = self.pool1(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool2(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.pool3(
            F.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        x = x.view(
            x.size(0),
            -1
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.dropout(x)

        x = self.fc2(x)

        return x