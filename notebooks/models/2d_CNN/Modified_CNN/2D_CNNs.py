#Optimized CNN
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN2DOptimized(nn.Module):
    def __init__(self, in_channels=1, num_outputs=3):  # Changed num_outputs to 3
        super(CNN2DOptimized, self).__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Conv Block 2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Conv Block 3
        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm2d(256)
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm2d(256)

        # Conv Block 4
        self.conv4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm2d(512)
        self.conv4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm2d(512)

        # Conv Block 5
        self.conv5a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5a = nn.BatchNorm2d(512)
        self.conv5b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm2d(512)

        # Global Average Pooling to reduce number of parameters
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        # Dropout layer for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Conv Block 1
        x = F.relu(self.bn1(self.conv1(x)))

        # Conv Block 2
        x = F.relu(self.bn2(self.conv2(x)))

        # Conv Block 3
        x = F.relu(self.bn3a(self.conv3a(x)))
        x = F.relu(self.bn3b(self.conv3b(x)))

        # Conv Block 4
        x = F.relu(self.bn4a(self.conv4a(x)))
        x = F.relu(self.bn4b(self.conv4b(x)))

        # Conv Block 5
        x = F.relu(self.bn5a(self.conv5a(x)))
        x = F.relu(self.bn5b(self.conv5b(x)))

        # Apply Global Average Pooling (GAP)
        x = self.gap(x)
        x = x.view(x.size(0), -1)  # Flatten the output for FC layers

        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout to FC layers
        return self.fc2(x)


# Optimized 3 conv blocks instead of 5 (simpler CNN)
class SimpleCNN16x16(nn.Module):
    def __init__(self, in_channels=1, num_outputs=3):
        super(SimpleCNN16x16, self).__init__()

        # 16x16 input
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # Output: 32 @ 8x8

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # Output: 64 @ 4x4

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # Output: 128 @ 2x2

        # GAP to 1x1 or flatten from 128 @ 2x2
        # GAP to 1x1 from 128 @ 2x2 output: (Batch, 128, 1, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64) # Input is from the GAP output channels
        self.fc2 = nn.Linear(64, num_outputs)
        self.dropout = nn.Dropout(0.2) # Reduced dropout as model is smaller

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))

        return self.fc2(x)

class MTF_CNN(nn.Module):
    def __init__(self, in_channels=1):
        super(MTF_CNN, self).__init__()
        
        # Conv1: 11×11 kernel, 6 filters, ReLU, MaxPool 2×2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=11, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv2: 11×11 kernel, 32 filters, ReLU, MaxPool 3×3
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=11, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )

        # Flatten → Dense layers: 10 → 10 → 1
        self.fc_layers = nn.Sequential(
            nn.Linear(30752, 10),  # matches flatten size from table (30,752)
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)  # flatten except batch dim
        x = self.fc_layers(x)
        return x
