import torch
import torch.nn as nn

class Self_CNN(nn.Module):
    def __init__(self, num_classes=24):
        super().__init__()

        # Block 1
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 224 → 112
            nn.Dropout(0.2) #0.2
        )

        # Block 2
        self.block_2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 112 → 56
            nn.Dropout(0.2)
        )

        # Block 3
        self.block_3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 56 → 28
            nn.Dropout(0.2)
        )

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output: (128, 1, 1)

        # Final Classifier Section
        self.classifier = nn.Sequential(
            nn.Flatten(),              # (128, 1, 1) → (128)
            nn.Linear(128, 64),
            nn.ReLU(), 
            nn.Dropout(0.4), 
            nn.Linear(64, num_classes)  # 24 classes
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x