import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import models


# ---------------- CNN FEATURE EXTRACTOR ----------------
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2   # Output = 128 channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# ---------------- REAL SWIN HYBRID ----------------
class HybridWaferModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # ---- CNN FRONTEND ----
        self.cnn = CNNFeatureExtractor(in_channels=1)

        # Convert CNN feature channels -> 3 channels for Swin
        self.channel_adapter = nn.Conv2d(128, 3, kernel_size=1)

        # ---- REAL SWIN BACKBONE ----
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            num_classes=0   # VERY IMPORTANT → removes classifier
        )

        # Swin Tiny final feature dim = 768
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # x = (B,1,64,360)
    

        # ---- CNN FEATURES ----
        x = self.cnn(x)   # (B,128,H,W)

        # ---- Resize for Swin ----
        x = F.interpolate(x, size=(224,224), mode="bilinear", align_corners=False)

        # ---- Convert channels ----
        x = self.channel_adapter(x)

        # ---- SWIN FEATURES ----
        features = self.swin(x)   # (B,768)

        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = HybridWaferModel(num_classes=9).cuda()

    dummy = torch.randn(2,1,64,360).cuda()
    out = model(dummy)

    print("Output shape:", out.shape)
