import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class SwinWaferModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()

        # Pure Swin Backbone directly taking 1 channel input
        # timm handles in_chans=1 correctly by summing pretrained weights along channel dim
        self.swin = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            in_chans=1,
            num_classes=num_classes
        )

    def forward(self, x):
        # x is (B, 1, 64, 360) polar strip
        
        # Swin expects 224x224 images
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        
        logits = self.swin(x)
        return logits


if __name__ == "__main__":
    model = SwinWaferModel(num_classes=9).cuda()

    dummy = torch.randn(2, 1, 64, 360).cuda()
    out = model(dummy)

    print("SwinWaferModel Output shape:", out.shape)
