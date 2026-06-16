"""
benchmark_inference.py
----------------------
Measures inference cost for BOTH disclosed embodiments, closing the latency gap in
INVENTION_DISCLOSURE.md §2.4:

  Embodiment B  pure Swin           -> imported from model.py (SwinWaferModel)
  Embodiment A  CNN-Swin hybrid     -> defined inline here (verbatim from commit a9c6dea),
                                       so the script runs without restoring model.py

Reports parameter count, model size (MB), warm latency (ms/sample) and throughput
(samples/sec) per batch size, on CPU and on CUDA if available. Writes eval_outputs/latency.json.

Run from the repository root:
    python -m experiments.benchmark_inference --batch-sizes 1 8 32 --iters 50
"""

import argparse
import json
import os
import time

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from model import SwinWaferModel  # Embodiment B


# ── Embodiment A: CNN-Swin hybrid (verbatim from git commit a9c6dea) ──
class CNNFeatureExtractor(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1, self.relu, self.maxpool = resnet.bn1, resnet.relu, resnet.maxpool
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2  # -> 128 ch

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer2(self.layer1(x))


class HybridWaferModel(nn.Module):
    def __init__(self, num_classes=9):
        super().__init__()
        self.cnn = CNNFeatureExtractor(in_channels=1)
        self.channel_adapter = nn.Conv2d(128, 3, kernel_size=1)
        self.swin = timm.create_model("swin_tiny_patch4_window7_224",
                                      pretrained=False, num_classes=0)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = self.channel_adapter(x)
        return self.classifier(self.swin(x))


def model_size_mb(model):
    p = sum(t.numel() * t.element_size() for t in model.parameters())
    b = sum(t.numel() * t.element_size() for t in model.buffers())
    return (p + b) / (1024 ** 2)


@torch.no_grad()
def bench(model, device, batch_sizes, iters, warmup):
    model.eval().to(device)
    results = {}
    for bs in batch_sizes:
        x = torch.randn(bs, 1, 64, 360, device=device)
        for _ in range(warmup):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        per_sample_ms = (elapsed / (iters * bs)) * 1000.0
        results[bs] = {"ms_per_sample": per_sample_ms,
                       "samples_per_sec": 1000.0 / per_sample_ms}
    return results


def main():
    ap = argparse.ArgumentParser(description="Inference benchmark for both embodiments")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    embodiments = {
        "B_pure_swin": SwinWaferModel(num_classes=9),
        "A_cnn_swin_hybrid": HybridWaferModel(num_classes=9),
    }
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

    report = {}
    for name, model in embodiments.items():
        report[name] = {
            "num_params": sum(p.numel() for p in model.parameters()),
            "model_size_mb": round(model_size_mb(model), 2),
            "devices": {},
        }
        for dev in devices:
            print(f"Benchmarking {name} on {dev} ...")
            report[name]["devices"][dev] = bench(
                model, dev, args.batch_sizes, args.iters, args.warmup)

    os.makedirs("eval_outputs", exist_ok=True)
    out = os.path.join("eval_outputs", "latency.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    # human-readable summary
    print("\n" + "=" * 64)
    for name, d in report.items():
        print(f"{name}: {d['num_params']/1e6:.2f}M params, {d['model_size_mb']} MB")
        for dev, by_bs in d["devices"].items():
            for bs, m in by_bs.items():
                print(f"  {dev:>4} bs={bs:<3} {m['ms_per_sample']:.3f} ms/sample "
                      f"({m['samples_per_sec']:.1f}/s)")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
