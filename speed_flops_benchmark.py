"""
speed_flops_benchmark.py
========================
Measures inference speed (ms/image) and FLOPs (GFLOPs) for
muClearGAN vs baseline architectures.

NO CHECKPOINTS NEEDED — uses random weights.
FLOPs and speed depend only on architecture, not learned values.

Install dependencies:
    pip install fvcore torch torchvision

Usage:
    python speed_flops_benchmark.py
    python speed_flops_benchmark.py --input_nc 1   # grayscale (BBBC006)
    python speed_flops_benchmark.py --input_nc 3   # RGB (RBC, FocusPath)
    python speed_flops_benchmark.py --input_size 512  # larger input
"""

import argparse
import time
import torch
import torch.nn as nn
import functools
import numpy as np

try:
    from fvcore.nn import FlopCountAnalysis
    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    print("Warning: fvcore not found. Install with: pip install fvcore")
    print("FLOPs will not be measured.\n")


# ──────────────────────────────────────────────
# Arguments
# ──────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nc", type=int, default=1,
                        help="Input channels: 1=grayscale, 3=RGB")
    parser.add_argument("--input_size", type=int, default=256,
                        help="Image size (square) for benchmark")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup passes before timing")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of timed passes")
    return parser.parse_args()


# ──────────────────────────────────────────────
# Architecture definitions
# (self-contained, no imports from the repo needed)
# ──────────────────────────────────────────────

def get_norm(norm_type="instance"):
    if norm_type == "instance":
        return functools.partial(nn.InstanceNorm2d, affine=False)
    return functools.partial(nn.BatchNorm2d, affine=True)


# ── Standard CycleGAN / DualGAN Generator (ResNet-9blocks) ──────────────────
class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        norm = get_norm()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), norm(dim), nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3), norm(dim),
        )
    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """Standard CycleGAN generator — 9 ResNet blocks, used by CycleGAN & DualGAN."""
    def __init__(self, input_nc=1, output_nc=1, ngf=64, n_blocks=9):
        super().__init__()
        norm = get_norm()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7), norm(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1), norm(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1), norm(ngf*4), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            layers.append(ResnetBlock(ngf*4))
        layers += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            norm(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1),
            norm(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ── U-Net Generator (used in µClearGAN G_A and FCL-GAN) ─────────────────────
class UnetSkipBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False):
        super().__init__()
        self.outermost = outermost
        norm = get_norm()
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, 4, stride=2, padding=1, bias=False)
        downrelu = nn.LeakyReLU(0.2, True)
        uprelu   = nn.ReLU(True)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, 4, stride=2, padding=1)
            down = [downconv]
            up   = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, 4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv]
            up   = [uprelu, upconv, norm(outer_nc)]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc, 4, stride=2, padding=1, bias=False)
            down = [downrelu, downconv, norm(inner_nc)]
            up   = [uprelu, upconv, norm(outer_nc)]
            model = down + [submodule] + up
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


def build_unet(input_nc=1, output_nc=1, ngf=64, n_down=8):
    """Build a U-Net generator with n_down downsample levels."""
    # Build from innermost outward
    unet = UnetSkipBlock(ngf*8, ngf*8, innermost=True)
    for _ in range(n_down - 5):
        unet = UnetSkipBlock(ngf*8, ngf*8, submodule=unet)
    unet = UnetSkipBlock(ngf*4, ngf*8, submodule=unet)
    unet = UnetSkipBlock(ngf*2, ngf*4, submodule=unet)
    unet = UnetSkipBlock(ngf,   ngf*2, submodule=unet)
    unet = UnetSkipBlock(output_nc, ngf, input_nc=input_nc, submodule=unet, outermost=True)
    return unet


# ── Blur Coefficient Estimator (µClearGAN specific) ──────────────────────────
class BlurCoefficientEstimator(nn.Module):
    """
    Extracts blur coefficients alpha in R^(3x22) from a blurry image.
    Multi-scale: processes original, W/2, W/4.
    Each scale: Conv+Norm+ReLU -> 6 ResBlocks -> 1x1 Conv -> Linear+Softmax
    """
    def __init__(self, input_nc=1, num_kernels=22):
        super().__init__()
        self.num_kernels = num_kernels
        self.scales = nn.ModuleList([
            self._build_branch(input_nc, num_kernels) for _ in range(3)
        ])
        self.pool = nn.MaxPool2d(2)

    def _build_branch(self, input_nc, num_kernels):
        norm = get_norm()
        layers = [nn.Conv2d(input_nc, 64, 3, padding=1), norm(64), nn.ReLU(True)]
        for _ in range(6):
            layers.append(ResnetBlock(64))
        layers += [
            nn.Conv2d(64, 32, 1),
            nn.AdaptiveAvgPool2d(1),
        ]
        return nn.Sequential(*layers)

    def forward(self, x):
        coeffs = []
        xi = x
        for branch in self.scales:
            feat = branch(xi).squeeze(-1).squeeze(-1)          # (B, 32)
            coeff = nn.functional.softmax(
                nn.functional.linear(feat, torch.ones(self.num_kernels, 32, device=x.device)), dim=-1
            )                                                   # (B, 22)
            coeffs.append(coeff.unsqueeze(1))
            xi = self.pool(xi)
        return torch.cat(coeffs, dim=1)                         # (B, 3, 22)


class MuClearGAN_GA(nn.Module):
    """
    µClearGAN restoration generator G_A (blurry -> sharp).
    Architecture: U-Net (same as standard CycleGAN G_A).
    The novelty is in G_B (degradation), so G_A is a U-Net.
    """
    def __init__(self, input_nc=1, output_nc=1, ngf=64):
        super().__init__()
        self.net = build_unet(input_nc, output_nc, ngf, n_down=8)

    def forward(self, x):
        return self.net(x)


class MuClearGAN_Full(nn.Module):
    """
    Full µClearGAN forward pass: Blur Coefficient Estimator + G_A.
    This is what runs at inference time for deblurring.
    """
    def __init__(self, input_nc=1, output_nc=1, ngf=64, num_kernels=22):
        super().__init__()
        self.estimator = BlurCoefficientEstimator(input_nc, num_kernels)
        self.GA = build_unet(input_nc, output_nc, ngf, n_down=8)

    def forward(self, x):
        _ = self.estimator(x)   # coefficients (used during training for G_B)
        return self.GA(x)       # restored sharp image


# ── FCL-GAN Generator (lightweight, fewer filters) ───────────────────────────
class FCLGANGenerator(nn.Module):
    """
    FCL-GAN uses a lightweight ResNet generator with ngf=32 (half of standard).
    Ref: Zhao et al., ACM MM 2022.
    """
    def __init__(self, input_nc=1, output_nc=1, ngf=32, n_blocks=6):
        super().__init__()
        # Same ResNet structure but smaller (ngf=32 vs 64, 6 blocks vs 9)
        norm = get_norm()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7), norm(ngf), nn.ReLU(True),
            nn.Conv2d(ngf, ngf*2, 3, stride=2, padding=1), norm(ngf*2), nn.ReLU(True),
            nn.Conv2d(ngf*2, ngf*4, 3, stride=2, padding=1), norm(ngf*4), nn.ReLU(True),
        ]
        for _ in range(n_blocks):
            layers.append(ResnetBlock(ngf*4))
        layers += [
            nn.ConvTranspose2d(ngf*4, ngf*2, 3, stride=2, padding=1, output_padding=1),
            norm(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 3, stride=2, padding=1, output_padding=1),
            norm(ngf), nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ──────────────────────────────────────────────
# Benchmark helpers
# ──────────────────────────────────────────────

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def measure_flops(model, dummy, device):
    if not FVCORE_AVAILABLE:
        return None
    try:
        flops = FlopCountAnalysis(model, dummy)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        return flops.total() / 1e9
    except Exception as e:
        print(f"  FLOPs error: {e}")
        return None


def measure_speed(model, dummy, device, warmup=10, runs=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    fps     = 1000.0 / mean_ms
    return mean_ms, std_ms, fps


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 65)
    print("  µClearGAN — Inference Speed & FLOPs Benchmark")
    print(f"  Device     : {device}")
    print(f"  Input      : {args.input_nc}×{args.input_size}×{args.input_size}")
    print(f"  Timing     : {args.warmup} warmup + {args.runs} timed runs")
    print("=" * 65)

    dummy = torch.randn(1, args.input_nc, args.input_size, args.input_size).to(device)

    # Define all models to benchmark
    models = {
        "CycleGAN   (ResNet-9)": ResnetGenerator(args.input_nc, args.input_nc, ngf=64, n_blocks=9),
        "DualGAN    (ResNet-9)": ResnetGenerator(args.input_nc, args.input_nc, ngf=64, n_blocks=9),
        "FCL-GAN    (ResNet-6)": FCLGANGenerator(args.input_nc, args.input_nc, ngf=32, n_blocks=6),
        "µClearGAN  (U-Net GA)": MuClearGAN_GA(args.input_nc, args.input_nc, ngf=64),
        "µClearGAN  (Full)    ": MuClearGAN_Full(args.input_nc, args.input_nc, ngf=64),
    }

    print(f"\n{'Model':<28} {'Params (M)':>10} {'GFLOPs':>10} {'ms/img':>10} {'FPS':>8}")
    print("-" * 70)

    results = {}
    for name, model in models.items():
        model = model.to(device)
        model.eval()

        params   = count_params(model)
        gflops   = measure_flops(model, dummy, device)
        ms, std, fps = measure_speed(model, dummy, device, args.warmup, args.runs)

        flops_str = f"{gflops:.2f}" if gflops is not None else "N/A"
        print(f"{name:<28} {params:>10.2f} {flops_str:>10} {ms:>8.2f}ms {fps:>8.1f}")

        results[name] = {"params": params, "gflops": gflops, "ms": ms, "std": std, "fps": fps}

    print("-" * 70)
    print("\nNotes:")
    print("  - CycleGAN and DualGAN share the same ResNet-9 generator architecture.")
    print("  - FCL-GAN uses a smaller ResNet-6 with ngf=32 (lightweight variant).")
    print("  - 'µClearGAN (U-Net GA)' is G_A only — the restoration generator at inference.")
    print("  - 'µClearGAN (Full)' includes the Blur Coefficient Estimator + G_A.")
    print("  - Random weights used — FLOPs/speed are architecture properties, not weight-dependent.")
    if not FVCORE_AVAILABLE:
        print("\n  Install fvcore to enable FLOPs: pip install fvcore")
    print()


if __name__ == "__main__":
    main()
