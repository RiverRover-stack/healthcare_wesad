"""
Student Models: Lightweight CNNs for Knowledge Distillation

Three architectures at different efficiency/accuracy trade-off points:

    Model           Params   Target Size  Architecture
    ─────────────────────────────────────────────────────────────
    MicroCNN         ~5K     < 20 KB      Depthwise-separable Conv1D (MobileNet-style)
    TinyCNN         ~15K     < 60 KB      3-layer standard Conv1D
    MiniCNN-LSTM    ~30K     <120 KB      2-layer Conv1D + single-layer LSTM

All models:
    - Input:  (batch, 6, 3840)  — 6 channels @ 64 Hz x 60s (same as teacher)
    - Output: (batch, 2)        — binary logits (baseline vs stress)
    - Use Global Average Pooling (not flatten) to minimise FC parameters
    - Quantization-friendly (no custom ops, all standard layers)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Building block: Depthwise-Separable 1D Convolution
# ─────────────────────────────────────────────────────────────────────────────

class _DepthwiseSep(nn.Module):
    """
    Depthwise-separable 1D conv (MobileNet-style).

    Two-step convolution that is ~8-9x cheaper than a standard Conv1d:
        1. Depthwise:  Conv1d(C, C, k, groups=C)  — one filter per channel
        2. Pointwise:  Conv1d(C, C_out, 1)         — 1x1 conv mixes channels
    Each step is followed by BN + ReLU.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            # Depthwise
            nn.Conv1d(in_ch, in_ch, kernel_size, padding=pad, groups=in_ch, bias=False),
            nn.BatchNorm1d(in_ch),
            nn.ReLU(),
            # Pointwise
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# MicroCNN  (~5K parameters)
# ─────────────────────────────────────────────────────────────────────────────

class MicroCNN(nn.Module):
    """
    MicroCNN: ~5.5K parameters.  Target deployment: microcontroller.

    Uses depthwise-separable convolutions throughout — each DS layer is
    ~8x cheaper than an equivalent standard Conv1d.

    Architecture:
        DS(6->16, k=7) -> MaxPool(2)
        DS(16->32, k=5) -> MaxPool(2)
        DS(32->64, k=3) -> GlobalAvgPool
        FC(64->32) -> ReLU
        FC(32->2)
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            _DepthwiseSep(in_channels, 16, kernel_size=7),
            nn.MaxPool1d(2),
            _DepthwiseSep(16, 32, kernel_size=5),
            nn.MaxPool1d(2),
            _DepthwiseSep(32, 64, kernel_size=3),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# TinyCNN  (~15K parameters)
# ─────────────────────────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    """
    TinyCNN: ~15K parameters.  Target deployment: smartphone / BLE SoC.

    Standard 3-layer Conv1D — straightforward and quantization-friendly.

    Architecture:
        Conv1d(6->24,  k=7) -> BN -> ReLU -> MaxPool(2)
        Conv1d(24->48, k=5) -> BN -> ReLU -> MaxPool(2)
        Conv1d(48->56, k=3) -> BN -> ReLU -> GlobalAvgPool
        FC(56->2)
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(24),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(24, 48, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(48),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(48, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(56),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(56, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# MiniCNN-LSTM  (~30K parameters)
# ─────────────────────────────────────────────────────────────────────────────

class MiniCNNLSTM(nn.Module):
    """
    MiniCNN-LSTM: ~29K parameters.  Hybrid CNN + recurrent model.

    The CNN layers extract local temporal features; the LSTM captures
    long-range sequential dependencies across the whole 60-second window.
    This architecture is particularly suited to physiological signals that
    have both local patterns (heartbeat peaks) and global trends (EDA rise).

    Architecture:
        Conv1d(6->32, k=7) -> BN -> ReLU -> MaxPool(4)   [3840 -> 960]
        Conv1d(32->64, k=5) -> BN -> ReLU -> MaxPool(4)  [960 -> 240]
        Permute -> LSTM(64, hidden=40, layers=1)          [processes T=240 steps]
        Take last hidden state (B, 40)
        FC(40->2)
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()
        self.conv_blocks = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),

            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(4),
        )
        # After two MaxPool1d(4): T = 3840 / 16 = 240
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=40,
            num_layers=1,
            batch_first=True,   # expects (B, T, features)
        )
        self.classifier = nn.Linear(40, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_blocks(x)         # (B, 64, 240)
        x = x.permute(0, 2, 1)          # (B, 240, 64)  -- LSTM wants (B, T, H)
        _, (h_n, _) = self.lstm(x)      # h_n: (1, B, 40)
        x = h_n.squeeze(0)              # (B, 40)
        return self.classifier(x)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Registry: all student classes in one place (used by train_students.py)
# ─────────────────────────────────────────────────────────────────────────────

STUDENT_REGISTRY = {
    'MicroCNN':     MicroCNN,
    'TinyCNN':      TinyCNN,
    'MiniCNN-LSTM': MiniCNNLSTM,
}
