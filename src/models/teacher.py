"""
Teacher CNN: Multi-Scale 1D-CNN for Stress Detection

Architecture rationale:
    Stress manifests at different physiological timescales:
      - Fast  (k=8):  ECG heartbeat cycles (~0.7s at 64Hz -> ~45 samples), motion artefacts
      - Medium (k=32): Respiration cycles (~4s at 64Hz -> ~256 samples)
      - Slow  (k=64): EDA skin conductance response (~10-30s), slow thermal drift

    Three parallel branches each capture one timescale, then their feature maps
    are concatenated for a unified classification decision. This is the core
    architectural novelty claimed in the paper.

Inputs:
    Tensor of shape (batch, 6, 3840) -- 6 channels @ 64 Hz x 60s

Outputs:
    Logits of shape (batch, 2) -- binary: 0=baseline, 1=stress

Parameter count: ~266K  (vs ~45K for the previous sequential design)
"""

import torch
import torch.nn as nn


class MultiScaleTeacherCNN(nn.Module):
    """
    Multi-Scale 1D-CNN teacher model.

    Three parallel convolutional branches, each with a different kernel size,
    followed by a shared classification head:

        Input (B, 6, 3840)
        |
        +-- Branch Small  (k= 8): Conv->BN->ReLU->Conv->BN->ReLU->GAP -> (B, 64)
        +-- Branch Medium (k=32): Conv->BN->ReLU->Conv->BN->ReLU->GAP -> (B, 64)
        +-- Branch Large  (k=64): Conv->BN->ReLU->Conv->BN->ReLU->GAP -> (B, 64)
        |
        Concat: (B, 192)
        FC(192->128) -> ReLU -> Dropout(0.3)
        FC(128->64)  -> ReLU
        FC(64->2)
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 2):
        super().__init__()

        self.branch_small  = self._make_branch(in_channels, 32, 64, kernel_size=8)
        self.branch_medium = self._make_branch(in_channels, 32, 64, kernel_size=32)
        self.branch_large  = self._make_branch(in_channels, 32, 64, kernel_size=64)

        # Fused embedding dim = 64 * 3 branches = 192
        embed_dim = 192

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    @staticmethod
    def _make_branch(in_channels: int, mid_channels: int, out_channels: int,
                     kernel_size: int) -> nn.Sequential:
        """
        Two-layer conv block with AdaptiveAvgPool at the end.
        Padding = (kernel_size - 1) // 2  gives 'same-ish' output length
        (for even kernels the output is 1 sample shorter, which is fine
        because AdaptiveAvgPool1d(1) collapses the time dimension anyway).
        """
        pad = (kernel_size - 1) // 2
        return nn.Sequential(
            nn.Conv1d(in_channels,   mid_channels,  kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(),
            nn.Conv1d(mid_channels,  out_channels,  kernel_size=kernel_size, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the 192-dim multi-scale embedding."""
        s = self.branch_small(x).squeeze(-1)    # (B, 64)
        m = self.branch_medium(x).squeeze(-1)   # (B, 64)
        l = self.branch_large(x).squeeze(-1)    # (B, 64)
        return torch.cat([s, m, l], dim=1)      # (B, 192)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self._embed(x))

    def forward_with_features(self, x: torch.Tensor):
        """
        Returns (logits, embedding).
        Used in feature-based knowledge distillation: the 192-dim embedding
        is the 'intermediate representation' the student tries to mimic.
        """
        embedding = self._embed(x)
        logits = self.classifier(embedding)
        return logits, embedding

    def freeze(self) -> None:
        """Freeze all parameters (call before using as KD teacher)."""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_teacher_cnn() -> MultiScaleTeacherCNN:
    """Return a MultiScaleTeacherCNN with default 6-channel input, binary output."""
    return MultiScaleTeacherCNN(in_channels=6, num_classes=2)
