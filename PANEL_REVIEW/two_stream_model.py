"""
two_stream_model.py -- Two-Stream Mamba + Vision-Mamba classifier.

Implements the whiteboard-sketch architecture (see architecture_spec.md):

    Branch 1 (pose):   (B, 29, 18) --Mamba--------------> hidden seq (B, 29, D)
    Branch 2 (vision):  (B, 30, 1, H, W) motion frames
                        --Vision Mamba------------------> hidden seq (B, 30, D)

    hidden_pose, hidden_vision --Cross Attention--> Feature Fusion --> head --> logits

Branch 1 reuses `MambaBlock` from the existing, already-trained
`fyp/mamba_model.py` (input_dim=18, seq_len=29) instead of redefining the SSM.
Branch 2 ("Vision Mamba") applies the same SSM block to per-frame visual tokens
produced by a small conv stem, after each frame has been diffed against that
video's base frame (see BASEFRAMES/extract_base_frames.py + base-frames memory)
to strip the static court and keep only player-movement signal.

This module is architecture-only: forward-pass shape-correct, not trained.
Run `smoke_test.py` in this folder to verify it end to end with random tensors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Reuse the existing, already-trained Mamba SSM block instead of redefining it.
# ---------------------------------------------------------------------------
_FYP_DIR = Path(__file__).resolve().parent.parent / "fyp"
if str(_FYP_DIR) not in sys.path:
    sys.path.insert(0, str(_FYP_DIR))

from mamba_model import MambaBlock  # noqa: E402  (path set above)


# ---------------------------------------------------------------------------
# Branch 1 -- Pose stream: "2 pos, 29x, 18 feature -> Mamba model"
# ---------------------------------------------------------------------------

class PoseMambaBranch(nn.Module):
    """29-frame, 18-feature pose sequence -> Mamba encoder.

    Same I/O contract as MambaClassifier's encoder half (embed + blocks + norm),
    but returns the *full* hidden sequence (needed as cross-attention keys/values)
    instead of pooling early.
    """

    def __init__(
        self,
        input_dim: int = 18,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state, d_conv=d_conv) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 29, input_dim) -> (B, 29, d_model)"""
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Branch 2 -- Vision stream: "Base frame - each clip - 30 -> Vision Mamba"
# ---------------------------------------------------------------------------

class FrameConvStem(nn.Module):
    """Patchify one motion frame (frame_t - base_frame) into a single d_model token.

    Small conv stack, not a full CNN backbone -- the temporal modelling is the
    Mamba blocks' job, this just needs a compact per-frame embedding.
    """

    def __init__(self, in_channels: int = 1, d_model: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2),  # 64->32
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),           # 32->16
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),           # 16->8
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),                                        # 8x8 -> 1x1
        )
        self.proj = nn.Linear(64, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, C, H, W) -> (N, d_model)   (N = batch * time, flattened by caller)"""
        z = self.net(x).flatten(1)   # (N, 64)
        return self.proj(z)          # (N, d_model)


class VisionMambaBranch(nn.Module):
    """30 motion frames/clip -> per-frame conv token -> Mamba encoder over time.

    "Vision Mamba" here = Vim-style temporal SSM over per-frame visual tokens
    (selective-scan replaces self-attention), reusing the same MambaBlock as
    branch 1 so both streams share one proven building block.
    """

    def __init__(
        self,
        in_channels: int = 1,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.stem = FrameConvStem(in_channels=in_channels, d_model=d_model)
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state, d_conv=d_conv) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, motion_frames: torch.Tensor) -> torch.Tensor:
        """motion_frames: (B, T=30, C, H, W) already diffed against the base frame.
        -> (B, 30, d_model)
        """
        B, T, C, H, W = motion_frames.shape
        tokens = self.stem(motion_frames.reshape(B * T, C, H, W))  # (B*T, d_model)
        h = tokens.reshape(B, T, -1)                                # (B, T, d_model)
        for block in self.blocks:
            h = block(h)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Cross Attention + Feature Fusion (the two circled blobs in the sketch)
# ---------------------------------------------------------------------------

class CrossAttentionFusion(nn.Module):
    """Bidirectional cross attention between the pose and vision hidden
    sequences, then fuse the two attended, pooled streams into one vector.

        pose  queries -> attends to -> vision keys/values
        vision queries -> attends to -> pose  keys/values
        concat(mean_pool(pose_attn), mean_pool(vision_attn)) -> Linear -> GELU -> d_model
    """

    def __init__(self, d_model: int = 64, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.pose_to_vision = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.vision_to_pose = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm_pose = nn.LayerNorm(d_model)
        self.norm_vision = nn.LayerNorm(d_model)
        self.fuse = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, pose_h: torch.Tensor, vision_h: torch.Tensor) -> torch.Tensor:
        """pose_h: (B, 29, D), vision_h: (B, 30, D) -> fused: (B, D)"""
        pose_attn, _ = self.pose_to_vision(pose_h, vision_h, vision_h)
        pose_attn = self.norm_pose(pose_h + pose_attn)

        vision_attn, _ = self.vision_to_pose(vision_h, pose_h, pose_h)
        vision_attn = self.norm_vision(vision_h + vision_attn)

        pooled = torch.cat([pose_attn.mean(dim=1), vision_attn.mean(dim=1)], dim=-1)  # (B, 2D)
        return self.fuse(pooled)  # (B, D)


# ---------------------------------------------------------------------------
# Full two-stream classifier
# ---------------------------------------------------------------------------

class TwoStreamClassifier(nn.Module):
    """Pose-Mamba branch + Vision-Mamba branch -> cross attention -> fusion -> head.

    I/O contract matches every other classifier in fyp/mamba_model.py:
        forward(pose_seq, motion_frames) -> logits (B, num_classes)
    """

    def __init__(
        self,
        pose_input_dim: int = 18,
        vision_in_channels: int = 1,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        n_heads: int = 4,
        num_classes: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pose_branch = PoseMambaBranch(
            input_dim=pose_input_dim, d_model=d_model, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv,
        )
        self.vision_branch = VisionMambaBranch(
            in_channels=vision_in_channels, d_model=d_model, n_layers=n_layers,
            d_state=d_state, d_conv=d_conv,
        )
        self.fusion = CrossAttentionFusion(d_model=d_model, n_heads=n_heads, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, pose_seq: torch.Tensor, motion_frames: torch.Tensor) -> torch.Tensor:
        """
        pose_seq:      (B, 29, 18)              -- branch 1 input
        motion_frames: (B, 30, C, H, W)          -- branch 2 input, frame - base_frame
        -> logits: (B, num_classes)
        """
        pose_h = self.pose_branch(pose_seq)          # (B, 29, D)
        vision_h = self.vision_branch(motion_frames)  # (B, 30, D)
        fused = self.fusion(pose_h, vision_h)          # (B, D)
        return self.head(fused)                         # (B, num_classes)


def build_two_stream_model(**kwargs) -> TwoStreamClassifier:
    """Factory, mirrors create_temporal_model()'s style in mamba_model.py."""
    return TwoStreamClassifier(**kwargs)
