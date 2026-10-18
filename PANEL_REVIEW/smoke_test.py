"""
smoke_test.py -- shape-correctness check for two_stream_model.py.

Not training, not real data: random tensors at the exact shapes the sketch
specifies (29x18 pose, 30-frame motion clips at 64x64), asserting every stage
produces the shape architecture_spec.md says it should. Also prints param count
per branch so the two streams' capacity can be sanity-checked at a glance.
"""

import torch

from two_stream_model import (
    PoseMambaBranch,
    VisionMambaBranch,
    CrossAttentionFusion,
    TwoStreamClassifier,
)


def count_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def main() -> None:
    torch.manual_seed(0)
    B, D = 4, 64
    pose_seq = torch.randn(B, 29, 18)
    motion_frames = torch.randn(B, 30, 1, 64, 64)

    print("=== Branch 1: PoseMambaBranch (29x18 -> Mamba) ===")
    pose_branch = PoseMambaBranch(input_dim=18, d_model=D)
    pose_h = pose_branch(pose_seq)
    assert pose_h.shape == (B, 29, D), pose_h.shape
    print(f"  input  {tuple(pose_seq.shape)} -> output {tuple(pose_h.shape)}  OK")
    print(f"  params: {count_params(pose_branch):,}")

    print("\n=== Branch 2: VisionMambaBranch (base-frame-diffed 30 frames -> Vision Mamba) ===")
    vision_branch = VisionMambaBranch(in_channels=1, d_model=D)
    vision_h = vision_branch(motion_frames)
    assert vision_h.shape == (B, 30, D), vision_h.shape
    print(f"  input  {tuple(motion_frames.shape)} -> output {tuple(vision_h.shape)}  OK")
    print(f"  params: {count_params(vision_branch):,}")

    print("\n=== Cross Attention + Feature Fusion ===")
    fusion = CrossAttentionFusion(d_model=D, n_heads=4)
    fused = fusion(pose_h, vision_h)
    assert fused.shape == (B, D), fused.shape
    print(f"  pose_h {tuple(pose_h.shape)} + vision_h {tuple(vision_h.shape)} -> fused {tuple(fused.shape)}  OK")
    print(f"  params: {count_params(fusion):,}")

    print("\n=== Full TwoStreamClassifier (end to end) ===")
    model = TwoStreamClassifier(num_classes=5)
    logits = model(pose_seq, motion_frames)
    assert logits.shape == (B, 5), logits.shape
    print(f"  pose {tuple(pose_seq.shape)} + motion {tuple(motion_frames.shape)} -> logits {tuple(logits.shape)}  OK")
    print(f"  total params: {count_params(model):,}")

    # backward pass sanity: gradients must flow through both branches
    loss = logits.sum()
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"parameters with no gradient: {no_grad}"
    print("\n  backward() OK -- gradients reached both branches, fusion, and head")

    print("\nALL SHAPE + GRADIENT CHECKS PASSED")


if __name__ == "__main__":
    main()
