"""
mamba_model.py – Mamba-based Sequence Classifier for Volleyball Analytics.

Implements the Mamba selective state-space model (SSM) in pure PyTorch
(no CUDA-specific kernels required) and wraps it in a sequence classifier
suitable for the 29-frame volleyball clip dataset produced by label_clips.py.

Architecture overview
---------------------
Input  : (batch, seq_len=29, input_dim=14)
          14 = ball_x, ball_y + p1_x, p1_y, …, p6_x, p6_y

Encoder: Linear projection → N × MambaBlock → LayerNorm

Head   : Mean-pool over time → Linear → 4-class softmax

References
----------
Gu, A. & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling with
Selective State Spaces. arXiv:2312.00752.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Selective State Space Model (S6) – core Mamba building block
# ---------------------------------------------------------------------------

class SelectiveSSM(nn.Module):
    """
    One Mamba block with selective scan.

    Parameters
    ----------
    d_model  : model (inner) dimension
    d_state  : SSM state dimension N  (paper default: 16)
    d_conv   : local depthwise-conv width (paper default: 4)
    dt_rank  : rank of the Δ projection  (default: ceil(d_model/16))
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        # ── Input expansion ────────────────────────────────────────────────
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)

        # ── Local context: depthwise conv along the time axis ─────────────
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=d_model,
            bias=True,
        )

        # ── SSM parameter projections ─────────────────────────────────────
        # Maps inner activation → (dt_rank + 2*d_state)
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)

        # dt projection: rank → d_model, with bias initialised so softplus(dt) ≈ 1
        dt_init_std = self.dt_rank ** -0.5
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        # Bias: softplus^{-1}(U[0.001, 0.1])
        dt_bias = torch.empty(d_model).uniform_(0.001, 0.1)
        self.dt_proj.bias = nn.Parameter(torch.log(torch.expm1(dt_bias)))

        # A: fixed diagonal state-transition matrix (learnable in log space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0)  # (1, N)
        A = A.expand(d_model, -1)  # (D, N)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection
        self.D = nn.Parameter(torch.ones(d_model))

        # ── Output projection ──────────────────────────────────────────────
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    # ------------------------------------------------------------------
    # Selective scan (sequential implementation; works without CUDA ext)
    # ------------------------------------------------------------------

    def _selective_scan(
        self,
        x: torch.Tensor,      # (B, L, D)
        delta: torch.Tensor,  # (B, L, D)
        A: torch.Tensor,      # (D, N)
        B: torch.Tensor,      # (B, L, N)
        C: torch.Tensor,      # (B, L, N)
    ) -> torch.Tensor:
        """Compute the discretised SSM recurrence and return (B, L, D)."""
        B_sz, L, D = x.shape
        N = A.shape[1]

        # Discretise with zero-order hold (ZOH)
        # Ā  = exp(Δ ⊗ A)              shape: (B, L, D, N)
        # B̄x = Δ ⊗ B ⊗ x              shape: (B, L, D, N)
        delta_A = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )  # (B, L, D, N)
        delta_Bx = (
            delta.unsqueeze(-1)               # (B, L, D, 1)
            * B.unsqueeze(2)                  # (B, L, 1, N)
            * x.unsqueeze(-1)                 # (B, L, D, 1)
        )  # (B, L, D, N)

        # Sequential scan
        h = x.new_zeros(B_sz, D, N)
        ys: list[torch.Tensor] = []
        for t in range(L):
            h = delta_A[:, t] * h + delta_Bx[:, t]   # (B, D, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, L, D)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, L, d_model)

        Returns
        -------
        out : (B, L, d_model)
        """
        B, L, _ = x.shape

        # Expand then gate
        xz = self.in_proj(x)                  # (B, L, 2*D)
        x_in, z = xz.chunk(2, dim=-1)         # each (B, L, D)

        # Depthwise conv along time axis (causal: trim right padding)
        x_conv = x_in.transpose(1, 2)          # (B, D, L)
        x_conv = self.conv1d(x_conv)[..., :L]  # causal crop
        x_conv = x_conv.transpose(1, 2)        # (B, L, D)
        x_conv = F.silu(x_conv)

        # SSM parameter projections
        bcdt = self.x_proj(x_conv)             # (B, L, dt_rank + 2*N)
        dt_rank = self.dt_rank
        delta_raw = bcdt[..., :dt_rank]        # (B, L, dt_rank)
        B_mat = bcdt[..., dt_rank: dt_rank + self.d_state]   # (B, L, N)
        C_mat = bcdt[..., dt_rank + self.d_state:]           # (B, L, N)

        # dt
        delta = F.softplus(self.dt_proj(delta_raw))          # (B, L, D)

        A = -torch.exp(self.A_log.float())     # (D, N)

        # SSM recurrence
        y = self._selective_scan(x_conv, delta, A, B_mat, C_mat)  # (B, L, D)

        # Skip connection with D
        y = y + x_conv * self.D               # (B, L, D)

        # Gate with SiLU of z-branch
        y = y * F.silu(z)                     # (B, L, D)

        return self.out_proj(y)               # (B, L, D)


# ---------------------------------------------------------------------------
# Mamba Block  (SSM + residual + layer-norm)
# ---------------------------------------------------------------------------

class MambaBlock(nn.Module):
    """Pre-norm residual Mamba block."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(d_model, d_state=d_state, d_conv=d_conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


# ---------------------------------------------------------------------------
# Full Mamba Sequence Classifier
# ---------------------------------------------------------------------------

# Label index → string mapping  (must match label_clips.py).
# "Unclassified" is the NOISE-SINK class (spec §4B): dead-ball phases,
# celebrations, huddles and service preparation are routed here instead of
# being forced into a tactical class (which would poison the gradients of the
# four real classes).  It sits at the END of the model-output vector so the
# four tactical indices are unchanged; its user-facing numeric index is 0.
LABEL_NAMES = [
    "Spacing Breakdown",
    "Delayed Support",
    "Coordinated Attack",
    "Coordinated Defense",
    "Unclassified",
]
NUM_CLASSES = len(LABEL_NAMES)

# Raw columns as stored in the clip CSVs (slot-ordered coordinate soup).
# NOTE: these are the columns READ from disk – the tensor actually fed to the
# model is the corrected 18-dim permutation-invariant descriptor produced by
# features.build_model_sequence() (see features.MODEL_FEATURE_COLS).
FEATURE_COLS = [
    "ball_x", "ball_y",
    "p1_x", "p1_y",
    "p2_x", "p2_y",
    "p3_x", "p3_y",
    "p4_x", "p4_y",
    "p5_x", "p5_y",
    "p6_x", "p6_y",
]

# Model-input width comes from the single-source-of-truth feature module.
try:
    from features import MODEL_INPUT_DIM as INPUT_DIM
except ImportError:  # pragma: no cover – allows importing this file standalone
    INPUT_DIM = 18


class MambaClassifier(nn.Module):
    """
    Mamba-based sequence classifier for volleyball clip labeling.

    Parameters
    ----------
    input_dim   : number of input features per frame (default 14)
    d_model     : model dimension (default 64)
    n_layers    : number of stacked Mamba blocks (default 4)
    d_state     : SSM state dimension per block (default 16)
    d_conv      : local conv width per block (default 4)
    num_classes : number of output classes (default 4)
    dropout     : dropout rate on the classifier head (default 0.1)
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        n_layers: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # ── Input embedding ───────────────────────────────────────────────
        self.embed = nn.Linear(input_dim, d_model)

        # ── Stack of Mamba blocks ─────────────────────────────────────────
        self.blocks = nn.ModuleList(
            [MambaBlock(d_model, d_state=d_state, d_conv=d_conv) for _ in range(n_layers)]
        )

        # ── Final layer-norm ──────────────────────────────────────────────
        self.norm = nn.LayerNorm(d_model)

        # ── Classifier head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        # Weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, input_dim)

        Returns
        -------
        logits : (batch, num_classes)
        """
        h = self.embed(x)                   # (B, L, d_model)
        for block in self.blocks:
            h = block(h)                    # (B, L, d_model)
        h = self.norm(h)                    # (B, L, d_model)
        h = h.mean(dim=1)                   # (B, d_model) – global avg pool
        return self.head(h)                 # (B, num_classes)

    # ------------------------------------------------------------------
    # Convenience: predict a single clip tensor
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[str, torch.Tensor]:
        """
        Predict the label for one clip.

        Parameters
        ----------
        x : (seq_len, input_dim)  or  (1, seq_len, input_dim)

        Returns
        -------
        label      : predicted class string
        probs      : (num_classes,) probability tensor
        """
        self.eval()
        if x.dim() == 2:
            x = x.unsqueeze(0)
        logits = self(x)                    # (1, num_classes)
        probs = F.softmax(logits, dim=-1)[0]
        label = LABEL_NAMES[probs.argmax().item()]
        return label, probs


# ---------------------------------------------------------------------------
# Transformer baseline – proves the temporal-model layer is swappable
# ---------------------------------------------------------------------------

class TransformerClassifier(nn.Module):
    """
    Standard Transformer-encoder sequence classifier with the SAME I/O contract
    as MambaClassifier: (batch, seq_len, input_dim) -> (batch, num_classes).

    Exists so the sequential layer can be benchmarked / swapped without
    touching file I/O or evaluation loops (spec §5B).  Select it with
    ``train_mamba.py --arch transformer``.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        n_layers: int = 4,
        n_heads: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        max_len: int = 64,
        # accepted-and-ignored so both architectures share one constructor call
        d_state: int = 16,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x) + self.pos[:, : x.shape[1]]
        h = self.encoder(h)
        h = self.norm(h).mean(dim=1)
        return self.head(h)


# ---------------------------------------------------------------------------
# Deep-learning comparison arms (spec §5B)
#
# Each class below keeps the MambaClassifier I/O contract EXACTLY:
#     forward(x): (batch, seq_len, input_dim) -> (batch, num_classes)  logits
# and accepts the same constructor kwargs create_temporal_model() forwards
# (unused ones are accepted-and-ignored, as TransformerClassifier already does),
# so training (train_mamba.py --arch ...), the checkpoint format, and the
# serve-time loader (interfaces.TorchTemporalClassifier -> pipeline.py) stay
# architecture-agnostic. Swapping arms is a pure retraining exercise.
# ---------------------------------------------------------------------------

try:  # torch>=2.1 exposes weight_norm through the parametrize API
    from torch.nn.utils.parametrizations import weight_norm as _weight_norm
except ImportError:  # pragma: no cover
    from torch.nn.utils import weight_norm as _weight_norm


# ── 1. Temporal Convolutional Network (Bai, Kolter & Koltun 2018) ───────────

class _Chomp1d(nn.Module):
    """Trim the `chomp` right-most timesteps that a left-padded dilated conv
    adds, keeping the convolution causal (output t sees only inputs <= t)."""

    def __init__(self, chomp: int) -> None:
        super().__init__()
        self.chomp = int(chomp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[..., : -self.chomp].contiguous() if self.chomp > 0 else x


class _TCNResidualBlock(nn.Module):
    """Two weight-normalised dilated causal convs + residual connection."""

    def __init__(self, channels: int, kernel: int, dilation: int,
                 dropout: float) -> None:
        super().__init__()
        pad = (kernel - 1) * dilation
        self.net = nn.Sequential(
            _weight_norm(nn.Conv1d(channels, channels, kernel,
                                   padding=pad, dilation=dilation)),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
            _weight_norm(nn.Conv1d(channels, channels, kernel,
                                   padding=pad, dilation=dilation)),
            _Chomp1d(pad), nn.ReLU(), nn.Dropout(dropout),
        )
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, L)
        return self.act(x + self.net(x))


class TCNClassifier(nn.Module):
    """Stack of dilated causal 1-D convolutions with an exponentially growing
    receptive field. Fewest parameters and the most data-efficient of the
    comparison arms. `--n_layers` = number of residual blocks (dilations
    1, 2, 4, ...); `--d_conv` = convolution kernel width.
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        n_layers: int = 4,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        d_conv: int = 4,
        d_state: int = 16,   # accepted-and-ignored (shared constructor call)
        n_heads: int = 4,    # accepted-and-ignored
        max_len: int = 64,   # accepted-and-ignored
    ) -> None:
        super().__init__()
        kernel = max(2, int(d_conv))
        self.embed = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([
            _TCNResidualBlock(d_model, kernel, dilation=2 ** i, dropout=dropout)
            for i in range(max(1, n_layers))
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x).transpose(1, 2)      # (B, d_model, L)
        for block in self.blocks:
            h = block(h)
        h = h.transpose(1, 2)                   # (B, L, d_model)
        h = self.norm(h).mean(dim=1)            # temporal average pool
        return self.head(h)


# ── 2. Bidirectional LSTM with additive attention pooling ───────────────────

class BiLSTMAttentionClassifier(nn.Module):
    """Bi-LSTM encoder + additive attention pool over time. The attention
    weights double as a per-frame saliency map. `--n_layers` is clamped to
    <= 2 (deeper stacked LSTMs overfit ~1k clips).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        n_layers: int = 2,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        d_state: int = 16,   # accepted-and-ignored
        d_conv: int = 4,     # accepted-and-ignored
        n_heads: int = 4,    # accepted-and-ignored
        max_len: int = 64,   # accepted-and-ignored
    ) -> None:
        super().__init__()
        n_rec = max(1, min(int(n_layers), 2))
        self.embed = nn.Linear(input_dim, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=n_rec, batch_first=True,
            bidirectional=True, dropout=dropout if n_rec > 1 else 0.0,
        )
        self.proj = nn.Linear(2 * d_model, d_model)
        self.attn = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        out, _ = self.lstm(h)                       # (B, L, 2*d_model)
        out = torch.tanh(self.proj(out))            # (B, L, d_model)
        scores = self.attn(out).squeeze(-1)         # (B, L)
        weights = torch.softmax(scores, dim=1).unsqueeze(-1)
        ctx = (out * weights).sum(dim=1)            # (B, d_model)
        return self.head(self.norm(ctx))


# ── 3. InceptionTime (Ismail Fawaz et al. 2020) ────────────────────────────

class _InceptionModule(nn.Module):
    """Bottleneck 1x1 conv, three parallel convs at different kernel sizes, and
    a max-pool branch, concatenated. GroupNorm replaces the paper's BatchNorm
    so the block is batch-size independent (train batches here can be small)
    and consistent with the LayerNorm used elsewhere in this module.
    """

    def __init__(self, in_ch: int, n_filters: int,
                 kernel_sizes: tuple[int, ...] = (5, 11, 23)) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(in_ch, n_filters, 1, bias=False)
        self.convs = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters, k, padding=k // 2, bias=False)
            for k in kernel_sizes
        ])
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, n_filters, 1, bias=False),
        )
        out_ch = n_filters * (len(kernel_sizes) + 1)
        self.norm = nn.GroupNorm(1, out_ch)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, C, L)
        length = x.shape[-1]
        z = self.bottleneck(x)
        branches = [conv(z)[..., :length] for conv in self.convs]
        branches.append(self.pool_branch(x)[..., :length])
        return self.act(self.norm(torch.cat(branches, dim=1)))


class InceptionTimeClassifier(nn.Module):
    """`--n_layers` Inception modules (multi-scale 1-D convolutions) with a
    residual shortcut every 3 modules, then global average pooling. Strongest
    single-network accuracy of the comparison arms; for the paper's full
    ensemble, train several seeds and average the logits. Classic depth is 6
    (`--n_layers 6`).
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        n_layers: int = 6,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.1,
        d_state: int = 16,   # accepted-and-ignored
        d_conv: int = 4,     # accepted-and-ignored
        n_heads: int = 4,    # accepted-and-ignored
        max_len: int = 64,   # accepted-and-ignored
    ) -> None:
        super().__init__()
        n_filters = max(8, d_model // 4)
        channels = n_filters * 4              # _InceptionModule output width
        depth = max(1, n_layers)
        self.stem = nn.Conv1d(input_dim, channels, 1, bias=False)
        self.inception = nn.ModuleList(
            [_InceptionModule(channels, n_filters) for _ in range(depth)]
        )
        self.shortcuts = nn.ModuleDict({
            str(d): nn.Sequential(nn.Conv1d(channels, channels, 1, bias=False),
                                  nn.GroupNorm(1, channels))
            for d in range(depth) if d % 3 == 2
        })
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x.transpose(1, 2))     # (B, channels, L)
        residual = h
        for d, module in enumerate(self.inception):
            h = module(h)
            if str(d) in self.shortcuts:
                h = torch.relu(h + self.shortcuts[str(d)](residual))
                residual = h
        h = self.drop(h.mean(dim=-1))         # global average pool over time
        return self.head(h)


def create_temporal_model(arch: str = "mamba", **kwargs) -> nn.Module:
    """Factory mapping an --arch string to a temporal model. Every architecture
    honours the same (batch, seq_len, input_dim) -> (batch, num_classes)
    contract, so the checkpoint format and interfaces.TorchTemporalClassifier
    never need to know which one was trained.
    """
    arch = (arch or "mamba").lower()
    builders = {
        "mamba": MambaClassifier,
        "transformer": TransformerClassifier,
        "tcn": TCNClassifier,
        "bilstm": BiLSTMAttentionClassifier,
        "inceptiontime": InceptionTimeClassifier,
    }
    if arch not in builders:
        raise ValueError(
            f"Unknown temporal architecture '{arch}' "
            f"(choose from: {', '.join(builders)})"
        )
    if arch == "mamba":
        kwargs.pop("n_heads", None)
    return builders[arch](**kwargs)

# temporal-model factory: mamba | transformer | tcn | bilstm | inceptiontime. sync-marker.
