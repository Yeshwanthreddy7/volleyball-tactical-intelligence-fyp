"""
drift.py - quantify distribution shift between source videos and detect it live.

WHY THIS EXISTS
---------------
Section 12.6 / 14.3 of Technical_Review_and_QA.md establish a "per-video
signature": the tactical classifier's leave-one-video-out score is much lower
than its k-fold score, and the stated explanation is that the model partly
learns WHICH CAMERA a clip came from rather than the tactic. Until this module,
that explanation rested entirely on an INDIRECT argument - a performance gap -
never on a direct measurement of the feature distributions themselves.

This module makes the direct measurement, using the two standard methods a
production ML system uses to detect that a new batch of inputs no longer
resembles what a model was trained on:

  * Kolmogorov-Smirnov two-sample test per feature - is the marginal
    distribution of this feature different between video A and video B?
  * Population Stability Index (PSI), the industry-standard single-number
    summary of how much a distribution has shifted, with the conventional
    reading:  <0.1 no meaningful shift · 0.1-0.25 moderate · >0.25 major.

Two uses:

  1. OFFLINE (this project's report): quantify how different the three source
     videos' feature distributions already are from each other, feature by
     feature. This is the missing direct evidence for the per-video-signature
     claim, and it should rank the SAME channels the classifier's own channel
     importance ranked highly if the story is coherent.

  2. LIVE (future matches): fit a reference distribution once from the training
     videos' pooled features, then score a new match's clips against it. A high
     drift score is an early, cheap warning that a new camera looks nothing
     like anything the classifier was trained on - BEFORE the classifier
     produces a confident-looking wrong label. This is the same role a
     production drift monitor plays for any deployed classifier (see
     `senior-ml-engineer` skill: "input data drift detection ... alerts fire
     before user-visible degradation").
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ChannelDrift:
    feature: str
    ks_statistic: float
    ks_pvalue: float
    psi: float
    reference_median: float
    other_median: float

    @property
    def shift_level(self) -> str:
        if self.psi < 0.1:
            return "none"
        if self.psi < 0.25:
            return "moderate"
        return "major"


@dataclass
class DriftReport:
    pair: tuple[str, str]
    channels: list[ChannelDrift] = field(default_factory=list)

    @property
    def mean_psi(self) -> float:
        vals = [c.psi for c in self.channels if np.isfinite(c.psi)]
        return float(np.mean(vals)) if vals else float("nan")

    @property
    def n_major_shift(self) -> int:
        return sum(1 for c in self.channels if c.shift_level == "major")

    def top(self, n: int = 10) -> list[ChannelDrift]:
        return sorted(self.channels, key=lambda c: -c.psi)[:n]


def _psi(reference: np.ndarray, other: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index between two 1-D samples.

    Bin edges are the REFERENCE sample's own decile boundaries (the standard
    construction), so PSI is asymmetric by design: it answers "how much has the
    distribution moved AWAY FROM the reference", which is exactly the question
    a live monitor needs answered (reference = training videos, other = the new
    match), and matches how §14.3's linear-scaling story would show up (a
    genuinely different camera scale shifts mass into bins the reference rarely
    used).
    """
    reference = reference[np.isfinite(reference)]
    other = other[np.isfinite(other)]
    if len(reference) < 5 or len(other) < 5:
        return float("nan")

    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if len(edges) < 3:
        return 0.0  # reference is (near-)constant: no informative binning possible

    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    oth_counts, _ = np.histogram(other, bins=edges)

    ref_frac = ref_counts / max(ref_counts.sum(), 1)
    oth_frac = oth_counts / max(oth_counts.sum(), 1)

    eps = 1e-4  # avoid log(0) / division by zero on an empty bin
    ref_frac = np.clip(ref_frac, eps, None)
    oth_frac = np.clip(oth_frac, eps, None)

    return float(np.sum((oth_frac - ref_frac) * np.log(oth_frac / ref_frac)))


def compare(reference: np.ndarray, other: np.ndarray,
           feature_names: list[str], pair: tuple[str, str] | None = None
           ) -> DriftReport:
    """
    Per-feature KS-test + PSI between two (n, d) feature matrices.

    `reference` and `other` must have the same number of columns as
    `feature_names`; rows are independent samples (one per clip/window).
    """
    from scipy.stats import ks_2samp

    reference = np.asarray(reference, dtype=float)
    other = np.asarray(other, dtype=float)
    if reference.shape[1] != len(feature_names) or other.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match the column count")

    report = DriftReport(pair=pair or ("reference", "other"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for j, name in enumerate(feature_names):
            r = reference[:, j]
            o = other[:, j]
            rf = r[np.isfinite(r)]
            of = o[np.isfinite(o)]
            if len(rf) < 5 or len(of) < 5:
                continue
            stat, p = ks_2samp(rf, of)
            report.channels.append(ChannelDrift(
                feature=name,
                ks_statistic=float(stat),
                ks_pvalue=float(p),
                psi=_psi(rf, of),
                reference_median=float(np.median(rf)),
                other_median=float(np.median(of)),
            ))
    return report


def compare_all_pairs(features_by_group: dict[str, np.ndarray],
                      feature_names: list[str]) -> list[DriftReport]:
    """Every ordered pair of groups (e.g. source videos), each direction -
    PSI is asymmetric so A->B and B->A are both informative."""
    keys = list(features_by_group)
    out = []
    for i, a in enumerate(keys):
        for b in keys:
            if a == b:
                continue
            out.append(compare(features_by_group[a], features_by_group[b],
                               feature_names, pair=(a, b)))
    return out


class ReferenceDriftMonitor:
    """
    Fit once on training-video features; score new clips/windows against it.

    Usage:
        mon = ReferenceDriftMonitor.fit(X_train, feature_names)
        result = mon.score(x_new_clip)          # single clip, 1-D
        result = mon.score(X_new_video)          # many clips, 2-D -> per-clip
    """

    def __init__(self, feature_names: list[str], reference: np.ndarray) -> None:
        self.feature_names = list(feature_names)
        self._ref = np.asarray(reference, dtype=float)
        self._mu = np.nanmean(self._ref, axis=0)
        self._sd = np.nanstd(self._ref, axis=0)
        self._sd[self._sd < 1e-9] = 1.0   # guard degenerate (constant) channels

    @classmethod
    def fit(cls, X: np.ndarray, feature_names: list[str]) -> "ReferenceDriftMonitor":
        return cls(feature_names, X)

    def score(self, x: np.ndarray) -> dict:
        """
        Mean |z-score| across channels, plus the worst-offending channels.

        A single-clip z-score is a much cheaper live check than a full KS-test
        (which needs a batch to compare against); it answers "does this one
        clip look like anything in the training distribution" per channel,
        aggregated into one number comparable across clips.
        """
        x = np.asarray(x, dtype=float).reshape(-1)
        z = np.abs((x - self._mu) / self._sd)
        z = np.where(np.isfinite(z), z, np.nan)
        order = np.argsort(np.nan_to_num(z, nan=-1.0))[::-1]
        top = [
            {"feature": self.feature_names[i], "abs_z": float(z[i])}
            for i in order[:5] if np.isfinite(z[i])
        ]
        return {
            "mean_abs_z": float(np.nanmean(z)),
            "max_abs_z": float(np.nanmax(z)) if np.any(np.isfinite(z)) else float("nan"),
            "top_offenders": top,
        }


__all__ = [
    "ChannelDrift", "DriftReport", "compare", "compare_all_pairs",
    "ReferenceDriftMonitor",
]
