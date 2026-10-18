"""
Tests for the reproduction harness (reproduce.py).

The harness is what a panel runs to verify every claim, so its own contract
matters: the stage registry must be complete, the check mechanism must actually
record failures (a gate that always passes is worse than no gate), and
METRICS.md must state the label-provenance caveat rather than presenting
agreement-with-a-heuristic as accuracy.

No video, no weights, no GPU.
"""
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "fyp"))

_spec = importlib.util.spec_from_file_location(
    "reproduce", os.path.join(ROOT, "reproduce.py"))
reproduce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reproduce)


def test_every_stage_is_registered_and_callable():
    assert set(reproduce.STAGES) == {0, 1, 2, 3, 4, 5, 6, 7, 8}
    assert all(callable(f) for f in reproduce.STAGES.values())


def test_check_records_failures_and_passes_cleanly(tmp_path):
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    assert ctx.failures == []

    assert reproduce.check(ctx, True, "a property that holds") is True
    assert ctx.failures == []

    assert reproduce.check(ctx, False, "a property that does not hold") is False
    assert ctx.failures == ["a property that does not hold"]


def test_sha_is_stable_and_none_for_missing_files(tmp_path):
    f = tmp_path / "artefact.bin"
    f.write_bytes(b"volleyball")
    a = reproduce._sha(str(f))
    assert a == reproduce._sha(str(f))          # deterministic
    assert len(a) == 12
    assert reproduce._sha(str(tmp_path / "nope.bin")) is None


def test_sha_changes_when_the_artefact_changes(tmp_path):
    """Provenance is worthless if two different weight files hash the same."""
    f = tmp_path / "w.pt"
    f.write_bytes(b"model-A")
    a = reproduce._sha(str(f))
    f.write_bytes(b"model-B")
    assert reproduce._sha(str(f)) != a


def test_ctx_creates_the_figure_directory(tmp_path):
    ctx = reproduce.Ctx(str(tmp_path / "out"), quick=False)
    assert os.path.isdir(ctx.figdir)
    assert ctx.fig("x.png").endswith(os.path.join("figures", "x.png"))


def test_metrics_document_states_the_label_provenance_caveat(tmp_path):
    """METRICS.md must state where the labels actually come from.

    An earlier version of this test asserted the OPPOSITE claim - that the
    document says the scores measure "agreement with a rule engine" and are
    "not tactical correctness". That was encoding a factual error: the labels
    are human annotations (labels.csv -> clip folder -> target_label, verified
    159/159), and `label_clips.py` never wrote them. The test is kept, pointed
    at the corrected claim, so the document cannot silently drift back.
    """
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    ctx.results["provenance"] = {"generated_at": "now", "packages": {},
                                 "artefacts": {}}
    reproduce.write_metrics(ctx)

    text = open(os.path.join(str(tmp_path), "METRICS.md"), encoding="utf-8").read()
    low = text.lower()
    # says the labels are human, and names the verified chain
    assert "human" in low
    assert "labels.csv" in low
    # names label_clips.py only to EXCLUDE it as the source
    assert "label_clips.py" in text
    assert "never wrote these" in low or "separate path" in low
    # keeps the honest remaining caveat
    assert "single annotator" in low or "inter-annotator" in low


def test_metrics_document_surfaces_failures(tmp_path):
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    ctx.failures.append("detector saw no players")
    reproduce.write_metrics(ctx)

    text = open(os.path.join(str(tmp_path), "METRICS.md"), encoding="utf-8").read()
    assert "CHECK(S) FAILED" in text
    assert "detector saw no players" in text


def test_metrics_document_reports_a_clean_run(tmp_path):
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    reproduce.write_metrics(ctx)
    text = open(os.path.join(str(tmp_path), "METRICS.md"), encoding="utf-8").read()
    assert "All automated checks passed" in text


def test_video_stages_are_declared_so_skip_video_is_meaningful():
    src = open(os.path.join(ROOT, "reproduce.py"), encoding="utf-8").read()
    assert "video_stages = {2, 3}" in src


def test_stage7_skips_gracefully_without_crashing_when_dataset_absent(tmp_path):
    """A machine without the raw (not git-committed, multi-hundred-MB)
    Roboflow export must not fail the whole reproduction run - stage7 should
    record detector_map=None and add no failure, not raise."""
    ctx = reproduce.Ctx(str(tmp_path), quick=True)
    orig = reproduce.DETECTOR_VAL_YAML
    try:
        reproduce.DETECTOR_VAL_YAML = os.path.join(
            "nonexistent_dir_xyz", "data.yaml")
        reproduce.stage7(ctx)
    finally:
        reproduce.DETECTOR_VAL_YAML = orig

    assert ctx.results.get("detector_map") is None
    assert ctx.failures == []


def test_stage8_config_references_the_drift_module():
    """stage8 must import the actual fyp/drift.py module it is documented as
    exercising, not a stale/renamed reference."""
    src = open(os.path.join(ROOT, "reproduce.py"), encoding="utf-8").read()
    assert "from drift import compare_all_pairs" in src


def test_only_a_full_run_refreshes_results_latest():
    """R4: a partial run (--stage or --skip-video) must NOT overwrite
    results/latest/ - the directory the report reads from - with its
    truncated output."""
    from argparse import Namespace
    full = Namespace(stage=None, skip_video=False)
    assert reproduce._should_refresh_latest(full) is True

    assert reproduce._should_refresh_latest(
        Namespace(stage=[5], skip_video=False)) is False
    assert reproduce._should_refresh_latest(
        Namespace(stage=None, skip_video=True)) is False
    assert reproduce._should_refresh_latest(
        Namespace(stage=[2, 3], skip_video=True)) is False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
