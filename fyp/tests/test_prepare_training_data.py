"""_iter_clip_files exclude-pattern behaviour (D5/Q1: drop the (plain)
source-video clips treated as a likely duplicate of source (3) without
moving files out of the labeled folder layout)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prepare_training_data import _iter_clip_files  # noqa: E402


def _make_dataset(tmp_path):
    names = {
        "Coordinated_Attack": [
            "videoplayback_clip_306.mp4",   # plain -> excluded
            "videoplayback (1)_clip_002.mp4",
            "videoplayback (3)_clip_010.mp4",
        ],
        "Spacing_Breakdown": [
            "videoplayback_clip_318.mp4",   # plain -> excluded
            "videoplayback (4)_clip_005.mp4",
        ],
    }
    for folder, files in names.items():
        d = tmp_path / folder
        d.mkdir()
        for f in files:
            (d / f).write_bytes(b"")
    return str(tmp_path)


def test_no_pattern_returns_all_clips(tmp_path):
    root = _make_dataset(tmp_path)
    clips = _iter_clip_files(root)
    assert len(clips) == 5


def test_exclude_pattern_drops_only_plain_clips(tmp_path):
    root = _make_dataset(tmp_path)
    clips = _iter_clip_files(root, exclude_pattern=r"^videoplayback_clip_[0-9]+\.mp4$")
    names = sorted(os.path.basename(p) for p, _, _ in clips)
    assert names == sorted([
        "videoplayback (1)_clip_002.mp4",
        "videoplayback (3)_clip_010.mp4",
        "videoplayback (4)_clip_005.mp4",
    ])


def test_exclude_pattern_never_matches_suffixed_sources(tmp_path):
    root = _make_dataset(tmp_path)
    clips = _iter_clip_files(root, exclude_pattern=r"^videoplayback_clip_[0-9]+\.mp4$")
    assert all("(" in os.path.basename(p) for p, _, _ in clips)


# --- R13: risky extraction defaults surfaced ---------------------------

def _ext_ns(**over):
    from argparse import Namespace
    base = dict(ball_model="fyp/volleyball_best.pt", auto_court=True, tracker="botsort")
    base.update(over)
    return Namespace(**base)


def test_risky_extraction_defaults_silent_when_correct():
    from prepare_training_data import _risky_extraction_defaults
    assert _risky_extraction_defaults(_ext_ns()) == []


def test_risky_extraction_defaults_flag_each_omission():
    from prepare_training_data import _risky_extraction_defaults
    assert any("ball-model" in r
               for r in _risky_extraction_defaults(_ext_ns(ball_model=None)))
    assert any("auto-court" in r
               for r in _risky_extraction_defaults(_ext_ns(auto_court=False)))
    assert any("bytetrack" in r
               for r in _risky_extraction_defaults(_ext_ns(tracker="bytetrack")))
