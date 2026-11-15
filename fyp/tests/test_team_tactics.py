import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "NEW_OUTPUTS" / "code"))
tt = pytest.importorskip("team_tactics")


def test_raw_rows_use_training_screen_scaling():
    # the model was trained on --court-coords linear: x = px/W*1800, y = py/H*900
    players = [[{"team": 1, "id": 3, "feet": np.float32([640, 540])},
                {"team": 0, "id": 1, "feet": np.float32([10, 10])}]]
    ball = np.float32([[320, 180]])
    rows = tt.raw_rows(players, ball, (1280, 720), 1)
    assert np.allclose(rows[0, :4], [450, 225, 900, 675])
    assert (rows[0, 4:] == 0).all()                 # opponent and empty slots stay 0


def _calls(p_raw, n=90, fps=30.0):
    P = np.asarray(p_raw, float) / tt.TRAIN_PRIOR
    P = np.repeat((P / P.sum())[None], 10, 0)
    starts = np.arange(0, 60, 6)
    return tt.frame_calls(n, fps, starts, starts + 28, P, np.ones(n, bool))


def test_confident_window_is_called():
    lab, conf = _calls([0.8, 0.1, 0.05, 0.05])
    assert (lab[10:60] == 0).all()
    assert conf[30].max() == pytest.approx(0.8, abs=0.05)


def test_unsure_window_still_gets_a_call():
    lab, _ = _calls([0.4, 0.3, 0.15, 0.15])
    assert (lab[10:60] >= 0).all()
    lab, _ = tt.frame_calls(90, 30.0, np.arange(0, 60, 6), np.arange(0, 60, 6) + 28,
                            np.repeat(np.array([[0.4, 0.3, 0.15, 0.15]]) / tt.TRAIN_PRIOR, 10, 0),
                            np.ones(90, bool), min_conf=0.5)
    assert (lab == -1).all()                         # a bar can still be passed explicitly


def test_rare_tactics_use_the_lower_bar():
    # Spacing Breakdown at 35%: below the 50% bar, above its own 20% bar -> shown
    lab, _ = _calls([0.35, 0.2, 0.1, 0.35])
    assert (lab[10:60] == 3).all()
    lab, _ = _calls([0.6, 0.27, 0.05, 0.08])
    assert (lab[10:60] == 0).all()
