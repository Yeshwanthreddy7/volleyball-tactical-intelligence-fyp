import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "NEW_OUTPUTS" / "code"))
nr = pytest.importorskip("new_render")


def test_ring_six_players_has_six_links_each_player_twice():
    pts = np.float32([[1, 10], [4, 10], [8, 10], [1, 16], [4, 16], [8, 16]])
    edges = nr.team_ring(pts)
    assert len(edges) == 6
    deg = np.bincount(np.array(edges).ravel(), minlength=6)
    assert (deg == 2).all()


@pytest.mark.parametrize("k,expected", [(0, 0), (1, 0), (2, 1), (3, 3), (5, 5)])
def test_ring_size(k, expected):
    pts = np.random.default_rng(0).uniform(0, 9, (k, 2)).astype(np.float32)
    assert len(nr.team_ring(pts)) == expected


def test_continuous_ball_recovers_flight_arc():
    t = np.arange(60.0)
    true = np.stack([100 + 5 * t, 600 - 20 * t + 0.4 * t ** 2], 1).astype(np.float32)
    gappy = true.copy()
    gappy[20:45] = np.nan
    out = nr.continuous_ball(gappy, np.ones(60, bool), 60)
    assert np.abs(out - true).max() < 1e-2


def test_continuous_ball_respects_max_gap_and_camera_cuts():
    b = np.full((40, 2), np.nan, np.float32)
    b[:5] = [[10, 10]] * 5
    b[30:] = [[50, 50]] * 10
    assert np.isnan(nr.continuous_ball(b, np.ones(40, bool), 10)[15, 0])
    ok = np.ones(40, bool)
    ok[20] = False
    assert np.isnan(nr.continuous_ball(b, ok, 60)[15, 0])
    assert not np.isnan(nr.continuous_ball(b, np.ones(40, bool), 60)[15, 0])


def test_in_near_half():
    assert nr.in_near_half({"court": np.float32([4.5, 13])})
    assert nr.in_near_half({"court": np.float32([9.3, 17.8])})      # on the line, within margin
    assert not nr.in_near_half({"court": np.float32([4.5, 5])})      # opponent half
    assert not nr.in_near_half({"court": np.float32([-2.5, 14])})    # bench / sideline


def test_all_four_tactics_are_shown():
    lab = np.array([0, 1, 2, 3, -1, 0])
    assert nr.shown_only(lab).tolist() == [0, 1, 2, 3, -1, 0]
    assert set(nr.SHOWN_TACTICS) == set(nr.tt.CLASSES)
    assert set(nr.CAPTION_BGR) == set(nr.SHOWN_TACTICS)


def test_caption_draws_inside_frame():
    img = np.zeros((720, 1280, 3), np.uint8)
    nr.draw_caption(img, "Coordinated Attack")
    assert img[:150, :500].any() and not img[200:].any()
