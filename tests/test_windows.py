import numpy as np

from audit.data.windows import make_windows, spread_scores, spread_step_scores, window_ends


def test_windows_cover_series_end():
    ends = window_ends(23, 5, 5)
    assert ends[0] == 4 and ends[-1] == 22


def test_make_windows_content():
    x = np.arange(20, dtype=float).reshape(10, 2)
    w, ends = make_windows(x, 3, 1)
    assert w.shape == (8, 3, 2) and ends[0] == 2
    assert w[1, :, 0].tolist() == [2.0, 4.0, 6.0]


def test_spread_scores_covers_every_timestep():
    ends = window_ends(12, 5, 5)             # 4, 9, 11
    out = spread_scores(np.array([1.0, 2.0, 3.0]), ends, 12)
    assert out.tolist() == [1] * 5 + [2] * 5 + [3] * 2


def test_spread_step_scores_keeps_each_timesteps_first_score():
    ends = window_ends(12, 5, 5)             # 4, 9, 11: the last window covers 7..11 and adds 10, 11
    s = np.arange(15, dtype=float).reshape(3, 5)
    assert spread_step_scores(s, ends, 12).tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 14]
