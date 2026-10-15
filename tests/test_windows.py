import numpy as np

from audit.data.windows import make_windows, spread_scores, window_ends


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
