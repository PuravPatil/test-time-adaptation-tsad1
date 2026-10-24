"""The upload loader: every supported format reads to the same table, and prepare() splits it."""
import json

import numpy as np
import pandas as pd
import pytest

from audit.data.upload import UploadError, guess_label_col, guess_time_col, inspect, prepare, read_table

T = 200


def _frame(with_time=True, with_label=True):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cpu": rng.normal(size=T), "mem": rng.normal(size=T)})
    if with_time:
        df.insert(0, "timestamp", pd.date_range("2026-01-01", periods=T, freq="min").astype(str))
    if with_label:
        lab = np.zeros(T, dtype=int)
        lab[150:160] = 1
        df["label"] = lab
    return df


def test_csv_with_header(tmp_path):
    p = tmp_path / "x.csv"
    _frame().to_csv(p, index=False)
    df = read_table(p)
    assert list(df.columns) == ["timestamp", "cpu", "mem", "label"] and len(df) == T


def test_headerless_txt_like_smd(tmp_path):
    p = tmp_path / "machine.txt"
    np.savetxt(p, np.arange(30, dtype=float).reshape(10, 3), delimiter=",")
    df = read_table(p)
    assert list(df.columns) == ["col_0", "col_1", "col_2"] and len(df) == 10


def test_whitespace_txt(tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("1.0 2.0\n3.0   4.0\n5.0\t6.0\n")
    df = read_table(p)
    assert df.shape == (3, 2) and df.iloc[2, 1] == 6.0


def test_tsv_semicolon_and_json_variants(tmp_path):
    f = _frame()
    (tmp_path / "x.tsv").write_text(f.to_csv(sep="\t", index=False))
    (tmp_path / "semi.txt").write_text(f.to_csv(sep=";", index=False))
    (tmp_path / "r.json").write_text(f.to_json(orient="records"))
    (tmp_path / "l.json").write_text(f.to_json(orient="records", lines=True))
    for name in ("x.tsv", "semi.txt", "r.json", "l.json"):
        df = read_table(tmp_path / name)
        assert df.shape == f.shape, name
        assert np.allclose(df["cpu"].astype(float), f["cpu"]), name


def test_npy_1d_2d_and_no_pickle(tmp_path):
    np.save(tmp_path / "a.npy", np.arange(5.0))
    np.save(tmp_path / "b.npy", np.ones((4, 3)))
    assert read_table(tmp_path / "a.npy").shape == (5, 1)
    assert read_table(tmp_path / "b.npy").shape == (4, 3)
    np.save(tmp_path / "obj.npy", np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(UploadError):
        read_table(tmp_path / "obj.npy")


def test_xlsx_and_parquet(tmp_path):
    f = _frame()
    pytest.importorskip("openpyxl")
    f.to_excel(tmp_path / "x.xlsx", index=False)
    assert read_table(tmp_path / "x.xlsx").shape == f.shape
    pytest.importorskip("pyarrow")
    f.to_parquet(tmp_path / "x.parquet")
    assert read_table(tmp_path / "x.parquet").shape == f.shape


def test_bad_files_are_upload_errors(tmp_path):
    (tmp_path / "x.pdf").write_text("nope")
    (tmp_path / "empty.csv").write_text("")
    (tmp_path / "broken.json").write_text("{not json")
    for name in ("x.pdf", "empty.csv", "broken.json"):
        with pytest.raises(UploadError):
            read_table(tmp_path / name)


def test_guesses_and_inspect():
    f = _frame()
    assert guess_time_col(f) == "timestamp" and guess_label_col(f) == "label"
    f2 = _frame().rename(columns={"timestamp": "when"})
    assert guess_time_col(f2) == "when"              # found by parsing, not by name
    assert guess_label_col(_frame(with_label=False)) is None
    info = inspect(f)
    assert info["n_rows"] == T and info["n_numeric"] == 3 and len(info["preview"]) == 5
    with pytest.raises(UploadError):
        inspect(f, max_rows=10)


def test_prepare_splits_in_time_order_and_scales_on_train():
    f = _frame().sample(frac=1.0, random_state=0)   # shuffled rows
    data, rep = prepare(f, time_col="auto", label_col="label", train_frac=0.5)
    assert rep["time_col"] == "timestamp" and rep["features"] == ["cpu", "mem"]
    assert rep["n_train"] == 100 and data.test.shape == (100, 2) and rep["has_labels"]
    assert data.test_labels.sum() == 10 and data.test_labels[50:60].all()   # rows 150-159, after sorting
    assert np.allclose(data.train.mean(0), 0, atol=1e-5) and np.allclose(data.train.std(0), 1, atol=1e-4)
    assert any("sorted" in w for w in rep["warnings"])


def test_prepare_without_labels_and_with_gaps_and_text_columns():
    f = _frame(with_time=False, with_label=False)
    f.loc[5, "cpu"] = np.nan
    f["host"] = "server-a"
    data, rep = prepare(f, time_col=None, label_col=None)
    assert not rep["has_labels"] and data.test_labels.sum() == 0
    assert rep["ignored"] == ["host"] and rep["missing_filled"] == 1
    assert np.isfinite(data.train).all()


def test_prepare_rejects_unusable_input():
    with pytest.raises(UploadError):
        prepare(pd.DataFrame({"text": ["a", "b"] * 100}))          # nothing numeric
    with pytest.raises(UploadError):
        prepare(_frame().head(30))                                    # too few rows
    with pytest.raises(UploadError):
        prepare(_frame(), label_col="missing")                        # unknown column
    with pytest.raises(UploadError):
        prepare(_frame(), feature_cols=["timestamp"], time_col=None)  # non-numeric feature


def test_auto_label_and_duplicate_column_names(tmp_path):
    p = tmp_path / "d.csv"
    p.write_text("a,a,anomaly\n" + "\n".join(f"{i},{i * 2},{int(i > 180)}" for i in range(T)))
    df = read_table(p)
    assert len(set(df.columns)) == 3
    data, rep = prepare(df, time_col=None, label_col="auto")
    assert rep["label_col"] == "anomaly" and rep["has_labels"]
    json.dumps(rep)   # the report must be JSON-serialisable for the API
