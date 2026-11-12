# Roll No: 2025AG05429 | DM4ML Assignment 2
"""Proves the validation stage quarantines every class of bad record.

We run the real pipeline validator against a synthetic raw dataset that
contains one example of each defect, then assert the clean outputs and
the error dataset both look right.
"""
import pandas as pd
import pytest

import src.config as config_mod
from src.validation import validate


@pytest.fixture()
def tmp_project(tmp_path, monkeypatch):
    """A throw-away project root with deliberately dirty raw CSVs."""
    monkeypatch.setattr(config_mod, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    real_cfg = (config_mod.Path(__file__).resolve().parents[1]
                / "configs" / "pipeline.yaml").read_text()
    (tmp_path / "configs" / "pipeline.yaml").write_text(real_cfg)

    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    genre_cols = {g: [0] * 3 for g in validate.GENRES}
    pd.DataFrame({
        "user_id": [1, 2, 2],              # duplicate user_id
        "age": [30, 250, 40],              # age 250 out of range
        "gender": ["M", "F", "X"],         # invalid gender
        "occupation": ["engineer", "artist", "doctor"],
        "zip_code": ["12345", "54321", "99999"],
    }).to_csv(raw / "users.csv", index=False)
    pd.DataFrame({
        "movie_id": [10, 20, 30], "title": ["A (1995)", "B (1996)", "C (1997)"],
        "release_date": ["01-Jan-1995", "01-Jan-1996", "01-Jan-1997"],
        "video_release_date": [None] * 3, "imdb_url": ["u"] * 3, **genre_cols,
    }).to_csv(raw / "items.csv", index=False)
    pd.DataFrame({
        #           ok    bad-rating  re-rating  orphan-usr orphan-mov bad-ts  exact-dup
        "user_id": [1,    1,          1,         777,       1,         1,      1],
        "movie_id": [10,  20,         20,        10,        999,       30,     10],
        "rating": [4,     9,          3,         4,         4,         3,      4],
        "timestamp": [880000000, 880000001, 880000005, 880000002,
                      880000003, 99, 880000000],
    }).to_csv(raw / "interactions.csv", index=False)
    return tmp_path


def test_all_defects_are_quarantined(tmp_project):
    report = validate.run()

    users_ok = pd.read_parquet(tmp_project / "data/processed/validated_users.parquet")
    inter_ok = pd.read_parquet(tmp_project / "data/processed/validated_interactions.parquet")

    assert set(users_ok["user_id"]) == {1}            # dup + bad age + bad gender removed
    assert report["entities"]["users"]["rows_quarantined"] == 2

    # invalid rating, invalid timestamp, orphans and the exact duplicate
    # event are all quarantined; the legitimate re-rating survives
    assert len(inter_ok) == 2
    assert set(inter_ok["movie_id"]) == {10, 20}
    err = pd.read_csv(tmp_project / "data/errors/interactions_errors.csv")
    assert "error_reasons" in err.columns and len(err) == 5
    reasons = " ".join(err["error_reasons"])
    assert "referential integrity" in reasons
