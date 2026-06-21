"""Snapshot store: save / load / miss / metadata round-trips."""

import pandas as pd

from data.store import SnapshotStore, _params_hash


def test_save_then_load_roundtrip(tmp_path):
    store = SnapshotStore(tmp_path)
    df = pd.DataFrame({"PLAYER": ["a", "b"], "PTS": [10, 20]})

    store.save("league_leaders", df, "2024-25")
    loaded = store.load("league_leaders", "2024-25")

    pd.testing.assert_frame_equal(loaded, df)


def test_load_miss_returns_none(tmp_path):
    store = SnapshotStore(tmp_path)
    assert store.load("league_leaders", "1999-00") is None
    assert store.exists("league_leaders", "1999-00") is False


def test_params_distinguish_snapshots(tmp_path):
    store = SnapshotStore(tmp_path)
    clutch = pd.DataFrame({"x": [1]})
    normal = pd.DataFrame({"x": [2]})

    store.save("player_shots", normal, "2024-25", params={"player_id": 1, "clutch_only": False})
    store.save("player_shots", clutch, "2024-25", params={"player_id": 1, "clutch_only": True})

    got = store.load("player_shots", "2024-25", params={"player_id": 1, "clutch_only": True})
    assert got["x"].iloc[0] == 1  # clutch_only=True snapshot, not the False one


def test_metadata_records_rows_and_time(tmp_path):
    store = SnapshotStore(tmp_path)
    df = pd.DataFrame({"x": range(5)})
    store.save("standings", df, "2024-25")

    meta = store.get_metadata("standings", "2024-25")
    assert meta["row_count"] == 5
    assert meta["season"] == "2024-25"
    assert meta["league"] == "nba"
    assert "fetched_at" in meta


def test_league_is_part_of_key(tmp_path):
    store = SnapshotStore(tmp_path)
    nba = pd.DataFrame({"x": [1]})
    store.save("standings", nba, "2024-25", league="nba")
    assert store.load("standings", "2024-25", league="euroleague") is None
    assert store.load("standings", "2024-25", league="nba") is not None


def test_params_hash_order_independent():
    assert _params_hash({"a": 1, "b": 2}) == _params_hash({"b": 2, "a": 1})
    assert _params_hash(None) == "default"
    assert _params_hash({}) == "default"
