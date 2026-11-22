from datetime import datetime

import pandas as pd
import pytest

from esa.load.raw_store import SHOTS, has_raw, write_raw

NOW = datetime(2026, 11, 15, 10, 30, 0)


def _write(df, raw_dir, now=NOW):
    return write_raw(
        df, source=SHOTS, season="2026-27", origin="api", tag="w", raw_dir=raw_dir, now=now
    )


def test_path_and_string_columns(tmp_path):
    df = pd.DataFrame({"GAME_ID": [22600001], "LOC_X": [-220], "NOTE": [None]})
    path = _write(df, tmp_path)
    assert path == tmp_path / "nba_shots" / "2026-27" / "2026-11-15" / "w_103000.parquet"
    out = pd.read_parquet(path)
    assert out.loc[0, "GAME_ID"] == "22600001" and out.loc[0, "LOC_X"] == "-220"
    assert pd.isna(out.loc[0, "NOTE"])
    assert out.loc[0, "_season"] == "2026-27" and out.loc[0, "_origin"] == "api"
    assert out.loc[0, "_loaded_at"] == pd.Timestamp(NOW)


def test_empty_table_writes_nothing(tmp_path):
    assert _write(pd.DataFrame({"GAME_ID": []}), tmp_path) is None
    assert has_raw(SHOTS, "2026-27", tmp_path) is False


def test_raw_files_do_not_change(tmp_path):
    df = pd.DataFrame({"GAME_ID": ["1"]})
    _write(df, tmp_path)
    with pytest.raises(FileExistsError):
        _write(df, tmp_path)
    assert has_raw(SHOTS, "2026-27", tmp_path) is True
