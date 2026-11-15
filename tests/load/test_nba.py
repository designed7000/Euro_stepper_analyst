from datetime import date

import pandas as pd
import pytest

from esa.load import nba
from esa.settings import Settings


def _fake(frame, seen):
    class Fake:
        def __init__(self, **kwargs):
            seen.update(kwargs)

        def get_data_frames(self):
            return [frame]

    return Fake


def _frame(columns, rows=1):
    return pd.DataFrame({c: ["x"] * rows for c in columns})


@pytest.fixture
def settings(tmp_path):
    return Settings(lake_dir=tmp_path, call_pause_s=0.0, call_attempts=1)


def test_fetch_shots_sends_window_and_regular_season(monkeypatch, settings):
    seen = {}
    monkeypatch.setattr(
        nba.shotchartdetail, "ShotChartDetail", _fake(_frame(nba.SHOT_COLUMNS), seen)
    )
    out = nba.fetch_shots("2026-27", date(2026, 11, 12), date(2026, 11, 15), settings)
    assert len(out) == 1
    assert seen["date_from_nullable"] == "11/12/2026"
    assert seen["date_to_nullable"] == "11/15/2026"
    assert seen["season_nullable"] == "2026-27"
    assert seen["season_type_all_star"] == "Regular Season"
    assert seen["context_measure_simple"] == "FGA"
    assert (seen["player_id"], seen["team_id"], seen["timeout"]) == (0, 0, 60)


def test_fetch_shots_full_season_sends_empty_dates(monkeypatch, settings):
    seen = {}
    monkeypatch.setattr(
        nba.shotchartdetail, "ShotChartDetail", _fake(_frame(nba.SHOT_COLUMNS), seen)
    )
    nba.fetch_shots("2026-27", None, None, settings)
    assert seen["date_from_nullable"] == "" and seen["date_to_nullable"] == ""


def test_missing_shot_column_raises_schema_error_with_names(monkeypatch, settings):
    frame = _frame(nba.SHOT_COLUMNS - {"LOC_X", "LOC_Y"})
    monkeypatch.setattr(nba.shotchartdetail, "ShotChartDetail", _fake(frame, {}))
    with pytest.raises(nba.SchemaError, match="LOC_X.*LOC_Y|LOC_Y.*LOC_X"):
        nba.fetch_shots("2026-27", None, None, settings)


def test_zero_rows_with_all_columns_is_valid(monkeypatch, settings):
    frame = _frame(nba.SHOT_COLUMNS, rows=0)
    monkeypatch.setattr(nba.shotchartdetail, "ShotChartDetail", _fake(frame, {}))
    assert nba.fetch_shots("2026-27", None, None, settings).empty


def test_fetch_player_index_includes_historical_players(monkeypatch, settings):
    seen = {}
    monkeypatch.setattr(
        nba.playerindex, "PlayerIndex", _fake(_frame(nba.PLAYER_INDEX_COLUMNS), seen)
    )
    assert len(nba.fetch_player_index("2026-27", settings)) == 1
    assert (seen["season"], seen["historical_nullable"], seen["timeout"]) == ("2026-27", 1, 60)


def test_missing_player_index_column_raises_schema_error(monkeypatch, settings):
    frame = _frame(nba.PLAYER_INDEX_COLUMNS - {"HEIGHT"})
    monkeypatch.setattr(nba.playerindex, "PlayerIndex", _fake(frame, {}))
    with pytest.raises(nba.SchemaError, match="HEIGHT"):
        nba.fetch_player_index("2026-27", settings)


def test_fetch_game_log_asks_for_team_rows(monkeypatch, settings):
    seen = {}
    monkeypatch.setattr(
        nba.leaguegamelog, "LeagueGameLog", _fake(_frame(nba.GAME_LOG_COLUMNS), seen)
    )
    assert len(nba.fetch_game_log("2026-27", settings)) == 1
    assert seen["player_or_team_abbreviation"] == "T"
    assert seen["season_type_all_star"] == "Regular Season"
    assert (seen["season"], seen["timeout"]) == ("2026-27", 60)


def test_missing_game_log_column_raises_schema_error(monkeypatch, settings):
    frame = _frame(nba.GAME_LOG_COLUMNS - {"MATCHUP"})
    monkeypatch.setattr(nba.leaguegamelog, "LeagueGameLog", _fake(frame, {}))
    with pytest.raises(nba.SchemaError, match="MATCHUP"):
        nba.fetch_game_log("2026-27", settings)
