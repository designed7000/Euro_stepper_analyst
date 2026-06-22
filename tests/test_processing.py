"""Data processing transforms (per-player and league-wide builders)."""

import pandas as pd

from data.processing import (
    process_player_data,
    calculate_player_metrics,
    add_zone_groups,
    get_top_players_by_position_smart,
    compute_league_zone_averages,
    build_advanced_stats,
    build_league_leaders,
    clean_standings,
    filter_mvp_stats,
)


def _player_shots():
    return pd.DataFrame({
        "SHOT_ZONE_BASIC": ["Restricted Area", "Above the Break 3", "Mid-Range"],
        "SHOT_ZONE_AREA": ["Center(C)", "Center(C)", "Left Side(L)"],
        "SHOT_ZONE_RANGE": ["Less Than 8 ft.", "24+ ft.", "16-24 ft."],
        "SHOT_MADE_FLAG": [1, 0, 1],
        "SHOT_TYPE": ["2PT Field Goal", "3PT Field Goal", "2PT Field Goal"],
        "LOC_X": [0, 50, -120],
        "LOC_Y": [10, 250, 150],
    })


def _league_avg():
    return pd.DataFrame({
        "SHOT_ZONE_BASIC": ["Restricted Area", "Above the Break 3", "Mid-Range"],
        "SHOT_ZONE_AREA": ["Center(C)", "Center(C)", "Left Side(L)"],
        "SHOT_ZONE_RANGE": ["Less Than 8 ft.", "24+ ft.", "16-24 ft."],
        "FG_PCT": [0.65, 0.36, 0.42],
    })


def test_process_player_data_adds_columns():
    df = process_player_data(_player_shots(), _league_avg())
    for col in ["LEAGUE_FG_PCT", "RELATIVE_EFFICIENCY", "SHOT_VALUE", "GSAA", "x", "y", "made"]:
        assert col in df.columns
    # 3PT shot gets SHOT_VALUE 3
    assert df.loc[df["SHOT_TYPE"] == "3PT Field Goal", "SHOT_VALUE"].iloc[0] == 3
    # coordinate transform divides by 10
    assert df["x"].iloc[1] == 5.0


def test_calculate_player_metrics():
    df = process_player_data(_player_shots(), _league_avg())
    m = calculate_player_metrics(df)
    assert m["attempts"] == 3
    assert m["makes"] == 2
    assert m["threes_attempted"] == 1
    assert m["fg_pct"] == 2 / 3


def test_calculate_metrics_empty_is_safe():
    empty = process_player_data(_player_shots().iloc[0:0], _league_avg())
    m = calculate_player_metrics(empty)
    assert m["attempts"] == 0
    assert m["fg_pct"] == 0


def test_add_zone_groups():
    df = add_zone_groups(_player_shots())
    assert list(df["zone_group"]) == ["Rim", "3-Point", "Mid-Range"]


def test_compute_league_zone_averages():
    shots = pd.DataFrame({
        "SHOT_ZONE_BASIC": ["Rim", "Rim", "Rim", "Mid-Range"],
        "SHOT_MADE_FLAG": [1, 0, 1, 1],
    })
    out = compute_league_zone_averages(shots)
    rim = out[out["SHOT_ZONE_BASIC"] == "Rim"].iloc[0]
    assert rim["league_fg_pct"] == 2 / 3
    assert rim["league_freq_pct"] == 3 / 4


def _base_df():
    return pd.DataFrame({
        "PLAYER_ID": [1, 2],
        "PLAYER_NAME": ["Big Man", "Point Guard"],
        "TEAM_ABBREVIATION": ["AAA", "BBB"],
        "GP": [40, 40],
        "MIN": [30, 30],
        "AST": [1.0, 8.0],
        "REB": [11.0, 3.0],
        "BLK": [2.0, 0.2],
        "FG3A": [0.5, 7.0],
        "FGA": [10.0, 15.0],
        "PTS": [20.0, 18.0],
        "STL": [0.5, 1.5],
        "TOV": [2.0, 3.0],
    })


def test_build_advanced_stats_filters_and_engineers():
    base = _base_df()
    adv = pd.DataFrame({
        "PLAYER_ID": [1, 2],
        "PLAYER_NAME": ["Big Man", "Point Guard"],
        "USG_PCT": [0.25, 0.28],
        "TS_PCT": [0.6, 0.58],
        "AST_PCT": [0.1, 0.35],
        "REB_PCT": [0.2, 0.05],
        "PACE": [99.0, 101.0],
    })
    out = build_advanced_stats(base, adv)
    assert "3P_AR" in out.columns
    assert len(out) == 2  # both pass MIN_GAMES/MIN_MINUTES


def test_build_league_leaders_assigns_positions():
    base = _base_df()
    adv = pd.DataFrame({
        "PLAYER_ID": [1, 2],
        "TS_PCT": [0.6, 0.58], "USG_PCT": [0.25, 0.28],
        "AST_TO": [0.5, 2.5], "AST_RATIO": [10, 30],
        "OFF_RATING": [110, 115], "DEF_RATING": [108, 112],
        "NET_RATING": [2, 3], "PIE": [0.1, 0.12],
    })
    per100 = pd.DataFrame({
        "PLAYER_ID": [1, 2], "AST": [2, 12], "TOV": [3, 4],
        "PTS": [28, 26], "STL": [1, 2], "BLK": [3, 0.3],
    })
    out = build_league_leaders(base, adv, per100)
    positions = dict(zip(out["PLAYER_NAME"], out["POSITION_GROUP"]))
    assert positions["Big Man"] == "Center"
    assert positions["Point Guard"] == "Guard"
    assert "AST_PER100" in out.columns


def test_clean_standings_builds_full_name_and_abbrev():
    raw = pd.DataFrame({
        "TeamID": [1], "TeamCity": ["Denver"], "TeamName": ["Nuggets"],
        "TeamSlug": ["DEN"], "WINS": [50], "LOSSES": [32], "WinPCT": [0.61],
    })
    out = clean_standings(raw)
    assert out["TEAM_FULL"].iloc[0] == "Denver Nuggets"
    assert "TEAM_ABBREVIATION" in out.columns


def test_filter_mvp_stats_drops_low_sample():
    totals = pd.DataFrame({"GP": [40, 5], "MIN": [1200, 50], "PTS": [800, 60]})
    pergame = pd.DataFrame({"GP": [40, 5], "MIN": [30, 8], "PTS": [20, 12]})
    ft, fp = filter_mvp_stats(totals, pergame)
    assert len(ft) == 1
    assert len(fp) == 1


def test_top_players_by_position_scoring():
    df = pd.DataFrame({
        "POSITION_GROUP": ["Guard", "Guard"],
        "PLAYER_NAME": ["A", "B"], "TEAM_ABBREVIATION": ["X", "Y"],
        "USG_PCT": [0.30, 0.20], "TS_PCT": [0.60, 0.50], "PTS": [28, 18],
        "AST_PER100": [8, 5], "AST_TO": [2.0, 1.5],
        "NET_RATING": [4, 1], "OFF_RATING": [115, 110], "DEF_RATING": [111, 109],
    })
    res = get_top_players_by_position_smart(df, "scoring", n=2)
    assert list(res["Guard"]["Player"]) == ["A", "B"]
