"""MVP ladder / DNA Production Index."""

import pandas as pd

from analysis.awards import AwardTracker


def _totals():
    return pd.DataFrame({
        "PTS": [2000, 1500], "REB": [400, 800], "AST": [600, 200],
        "STL": [100, 60], "BLK": [40, 150], "TOV": [250, 180],
    })


def _pergame():
    return pd.DataFrame({
        "PLAYER_NAME": ["Star Guard", "Big Center"],
        "TEAM_ABBREVIATION": ["AAA", "BBB"],
        "TEAM_ID": [10, 20],
        "PTS": [28.0, 22.0], "REB": [6.0, 12.0], "AST": [9.0, 3.0],
        "STL": [1.5, 0.8], "BLK": [0.5, 2.2], "TOV": [3.2, 2.4],
    })


def _standings():
    return pd.DataFrame({
        "TeamID": [10, 20], "WINS": [55, 30], "LOSSES": [27, 52],
        "WinPCT": [0.67, 0.37],
    })


def test_scarcity_weights_rarer_stats_weighted_higher():
    tracker = AwardTracker()
    w = tracker.calculate_scarcity_weights(_totals())
    # Blocks are rarer than rebounds in the totals -> higher base exchange rate,
    # even after the 0.6 dampening modifier here blocks total (190) << reb (1200).
    assert w["BLK"] > w["REB"]
    assert w["PTS"] == 1.0


def test_mvp_ladder_ranks_and_columns():
    tracker = AwardTracker(_pergame(), _standings())
    ladder = tracker.calculate_mvp_ladder(top_n=2)
    assert len(ladder) == 2
    assert list(ladder["Rank"]) == [1, 2]
    for col in ["MVP_SCORE", "RAW_VALUE", "RECORD", "WIN_PCT"]:
        assert col in ladder.columns
    # Record string is built from wins-losses
    assert ladder["RECORD"].iloc[0] in ("55-27", "30-52")


def test_mvp_ladder_empty_without_data():
    assert AwardTracker().calculate_mvp_ladder().empty
