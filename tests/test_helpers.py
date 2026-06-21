"""Player matching and stat helpers. Uses nba_api's bundled static player list
(local JSON, no network)."""

import pandas as pd

from utils.helpers import get_player_id, normalize_metric, safe_val, estimate_position


def test_get_player_id_exact_match():
    pid, name, msg = get_player_id("LeBron James")
    assert pid is not None
    assert name == "LeBron James"
    assert msg is None


def test_get_player_id_fuzzy_typo():
    pid, name, msg = get_player_id("lebron jame")
    assert pid is not None
    assert name == "LeBron James"
    assert msg is not None  # correction message present


def test_get_player_id_unknown_returns_none():
    pid, name, msg = get_player_id("zzz not a player zzz")
    assert pid is None
    assert name is None


def test_normalize_metric_clamps():
    assert normalize_metric(50, 0, 100) == 50
    assert normalize_metric(-5, 0, 100) == 0
    assert normalize_metric(150, 0, 100) == 100


def test_safe_val_handles_nan():
    assert safe_val(float("nan"), default=7) == 7
    assert safe_val(3.5) == 3.5


def test_estimate_position_center_and_guard():
    center = pd.Series({"AST": 1, "REB": 11, "BLK": 2, "FG3A": 0.5, "FGA": 8})
    guard = pd.Series({"AST": 8, "REB": 3, "BLK": 0.2, "FG3A": 7, "FGA": 15})
    assert estimate_position(center) == "Center"
    assert estimate_position(guard) == "Guard"
