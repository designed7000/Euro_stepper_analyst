"""
Data repository: the single data interface the app imports.

Mirrors the signatures of the old data/api.py so the rest of the app is unaware
of the snapshot layer. Two access patterns:

  - League-wide datasets (leaders, standings, advanced stats, zone averages, MVP)
    are pre-built by the refresh job. The repository only reads them; a miss
    raises SnapshotMissing telling you to run the refresh. It never fetches these
    live, so user traffic is decoupled from the NBA API's rate limits.

  - Per-player shots can't be pre-fetched for every player, so they use
    read-through caching: check the store, and on a miss fetch once, save, return.
"""

from config import STORE_DIR
from data import fetch
from data.store import SnapshotStore

_store = SnapshotStore(STORE_DIR)


class SnapshotMissing(Exception):
    """Raised when a league-wide snapshot has not been built yet."""


def _require(dataset, season, params=None):
    """Load a league-wide snapshot or raise a clear 'run refresh' error."""
    df = _store.load(dataset, season, params)
    if df is None:
        raise SnapshotMissing(
            f"No '{dataset}' snapshot for {season}. "
            f"Run: python -m data.refresh --season {season}"
        )
    return df


# --- LEAGUE-WIDE (read-only, served from snapshots) ----------------------

def get_league_leaders(season):
    """League leaders with positions and advanced metrics."""
    return _require("league_leaders", season)


def get_standings(season):
    """Team standings with win percentage."""
    return _require("standings", season)


def get_advanced_stats(season):
    """Merged base + advanced player stats for similarity analysis."""
    return _require("advanced_stats", season)


def get_league_averages(season):
    """Per-zone league FG% and frequency."""
    return _require("league_zone_averages", season)


def get_mvp_stats(season):
    """Player totals and per-game frames for MVP scoring."""
    return _require("mvp_totals", season), _require("mvp_pergame", season)


# --- PER-PLAYER (read-through cached) -------------------------------------

def get_player_shots(player_id, season, clutch_only=False):
    """Shot chart data for a player, plus the league averages frame.

    Read-through cached: returns the stored snapshot if present, otherwise
    fetches once, saves, and returns. The league-averages frame depends only on
    (season, clutch_only), so it is stored once rather than per player.

    Returns:
        tuple: (player_shots_df, league_averages_df)
    """
    shot_params = {"player_id": player_id, "clutch_only": clutch_only}
    avg_params = {"clutch_only": clutch_only}

    shots = _store.load("player_shots", season, shot_params)
    league_avg = _store.load("player_shot_league_avg", season, avg_params)

    if shots is None or league_avg is None:
        shots, league_avg = fetch.get_player_shots(player_id, season, clutch_only)
        _store.save("player_shots", shots, season, shot_params)
        _store.save("player_shot_league_avg", league_avg, season, avg_params)

    return shots, league_avg
