"""Fetchers for the three stats.nba.com endpoints that part 1 uses.

Each fetcher has a timeout, retries, a pause after the call, and a check
that the response has the columns that the warehouse needs.
"""

import time
from collections.abc import Callable
from datetime import date

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, playerindex, shotchartdetail

from esa.load.retry import LoadError, call_with_retry
from esa.settings import Settings

REGULAR_SEASON = "Regular Season"

SHOT_COLUMNS = {
    "GAME_ID", "GAME_EVENT_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "PERIOD",
    "MINUTES_REMAINING", "SECONDS_REMAINING", "ACTION_TYPE", "SHOT_TYPE",
    "SHOT_ZONE_BASIC", "SHOT_ZONE_AREA", "SHOT_ZONE_RANGE", "SHOT_DISTANCE",
    "LOC_X", "LOC_Y", "SHOT_ATTEMPTED_FLAG", "SHOT_MADE_FLAG", "GAME_DATE", "HTM", "VTM",
}  # fmt: skip
PLAYER_INDEX_COLUMNS = {
    "PERSON_ID", "PLAYER_FIRST_NAME", "PLAYER_LAST_NAME", "PLAYER_SLUG", "TEAM_ID",
    "POSITION", "HEIGHT", "FROM_YEAR", "TO_YEAR",
}  # fmt: skip
GAME_LOG_COLUMNS = {
    "TEAM_ID", "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_ID", "GAME_DATE", "MATCHUP",
}  # fmt: skip


class SchemaError(LoadError):
    """An endpoint response does not have a column that the warehouse needs."""


def _call(make: Callable[[], object], settings: Settings) -> pd.DataFrame:
    frame = call_with_retry(
        lambda: make().get_data_frames()[0], attempts=settings.call_attempts
    )
    time.sleep(settings.call_pause_s)
    return frame


def check_columns(frame: pd.DataFrame, required: set[str], source: str) -> pd.DataFrame:
    """Return the frame, or raise SchemaError that names each missing column."""
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SchemaError(f"{source} response has missing columns: {', '.join(missing)}")
    return frame


def _nba_date(d: date | None) -> str:
    return d.strftime("%m/%d/%Y") if d else ""


def fetch_shots(
    season: str, date_from: date | None, date_to: date | None, settings: Settings
) -> pd.DataFrame:
    """All regular season shots of the league in the window. None means no limit."""
    frame = _call(
        lambda: shotchartdetail.ShotChartDetail(
            team_id=0,
            player_id=0,
            context_measure_simple="FGA",
            season_nullable=season,
            season_type_all_star=REGULAR_SEASON,
            date_from_nullable=_nba_date(date_from),
            date_to_nullable=_nba_date(date_to),
            timeout=settings.call_timeout_s,
        ),
        settings,
    )
    return check_columns(frame, SHOT_COLUMNS, "ShotChartDetail")


def fetch_player_index(season: str, settings: Settings) -> pd.DataFrame:
    """All players, current and historical, with position and height."""
    frame = _call(
        lambda: playerindex.PlayerIndex(
            season=season, historical_nullable=1, timeout=settings.call_timeout_s
        ),
        settings,
    )
    return check_columns(frame, PLAYER_INDEX_COLUMNS, "PlayerIndex")


def fetch_game_log(season: str, settings: Settings) -> pd.DataFrame:
    """One row for each team in each regular season game."""
    frame = _call(
        lambda: leaguegamelog.LeagueGameLog(
            season=season,
            season_type_all_star=REGULAR_SEASON,
            player_or_team_abbreviation="T",
            timeout=settings.call_timeout_s,
        ),
        settings,
    )
    return check_columns(frame, GAME_LOG_COLUMNS, "LeagueGameLog")
