"""The two load runs: the history backfill and the daily load."""

from collections.abc import Callable
from datetime import date, datetime

import pandas as pd

from esa.load.archive import download, fetch_index, read_shot_archive, shot_archive_name
from esa.load.nba import fetch_game_log, fetch_player_index, fetch_shots
from esa.load.raw_store import GAME_LOG, PLAYER_INDEX, SHOTS, has_raw, write_raw
from esa.load.retry import LoadError
from esa.load.window import plan_date_from
from esa.seasons import season_label, season_start_year
from esa.settings import Settings
from esa.warehouse import run_log
from esa.warehouse.db import connect


def _last_game_date(shots: pd.DataFrame) -> date | None:
    """Latest GAME_DATE (format YYYYMMDD) of a shot table, or None if it is empty."""
    if shots.empty:
        return None
    try:
        return pd.to_datetime(shots["GAME_DATE"].astype("string"), format="%Y%m%d").max().date()
    except ValueError as exc:
        raise LoadError(f"Shot GAME_DATE does not have the format YYYYMMDD: {exc}") from exc


def _logged_load[T](
    settings: Settings, season: str, now: datetime, body: Callable[[date | None], T]
) -> T:
    """Run a load body with a run_log row. The body gets the last good game date
    and returns (result, last game date)."""
    con = connect(settings)
    try:
        run_log.ensure(con)
        last_good = run_log.last_good_game_date(con, season)
        run_id = run_log.start(con, "load", season, now)
        try:
            result, last = body(last_good)
        except Exception as exc:
            run_log.finish(
                con, run_id, status="failed", last_game_date=None, message=str(exc), now=now
            )
            raise
        run_log.finish(con, run_id, status="ok", last_game_date=last, message=None, now=now)
        return result
    finally:
        con.close()


def run_daily_load(settings: Settings, now: datetime) -> date | None:
    """Load the recent days of the current season. Return the last game date."""
    today = now.date()
    season = season_label(season_start_year(today))

    def body(last_good: date | None):
        date_from = plan_date_from(last_good, today, settings.lookback_days)
        # Get all the tables before the first write, so a failed run writes nothing.
        shots = fetch_shots(season, date_from, today, settings)
        game_log = fetch_game_log(season, settings)
        player_index = fetch_player_index(season, settings)
        last = _last_game_date(shots) or last_good
        tag = "window" if date_from else "full"
        common = {"season": season, "origin": "api", "raw_dir": settings.raw_dir, "now": now}
        write_raw(shots, source=SHOTS, tag=tag, **common)
        write_raw(game_log, source=GAME_LOG, tag="full", **common)
        write_raw(player_index, source=PLAYER_INDEX, tag="full", **common)
        return last, last

    return _logged_load(settings, season, now, body)


def run_backfill(settings: Settings, now: datetime) -> None:
    """Load the history from settings.first_season to the current season.

    A past season that already has raw files is skipped, so a second run
    continues an interrupted backfill and does not download again.
    """
    raw_dir = settings.raw_dir
    current_year = season_start_year(now.date())
    current = season_label(current_year)

    def body(last_good: date | None):
        index = fetch_index(settings)
        last = last_good
        for year in range(settings.first_season, current_year + 1):
            season = season_label(year)
            past = year < current_year
            shots = game_log = None
            origin = "api"
            # Get the two tables of a season before the first write of that season.
            if not (past and has_raw(SHOTS, season, raw_dir)):
                name = shot_archive_name(year)
                if name in index:
                    shots = read_shot_archive(download(index[name], settings), name)
                    origin = "archive"
                else:
                    shots = fetch_shots(season, None, None, settings)
            if not (past and has_raw(GAME_LOG, season, raw_dir)):
                game_log = fetch_game_log(season, settings)
            if shots is not None:
                if not past:
                    last = _last_game_date(shots) or last
                tag = "archive" if origin == "archive" else "full"
                write_raw(
                    shots, source=SHOTS, season=season, origin=origin, tag=tag,
                    raw_dir=raw_dir, now=now,
                )  # fmt: skip
            if game_log is not None:
                write_raw(
                    game_log, source=GAME_LOG, season=season, origin="api", tag="full",
                    raw_dir=raw_dir, now=now,
                )  # fmt: skip
        player_index = fetch_player_index(current, settings)
        write_raw(
            player_index, source=PLAYER_INDEX, season=current, origin="api", tag="full",
            raw_dir=raw_dir, now=now,
        )  # fmt: skip
        return None, last

    _logged_load(settings, current, now, body)
