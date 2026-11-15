import io
import tarfile
from datetime import date, datetime, timedelta

import pandas as pd
import pytest

from esa.load import nba
from esa.load.raw_store import GAME_LOG, PLAYER_INDEX, SHOTS, has_raw
from esa.load.retry import LoadError
from esa.load.run import run_backfill, run_daily_load
from esa.settings import Settings
from esa.warehouse import run_log
from esa.warehouse.db import connect

NOW = datetime(2026, 11, 15, 10, 0, 0)


def _frame(columns, **values):
    return pd.DataFrame({c: [values.get(c, "x")] for c in sorted(columns)})


def _tar_xz(member: str, frame: pd.DataFrame) -> bytes:
    data = frame.to_csv(index=False).encode()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        info = tarfile.TarInfo(member)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class Env:
    def __init__(self, tmp_path):
        self.settings = Settings(lake_dir=tmp_path, first_season=2025, call_pause_s=0.0)
        self.shots = _frame(nba.SHOT_COLUMNS, GAME_DATE="20261114")
        self.game_log = _frame(nba.GAME_LOG_COLUMNS)
        self.player_index = _frame(nba.PLAYER_INDEX_COLUMNS)
        self.index: dict[str, str] = {}
        self.shot_calls, self.game_log_calls, self.downloads = [], [], []
        self.game_log_error: Exception | None = None

    def fetch_shots(self, season, date_from, date_to, settings):
        self.shot_calls.append((season, date_from, date_to))
        return self.shots.copy()

    def fetch_game_log(self, season, settings):
        if self.game_log_error:
            raise self.game_log_error
        self.game_log_calls.append(season)
        return self.game_log.copy()

    def fetch_player_index(self, season, settings):
        return self.player_index.copy()

    def fetch_index(self, settings):
        return self.index

    def download(self, url, settings):
        self.downloads.append(url)
        year = url.rsplit("/", 1)[1].removesuffix(".tar.xz")
        archived = _frame(nba.SHOT_COLUMNS, GAME_DATE=f"{int(year) + 1}0410")
        return _tar_xz(f"shotdetail_{year}.csv", archived)


@pytest.fixture
def env(tmp_path, monkeypatch):
    e = Env(tmp_path)
    for name in ("fetch_shots", "fetch_game_log", "fetch_player_index", "fetch_index", "download"):
        monkeypatch.setattr(f"esa.load.run.{name}", getattr(e, name))
    return e


def test_daily_load_first_run_loads_full_season(env):
    last = run_daily_load(env.settings, NOW)
    assert env.shot_calls == [("2026-27", None, date(2026, 11, 15))]
    assert last == date(2026, 11, 14)
    for source in (SHOTS, GAME_LOG, PLAYER_INDEX):
        assert has_raw(source, "2026-27", env.settings.raw_dir)


def test_daily_load_second_run_uses_three_day_window(env):
    run_daily_load(env.settings, NOW)
    run_daily_load(env.settings, NOW + timedelta(days=1))
    assert env.shot_calls[1] == ("2026-27", date(2026, 11, 13), date(2026, 11, 16))


def test_day_with_no_games_keeps_last_good_date(env):
    run_daily_load(env.settings, NOW)
    env.shots = env.shots.iloc[0:0]
    tomorrow = NOW + timedelta(days=1)
    assert run_daily_load(env.settings, tomorrow) == date(2026, 11, 14)
    con = connect(env.settings)
    assert run_log.last_good_game_date(con, "2026-27") == date(2026, 11, 14)
    assert con.sql("select count(*) from ops.run_log where status = 'ok'").fetchall() == [(2,)]


def test_failure_on_second_call_writes_nothing_and_logs_failed(env):
    env.game_log_error = LoadError("timeout")
    with pytest.raises(LoadError):
        run_daily_load(env.settings, NOW)
    assert not has_raw(SHOTS, "2026-27", env.settings.raw_dir)
    con = connect(env.settings)
    assert con.sql("select status, message from ops.run_log").fetchall() == [("failed", "timeout")]
    assert run_log.last_good_game_date(con, "2026-27") is None


def test_backfill_uses_archive_when_listed_and_api_when_not(env):
    env.index = {"shotdetail_2025": "https://x/2025.tar.xz"}  # 2026 is not in the archive
    run_backfill(env.settings, NOW)
    assert env.downloads == ["https://x/2025.tar.xz"]
    assert env.shot_calls == [("2026-27", None, None)]
    assert sorted(env.game_log_calls) == ["2025-26", "2026-27"]
    archived = pd.read_parquet(next((env.settings.raw_dir / SHOTS / "2025-26").glob("*/*.parquet")))
    assert archived.loc[0, "_origin"] == "archive" and archived.loc[0, "GAME_DATE"] == "20260410"
    assert has_raw(PLAYER_INDEX, "2026-27", env.settings.raw_dir)


def test_backfill_second_run_skips_past_seasons(env):
    env.index = {"shotdetail_2025": "https://x/2025.tar.xz"}
    run_backfill(env.settings, NOW)
    run_backfill(env.settings, NOW + timedelta(hours=1))
    assert env.downloads == ["https://x/2025.tar.xz"]  # not downloaded again


def test_daily_load_after_backfill_uses_a_window(env):
    run_backfill(env.settings, NOW)
    env.shot_calls.clear()
    run_daily_load(env.settings, NOW + timedelta(days=1))
    assert env.shot_calls == [("2026-27", date(2026, 11, 13), date(2026, 11, 16))]
