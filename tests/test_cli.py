import time
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from typer.testing import CliRunner

from esa.cli import app
from esa.load.nba import SchemaError
from esa.load.retry import LoadError
from esa.settings import Settings
from esa.warehouse.db import connect


@pytest.fixture(autouse=True)
def lake_in_tmp(tmp_path, monkeypatch):
    """No CLI test can touch the real lake."""
    monkeypatch.setenv("ESA_LAKE_DIR", str(tmp_path))
    monkeypatch.setenv("ESA_CALL_PAUSE_S", "0")


def _raise(exc):
    def fn(*args, **kwargs):
        raise exc

    return fn


def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.startswith("esa ")


def test_daily_stops_before_build_when_load_fails(monkeypatch):
    built = []
    monkeypatch.setattr("esa.cli.run_daily_load", _raise(LoadError("timeout")))
    monkeypatch.setattr("esa.cli.build_warehouse", lambda s: built.append(1))
    result = CliRunner().invoke(app, ["daily"])
    assert result.exit_code == 1 and "timeout" in result.output and built == []


def test_daily_runs_load_then_build(monkeypatch):
    order = []
    monkeypatch.setattr("esa.cli.run_daily_load", lambda s, now: order.append("load"))
    monkeypatch.setattr("esa.cli.build_warehouse", lambda s: order.append("build"))
    assert CliRunner().invoke(app, ["daily"]).exit_code == 0 and order == ["load", "build"]


def test_daily_build_row_has_its_own_start_and_finish_time(monkeypatch):
    times = iter(datetime(2026, 11, 22, 10, minute) for minute in range(10))
    monkeypatch.setattr("esa.cli.utc_now", lambda: next(times))
    monkeypatch.setattr("esa.cli.run_daily_load", lambda s, now: None)
    monkeypatch.setattr("esa.cli.build_warehouse", lambda s: None)
    assert CliRunner().invoke(app, ["daily"]).exit_code == 0
    con = connect(Settings())
    # 10:00 is the start of the load, so the build starts at 10:01
    assert con.sql("select started_at, finished_at from ops.run_log").fetchall() == [
        (datetime(2026, 11, 22, 10, 1), datetime(2026, 11, 22, 10, 2))
    ]


def test_build_records_a_run_log_row(monkeypatch):
    monkeypatch.setattr("esa.cli.build_warehouse", lambda s: None)
    assert CliRunner().invoke(app, ["build"]).exit_code == 0
    con = connect(Settings())
    assert con.sql("select command, status from ops.run_log").fetchall() == [("build", "ok")]


def test_failed_build_is_recorded_and_exits_1(monkeypatch):
    monkeypatch.setattr("esa.cli.build_warehouse", _raise(RuntimeError("dbt broke")))
    result = CliRunner().invoke(app, ["build"])
    assert result.exit_code == 1 and "dbt broke" in result.output
    con = connect(Settings())
    assert con.sql("select status, message from ops.run_log").fetchall() == [
        ("failed", "dbt broke")
    ]


def test_load_backfill_flag(monkeypatch):
    called = []
    monkeypatch.setattr("esa.cli.run_backfill", lambda s, now: called.append("backfill"))
    monkeypatch.setattr("esa.cli.run_daily_load", lambda s, now: called.append("daily"))
    CliRunner().invoke(app, ["load", "--backfill"])
    assert called == ["backfill"]


@pytest.fixture
def new_york_time(monkeypatch):
    """A local zone that is never UTC, so a test can see local time used as UTC."""
    monkeypatch.setenv("TZ", "America/New_York")
    time.tzset()
    yield
    monkeypatch.undo()
    time.tzset()


def test_load_uses_utc_time(monkeypatch, new_york_time):
    # Local time goes back one hour in autumn, so a later load could look older.
    times = []
    monkeypatch.setattr("esa.cli.run_daily_load", lambda s, now: times.append(now))
    CliRunner().invoke(app, ["load"])
    utc = datetime.now(UTC).replace(tzinfo=None)
    assert times[0].tzinfo is None and abs(times[0] - utc) < timedelta(seconds=5)


def test_load_without_flag_is_the_daily_load(monkeypatch):
    called = []
    monkeypatch.setattr("esa.cli.run_backfill", lambda s, now: called.append("backfill"))
    monkeypatch.setattr("esa.cli.run_daily_load", lambda s, now: called.append("daily"))
    CliRunner().invoke(app, ["load"])
    assert called == ["daily"]


def test_probe_reports_each_source_and_fails_on_error(monkeypatch):
    ok = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr("esa.cli.fetch_index", lambda s: {"shotdetail_2015": "u"})
    monkeypatch.setattr("esa.cli.fetch_shots", lambda *a: ok)
    monkeypatch.setattr("esa.cli.fetch_game_log", lambda *a: ok)
    monkeypatch.setattr("esa.cli.fetch_player_index", _raise(SchemaError("missing: HEIGHT")))
    result = CliRunner().invoke(app, ["probe"])
    assert result.exit_code == 1
    assert "nba_shots" in result.output and "missing: HEIGHT" in result.output


def test_probe_succeeds_when_all_sources_answer(monkeypatch):
    ok = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr("esa.cli.fetch_index", lambda s: {"shotdetail_2015": "u"})
    for name in ("fetch_shots", "fetch_game_log", "fetch_player_index"):
        monkeypatch.setattr(f"esa.cli.{name}", lambda *a: ok)
    result = CliRunner().invoke(app, ["probe"])
    assert result.exit_code == 0 and result.output.count("OK") == 4
