from datetime import date, datetime

from esa.settings import Settings
from esa.warehouse import run_log
from esa.warehouse.db import connect

T = datetime(2026, 11, 15, 10, 0, 0)


def test_last_good_date_ignores_failed_runs_and_other_seasons(tmp_path):
    con = connect(Settings(lake_dir=tmp_path))
    run_log.ensure(con)
    a = run_log.start(con, "load", "2026-27", T)
    run_log.finish(con, a, status="ok", last_game_date=date(2026, 11, 13), message=None, now=T)
    b = run_log.start(con, "load", "2026-27", T)
    run_log.finish(
        con, b, status="failed", last_game_date=date(2026, 11, 14), message="timeout", now=T
    )
    c = run_log.start(con, "load", "2025-26", T)
    run_log.finish(con, c, status="ok", last_game_date=date(2026, 4, 12), message=None, now=T)
    assert run_log.last_good_game_date(con, "2026-27") == date(2026, 11, 13)
    assert run_log.last_good_game_date(con, "2024-25") is None


def test_ensure_is_idempotent(tmp_path):
    con = connect(Settings(lake_dir=tmp_path))
    run_log.ensure(con)
    run_log.ensure(con)
    assert con.sql("select count(*) from ops.run_log").fetchall() == [(0,)]


def test_connect_makes_the_lake_dir(tmp_path):
    settings = Settings(lake_dir=tmp_path / "new" / "lake")
    connect(settings).close()
    assert settings.duckdb_path.exists()
