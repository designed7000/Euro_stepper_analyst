"""Command line entry point for the esa platform."""

from collections.abc import Callable
from datetime import datetime, timedelta
from importlib.metadata import version as package_version

import typer

from esa.clock import utc_now
from esa.load.archive import fetch_index
from esa.load.nba import fetch_game_log, fetch_player_index, fetch_shots
from esa.load.raw_store import GAME_LOG, PLAYER_INDEX, SHOTS
from esa.load.run import run_backfill, run_daily_load
from esa.seasons import season_label, season_start_year
from esa.settings import Settings
from esa.warehouse import run_log
from esa.warehouse.build import build_warehouse
from esa.warehouse.db import connect

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main() -> None:
    """Euro Stepper Analyst data platform."""


def _fail(exc: Exception) -> typer.Exit:
    typer.echo(f"error: {exc}", err=True)
    return typer.Exit(code=1)


def _load(settings: Settings, now: datetime, backfill: bool) -> None:
    if backfill:
        run_backfill(settings, now)
    else:
        run_daily_load(settings, now)


def _build(settings: Settings) -> None:
    """Build the warehouse and record the run, with its own start and finish time.

    The run log connection is closed during the build: dbt runs in its own
    process, and DuckDB permits only one process to write to a file.
    """
    con = connect(settings)
    try:
        run_log.ensure(con)
        run_id = run_log.start(con, "build", None, utc_now())
    finally:
        con.close()

    error: Exception | None = None
    try:
        build_warehouse(settings)
    except Exception as exc:  # noqa: BLE001 - any failure is recorded, then raised again
        error = exc

    con = connect(settings)
    try:
        run_log.finish(
            con,
            run_id,
            status="failed" if error else "ok",
            last_game_date=None,
            message=str(error) if error else None,
            now=utc_now(),
        )
    finally:
        con.close()
    if error:
        raise error


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(f"esa {package_version('esa')}")


@app.command()
def load(
    backfill: bool = typer.Option(False, "--backfill", help="Load the full history."),
) -> None:
    """Get new data from the NBA and write raw files."""
    try:
        _load(Settings(), utc_now(), backfill)
    except Exception as exc:
        raise _fail(exc) from exc


@app.command()
def build() -> None:
    """Build the warehouse tables from the raw files and run the data tests."""
    try:
        _build(Settings())
    except Exception as exc:
        raise _fail(exc) from exc


@app.command()
def daily() -> None:
    """Do the daily run: load, then build. A failed step stops the run."""
    settings, now = Settings(), utc_now()
    try:
        _load(settings, now, backfill=False)
        _build(settings)
    except Exception as exc:
        raise _fail(exc) from exc


@app.command()
def probe() -> None:
    """Check that the archive index and the three NBA endpoints answer correctly."""
    settings = Settings()
    today = utc_now().date()
    season = season_label(season_start_year(today))
    window_start = today - timedelta(days=settings.lookback_days)
    checks: list[tuple[str, Callable[[], int]]] = [
        ("archive_index", lambda: len(fetch_index(settings))),
        (SHOTS, lambda: len(fetch_shots(season, window_start, today, settings))),
        (GAME_LOG, lambda: len(fetch_game_log(season, settings))),
        (PLAYER_INDEX, lambda: len(fetch_player_index(season, settings))),
    ]
    failed = False
    for name, check in checks:
        try:
            typer.echo(f"{name}: {check()} rows OK")
        except Exception as exc:  # noqa: BLE001 - report each source, then exit 1
            failed = True
            typer.echo(f"{name}: FAILED {exc}")
    if failed:
        raise typer.Exit(code=1)
