"""The ops.run_log table: one row for each run of a CLI command."""

from datetime import date, datetime
from typing import Literal

import duckdb

Status = Literal["ok", "failed"]


def ensure(con: duckdb.DuckDBPyConnection) -> None:
    """Make the table if it is absent."""
    con.execute("create schema if not exists ops")
    con.execute("create sequence if not exists ops.run_id_seq")
    con.execute(
        """
        create table if not exists ops.run_log (
            run_id bigint primary key,
            command varchar not null,
            season varchar,
            started_at timestamp not null,
            finished_at timestamp,
            status varchar not null,
            last_game_date date,
            message varchar
        )
        """
    )


def start(con: duckdb.DuckDBPyConnection, command: str, season: str | None, now: datetime) -> int:
    """Record the start of a run and return its id."""
    # fetchall, not fetchone: DuckDB keeps a result open until all rows are read.
    rows = con.execute(
        """
        insert into ops.run_log (run_id, command, season, started_at, status)
        values (nextval('ops.run_id_seq'), ?, ?, ?, 'running')
        returning run_id
        """,
        [command, season, now],
    ).fetchall()
    return rows[0][0]


def finish(
    con: duckdb.DuckDBPyConnection,
    run_id: int,
    *,
    status: Status,
    last_game_date: date | None,
    message: str | None,
    now: datetime,
) -> None:
    """Record the result of a run."""
    con.execute(
        """
        update ops.run_log
        set status = ?, last_game_date = ?, message = ?, finished_at = ?
        where run_id = ?
        """,
        [status, last_game_date, message, now, run_id],
    )


def last_good_game_date(con: duckdb.DuckDBPyConnection, season: str) -> date | None:
    """Latest game date of a successful load of the season."""
    # fetchall, not fetchone: DuckDB keeps a result open until all rows are read.
    rows = con.execute(
        """
        select max(last_game_date) from ops.run_log
        where command = 'load' and status = 'ok' and season = ?
        """,
        [season],
    ).fetchall()
    return rows[0][0]
