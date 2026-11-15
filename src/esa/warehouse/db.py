"""Connection to the local DuckDB warehouse file."""

import duckdb

from esa.settings import Settings


def connect(settings: Settings) -> duckdb.DuckDBPyConnection:
    """Open the warehouse. Makes the lake directory if it is absent."""
    settings.lake_dir.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(str(settings.duckdb_path))
