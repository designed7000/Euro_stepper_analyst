"""Build the warehouse tables with dbt."""

import os
import subprocess
import sys
from pathlib import Path

from esa.load.raw_store import GAME_LOG, PLAYER_INDEX, SHOTS
from esa.settings import REPO_ROOT, Settings

DBT_DIR = REPO_ROOT / "warehouse"
ERROR_TAIL_LINES = 25


class BuildError(Exception):
    """The warehouse build failed."""


class EmptyLakeError(BuildError):
    """A source has no raw file, so there is nothing to build."""


def build_warehouse(settings: Settings) -> None:
    """Run `dbt build`: all models and all data tests. Raise BuildError on any failure.

    dbt runs in its own process. dbt-duckdb keeps its connection open until the
    process ends, and DuckDB permits only one configuration of a file in a process.
    """
    for source in (SHOTS, PLAYER_INDEX, GAME_LOG):
        if not any((settings.raw_dir / source).glob("*/*/*.parquet")):
            raise EmptyLakeError(f"No raw files for {source}. Run: esa load --backfill")

    settings.lake_dir.mkdir(parents=True, exist_ok=True)
    dbt = Path(sys.executable).parent / "dbt"
    result = subprocess.run(
        [str(dbt), "--quiet", "--no-use-colors", "build",
         "--project-dir", str(DBT_DIR), "--profiles-dir", str(DBT_DIR)],
        # The dbt profile and the dbt sources read these two variables.
        env=os.environ | {
            "ESA_DUCKDB_PATH": str(settings.duckdb_path),
            "ESA_RAW_DIR": str(settings.raw_dir),
        },
        capture_output=True,
        check=False,  # the return code is examined below
        text=True,
    )  # fmt: skip
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).strip().splitlines()[-ERROR_TAIL_LINES:])
        raise BuildError(f"dbt build failed:\n{tail}")
