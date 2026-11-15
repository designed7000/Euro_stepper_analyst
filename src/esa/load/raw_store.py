"""Immutable raw landing zone: one Parquet file for each load of each source."""

from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd

SHOTS = "nba_shots"
PLAYER_INDEX = "nba_player_index"
GAME_LOG = "nba_game_log"

Origin = Literal["api", "archive"]


def write_raw(
    df: pd.DataFrame,
    *,
    source: str,
    season: str,
    origin: Origin,
    tag: str,
    raw_dir: Path,
    now: datetime,
) -> Path | None:
    """Write one raw file and return its path. An empty table writes nothing.

    Each source column is stored as a string, so that files from different
    origins have the same types. Staging models do the casts.
    """
    if df.empty:
        return None
    path = raw_dir / source / season / now.date().isoformat() / f"{tag}_{now:%H%M%S}.parquet"
    if path.exists():
        raise FileExistsError(f"Raw files do not change: {path}")
    out = df.astype("string")
    out["_season"] = season
    out["_origin"] = origin
    out["_loaded_at"] = pd.Timestamp(now)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return path


def has_raw(source: str, season: str, raw_dir: Path) -> bool:
    """Return True if the source has one raw file or more for the season."""
    return any((raw_dir / source / season).glob("*/*.parquet"))
