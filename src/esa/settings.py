"""Runtime settings. Each field can be set with an ESA_<NAME> environment variable."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ESA_")

    lake_dir: Path = REPO_ROOT / "lake"
    first_season: int = 2015
    lookback_days: int = 3
    call_pause_s: float = 1.0
    call_timeout_s: int = 60
    call_attempts: int = 4

    @property
    def raw_dir(self) -> Path:
        return self.lake_dir / "raw"

    @property
    def duckdb_path(self) -> Path:
        return self.lake_dir / "warehouse.duckdb"
