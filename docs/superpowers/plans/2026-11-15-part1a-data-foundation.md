# Part 1A: Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `esa load` and `esa build` fill a local DuckDB warehouse with tested `dim_player`, `dim_team`, `dim_game`, and `fct_shot` tables from NBA shot data.

**Architecture:** A Python package (`esa`) loads NBA data to immutable Parquet files in `lake/raw/`. A dbt project (`warehouse/`) builds `staging` and `core` tables in one DuckDB file. A Typer CLI runs the steps and records each run in a `run_log` table.

**Tech Stack:** Python 3.12, uv, nba_api, pandas, pyarrow, DuckDB, dbt-duckdb, Typer, pydantic-settings, pytest, ruff, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-11-15-platform-skeleton-shot-dna-design.md` (sections 5.1, 5.2, 5.4, 8, 9).

**Sequence of part 1 plans:** 1A data foundation (this plan) → 1B module runner and Shot DNA model → 1C publish and API → 1D web, hosting, and schedule. Plans 1B to 1D are written after 1A is complete, because they use its real table shapes.

**Not in this plan (spec items that later plans own):** the `run` and `publish` commands (1B, 1C), the port of the zone mapping from `legacy/config.py` (1B), the port of the court geometry from `legacy/charts/court.py` (1D), and the `launchd` schedule (1D).

## Global Constraints

- All calls to stats.nba.com come from the owner's computer. Tests and CI make no network call.
- No raw data in Git. `lake/` is in `.gitignore`.
- Raw files do not change after they are written.
- Each run loads the last 3 days again.
- Regular season games only.
- Published seasons start at 2015-16 (`first_season = 2015`).
- Each NBA call has a timeout, retries with backoff, and a pause between calls.
- Each command is idempotent. A second run on the same day gives the same warehouse.
- If a step fails, the run stops and the next step does not start.
- Python 3.12. Add dependencies with `uv add` so that `uv.lock` pins the versions.
- Downloads: this plan downloads shot archives from `github.com/shufinskiy/nba_data` (Apache-2.0). Read them in memory. Do not extract them to disk.

## Review Focus

1. **Endpoint schema drift.** An NBA endpoint returns a table with a required column missing. Expected: the load fails with an error that names the missing columns, and writes no raw file for that call. (Task 4)
2. **Day with no games.** The shot call returns zero rows. Expected: the load succeeds, writes no shot file, and the last good game date does not change. (Tasks 3, 7)
3. **Game ID with no leading zeros.** The archive CSV gives `22600001` and the API gives `0022600001`. Expected: one game, ID `0022600001`. (Task 8)
4. **Same shot loaded two times with different values.** A stat correction changes a shot after the first load. Expected: one row, and the latest load wins. (Task 8)
5. **Failure in the middle of a load.** The second NBA call fails after all retries. Expected: `run_log` status `failed`, `esa daily` does not build, and the next run starts from the last good game date. (Tasks 7, 9)

## File Structure

```
pyproject.toml, uv.lock, .python-version (3.12), .gitignore
legacy/                      old Streamlit app, moved unchanged
src/esa/
  settings.py                Settings (paths, limits)
  seasons.py                 season label helpers
  cli.py                     Typer commands
  load/retry.py              call_with_retry, LoadError
  load/raw_store.py          write_raw, has_raw, source names
  load/nba.py                three endpoint fetchers, SchemaError
  load/archive.py            archive index and in-memory reader
  load/window.py             plan_date_from
  load/run.py                run_backfill, run_daily_load
  warehouse/db.py            connect
  warehouse/run_log.py       run_log table functions
  warehouse/build.py         build_warehouse (calls dbt)
warehouse/                   dbt project: dbt_project.yml, profiles.yml, macros/, models/staging/, models/core/
tests/                       one test file for each source file
.github/workflows/ci.yml
```

---

### Task 1: Project skeleton and legacy move

**Files:**
- Move: `app.py`, `config.py`, `requirements.txt`, `analysis/`, `charts/`, `data/`, `utils/`, `.devcontainer/` → `legacy/`
- Create: `pyproject.toml`, `src/esa/__init__.py`, `src/esa/cli.py`, `tests/test_cli.py`, `.github/workflows/ci.yml`
- Modify: `.gitignore`, `.python-version`

**Interfaces:**
- Produces: `esa.cli.app` (a `typer.Typer`), console script `esa`.

- [ ] **Step 1: Make the branch and move the old code**

```bash
git checkout -b platform-v2
mkdir legacy
git mv app.py config.py requirements.txt analysis charts data utils .devcontainer legacy/
```

- [ ] **Step 2: Make the package**

Set `.python-version` to `3.12`. Write `pyproject.toml` with project name `esa`, `requires-python = ">=3.12,<3.13"`, a `src` layout, the script `esa = "esa.cli:app"`, pytest `testpaths = ["tests"]`, and ruff `line-length = 100`. Then:

```bash
uv add nba_api pandas pyarrow duckdb dbt-duckdb typer pydantic-settings
uv add --dev pytest ruff
```

Replace `.gitignore` with: `.venv`, `__pycache__`, `.DS_Store`, `lake/`, `warehouse/target/`, `warehouse/logs/`, `warehouse/dbt_packages/`, `.env`.

- [ ] **Step 3: Write the failing test** in `tests/test_cli.py`

```python
from typer.testing import CliRunner
from esa.cli import app

def test_version_command():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.output.startswith("esa ")
```

- [ ] **Step 4: Run it.** `uv run pytest tests/test_cli.py -v` → FAIL (`esa.cli` not found).

- [ ] **Step 5: Implement** `app = typer.Typer(no_args_is_help=True)` and the command `version()` in `src/esa/cli.py`. It prints `esa <package version>`.

- [ ] **Step 6: Run it.** `uv run pytest -v` → PASS. `uv run ruff check .` → no errors (exclude `legacy/` in the ruff config).

- [ ] **Step 7: Add CI** in `.github/workflows/ci.yml`: on push and pull request, `astral-sh/setup-uv`, `uv sync`, `uv run ruff check .`, `uv run pytest`.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "chore: new package skeleton, move Streamlit app to legacy"
```

---

### Task 2: Settings and season helpers

**Files:**
- Create: `src/esa/settings.py`, `src/esa/seasons.py`, `tests/test_settings.py`, `tests/test_seasons.py`

**Interfaces:**
- Produces:
  - `Settings(BaseSettings)` with env prefix `ESA_` and fields `lake_dir: Path` (default `<repo root>/lake`), `first_season: int = 2015`, `lookback_days: int = 3`, `call_pause_s: float = 1.0`, `call_timeout_s: int = 60`, `call_attempts: int = 4`; properties `raw_dir -> Path` (`lake_dir/"raw"`) and `duckdb_path -> Path` (`lake_dir/"warehouse.duckdb"`).
  - `REPO_ROOT: Path` in `settings.py`.
  - `season_label(start_year: int) -> str`, `season_start_year(d: date) -> int`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_seasons.py
from datetime import date
from esa.seasons import season_label, season_start_year

def test_season_label():
    assert season_label(2015) == "2015-16"
    assert season_label(1999) == "1999-00"
    assert season_label(2026) == "2026-27"

def test_season_start_year():
    assert season_start_year(date(2026, 11, 15)) == 2026
    assert season_start_year(date(2027, 4, 1)) == 2026
    assert season_start_year(date(2026, 9, 30)) == 2025   # offseason maps to the last season
```

```python
# tests/test_settings.py
from pathlib import Path
from esa.settings import Settings

def test_paths_derive_from_lake_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ESA_LAKE_DIR", str(tmp_path))
    s = Settings()
    assert s.raw_dir == tmp_path / "raw"
    assert s.duckdb_path == tmp_path / "warehouse.duckdb"
    assert (s.first_season, s.lookback_days) == (2015, 3)
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_seasons.py tests/test_settings.py -v` → FAIL.
- [ ] **Step 3: Implement** the interfaces. The season changes on 1 October.
- [ ] **Step 4: Run.** Same command → PASS.
- [ ] **Step 5: Commit.** `git commit -am "feat: settings and season helpers"` (after `git add`).

---

### Task 3: Raw store

**Files:**
- Create: `src/esa/load/__init__.py`, `src/esa/load/raw_store.py`, `tests/load/test_raw_store.py`

**Interfaces:**
- Produces:
  - Constants `SHOTS = "nba_shots"`, `PLAYER_INDEX = "nba_player_index"`, `GAME_LOG = "nba_game_log"`.
  - `write_raw(df: pd.DataFrame, *, source: str, season: str, origin: Literal["api", "archive"], tag: str, raw_dir: Path, now: datetime) -> Path | None`
  - `has_raw(source: str, season: str, raw_dir: Path) -> bool`

Rules that the tests pin: the path is `raw_dir/<source>/<season>/<now.date()>/<tag>_<HHMMSS>.parquet`. Each source column is stored as a string, and nulls stay null. Three columns are added: `_season`, `_origin`, `_loaded_at`. An empty table writes no file and returns `None`. An existing path raises `FileExistsError`.

- [ ] **Step 1: Write the failing tests**

```python
from datetime import datetime
import pandas as pd, pytest
from esa.load.raw_store import write_raw, has_raw, SHOTS

NOW = datetime(2026, 11, 15, 10, 30, 0)

def _write(df, raw_dir, now=NOW):
    return write_raw(df, source=SHOTS, season="2026-27", origin="api", tag="w", raw_dir=raw_dir, now=now)

def test_path_and_string_columns(tmp_path):
    path = _write(pd.DataFrame({"GAME_ID": [22600001], "LOC_X": [-220], "NOTE": [None]}), tmp_path)
    assert path == tmp_path / "nba_shots" / "2026-27" / "2026-11-15" / "w_103000.parquet"
    out = pd.read_parquet(path)
    assert out.loc[0, "GAME_ID"] == "22600001" and out.loc[0, "LOC_X"] == "-220"
    assert pd.isna(out.loc[0, "NOTE"])
    assert out.loc[0, "_season"] == "2026-27" and out.loc[0, "_origin"] == "api"
    assert out.loc[0, "_loaded_at"] == pd.Timestamp(NOW)

def test_empty_table_writes_nothing(tmp_path):
    assert _write(pd.DataFrame({"GAME_ID": []}), tmp_path) is None
    assert has_raw(SHOTS, "2026-27", tmp_path) is False

def test_raw_files_do_not_change(tmp_path):
    df = pd.DataFrame({"GAME_ID": ["1"]})
    _write(df, tmp_path)
    with pytest.raises(FileExistsError):
        _write(df, tmp_path)
    assert has_raw(SHOTS, "2026-27", tmp_path) is True
```

- [ ] **Step 2: Run.** `uv run pytest tests/load/test_raw_store.py -v` → FAIL.
- [ ] **Step 3: Implement** the two functions.
- [ ] **Step 4: Run.** → PASS.
- [ ] **Step 5: Commit.** `feat: immutable raw store`.

---

### Task 4: Retry wrapper and NBA fetchers

**Files:**
- Create: `src/esa/load/retry.py`, `src/esa/load/nba.py`, `tests/load/test_retry.py`, `tests/load/test_nba.py`

**Interfaces:**
- Consumes: `Settings` (Task 2).
- Produces:
  - `class LoadError(Exception)`; `call_with_retry(fn: Callable[[], T], *, attempts: int, base_delay_s: float = 2.0, sleep: Callable[[float], None] = time.sleep) -> T`. The delay before retry `i` (from 0) is `base_delay_s * 2**i`. After the last attempt it raises `LoadError` from the last exception.
  - `class SchemaError(LoadError)`.
  - `fetch_shots(season: str, date_from: date | None, date_to: date | None, settings: Settings) -> pd.DataFrame`
  - `fetch_player_index(season: str, settings: Settings) -> pd.DataFrame`
  - `fetch_game_log(season: str, settings: Settings) -> pd.DataFrame`
  - Required column sets:
    - `SHOT_COLUMNS = {"GAME_ID","GAME_EVENT_ID","PLAYER_ID","PLAYER_NAME","TEAM_ID","PERIOD","MINUTES_REMAINING","SECONDS_REMAINING","ACTION_TYPE","SHOT_TYPE","SHOT_ZONE_BASIC","SHOT_ZONE_AREA","SHOT_ZONE_RANGE","SHOT_DISTANCE","LOC_X","LOC_Y","SHOT_ATTEMPTED_FLAG","SHOT_MADE_FLAG","GAME_DATE"}`
    - `PLAYER_INDEX_COLUMNS = {"PERSON_ID","PLAYER_FIRST_NAME","PLAYER_LAST_NAME","PLAYER_SLUG","TEAM_ID","POSITION","HEIGHT","FROM_YEAR","TO_YEAR"}`
    - `GAME_LOG_COLUMNS = {"TEAM_ID","TEAM_ABBREVIATION","TEAM_NAME","GAME_ID","GAME_DATE","MATCHUP"}`

Endpoint calls (all through `call_with_retry`, each followed by `sleep(settings.call_pause_s)`, each with `timeout=settings.call_timeout_s`):
- Shots: `shotchartdetail.ShotChartDetail(team_id=0, player_id=0, context_measure_simple="FGA", season_nullable=season, season_type_all_star="Regular Season", date_from_nullable=<MM/DD/YYYY or "">, date_to_nullable=<MM/DD/YYYY or "">)`, first data frame.
- Player index: `playerindex.PlayerIndex(season=season, historical_nullable=1)`, first data frame.
- Game log: `leaguegamelog.LeagueGameLog(season=season, season_type_all_star="Regular Season", player_or_team_abbreviation="T")`, first data frame.

- [ ] **Step 1: Write the failing tests**

```python
# tests/load/test_retry.py
import pytest
from esa.load.retry import call_with_retry, LoadError

def test_succeeds_on_third_attempt():
    calls, sleeps = [], []
    def fn():
        calls.append(1)
        if len(calls) < 3: raise TimeoutError("slow")
        return "ok"
    assert call_with_retry(fn, attempts=4, base_delay_s=2.0, sleep=sleeps.append) == "ok"
    assert sleeps == [2.0, 4.0]

def test_raises_load_error_after_last_attempt():
    def fn(): raise TimeoutError("slow")
    with pytest.raises(LoadError):
        call_with_retry(fn, attempts=2, base_delay_s=0.0, sleep=lambda s: None)
```

```python
# tests/load/test_nba.py
from datetime import date
import pandas as pd, pytest
from esa.load import nba
from esa.settings import Settings

def _fake(frame, seen):
    class Fake:
        def __init__(self, **kwargs): seen.update(kwargs)
        def get_data_frames(self): return [frame]
    return Fake

@pytest.fixture
def settings(tmp_path):
    return Settings(lake_dir=tmp_path, call_pause_s=0.0, call_attempts=1)

def test_fetch_shots_sends_window_and_regular_season(monkeypatch, settings):
    seen = {}
    frame = pd.DataFrame({c: ["x"] for c in nba.SHOT_COLUMNS})
    monkeypatch.setattr(nba.shotchartdetail, "ShotChartDetail", _fake(frame, seen))
    out = nba.fetch_shots("2026-27", date(2026, 11, 12), date(2026, 11, 15), settings)
    assert len(out) == 1
    assert seen["date_from_nullable"] == "11/12/2026" and seen["date_to_nullable"] == "11/15/2026"
    assert seen["season_type_all_star"] == "Regular Season"
    assert (seen["player_id"], seen["team_id"], seen["timeout"]) == (0, 0, 60)

def test_fetch_shots_full_season_sends_empty_dates(monkeypatch, settings):
    seen = {}
    frame = pd.DataFrame({c: ["x"] for c in nba.SHOT_COLUMNS})
    monkeypatch.setattr(nba.shotchartdetail, "ShotChartDetail", _fake(frame, seen))
    nba.fetch_shots("2026-27", None, None, settings)
    assert seen["date_from_nullable"] == "" and seen["date_to_nullable"] == ""

def test_missing_column_raises_schema_error_with_names(monkeypatch, settings):   # Review Focus 1
    frame = pd.DataFrame({c: ["x"] for c in nba.SHOT_COLUMNS - {"LOC_X", "LOC_Y"}})
    monkeypatch.setattr(nba.shotchartdetail, "ShotChartDetail", _fake(frame, {}))
    with pytest.raises(nba.SchemaError, match="LOC_X.*LOC_Y|LOC_Y.*LOC_X"):
        nba.fetch_shots("2026-27", None, None, settings)

def test_zero_rows_with_all_columns_is_valid(monkeypatch, settings):
    frame = pd.DataFrame({c: [] for c in nba.SHOT_COLUMNS})
    monkeypatch.setattr(nba.shotchartdetail, "ShotChartDetail", _fake(frame, {}))
    assert nba.fetch_shots("2026-27", None, None, settings).empty
```

Add one test each for `fetch_player_index` (asserts `historical_nullable == 1`) and `fetch_game_log` (asserts `player_or_team_abbreviation == "T"`), in the same style, with a missing-column case for each.

- [ ] **Step 2: Run.** `uv run pytest tests/load/test_retry.py tests/load/test_nba.py -v` → FAIL.
- [ ] **Step 3: Implement** `retry.py` and `nba.py`. Import the endpoint modules at module level (`from nba_api.stats.endpoints import shotchartdetail, playerindex, leaguegamelog`) so that the tests can replace the classes.
- [ ] **Step 4: Run.** → PASS.
- [ ] **Step 5: Commit.** `feat: NBA fetchers with retry and schema check`.

---

### Task 5: Archive reader

**Files:**
- Create: `src/esa/load/archive.py`, `tests/load/test_archive.py`

**Interfaces:**
- Consumes: `Settings`, `call_with_retry`, `LoadError`.
- Produces:
  - `ARCHIVE_INDEX_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"`
  - `parse_index(text: str) -> dict[str, str]` — each line is `name=url`; blank lines are ignored.
  - `shot_archive_name(start_year: int) -> str` → `"shotdetail_2015"` (regular season name, no `po_`).
  - `read_shot_archive(blob: bytes, name: str) -> pd.DataFrame` — opens the `.tar.xz` bytes in memory, reads only the member `<name>.csv` with `dtype=str`, and raises `LoadError` if the member is absent.
  - `download(url: str, settings: Settings) -> bytes` — `urllib.request.urlopen` with the timeout, through `call_with_retry`.
  - `fetch_index(settings: Settings) -> dict[str, str]`

- [ ] **Step 1: Write the failing tests**

```python
import io, tarfile
import pytest
from esa.load.archive import parse_index, read_shot_archive, shot_archive_name
from esa.load.retry import LoadError

def _tar_xz(member: str, text: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        data = text.encode()
        info = tarfile.TarInfo(member); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()

def test_parse_index():
    idx = parse_index("shotdetail_2015=https://x/a.tar.xz\n\nnbastats_2015=https://x/b.tar.xz\n")
    assert idx == {"shotdetail_2015": "https://x/a.tar.xz", "nbastats_2015": "https://x/b.tar.xz"}

def test_archive_name_is_regular_season():
    assert shot_archive_name(2015) == "shotdetail_2015"

def test_reads_named_member_as_strings():
    blob = _tar_xz("shotdetail_2015.csv", "GAME_ID,LOC_X\n0021500001,-220\n")
    df = read_shot_archive(blob, "shotdetail_2015")
    assert df.loc[0, "GAME_ID"] == "0021500001" and df.loc[0, "LOC_X"] == "-220"

def test_other_member_names_are_rejected():
    blob = _tar_xz("../evil.csv", "a\n1\n")
    with pytest.raises(LoadError):
        read_shot_archive(blob, "shotdetail_2015")
```

- [ ] **Step 2: Run.** `uv run pytest tests/load/test_archive.py -v` → FAIL.
- [ ] **Step 3: Implement.** Use `tar.extractfile(member)`. Do not call `extract` or `extractall`.
- [ ] **Step 4: Run.** → PASS.
- [ ] **Step 5: Commit.** `feat: in-memory shot archive reader`.

---

### Task 6: Warehouse connection, run log, and load window

**Files:**
- Create: `src/esa/warehouse/__init__.py`, `src/esa/warehouse/db.py`, `src/esa/warehouse/run_log.py`, `src/esa/load/window.py`, `tests/warehouse/test_run_log.py`, `tests/load/test_window.py`

**Interfaces:**
- Produces:
  - `connect(settings: Settings) -> duckdb.DuckDBPyConnection` — makes `lake_dir` if absent.
  - Table `ops.run_log(run_id BIGINT, command VARCHAR, season VARCHAR, started_at TIMESTAMP, finished_at TIMESTAMP, status VARCHAR, last_game_date DATE, message VARCHAR)`. `status` is `running`, `ok`, or `failed`.
  - `run_log.ensure(con) -> None`
  - `run_log.start(con, command: str, season: str | None, now: datetime) -> int`
  - `run_log.finish(con, run_id: int, *, status: Literal["ok", "failed"], last_game_date: date | None, message: str | None, now: datetime) -> None`
  - `run_log.last_good_game_date(con, season: str) -> date | None` — the maximum `last_game_date` of rows with `command = 'load'`, `status = 'ok'`, and that season.
  - `plan_date_from(last_good: date | None, today: date, lookback_days: int) -> date | None` — `None` means "load the full season".

- [ ] **Step 1: Write the failing tests**

```python
# tests/load/test_window.py
from datetime import date
from esa.load.window import plan_date_from

def test_normal_day_reloads_three_days():
    assert plan_date_from(date(2026, 11, 14), date(2026, 11, 15), 3) == date(2026, 11, 12)

def test_gap_starts_from_last_good_date():
    assert plan_date_from(date(2026, 11, 1), date(2026, 11, 15), 3) == date(2026, 11, 1)

def test_first_load_of_a_season_is_full():
    assert plan_date_from(None, date(2026, 11, 15), 3) is None
```

```python
# tests/warehouse/test_run_log.py
from datetime import date, datetime
from esa.settings import Settings
from esa.warehouse import run_log
from esa.warehouse.db import connect

T = datetime(2026, 11, 15, 10, 0, 0)

def test_last_good_date_ignores_failed_runs_and_other_seasons(tmp_path):
    con = connect(Settings(lake_dir=tmp_path)); run_log.ensure(con)
    a = run_log.start(con, "load", "2026-27", T)
    run_log.finish(con, a, status="ok", last_game_date=date(2026, 11, 13), message=None, now=T)
    b = run_log.start(con, "load", "2026-27", T)
    run_log.finish(con, b, status="failed", last_game_date=date(2026, 11, 14), message="timeout", now=T)
    c = run_log.start(con, "load", "2025-26", T)
    run_log.finish(con, c, status="ok", last_game_date=date(2026, 4, 12), message=None, now=T)
    assert run_log.last_good_game_date(con, "2026-27") == date(2026, 11, 13)
    assert run_log.last_good_game_date(con, "2024-25") is None

def test_ensure_is_idempotent(tmp_path):
    con = connect(Settings(lake_dir=tmp_path)); run_log.ensure(con); run_log.ensure(con)
```

- [ ] **Step 2: Run.** `uv run pytest tests/load/test_window.py tests/warehouse/test_run_log.py -v` → FAIL.
- [ ] **Step 3: Implement.** `plan_date_from` returns `min(today - lookback_days, last_good)`.
- [ ] **Step 4: Run.** → PASS.
- [ ] **Step 5: Commit.** `feat: run log and load window`.

---

### Task 7: Load runs

**Files:**
- Create: `src/esa/load/run.py`, `tests/load/test_run.py`

**Interfaces:**
- Consumes: Tasks 2 to 6, with the exact names in their Interfaces blocks.
- Produces:
  - `run_backfill(settings: Settings, now: datetime) -> None`
  - `run_daily_load(settings: Settings, now: datetime) -> date | None` — returns the last game date in the loaded shots, or the earlier last good date if the window had no shots.

Behavior:

`run_backfill`: for each start year from `settings.first_season` to the current season:
1. Skip the shots of a past season if `has_raw(SHOTS, season, raw_dir)` is true.
2. If `shot_archive_name(year)` is in the index, download and read the archive, and write it with `origin="archive"`, `tag="archive"`.
3. If not, call `fetch_shots(season, None, None, settings)` and write it with `origin="api"`, `tag="full"`.
4. Call `fetch_game_log(season, settings)` and write it with `tag="full"`.

Then call `fetch_player_index(<current season>, settings)` one time and write it. Record one `run_log` row with `command="load"` and the current season.

`run_daily_load`: for the current season, get `last_good_game_date`, calculate `plan_date_from`, call `fetch_shots(season, date_from, now.date(), settings)`, `fetch_game_log`, and `fetch_player_index`, and write each result. `GAME_DATE` in shots has the format `YYYYMMDD`.

Both functions: fetch all three tables before the first write of the run. On any exception, call `run_log.finish(status="failed", message=str(exc))` and raise the exception again.

- [ ] **Step 1: Write the failing tests.** Replace `esa.load.run.fetch_shots`, `fetch_game_log`, `fetch_player_index`, `fetch_index`, and `download` with fakes through `monkeypatch`. Use `Settings(lake_dir=tmp_path, first_season=2025)` and `NOW = datetime(2026, 11, 15, 10, 0, 0)`.

```python
def test_daily_load_first_run_loads_full_season(env):          # env = fixture with the fakes
    last = run_daily_load(env.settings, NOW)
    assert env.shot_calls == [("2026-27", None, date(2026, 11, 15))]
    assert last == date(2026, 11, 14)                           # fake shots have GAME_DATE "20261114"
    assert has_raw(SHOTS, "2026-27", env.settings.raw_dir)

def test_daily_load_second_run_uses_three_day_window(env):
    run_daily_load(env.settings, NOW)
    run_daily_load(env.settings, NOW + timedelta(days=1))
    assert env.shot_calls[1] == ("2026-27", date(2026, 11, 13), date(2026, 11, 16))

def test_day_with_no_games_keeps_last_good_date(env):            # Review Focus 2
    run_daily_load(env.settings, NOW)
    env.shots = env.shots.iloc[0:0]
    assert run_daily_load(env.settings, NOW + timedelta(days=1)) == date(2026, 11, 14)

def test_failure_on_second_call_writes_nothing_and_logs_failed(env):   # Review Focus 5
    env.game_log_error = LoadError("timeout")
    with pytest.raises(LoadError):
        run_daily_load(env.settings, NOW)
    assert not has_raw(SHOTS, "2026-27", env.settings.raw_dir)
    con = connect(env.settings)
    assert con.sql("select status, message from ops.run_log").fetchall() == [("failed", "timeout")]
    assert run_log.last_good_game_date(con, "2026-27") is None

def test_backfill_uses_archive_when_listed_and_api_when_not(env):
    env.index = {"shotdetail_2025": "https://x/2025.tar.xz"}       # 2026 is not in the archive
    run_backfill(env.settings, NOW)
    assert env.downloads == ["https://x/2025.tar.xz"]
    assert env.shot_calls == [("2026-27", None, None)]
    assert sorted(env.game_log_calls) == ["2025-26", "2026-27"]

def test_backfill_second_run_skips_past_seasons(env):
    env.index = {"shotdetail_2025": "https://x/2025.tar.xz"}
    run_backfill(env.settings, NOW)
    run_backfill(env.settings, NOW + timedelta(hours=1))
    assert env.downloads == ["https://x/2025.tar.xz"]              # not downloaded again
```

- [ ] **Step 2: Run.** `uv run pytest tests/load/test_run.py -v` → FAIL.
- [ ] **Step 3: Implement** `run.py`. Close the DuckDB connection before the function returns.
- [ ] **Step 4: Run.** → PASS.
- [ ] **Step 5: Commit.** `feat: backfill and daily load runs`.

---

### Task 8: dbt warehouse

**Files:**
- Create: `warehouse/dbt_project.yml`, `warehouse/profiles.yml`, `warehouse/macros/generate_schema_name.sql`, `warehouse/models/staging/sources.yml`, `warehouse/models/staging/stg_shots.sql`, `stg_player_index.sql`, `stg_game_log.sql`, `warehouse/models/staging/schema.yml`, `warehouse/models/core/dim_player.sql`, `dim_team.sql`, `dim_game.sql`, `fct_shot.sql`, `warehouse/models/core/schema.yml`
- Create: `src/esa/warehouse/build.py`, `tests/warehouse/test_build.py`

**Interfaces:**
- Consumes: `Settings`, `REPO_ROOT`, `write_raw`, source name constants.
- Produces:
  - `class BuildError(Exception)`; `class EmptyLakeError(BuildError)`.
  - `build_warehouse(settings: Settings) -> None` — raises `EmptyLakeError("No raw files. Run: esa load --backfill")` if any of the three sources has no Parquet file. Sets the environment variables `ESA_DUCKDB_PATH` and `ESA_RAW_DIR`, then runs `dbtRunner().invoke(["build", "--project-dir", <REPO_ROOT/warehouse>, "--profiles-dir", <REPO_ROOT/warehouse>])`. Raises `BuildError` if dbt reports a failure.
  - DuckDB tables (exact schemas `staging` and `core`; the macro removes the default `main_` prefix):

| Table | Columns |
|---|---|
| `core.fct_shot` | `shot_id` (`game_id-game_event_id`), `game_id` VARCHAR(10), `game_event_id` INT, `player_id` BIGINT, `team_id` BIGINT, `season` VARCHAR, `game_date` DATE, `period` INT, `seconds_remaining` INT, `action_type` VARCHAR, `shot_value` INT (2 or 3), `zone_basic`, `zone_area`, `zone_range` VARCHAR, `shot_distance_ft` INT, `loc_x_ft` DOUBLE, `loc_y_ft` DOUBLE, `made` BOOLEAN |
| `core.dim_player` | `player_id` BIGINT, `full_name`, `slug`, `position` VARCHAR, `height_in` INT, `from_year` INT, `to_year` INT, `team_id` BIGINT |
| `core.dim_team` | `team_id` BIGINT, `abbreviation`, `name` VARCHAR |
| `core.dim_game` | `game_id` VARCHAR(10), `season` VARCHAR, `game_date` DATE, `home_team_id` BIGINT, `away_team_id` BIGINT |

Model rules:
- Sources read `read_parquet('<ESA_RAW_DIR>/<source>/*/*/*.parquet', union_by_name = true)` through the dbt-duckdb `external_location` source setting.
- `game_id = lpad(GAME_ID, 10, '0')` in all staging models.
- Each staging model keeps one row for each key, with the latest `_loaded_at`. Keys: shots `(game_id, game_event_id)`, player index `PERSON_ID`, game log `(game_id, TEAM_ID)`.
- `loc_x_ft = LOC_X / 10.0`, `loc_y_ft = LOC_Y / 10.0`. `seconds_remaining = MINUTES_REMAINING * 60 + SECONDS_REMAINING`. `shot_value = 3` if `SHOT_TYPE` starts with `3PT`, else `2`. `made = (SHOT_MADE_FLAG = '1')`. `season = _season`. Shot `GAME_DATE` has the format `YYYYMMDD`. Game log `GAME_DATE` has the format `YYYY-MM-DD`.
- `fct_shot` keeps only rows with `SHOT_ATTEMPTED_FLAG = '1'`.
- `dim_game`: the home team is the row whose `MATCHUP` contains ` vs. `. The away team is the row whose `MATCHUP` contains ` @ `.
- `dim_player`: all players of the player index, plus each `player_id` that is in shots and not in the index (name from `PLAYER_NAME`, other columns null). `height_in` converts `6-7` to `79`.
- dbt tests in `schema.yml`: `unique` and `not_null` on each primary key; `relationships` from `fct_shot.player_id` to `dim_player` and from `fct_shot.game_id` to `dim_game`; `accepted_values` `[2, 3]` on `shot_value`; `not_null` on `dim_game.home_team_id` and `away_team_id`.

- [ ] **Step 1: Write the failing integration test** in `tests/warehouse/test_build.py`. The fixture writes raw files with `write_raw` into `tmp_path` and returns `Settings(lake_dir=tmp_path)`:

  - Shots, first file (`now = T1`): (`GAME_ID "0022600001"`, event `"10"`, player `"1"`, `SHOT_TYPE "3PT Field Goal"`, `LOC_X "-220"`, `LOC_Y "15"`, made `"1"`, `MINUTES_REMAINING "2"`, `SECONDS_REMAINING "30"`, `GAME_DATE "20261114"`); (`GAME_ID "22600001"`, event `"11"`, player `"2"`, `SHOT_TYPE "2PT Field Goal"`, made `"0"`); (`GAME_ID "0022600002"`, event `"5"`, player `"1"`, 2PT, made `"1"`). All have `SHOT_ATTEMPTED_FLAG "1"`.
  - Shots, second file (`now = T1 + 1 day`): the event `"10"` row again with made `"0"`.
  - Game log: game `0022600001` rows (`TEAM_ID "100"`, `"BOS"`, `MATCHUP "BOS vs. NYK"`) and (`"200"`, `"NYK"`, `"NYK @ BOS"`); game `0022600002` rows (`"200"`, `"NYK vs. BOS"`) and (`"100"`, `"BOS @ NYK"`). `GAME_DATE "2026-11-14"`.
  - Player index: only `PERSON_ID "1"`, `HEIGHT "6-7"`, `POSITION "G-F"`.

```python
def q(settings, sql):
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try: return con.sql(sql).fetchall()
    finally: con.close()

def test_fct_shot_is_clean(lake):
    build_warehouse(lake)
    assert q(lake, "select count(*) from core.fct_shot") == [(3,)]
    # Review Focus 4: the later load wins
    assert q(lake, "select made from core.fct_shot where shot_id = '0022600001-10'") == [(False,)]
    # Review Focus 3: game id is padded and joins to one game
    assert q(lake, "select count(distinct game_id) from core.fct_shot") == [(2,)]
    assert q(lake, "select loc_x_ft, loc_y_ft, shot_value, seconds_remaining, game_date "
                   "from core.fct_shot where shot_id = '0022600001-10'") == [(-22.0, 1.5, 3, 150, date(2026, 11, 14))]

def test_dims(lake):
    build_warehouse(lake)
    assert q(lake, "select home_team_id, away_team_id from core.dim_game where game_id = '0022600001'") == [(100, 200)]
    assert q(lake, "select height_in, position from core.dim_player where player_id = 1") == [(79, "G-F")]
    assert q(lake, "select position from core.dim_player where player_id = 2") == [(None,)]   # in shots, not in index
    assert q(lake, "select count(*) from core.dim_team") == [(2,)]

def test_second_build_gives_same_result(lake):
    build_warehouse(lake); build_warehouse(lake)
    assert q(lake, "select count(*) from core.fct_shot") == [(3,)]

def test_empty_lake_gives_clear_error(tmp_path):
    with pytest.raises(EmptyLakeError, match="esa load --backfill"):
        build_warehouse(Settings(lake_dir=tmp_path))
```

- [ ] **Step 2: Run.** `uv run pytest tests/warehouse/test_build.py -v` → FAIL.
- [ ] **Step 3: Write the dbt project.** `profiles.yml` has one `duckdb` target with `path: "{{ env_var('ESA_DUCKDB_PATH') }}"`. `dbt_project.yml` sets `+schema: staging` and `+schema: core` for the two folders, with `staging` as views and `core` as tables.
- [ ] **Step 4: Implement** `build.py`.
- [ ] **Step 5: Run.** `uv run pytest tests/warehouse/test_build.py -v` → PASS.
- [ ] **Step 6: Commit.** `feat: dbt warehouse with staging and core models`.

---

### Task 9: CLI commands and live check

**Files:**
- Modify: `src/esa/cli.py`, `tests/test_cli.py`
- Modify: `README.md` (replace the install section with the four commands below; leave the rest for plan 1D)

**Interfaces:**
- Consumes: `run_backfill`, `run_daily_load`, `build_warehouse`, `fetch_*`, `fetch_index`, the column sets, `Settings`.
- Produces: CLI commands
  - `esa load [--backfill]` — `run_backfill` if the flag is set, else `run_daily_load`.
  - `esa build` — `build_warehouse`, with a `run_log` row (`command="build"`).
  - `esa daily` — `load` then `build`. Plans 1B and 1C add `run` and `publish` to this sequence.
  - `esa probe` — live check. It calls the archive index and the three fetchers (shots for the last 3 days), and prints one line for each: source, row count, `OK` or the error.
  - Each command exits with code 0 on success and 1 on any error, and prints the error message on one line.

- [ ] **Step 1: Write the failing tests** (replace the run functions with fakes through `monkeypatch` on `esa.cli`):

```python
def test_daily_stops_before_build_when_load_fails(monkeypatch):    # Review Focus 5
    built = []
    monkeypatch.setattr("esa.cli.run_daily_load", lambda s, now: (_ for _ in ()).throw(LoadError("timeout")))
    monkeypatch.setattr("esa.cli.build_warehouse", lambda s: built.append(1))
    result = CliRunner().invoke(app, ["daily"])
    assert result.exit_code == 1 and "timeout" in result.output and built == []

def test_daily_runs_load_then_build(monkeypatch):
    order = []
    monkeypatch.setattr("esa.cli.run_daily_load", lambda s, now: order.append("load"))
    monkeypatch.setattr("esa.cli.build_warehouse", lambda s: order.append("build"))
    assert CliRunner().invoke(app, ["daily"]).exit_code == 0 and order == ["load", "build"]

def test_build_records_a_run_log_row(monkeypatch, tmp_path):
    monkeypatch.setenv("ESA_LAKE_DIR", str(tmp_path))
    monkeypatch.setattr("esa.cli.build_warehouse", lambda s: None)
    assert CliRunner().invoke(app, ["build"]).exit_code == 0
    con = connect(Settings())
    assert con.sql("select command, status from ops.run_log").fetchall() == [("build", "ok")]

def test_load_backfill_flag(monkeypatch):
    called = []
    monkeypatch.setattr("esa.cli.run_backfill", lambda s, now: called.append("backfill"))
    monkeypatch.setattr("esa.cli.run_daily_load", lambda s, now: called.append("daily"))
    CliRunner().invoke(app, ["load", "--backfill"])
    assert called == ["backfill"]

def test_probe_reports_each_source_and_fails_on_error(monkeypatch):
    ok = pd.DataFrame({"a": [1, 2]})
    monkeypatch.setattr("esa.cli.fetch_index", lambda s: {"shotdetail_2015": "u"})
    monkeypatch.setattr("esa.cli.fetch_shots", lambda *a: ok)
    monkeypatch.setattr("esa.cli.fetch_game_log", lambda *a: ok)
    monkeypatch.setattr("esa.cli.fetch_player_index", lambda *a: (_ for _ in ()).throw(SchemaError("missing: HEIGHT")))
    result = CliRunner().invoke(app, ["probe"])
    assert result.exit_code == 1
    assert "nba_shots" in result.output and "missing: HEIGHT" in result.output
```

- [ ] **Step 2: Run.** `uv run pytest tests/test_cli.py -v` → FAIL.
- [ ] **Step 3: Implement** the four commands.
- [ ] **Step 4: Run all checks.** `uv run pytest -v` → all PASS. `uv run ruff check .` → no errors.
- [ ] **Step 5: Commit.** `feat: load, build, daily and probe commands`.

- [ ] **Step 6: Live check on the owner's computer.** This step makes real network calls. It is the risk check from spec section 5.1.

Run: `uv run esa probe`
Expected: four lines with `OK`, and exit code 0. If a line shows missing columns, stop and correct the column sets in `nba.py` (and the staging models) before the next step.

Run: `uv run esa load --backfill`
Expected: exit code 0. `lake/raw/nba_shots/` has one folder for each season from `2015-16` to `2026-27`.

Run: `uv run esa build`
Expected: exit code 0, and all dbt tests pass.

Run:
```bash
uv run python -c "import duckdb; c=duckdb.connect('lake/warehouse.duckdb', read_only=True); print(c.sql('select season, count(*) shots, count(distinct game_id) games from core.fct_shot group by 1 order by 1'))"
```
Expected: each complete season has about 1,230 games (2019-20 and 2020-21 have fewer) and between 190,000 and 230,000 shots. Report the table to the owner. A season far outside these ranges is a defect to examine, not to accept.

Run: `uv run esa daily` two times.
Expected: exit code 0 both times, and the shot counts are the same after each run.

- [ ] **Step 7: Commit** any corrections from the live check. `fix: align column sets with live endpoints` (only if there were changes).
