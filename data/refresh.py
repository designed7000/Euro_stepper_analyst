"""
Snapshot refresh CLI.

Populates the league-wide snapshots the app serves. This is the time-triggered
"data pipeline" — run it on a schedule (cron / launchd / CI) whenever games have
been played. It is the only code path that calls the NBA API in bulk, so all the
rate-limit pacing and retry logic lives here, not in fetch.py.

Usage:
    python -m data.refresh                      # current season
    python -m data.refresh --season 2024-25
    python -m data.refresh --datasets leaders standings
"""

import time
import argparse

from config import STORE_DIR, current_season
from data import fetch, processing
from data.store import SnapshotStore

# Pacing between NBA API calls (seconds) to respect rate limits.
PACE_SECONDS = 1.0


def _with_retry(fn, *args, attempts=3, base_sleep=2.0, **kwargs):
    """Call fn with exponential backoff on failure."""
    for i in range(attempts):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - surface only after retries
            if i == attempts - 1:
                raise
            wait = base_sleep * (2 ** i)
            print(f"  ! {fn.__name__} failed ({exc}); retrying in {wait:.0f}s")
            time.sleep(wait)


def _pace():
    time.sleep(PACE_SECONDS)


# --- DATASET REFRESHERS --------------------------------------------------
# Each returns a dict of {dataset_name: dataframe} to be saved.

def refresh_leaders(season):
    base = _with_retry(fetch.get_player_dash_stats, season, 'Base', 'PerGame')
    _pace()
    adv = _with_retry(fetch.get_player_dash_stats, season, 'Advanced', 'PerGame')
    _pace()
    per100 = _with_retry(fetch.get_player_dash_stats, season, 'Base', 'Per100Possessions')
    return {"league_leaders": processing.build_league_leaders(base, adv, per100)}


def refresh_advanced(season):
    base = _with_retry(fetch.get_player_dash_stats, season, 'Base', 'PerGame')
    _pace()
    adv = _with_retry(fetch.get_player_dash_stats, season, 'Advanced', 'PerGame')
    return {"advanced_stats": processing.build_advanced_stats(base, adv)}


def refresh_standings(season):
    raw = _with_retry(fetch.get_standings, season)
    return {"standings": processing.clean_standings(raw)}


def refresh_zone_averages(season):
    league_shots = _with_retry(fetch.get_league_shot_chart, season)
    return {"league_zone_averages": processing.compute_league_zone_averages(league_shots)}


def refresh_mvp(season):
    totals = _with_retry(fetch.get_player_dash_stats, season, 'Base', 'Totals')
    _pace()
    pergame = _with_retry(fetch.get_player_dash_stats, season, 'Base', 'PerGame')
    mvp_totals, mvp_pergame = processing.filter_mvp_stats(totals, pergame)
    return {"mvp_totals": mvp_totals, "mvp_pergame": mvp_pergame}


REFRESHERS = {
    "leaders": refresh_leaders,
    "advanced": refresh_advanced,
    "standings": refresh_standings,
    "zone_averages": refresh_zone_averages,
    "mvp": refresh_mvp,
}


def run(season, datasets=None, store=None):
    """Refresh the requested datasets (all by default) for one season."""
    store = store or SnapshotStore(STORE_DIR)
    datasets = datasets or list(REFRESHERS.keys())

    print(f"Refreshing {season} -> {STORE_DIR}")
    for name in datasets:
        refresher = REFRESHERS.get(name)
        if refresher is None:
            print(f"  ? unknown dataset '{name}', skipping")
            continue
        print(f"  - {name} ...")
        try:
            for dataset_name, df in refresher(season).items():
                store.save(dataset_name, df, season)
                print(f"    saved {dataset_name}: {len(df)} rows")
        except Exception as exc:  # noqa: BLE001 - keep going on partial failure
            print(f"    FAILED {name}: {exc}")
        _pace()
    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Refresh NBA league-wide snapshots.")
    parser.add_argument("--season", default=current_season(),
                        help="Season string, e.g. 2024-25 (default: current).")
    parser.add_argument("--datasets", nargs="*", choices=list(REFRESHERS.keys()),
                        help="Subset of datasets to refresh (default: all).")
    args = parser.parse_args()
    run(args.season, args.datasets)


if __name__ == "__main__":
    main()
