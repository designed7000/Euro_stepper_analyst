"""
Snapshot store: the persistence layer that replaces Streamlit's @st.cache_data.

Each snapshot is one Parquet file plus a manifest entry recording when it was
fetched and how many rows it holds. Snapshots are keyed by
(league, dataset, season, params) so the same dataset can coexist for several
seasons, and parameterised datasets (e.g. player shots with clutch_only) get
distinct files.

Storage format choice: Parquet + a JSON manifest.
  - Parquet preserves pandas dtypes faithfully (unlike SQLite's to_sql round-trip,
    which mangles datetimes and nullable ints) and is columnar/compressed.
  - The manifest keeps lightweight metadata queryable without opening the data.
The 'league' field is fixed to 'nba' for now but is part of the key so Euroleague
can be added later without re-keying anything.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def _params_hash(params):
    """Stable short hash of a params dict (order-independent). None -> 'default'."""
    if not params:
        return "default"
    encoded = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(encoded.encode()).hexdigest()[:12]


class SnapshotStore:
    """File-backed store for pre-fetched DataFrames."""

    def __init__(self, base_dir):
        self.base = Path(base_dir)
        self.manifest_path = self.base / "manifest.json"

    # --- paths & manifest ------------------------------------------------

    def _path(self, dataset, season, params=None, league="nba"):
        filename = f"{season}__{_params_hash(params)}.parquet"
        return self.base / league / dataset / filename

    def _manifest_key(self, dataset, season, params=None, league="nba"):
        return f"{league}/{dataset}/{season}/{_params_hash(params)}"

    def _read_manifest(self):
        if not self.manifest_path.exists():
            return {}
        try:
            return json.loads(self.manifest_path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_manifest(self, manifest):
        self.base.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    # --- public API ------------------------------------------------------

    def save(self, dataset, df, season, params=None, league="nba"):
        """Persist a DataFrame snapshot and record it in the manifest."""
        path = self._path(dataset, season, params, league)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)

        manifest = self._read_manifest()
        manifest[self._manifest_key(dataset, season, params, league)] = {
            "league": league,
            "dataset": dataset,
            "season": season,
            "params": params or {},
            "path": str(path.relative_to(self.base)),
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "row_count": int(len(df)),
        }
        self._write_manifest(manifest)
        return path

    def load(self, dataset, season, params=None, league="nba"):
        """Return a stored snapshot, or None on a cache miss."""
        path = self._path(dataset, season, params, league)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def exists(self, dataset, season, params=None, league="nba"):
        """True if a snapshot is present on disk."""
        return self._path(dataset, season, params, league).exists()

    def get_metadata(self, dataset, season, params=None, league="nba"):
        """Return the manifest entry for a snapshot, or None."""
        manifest = self._read_manifest()
        return manifest.get(self._manifest_key(dataset, season, params, league))
