from esa.settings import Settings


def test_paths_derive_from_lake_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("ESA_LAKE_DIR", str(tmp_path))
    s = Settings()
    assert s.raw_dir == tmp_path / "raw"
    assert s.duckdb_path == tmp_path / "warehouse.duckdb"
    assert (s.first_season, s.lookback_days) == (2015, 3)
