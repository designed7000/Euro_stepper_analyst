"""Season derivation logic."""

from datetime import date

from config import current_season, recent_seasons, historical_seasons, season_string


def test_october_starts_new_season():
    assert current_season(date(2025, 10, 1)) == "2025-26"


def test_before_october_is_previous_start_year():
    assert current_season(date(2026, 6, 21)) == "2025-26"
    assert current_season(date(2026, 1, 5)) == "2025-26"


def test_september_is_still_previous_season():
    assert current_season(date(2025, 9, 30)) == "2024-25"


def test_season_string_rollover_decade():
    assert season_string(2009) == "2009-10"
    assert season_string(1999) == "1999-00"


def test_recent_seasons_newest_first():
    seasons = recent_seasons(6, today=date(2026, 6, 21))
    assert seasons[0] == "2025-26"
    assert len(seasons) == 6
    assert seasons[-1] == "2020-21"


def test_historical_excludes_current():
    hist = historical_seasons(4, today=date(2026, 6, 21))
    assert "2025-26" not in hist
    assert hist[0] == "2024-25"
    assert len(hist) == 4
