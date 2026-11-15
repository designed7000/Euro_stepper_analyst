from datetime import date

from esa.seasons import season_label, season_start_year


def test_season_label():
    assert season_label(2015) == "2015-16"
    assert season_label(1999) == "1999-00"
    assert season_label(2026) == "2026-27"


def test_season_start_year():
    assert season_start_year(date(2026, 11, 15)) == 2026
    assert season_start_year(date(2027, 4, 1)) == 2026
    assert season_start_year(date(2026, 9, 30)) == 2025  # offseason maps to the last season
