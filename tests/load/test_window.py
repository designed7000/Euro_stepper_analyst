from datetime import date

from esa.load.window import plan_date_from


def test_normal_day_reloads_three_days():
    assert plan_date_from(date(2026, 11, 14), date(2026, 11, 15), 3) == date(2026, 11, 12)


def test_gap_starts_from_last_good_date():
    assert plan_date_from(date(2026, 11, 1), date(2026, 11, 15), 3) == date(2026, 11, 1)


def test_first_load_of_a_season_is_full():
    assert plan_date_from(None, date(2026, 11, 15), 3) is None
