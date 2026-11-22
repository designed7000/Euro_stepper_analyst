"""Helpers for NBA season names. A season is named by the year in which it starts."""

from datetime import date

SEASON_FIRST_MONTH = 10  # the season changes on 1 October


def season_label(start_year: int) -> str:
    """Return the NBA label of a season: 2015 -> '2015-16'."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"


def season_start_year(d: date) -> int:
    """Return the start year of the season that contains the date."""
    return d.year if d.month >= SEASON_FIRST_MONTH else d.year - 1
