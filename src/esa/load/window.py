"""Decide which dates a daily load must get."""

from datetime import date, timedelta


def plan_date_from(last_good: date | None, today: date, lookback_days: int) -> date | None:
    """Return the first date to load. None means "load the full season".

    A normal run loads the last `lookback_days` again, because the NBA corrects
    stats. If runs were missed, the load starts from the last good game date.
    """
    if last_good is None:
        return None
    return min(today - timedelta(days=lookback_days), last_good)
