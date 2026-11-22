from datetime import UTC, datetime, timedelta

from esa.clock import utc_now


def test_utc_now_is_naive_utc():
    # Naive, so it has the same type as the _loaded_at of the files in the lake.
    now = utc_now()
    assert now.tzinfo is None
    assert abs(now - datetime.now(UTC).replace(tzinfo=None)) < timedelta(seconds=5)
