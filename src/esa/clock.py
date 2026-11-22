"""The clock of the platform: UTC, so that time always goes forward."""

from datetime import UTC, datetime


def utc_now() -> datetime:
    """The time now in UTC, with no time zone.

    Local time goes back one hour in autumn, so a later load could get an
    earlier _loaded_at than an earlier load. UTC does not go back. The value
    is naive, so it has the same type as the _loaded_at of the existing raw files.
    """
    return datetime.now(UTC).replace(tzinfo=None)
