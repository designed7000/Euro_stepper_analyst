"""Reader for the shufinskiy/nba_data shot archives (history from 1996-97).

An archive is a .tar.xz file with one CSV. We read the CSV in memory and
never extract files to disk, so a bad member name cannot write outside.
"""

import io
import tarfile

import pandas as pd
import requests

from esa.load.retry import LoadError, call_with_retry
from esa.settings import Settings

ARCHIVE_INDEX_URL = "https://raw.githubusercontent.com/shufinskiy/nba_data/main/list_data.txt"


def parse_index(text: str) -> dict[str, str]:
    """Parse the index file. Each line is `name=url`."""
    index = {}
    for line in text.splitlines():
        name, sep, url = line.strip().partition("=")
        if sep and name and url:
            index[name] = url
    return index


def shot_archive_name(start_year: int) -> str:
    """Name of the regular season shot archive of a season."""
    return f"shotdetail_{start_year}"


def read_shot_archive(blob: bytes, name: str) -> pd.DataFrame:
    """Read the member `<name>.csv` of a .tar.xz archive. All columns are strings."""
    wanted = f"{name}.csv"
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:xz") as tar:
            member = next((m for m in tar.getmembers() if m.name == wanted and m.isfile()), None)
            if member is None:
                raise LoadError(f"Archive {name} has no member {wanted}")
            return pd.read_csv(tar.extractfile(member), dtype="string")
    except tarfile.TarError as exc:
        raise LoadError(f"Archive {name} is not a valid .tar.xz file: {exc}") from exc


def download(url: str, settings: Settings) -> bytes:
    """Download one file, with the timeout and retries of the settings."""

    def get() -> bytes:
        response = requests.get(url, timeout=settings.call_timeout_s)
        response.raise_for_status()
        return response.content

    return call_with_retry(get, attempts=settings.call_attempts)


def fetch_index(settings: Settings) -> dict[str, str]:
    """Download and parse the archive index."""
    return parse_index(download(ARCHIVE_INDEX_URL, settings).decode("utf-8"))
