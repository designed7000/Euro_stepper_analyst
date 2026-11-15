import io
import tarfile

import pytest

from esa.load.archive import parse_index, read_shot_archive, shot_archive_name
from esa.load.retry import LoadError


def _tar_xz(member: str, text: str) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:xz") as tar:
        data = text.encode()
        info = tarfile.TarInfo(member)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_parse_index():
    idx = parse_index("shotdetail_2015=https://x/a.tar.xz\n\nnbastats_2015=https://x/b.tar.xz\n")
    assert idx == {"shotdetail_2015": "https://x/a.tar.xz", "nbastats_2015": "https://x/b.tar.xz"}


def test_archive_name_is_regular_season():
    assert shot_archive_name(2015) == "shotdetail_2015"


def test_reads_named_member_as_strings():
    blob = _tar_xz("shotdetail_2015.csv", "GAME_ID,LOC_X\n0021500001,-220\n")
    df = read_shot_archive(blob, "shotdetail_2015")
    assert df.loc[0, "GAME_ID"] == "0021500001" and df.loc[0, "LOC_X"] == "-220"


def test_other_member_names_are_rejected():
    blob = _tar_xz("../evil.csv", "a\n1\n")
    with pytest.raises(LoadError):
        read_shot_archive(blob, "shotdetail_2015")
