from datetime import date, datetime, timedelta

import duckdb
import pandas as pd
import pytest

from esa.load.raw_store import GAME_LOG, PLAYER_INDEX, SHOTS, write_raw
from esa.settings import Settings
from esa.warehouse.build import BuildError, EmptyLakeError, build_warehouse

T1 = datetime(2026, 11, 15, 10, 0, 0)
SEASON = "2026-27"


def _shot(**values):
    row = {
        "GAME_ID": "0022600001", "GAME_EVENT_ID": "10", "PLAYER_ID": "1", "PLAYER_NAME": "Al Bo",
        "TEAM_ID": "100", "PERIOD": "1", "MINUTES_REMAINING": "2", "SECONDS_REMAINING": "30",
        "ACTION_TYPE": "Jump Shot", "SHOT_TYPE": "2PT Field Goal", "SHOT_ZONE_BASIC": "Mid-Range",
        "SHOT_ZONE_AREA": "Center(C)", "SHOT_ZONE_RANGE": "16-24 ft.", "SHOT_DISTANCE": "22",
        "LOC_X": "-220", "LOC_Y": "15", "SHOT_ATTEMPTED_FLAG": "1", "SHOT_MADE_FLAG": "1",
        "GAME_DATE": "20261114", "HTM": "BOS", "VTM": "NYK",
    }  # fmt: skip
    return row | values


def _game(game_id, team_id, abbr, matchup):
    return {
        "GAME_ID": game_id, "TEAM_ID": team_id, "TEAM_ABBREVIATION": abbr,
        "TEAM_NAME": f"{abbr} Team", "GAME_DATE": "2026-11-14", "MATCHUP": matchup,
    }  # fmt: skip


@pytest.fixture
def lake(tmp_path):
    settings = Settings(lake_dir=tmp_path)

    def write(rows, source, now=T1):
        write_raw(
            pd.DataFrame(rows), source=source, season=SEASON, origin="api", tag="t",
            raw_dir=settings.raw_dir, now=now,
        )  # fmt: skip

    write(
        [
            _shot(SHOT_TYPE="3PT Field Goal"),
            # archive style game id with no leading zeros, and a player who is not in the index
            _shot(GAME_ID="22600001", GAME_EVENT_ID="11", PLAYER_ID="2", PLAYER_NAME="Cy Du",
                  SHOT_MADE_FLAG="0"),
            _shot(GAME_ID="0022600002", GAME_EVENT_ID="5"),
        ],
        SHOTS,
    )  # fmt: skip
    # a stat correction: game 1 loaded one day later, and shot 10 is now a miss.
    # A window load holds all the shots of each game in the window.
    write(
        [
            _shot(SHOT_TYPE="3PT Field Goal", SHOT_MADE_FLAG="0"),
            _shot(GAME_EVENT_ID="11", PLAYER_ID="2", PLAYER_NAME="Cy Du", SHOT_MADE_FLAG="0"),
        ],
        SHOTS,
        T1 + timedelta(days=1),
    )
    write(
        [
            _game("0022600001", "100", "BOS", "BOS vs. NYK"),
            _game("0022600001", "200", "NYK", "NYK @ BOS"),
            _game("0022600002", "200", "NYK", "NYK vs. BOS"),
            _game("0022600002", "100", "BOS", "BOS @ NYK"),
        ],
        GAME_LOG,
    )
    write(
        [{
            "PERSON_ID": "1", "PLAYER_FIRST_NAME": "Al", "PLAYER_LAST_NAME": "Bo",
            "PLAYER_SLUG": "al-bo", "TEAM_ID": "100", "POSITION": "G-F", "HEIGHT": "6-7",
            "FROM_YEAR": "2020", "TO_YEAR": "2026",
        }],
        PLAYER_INDEX,
    )  # fmt: skip
    return settings


def q(settings, sql):
    con = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        return con.sql(sql).fetchall()
    finally:
        con.close()


def test_fct_shot_is_clean(lake):
    build_warehouse(lake)
    assert q(lake, "select count(*) from core.fct_shot") == [(3,)]
    # the later load wins
    assert q(lake, "select made from core.fct_shot where shot_id = '0022600001-10'") == [(False,)]
    # the game id is padded and joins to one game
    assert q(lake, "select count(distinct game_id) from core.fct_shot") == [(2,)]
    assert q(
        lake,
        "select loc_x_ft, loc_y_ft, shot_value, seconds_remaining, game_date, season "
        "from core.fct_shot where shot_id = '0022600001-10'",
    ) == [(-22.0, 1.5, 3, 150, date(2026, 11, 14), SEASON)]


def test_shot_missing_from_a_later_load_of_its_game_is_removed(lake):
    # The NBA rescinded shot 11 of game 1, so the next window load has the game without it.
    write_raw(
        pd.DataFrame([_shot(SHOT_TYPE="3PT Field Goal", SHOT_MADE_FLAG="0")]), source=SHOTS,
        season=SEASON, origin="api", tag="window", raw_dir=lake.raw_dir,
        now=T1 + timedelta(days=2),
    )  # fmt: skip
    build_warehouse(lake)
    # game 2 is not in the later load, so its shot stays
    assert q(lake, "select shot_id from core.fct_shot order by 1") == [
        ("0022600001-10",),
        ("0022600002-5",),
    ]


def test_dims(lake):
    build_warehouse(lake)
    assert q(
        lake, "select home_team_id, away_team_id from core.dim_game where game_id = '0022600001'"
    ) == [(100, 200)]
    assert q(
        lake, "select full_name, height_in, position from core.dim_player where player_id = 1"
    ) == [("Al Bo", 79, "G-F")]
    # a player who is in shots and not in the index
    assert q(lake, "select full_name, position from core.dim_player where player_id = 2") == [
        ("Cy Du", None)
    ]
    assert q(lake, "select abbreviation from core.dim_team order by team_id") == [
        ("BOS",),
        ("NYK",),
    ]


def test_second_build_gives_same_result(lake):
    build_warehouse(lake)
    build_warehouse(lake)
    assert q(lake, "select count(*) from core.fct_shot") == [(3,)]


def test_empty_lake_gives_clear_error(tmp_path):
    with pytest.raises(EmptyLakeError, match="esa load --backfill"):
        build_warehouse(Settings(lake_dir=tmp_path))


def test_failed_data_test_raises_build_error_that_names_the_test(lake):
    # a shot of a game that is not in the game log
    write_raw(
        pd.DataFrame([_shot(GAME_ID="0022600099", GAME_EVENT_ID="1")]), source=SHOTS,
        season=SEASON, origin="api", tag="orphan", raw_dir=lake.raw_dir, now=T1,
    )  # fmt: skip
    with pytest.raises(BuildError, match="relationships_fct_shot_game_id"):
        build_warehouse(lake)


def test_neutral_site_game_takes_the_home_team_from_the_shot_data(lake):
    # At a neutral site the game log shows '@' for the two teams (seen in real data:
    # Mexico City, Paris, NBA Cup semifinals). The shot data names the designated home team.
    common = {"season": SEASON, "origin": "api", "tag": "neutral", "raw_dir": lake.raw_dir}
    write_raw(
        pd.DataFrame([
            _game("0022600003", "100", "BOS", "BOS @ NYK"),
            _game("0022600003", "200", "NYK", "NYK @ BOS"),
        ]),
        source=GAME_LOG, now=T1, **common,
    )  # fmt: skip
    write_raw(
        pd.DataFrame([_shot(GAME_ID="0022600003", GAME_EVENT_ID="7", HTM="NYK", VTM="BOS")]),
        source=SHOTS, now=T1, **common,
    )  # fmt: skip
    build_warehouse(lake)
    assert q(
        lake,
        "select game_id, home_team_id, away_team_id, is_neutral_site from core.dim_game order by 1",
    ) == [
        ("0022600001", 100, 200, False),
        ("0022600002", 200, 100, False),
        ("0022600003", 200, 100, True),
    ]
