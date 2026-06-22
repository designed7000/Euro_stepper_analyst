"""
Raw NBA API calls.

This layer is fetch-only: every function wraps a single nba_api endpoint and
returns exactly what the API gives back. No caching, no rate-limit pacing, and
no transforms — those belong to the refresh orchestrator (pacing/retries) and
to data/processing.py (shaping the raw frames into app-facing datasets).
"""

from nba_api.stats.endpoints import (
    shotchartdetail,
    leaguedashplayerstats,
    leaguestandings,
)

# Default per-request network timeout (seconds). nba_api hangs without one.
DEFAULT_TIMEOUT = 30


def get_player_shots(player_id, season, clutch_only=False, timeout=DEFAULT_TIMEOUT):
    """Fetch shot chart data for a specific player.

    Args:
        player_id: NBA player ID.
        season: Season string (e.g. '2024-25').
        clutch_only: If True, only fetch clutch-time shots.
        timeout: Network timeout in seconds.

    Returns:
        tuple: (player_shots_df, league_averages_df) — both raw API frames.
    """
    clutch_time = 'Last 5 Minutes' if clutch_only else None
    ahead_behind = 'Ahead or Behind' if clutch_only else None

    shot_data = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        context_measure_simple='FGA',
        season_nullable=season,
        clutch_time_nullable=clutch_time,
        ahead_behind_nullable=ahead_behind,
        timeout=timeout,
    )
    frames = shot_data.get_data_frames()
    return frames[0], frames[1]


def get_league_shot_chart(season, timeout=DEFAULT_TIMEOUT):
    """Fetch every shot in the league for a season (player_id=0).

    This is the raw, ~200k-row frame. Aggregation into zone averages happens in
    processing.compute_league_zone_averages.

    Args:
        season: Season string (e.g. '2024-25').
        timeout: Network timeout in seconds.

    Returns:
        DataFrame: Raw league-wide shot chart.
    """
    shot_data = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,  # 0 = all players
        context_measure_simple='FGA',
        season_nullable=season,
        timeout=timeout,
    )
    return shot_data.get_data_frames()[0]


def get_player_dash_stats(season, measure_type='Base', per_mode='PerGame',
                          timeout=DEFAULT_TIMEOUT):
    """Fetch league-wide player stats for a season.

    Generic wrapper over LeagueDashPlayerStats. The various league-wide datasets
    (base / advanced / per-100 / totals) differ only by these two params.

    Args:
        season: Season string (e.g. '2024-25').
        measure_type: 'Base' or 'Advanced'.
        per_mode: 'PerGame', 'Per100Possessions', or 'Totals'.
        timeout: Network timeout in seconds.

    Returns:
        DataFrame: Raw player stats frame.
    """
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense=measure_type,
        per_mode_detailed=per_mode,
        timeout=timeout,
    )
    return stats.get_data_frames()[0]


def get_standings(season, timeout=DEFAULT_TIMEOUT):
    """Fetch raw league standings for a season.

    Args:
        season: Season string (e.g. '2024-25').
        timeout: Network timeout in seconds.

    Returns:
        DataFrame: Raw standings frame.
    """
    standings = leaguestandings.LeagueStandings(
        season=season,
        league_id='00',
        timeout=timeout,
    )
    return standings.get_data_frames()[0]
