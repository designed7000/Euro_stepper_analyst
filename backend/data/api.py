"""
NBA API data fetching functions.
All API calls are cached using a TTL cache (replacing Streamlit's @st.cache_data).
"""

import time
from nba_api.stats.endpoints import shotchartdetail, leaguedashplayerstats, leaguestandings

from utils.helpers import estimate_position
from config import MIN_GAMES, MIN_MINUTES, MIN_MINUTES_LEADERS
from cache import ttl_cache


@ttl_cache(ttl=3600)
def get_player_shots(player_id, season, clutch_only=False):
    """Fetch shot chart data for a specific player.

    Args:
        player_id: NBA player ID
        season: Season string (e.g., '2024-25')
        clutch_only: If True, only fetch clutch time shots

    Returns:
        tuple: (player_shots_df, league_averages_df)
    """
    clutch_time = 'Last 5 Minutes' if clutch_only else None
    ahead_behind = 'Ahead or Behind' if clutch_only else None

    shot_data = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        context_measure_simple='FGA',
        season_nullable=season,
        clutch_time_nullable=clutch_time,
        ahead_behind_nullable=ahead_behind
    )
    frames = shot_data.get_data_frames()
    return frames[0], frames[1]


@ttl_cache(ttl=3600)
def get_league_averages(season):
    """Fetch league-wide shot data to calculate zone averages.

    Args:
        season: Season string (e.g., '2024-25')

    Returns:
        DataFrame with league FG% and frequency by zone
    """
    shot_data = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,
        context_measure_simple='FGA',
        season_nullable=season
    )
    league_df = shot_data.get_data_frames()[0]

    total_league_attempts = len(league_df)
    zone_avgs = league_df.groupby('SHOT_ZONE_BASIC').agg(
        league_makes=('SHOT_MADE_FLAG', 'sum'),
        league_attempts=('SHOT_MADE_FLAG', 'count')
    ).reset_index()
    zone_avgs['league_fg_pct'] = zone_avgs['league_makes'] / zone_avgs['league_attempts']
    zone_avgs['league_freq_pct'] = zone_avgs['league_attempts'] / total_league_attempts

    return zone_avgs[['SHOT_ZONE_BASIC', 'league_fg_pct', 'league_freq_pct']]


@ttl_cache(ttl=3600)
def get_advanced_stats(season):
    """Fetch and merge base + advanced stats for all players.

    Args:
        season: Season string (e.g., '2024-25')

    Returns:
        DataFrame with merged base and advanced stats
    """
    base_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame'
    )
    base_df = base_stats.get_data_frames()[0]
    time.sleep(0.6)

    adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame'
    )
    adv_df = adv_stats.get_data_frames()[0]

    merged = base_df.merge(
        adv_df[['PLAYER_ID', 'PLAYER_NAME', 'USG_PCT', 'TS_PCT', 'AST_PCT', 'REB_PCT', 'PACE']],
        on=['PLAYER_ID', 'PLAYER_NAME'],
        how='left'
    )

    merged['3P_AR'] = merged['FG3A'] / merged['FGA']
    merged['3P_AR'] = merged['3P_AR'].fillna(0)
    merged = merged[(merged['GP'] >= MIN_GAMES) & (merged['MIN'] >= MIN_MINUTES)]

    return merged


@ttl_cache(ttl=3600)
def get_league_leaders(season):
    """Fetch league leaders with position data and advanced metrics.

    Args:
        season: Season string (e.g., '2024-25')

    Returns:
        DataFrame with player stats and estimated positions
    """
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame'
    )
    df = stats.get_data_frames()[0]
    time.sleep(0.4)

    adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame'
    )
    adv_df = adv_stats.get_data_frames()[0]
    time.sleep(0.4)

    per100_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='Per100Possessions'
    )
    per100_df = per100_stats.get_data_frames()[0]

    adv_cols = ['PLAYER_ID', 'TS_PCT', 'USG_PCT', 'AST_TO', 'AST_RATIO',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PIE']
    df = df.merge(adv_df[adv_cols], on='PLAYER_ID', how='left')

    per100_cols = ['PLAYER_ID', 'AST', 'TOV', 'PTS', 'STL', 'BLK']
    per100_renamed = per100_df[per100_cols].copy()
    per100_renamed.columns = ['PLAYER_ID', 'AST_PER100', 'TOV_PER100', 'PTS_PER100', 'STL_PER100', 'BLK_PER100']
    df = df.merge(per100_renamed, on='PLAYER_ID', how='left')

    df = df[(df['GP'] >= MIN_GAMES) & (df['MIN'] >= MIN_MINUTES_LEADERS)]
    df['POSITION_GROUP'] = df.apply(estimate_position, axis=1)
    df = df[df['POSITION_GROUP'].notna()]

    return df


@ttl_cache(ttl=3600)
def get_standings(season):
    """Fetch league standings with team records.

    Args:
        season: Season string (e.g., '2024-25')

    Returns:
        DataFrame with team standings including win percentage
    """
    standings = leaguestandings.LeagueStandings(
        season=season,
        league_id='00'
    )
    df = standings.get_data_frames()[0]

    cols_to_keep = ['TeamID', 'TeamCity', 'TeamName', 'TeamSlug', 'Conference',
                    'WINS', 'LOSSES', 'WinPCT', 'HOME', 'ROAD']
    available_cols = [c for c in cols_to_keep if c in df.columns]
    standings_df = df[available_cols].copy()

    if 'TeamCity' in standings_df.columns and 'TeamName' in standings_df.columns:
        standings_df['TEAM_FULL'] = standings_df['TeamCity'] + ' ' + standings_df['TeamName']

    if 'TeamSlug' in standings_df.columns:
        standings_df.rename(columns={'TeamSlug': 'TEAM_ABBREVIATION'}, inplace=True)

    return standings_df


@ttl_cache(ttl=3600)
def get_mvp_stats(season):
    """Fetch player stats for MVP calculations (totals + per-game).

    Args:
        season: Season string (e.g., '2024-25')

    Returns:
        tuple: (totals_df, pergame_df)
    """
    totals_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='Totals'
    )
    totals_df = totals_stats.get_data_frames()[0]
    time.sleep(0.4)

    pergame_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame'
    )
    pergame_df = pergame_stats.get_data_frames()[0]

    totals_df = totals_df[(totals_df['GP'] >= MIN_GAMES) & (totals_df['MIN'] / totals_df['GP'] >= MIN_MINUTES)]
    pergame_df = pergame_df[(pergame_df['GP'] >= MIN_GAMES) & (pergame_df['MIN'] >= MIN_MINUTES)]

    return totals_df, pergame_df
