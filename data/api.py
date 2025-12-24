"""
NBA API data fetching functions.
All API calls are cached using Streamlit's cache_data decorator.
"""

import time
import streamlit as st
from nba_api.stats.endpoints import shotchartdetail, leaguedashplayerstats

from utils.helpers import estimate_position
from config import MIN_GAMES, MIN_MINUTES, MIN_MINUTES_LEADERS


@st.cache_data
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
    player_shots = frames[0]
    league_averages = frames[1]
    return player_shots, league_averages


@st.cache_data
def get_league_averages(season):
    """Fetch league-wide shot data to calculate zone averages.
    
    Args:
        season: Season string (e.g., '2024-25')
        
    Returns:
        DataFrame with league FG% and frequency by zone
    """
    shot_data = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,  # 0 = all players
        context_measure_simple='FGA',
        season_nullable=season
    )
    league_df = shot_data.get_data_frames()[0]
    
    # Calculate FG% by zone
    total_league_attempts = len(league_df)
    zone_avgs = league_df.groupby('SHOT_ZONE_BASIC').agg(
        league_makes=('SHOT_MADE_FLAG', 'sum'),
        league_attempts=('SHOT_MADE_FLAG', 'count')
    ).reset_index()
    zone_avgs['league_fg_pct'] = zone_avgs['league_makes'] / zone_avgs['league_attempts']
    zone_avgs['league_freq_pct'] = zone_avgs['league_attempts'] / total_league_attempts
    
    return zone_avgs[['SHOT_ZONE_BASIC', 'league_fg_pct', 'league_freq_pct']]


@st.cache_data
def get_advanced_stats(season):
    """Fetch and merge base + advanced stats for all players in a season.
    
    Args:
        season: Season string (e.g., '2024-25')
        
    Returns:
        DataFrame with merged base and advanced stats
    """
    # Call 1: Base stats (for FG3A, FGA)
    base_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame'
    )
    base_df = base_stats.get_data_frames()[0]
    time.sleep(0.6)  # Respect API rate limits
    
    # Call 2: Advanced stats (for USG_PCT, TS_PCT, AST_PCT, etc.)
    adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame'
    )
    adv_df = adv_stats.get_data_frames()[0]
    
    # Merge on PLAYER_ID and PLAYER_NAME
    merged = base_df.merge(
        adv_df[['PLAYER_ID', 'PLAYER_NAME', 'USG_PCT', 'TS_PCT', 'AST_PCT', 'REB_PCT', 'PACE']],
        on=['PLAYER_ID', 'PLAYER_NAME'],
        how='left'
    )
    
    # Feature Engineering: 3-Point Attempt Rate
    merged['3P_AR'] = merged['FG3A'] / merged['FGA']
    merged['3P_AR'] = merged['3P_AR'].fillna(0)
    
    # Filtering: Remove low-sample players
    merged = merged[(merged['GP'] >= MIN_GAMES) & (merged['MIN'] >= MIN_MINUTES)]
    
    return merged


@st.cache_data
def get_league_leaders(season):
    """Fetch league leaders with position data and advanced metrics.
    
    Args:
        season: Season string (e.g., '2024-25')
        
    Returns:
        DataFrame with player stats and estimated positions
    """
    # Get base stats (per game)
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame'
    )
    df = stats.get_data_frames()[0]
    time.sleep(0.4)
    
    # Get advanced stats (USG%, TS%, OFF/DEF Rating, AST_TO, etc.)
    adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame'
    )
    adv_df = adv_stats.get_data_frames()[0]
    time.sleep(0.4)
    
    # Get per-100 possessions stats for pace-adjusted metrics
    per100_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='Per100Possessions'
    )
    per100_df = per100_stats.get_data_frames()[0]
    
    # Merge advanced stats
    adv_cols = ['PLAYER_ID', 'TS_PCT', 'USG_PCT', 'AST_TO', 'AST_RATIO', 
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PIE']
    df = df.merge(adv_df[adv_cols], on='PLAYER_ID', how='left')
    
    # Merge per-100 stats (rename to avoid conflicts)
    per100_cols = ['PLAYER_ID', 'AST', 'TOV', 'PTS', 'STL', 'BLK']
    per100_renamed = per100_df[per100_cols].copy()
    per100_renamed.columns = ['PLAYER_ID', 'AST_PER100', 'TOV_PER100', 'PTS_PER100', 'STL_PER100', 'BLK_PER100']
    df = df.merge(per100_renamed, on='PLAYER_ID', how='left')
    
    # Filter: minimum games and minutes
    df = df[(df['GP'] >= MIN_GAMES) & (df['MIN'] >= MIN_MINUTES_LEADERS)]
    
    # Add position estimation
    df['POSITION_GROUP'] = df.apply(estimate_position, axis=1)
    
    # Remove players without position data
    df = df[df['POSITION_GROUP'].notna()]
    
    return df
