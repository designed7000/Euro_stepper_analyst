"""
Data processing and transformation functions.

This module shapes raw API frames (from data/fetch.py) into the app-facing
datasets. Transforms here run once at refresh time and the result is what gets
stored as a snapshot.
"""

import numpy as np
import pandas as pd

from config import ZONE_MAPPING, ZONE_ORDER, MIN_GAMES, MIN_MINUTES, MIN_MINUTES_LEADERS
from utils.helpers import estimate_position


# --- LEAGUE-WIDE DATASET BUILDERS ----------------------------------------
# These take raw frames from data/fetch.py and produce the snapshot that the
# repository serves. They contain the transforms that previously lived inside
# the cached API functions.

def compute_league_zone_averages(league_shot_df):
    """Aggregate raw league-wide shots into per-zone FG% and frequency.

    Args:
        league_shot_df: Raw league shot chart (fetch.get_league_shot_chart).

    Returns:
        DataFrame with ['SHOT_ZONE_BASIC', 'league_fg_pct', 'league_freq_pct'].
    """
    total_attempts = len(league_shot_df)
    zone_avgs = league_shot_df.groupby('SHOT_ZONE_BASIC').agg(
        league_makes=('SHOT_MADE_FLAG', 'sum'),
        league_attempts=('SHOT_MADE_FLAG', 'count')
    ).reset_index()
    zone_avgs['league_fg_pct'] = zone_avgs['league_makes'] / zone_avgs['league_attempts']
    zone_avgs['league_freq_pct'] = zone_avgs['league_attempts'] / total_attempts
    return zone_avgs[['SHOT_ZONE_BASIC', 'league_fg_pct', 'league_freq_pct']]


def build_advanced_stats(base_df, adv_df):
    """Merge base + advanced player stats for similarity/style analysis.

    Args:
        base_df: Raw Base PerGame frame.
        adv_df: Raw Advanced PerGame frame.

    Returns:
        DataFrame with merged stats, 3P attempt rate, and low-sample players removed.
    """
    merged = base_df.merge(
        adv_df[['PLAYER_ID', 'PLAYER_NAME', 'USG_PCT', 'TS_PCT', 'AST_PCT', 'REB_PCT', 'PACE']],
        on=['PLAYER_ID', 'PLAYER_NAME'],
        how='left'
    )

    # Feature engineering: 3-point attempt rate
    merged['3P_AR'] = (merged['FG3A'] / merged['FGA']).fillna(0)

    # Remove low-sample players
    merged = merged[(merged['GP'] >= MIN_GAMES) & (merged['MIN'] >= MIN_MINUTES)]
    return merged


def build_league_leaders(base_df, adv_df, per100_df):
    """Build the league leaders dataset with positions and advanced metrics.

    Args:
        base_df: Raw Base PerGame frame.
        adv_df: Raw Advanced PerGame frame.
        per100_df: Raw Base Per100Possessions frame.

    Returns:
        DataFrame with merged stats, estimated positions, low-sample players removed.
    """
    df = base_df

    adv_cols = ['PLAYER_ID', 'TS_PCT', 'USG_PCT', 'AST_TO', 'AST_RATIO',
                'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PIE']
    df = df.merge(adv_df[adv_cols], on='PLAYER_ID', how='left')

    per100_cols = ['PLAYER_ID', 'AST', 'TOV', 'PTS', 'STL', 'BLK']
    per100_renamed = per100_df[per100_cols].copy()
    per100_renamed.columns = ['PLAYER_ID', 'AST_PER100', 'TOV_PER100',
                              'PTS_PER100', 'STL_PER100', 'BLK_PER100']
    df = df.merge(per100_renamed, on='PLAYER_ID', how='left')

    # Filter: minimum games and minutes
    df = df[(df['GP'] >= MIN_GAMES) & (df['MIN'] >= MIN_MINUTES_LEADERS)]

    # Estimate positions and drop rows without one
    df['POSITION_GROUP'] = df.apply(estimate_position, axis=1)
    df = df[df['POSITION_GROUP'].notna()]
    return df


def clean_standings(raw_df):
    """Select and rename the standings columns the app uses.

    Args:
        raw_df: Raw standings frame (fetch.get_standings).

    Returns:
        DataFrame with relevant columns, a TEAM_FULL name, and TEAM_ABBREVIATION.
    """
    cols_to_keep = ['TeamID', 'TeamCity', 'TeamName', 'TeamSlug', 'Conference',
                    'WINS', 'LOSSES', 'WinPCT', 'HOME', 'ROAD']
    available_cols = [c for c in cols_to_keep if c in raw_df.columns]
    standings_df = raw_df[available_cols].copy()

    if 'TeamCity' in standings_df.columns and 'TeamName' in standings_df.columns:
        standings_df['TEAM_FULL'] = standings_df['TeamCity'] + ' ' + standings_df['TeamName']

    if 'TeamSlug' in standings_df.columns:
        standings_df.rename(columns={'TeamSlug': 'TEAM_ABBREVIATION'}, inplace=True)

    return standings_df


def filter_mvp_stats(totals_df, pergame_df):
    """Apply the sample-size filter to the MVP totals/per-game frames.

    Args:
        totals_df: Raw Base Totals frame.
        pergame_df: Raw Base PerGame frame.

    Returns:
        tuple: (filtered_totals_df, filtered_pergame_df).
    """
    totals_df = totals_df[
        (totals_df['GP'] >= MIN_GAMES) & (totals_df['MIN'] / totals_df['GP'] >= MIN_MINUTES)
    ]
    pergame_df = pergame_df[
        (pergame_df['GP'] >= MIN_GAMES) & (pergame_df['MIN'] >= MIN_MINUTES)
    ]
    return totals_df, pergame_df


# --- PER-PLAYER SHOT PROCESSING ------------------------------------------


def process_player_data(df, league_avg_df):
    """Process player shot data: merge with league averages, calculate metrics.
    
    Args:
        df: Player shot data DataFrame
        league_avg_df: League averages DataFrame
        
    Returns:
        Processed DataFrame with relative efficiency and other metrics
    """
    # Merge with league averages to calculate relative efficiency per shot
    merge_cols = ['SHOT_ZONE_BASIC', 'SHOT_ZONE_AREA', 'SHOT_ZONE_RANGE']
    df = df.merge(
        league_avg_df[merge_cols + ['FG_PCT']], 
        on=merge_cols, 
        how='left',
        suffixes=('', '_LEAGUE')
    )
    df.rename(columns={'FG_PCT': 'LEAGUE_FG_PCT'}, inplace=True)
    
    # Calculate relative efficiency
    df['RELATIVE_EFFICIENCY'] = df['SHOT_MADE_FLAG'] - df['LEAGUE_FG_PCT']
    
    # Calculate SHOT_VALUE and GSAA. Use np.where rather than .apply so the
    # column is always integer-typed: on an empty frame .apply preserves the
    # (Arrow string) SHOT_TYPE dtype, which then breaks the multiplication below.
    is_three = df['SHOT_TYPE'].astype(str).str.contains('3PT')
    df['SHOT_VALUE'] = np.where(is_three, 3, 2)
    df['GSAA'] = (df['SHOT_MADE_FLAG'] * df['SHOT_VALUE']) - (df['LEAGUE_FG_PCT'] * df['SHOT_VALUE'])
    
    # Coordinate transformation
    df['x'] = df['LOC_X'] / 10
    df['y'] = df['LOC_Y'] / 10
    df['made'] = df['SHOT_MADE_FLAG'].map({1: 'Made', 0: 'Missed'})
    
    return df


def calculate_player_metrics(df):
    """Calculate summary metrics for a player.
    
    Args:
        df: Processed player shot data DataFrame
        
    Returns:
        dict: Dictionary of player metrics
    """
    makes = len(df[df['SHOT_MADE_FLAG'] == 1])
    attempts = len(df)
    threes_made = len(df[(df['SHOT_TYPE'] == '3PT Field Goal') & (df['SHOT_MADE_FLAG'] == 1)])
    efg = (makes + (0.5 * threes_made)) / attempts if attempts > 0 else 0
    fg_pct = makes / attempts if attempts > 0 else 0
    total_gsaa = df['GSAA'].sum()
    threes_attempted = len(df[df['SHOT_TYPE'] == '3PT Field Goal'])
    
    return {
        'attempts': attempts,
        'makes': makes,
        'fg_pct': fg_pct,
        'efg': efg,
        'gsaa': total_gsaa,
        'threes_made': threes_made,
        'threes_attempted': threes_attempted
    }


def calculate_zone_stats(df, player_name):
    """Calculate stats by zone group for a player.
    
    Args:
        df: Player shot data DataFrame with zone_group column
        player_name: Player name for labeling
        
    Returns:
        DataFrame with zone statistics
    """
    stats = []
    for zone in ZONE_ORDER:
        zone_df = df[df['zone_group'] == zone]
        count = len(zone_df)
        makes = zone_df['SHOT_MADE_FLAG'].sum()
        fg_pct = (makes / count * 100) if count > 0 else 0
        freq_pct = (count / len(df) * 100) if len(df) > 0 else 0
        stats.append({
            'zone': zone,
            'count': count,
            'makes': makes,
            'fg_pct': fg_pct,
            'freq_pct': freq_pct,
            'player': player_name
        })
    return pd.DataFrame(stats)


def add_zone_groups(df):
    """Add zone_group column to DataFrame based on SHOT_ZONE_BASIC.
    
    Args:
        df: Shot data DataFrame
        
    Returns:
        DataFrame with zone_group column added
    """
    df['zone_group'] = df['SHOT_ZONE_BASIC'].map(ZONE_MAPPING)
    return df


def get_top_players_by_position_smart(df, category, n=5):
    """Get top N players per position using smart composite metrics.
    
    Args:
        df: League leaders DataFrame
        category: One of 'scoring', 'playmaking', 'impact'
        n: Number of players to return per position
        
    Returns:
        dict: Dictionary with position keys and DataFrames as values
    """
    results = {}
    
    for pos in ['Guard', 'Forward', 'Center']:
        pos_df = df[df['POSITION_GROUP'] == pos].copy()
        
        if pos_df.empty:
            continue
        
        if category == 'scoring':
            # Scoring Impact = USG% × TS% (load × efficiency)
            pos_df['_score'] = (pos_df['USG_PCT'].fillna(0.2) * pos_df['TS_PCT'].fillna(0.5))
            pos_df = pos_df.nlargest(n, '_score')
            pos_df['Value'] = pos_df.apply(
                lambda r: f"{r['PTS']:.1f} pts | {r['USG_PCT']*100:.1f}% USG | {r['TS_PCT']*100:.1f}% TS", axis=1
            )
            
        elif category == 'playmaking':
            # Playmaking = AST per 100 poss weighted by AST/TO ratio
            pos_df['_score'] = pos_df['AST_PER100'].fillna(0) * (1 + pos_df['AST_TO'].fillna(1) * 0.2)
            pos_df = pos_df.nlargest(n, '_score')
            pos_df['Value'] = pos_df.apply(
                lambda r: f"{r['AST_PER100']:.1f} ast/100 | {r['AST_TO']:.2f} A/TO", axis=1
            )
            
        elif category == 'impact':
            # Two-Way Impact = Net Rating
            pos_df['_score'] = pos_df['NET_RATING'].fillna(0)
            pos_df = pos_df.nlargest(n, '_score')
            pos_df['Value'] = pos_df.apply(
                lambda r: f"+{r['NET_RATING']:.1f} NET | {r['OFF_RATING']:.0f} OFF | {r['DEF_RATING']:.0f} DEF" 
                          if r['NET_RATING'] >= 0 
                          else f"{r['NET_RATING']:.1f} NET | {r['OFF_RATING']:.0f} OFF | {r['DEF_RATING']:.0f} DEF", 
                axis=1
            )
        
        results[pos] = pos_df[['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Value']].copy()
        results[pos].columns = ['Player', 'Team', 'Stats']
    
    return results
