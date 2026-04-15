"""
Data processing and transformation functions.
"""

import pandas as pd

from config import ZONE_MAPPING, ZONE_ORDER


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
    
    # Calculate SHOT_VALUE and GSAA
    df['SHOT_VALUE'] = df['SHOT_TYPE'].apply(lambda x: 3 if '3PT' in str(x) else 2)
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
