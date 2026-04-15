"""
Utility functions for player name matching and data normalization.
"""

from difflib import SequenceMatcher, get_close_matches
from nba_api.stats.static import players


def get_all_player_names():
    """Get all NBA player names for fuzzy matching."""
    nba_players = players.get_players()
    return {p['full_name'].lower(): p for p in nba_players}


def get_player_id(name):
    """Find player ID with fuzzy matching support.
    
    Returns:
        tuple: (player_id, corrected_name, match_message)
    """
    nba_players = players.get_players()
    name_lower = name.lower().strip()
    
    # 1. Exact match (case-insensitive)
    for p in nba_players:
        if p['full_name'].lower() == name_lower:
            return p['id'], p['full_name'], None
    
    # 2. Partial match (first name or last name matches)
    for p in nba_players:
        full_name_lower = p['full_name'].lower()
        # Check if input contains both parts of the name (handles "steph curry" -> "Stephen Curry")
        name_parts = name_lower.split()
        player_parts = full_name_lower.split()
        if len(name_parts) >= 2 and len(player_parts) >= 2:
            # Check if first letters match and last name is close
            if (name_parts[0][0] == player_parts[0][0] and 
                SequenceMatcher(None, name_parts[-1], player_parts[-1]).ratio() > 0.8):
                return p['id'], p['full_name'], f"Showing results for: {p['full_name']}"
    
    # 3. Fuzzy match using difflib
    all_names = [p['full_name'] for p in nba_players]
    all_names_lower = [n.lower() for n in all_names]
    
    # Find close matches
    close_matches = get_close_matches(name_lower, all_names_lower, n=1, cutoff=0.6)
    
    if close_matches:
        # Find the original name with correct casing
        matched_idx = all_names_lower.index(close_matches[0])
        matched_player = nba_players[matched_idx]
        return matched_player['id'], matched_player['full_name'], f"Showing results for: {matched_player['full_name']}"
    
    # 4. Try matching just last name
    for p in nba_players:
        player_parts = p['full_name'].lower().split()
        if len(player_parts) >= 2:
            if SequenceMatcher(None, name_lower, player_parts[-1]).ratio() > 0.85:
                return p['id'], p['full_name'], f"Showing results for: {p['full_name']}"
    
    return None, None, None


def normalize_metric(val, min_val, max_val):
    """Normalize a value to 0-100 scale based on given range."""
    return max(0, min(100, ((val - min_val) / (max_val - min_val)) * 100))


def safe_val(val, default=0):
    """Safely extract a numeric value, handling NaN."""
    import pandas as pd
    if pd.isna(val):
        return default
    return float(val)


def estimate_position(row):
    """Estimate player position based on stats (heuristic method).
    
    Args:
        row: DataFrame row with player stats
        
    Returns:
        str: 'Guard', 'Forward', or 'Center'
    """
    ast = float(row.get('AST', 0) or 0)
    reb = float(row.get('REB', 0) or 0)
    blk = float(row.get('BLK', 0) or 0)
    fg3a = float(row.get('FG3A', 0) or 0)
    fga = float(row.get('FGA', 1) or 1)
    
    three_rate = fg3a / fga if fga > 0 else 0
    
    # Centers: High rebounds OR high blocks
    if (reb >= 8) or (reb >= 6 and blk >= 1.2) or (blk >= 1.5):
        return 'Center'
    # Guards: High assists OR high 3pt rate with low rebounds
    elif ast >= 4.5 or (three_rate > 0.4 and reb < 5) or (reb < 4 and ast >= 3):
        return 'Guard'
    # Forwards: Everything else
    else:
        return 'Forward'
