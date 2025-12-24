"""
Player similarity analysis using machine learning.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


def build_similarity_model(stats_df):
    """Build the ML pipeline for player similarity.
    
    Args:
        stats_df: DataFrame with player advanced stats
        
    Returns:
        tuple: (nn_model, scaler, feature_cols)
    """
    # Feature columns for "Style Vector"
    feature_cols = ['USG_PCT', 'TS_PCT', 'AST_PCT', 'REB_PCT', 'PACE', '3P_AR']
    
    # Extract features and handle NaN
    X = stats_df[feature_cols].fillna(0).values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit NearestNeighbors model
    nn_model = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
    nn_model.fit(X_scaled)
    
    return nn_model, scaler, feature_cols


def find_similar_players(player_name, stats_df, nn_model, scaler, feature_cols):
    """Find the 5 most similar players to the given player.
    
    Args:
        player_name: Name of the player to find matches for
        stats_df: DataFrame with player stats
        nn_model: Fitted NearestNeighbors model
        scaler: Fitted StandardScaler
        feature_cols: List of feature column names
        
    Returns:
        tuple: (similar_players_list, player_row, error_message)
    """
    # Find the player in the dataset
    player_row = stats_df[stats_df['PLAYER_NAME'].str.lower() == player_name.lower()]
    
    if player_row.empty:
        return None, None, "Player not found in current season stats"
    
    # Get player's feature vector
    player_features = player_row[feature_cols].fillna(0).values
    player_scaled = scaler.transform(player_features)
    
    # Find nearest neighbors
    distances, indices = nn_model.kneighbors(player_scaled)
    
    # Get similar players (exclude the player themselves - index 0)
    similar_players = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if i == 0:  # Skip self
            continue
        player_data = stats_df.iloc[idx]
        similarity_score = max(0, 100 - (dist * 20))  # Convert distance to similarity %
        similar_players.append({
            'Rank': i,
            'Player': player_data['PLAYER_NAME'],
            'Team': player_data['TEAM_ABBREVIATION'],
            'Similarity': f"{similarity_score:.1f}%",
            'USG%': f"{player_data['USG_PCT']:.1f}",
            'TS%': f"{player_data['TS_PCT']:.1f}",
            '3P Rate': f"{player_data['3P_AR']:.1%}",
            '_distance': dist,
            '_idx': idx
        })
    
    return similar_players, player_row.iloc[0], None


def get_player_style_values(player_data):
    """Extract style values from player data for radar chart.
    
    Args:
        player_data: Series/row with player stats
        
    Returns:
        list: [USG, TS, AST, REB, PACE, 3P_RATE]
    """
    import pandas as pd
    
    def safe_val(val, default=0):
        if pd.isna(val):
            return default
        return float(val)
    
    return [
        safe_val(player_data['USG_PCT']),
        safe_val(player_data['TS_PCT']),
        safe_val(player_data['AST_PCT']),
        safe_val(player_data['REB_PCT']),
        safe_val(player_data['PACE']),
        safe_val(player_data['3P_AR']) * 100  # Convert decimal to percentage
    ]
