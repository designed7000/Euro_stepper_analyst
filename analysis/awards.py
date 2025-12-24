"""
Award tracking analysis - MVP ladder calculations using DNA Production Index.
"""

import pandas as pd
import numpy as np
from difflib import get_close_matches


class AwardTracker:
    """Handles MVP and award calculations using scarcity-based valuation."""
    
    def __init__(self, players_df=None, standings_df=None):
        self.scarcity_weights = {}
        self.players_df = players_df
        self.standings_df = standings_df
        
    def calculate_scarcity_weights(self, totals_df):
        """Calculate scarcity weights based on league totals.
        
        The "Exchange Rate" for each stat relative to Points.
        Formula: Weight_Stat = (Total_League_Points / Total_League_Stat) * Impact_Modifier
        
        Impact Modifiers balance the raw scarcity to avoid center/big-man bias:
        - Assists: 1.5x (reward playmaking/creation)
        - Rebounds: 0.7x (slightly devalue raw board crashing)
        - Blocks: 0.6x (dampen scarcity so high-block players don't break the model)
        - Steals: 1.0x (keep as is)
        
        Args:
            totals_df: DataFrame with player season totals
            
        Returns:
            dict: Scarcity weights for each stat category
        """
        # Sum up league totals
        total_pts = totals_df['PTS'].sum()
        total_reb = totals_df['REB'].sum()
        total_ast = totals_df['AST'].sum()
        total_stl = totals_df['STL'].sum()
        total_blk = totals_df['BLK'].sum()
        total_tov = totals_df['TOV'].sum()
        
        # Impact modifiers to balance position bias
        IMPACT_MODIFIERS = {
            'PTS': 1.0,   # Baseline
            'AST': 1.5,   # Reward playmaking/creation
            'REB': 0.7,   # Slightly devalue raw board crashing
            'STL': 1.0,   # Keep as is
            'BLK': 0.6,   # Dampen scarcity value for blocks
            'TOV': 1.0    # Penalty modifier
        }
        
        # Calculate base exchange rates then apply impact modifiers
        self.scarcity_weights = {
            'PTS': 1.0 * IMPACT_MODIFIERS['PTS'],
            'REB': (total_pts / total_reb if total_reb > 0 else 1.0) * IMPACT_MODIFIERS['REB'],
            'AST': (total_pts / total_ast if total_ast > 0 else 1.0) * IMPACT_MODIFIERS['AST'],
            'STL': (total_pts / total_stl if total_stl > 0 else 1.0) * IMPACT_MODIFIERS['STL'],
            'BLK': (total_pts / total_blk if total_blk > 0 else 1.0) * IMPACT_MODIFIERS['BLK'],
            'TOV': (total_pts / total_tov if total_tov > 0 else 1.0) * IMPACT_MODIFIERS['TOV']
        }
        
        return self.scarcity_weights
    
    def _match_team_to_standings(self, team_id, standings_df):
        """Match a team ID to standings data.
        
        Args:
            team_id: Team ID (e.g., 1610612744)
            standings_df: Standings DataFrame
            
        Returns:
            dict: {'win_pct': float, 'wins': int, 'losses': int} or None if not found
        """
        # Match on TeamID
        if 'TeamID' in standings_df.columns:
            match = standings_df[standings_df['TeamID'] == team_id]
            if not match.empty:
                row = match.iloc[0]
                return {
                    'win_pct': row['WinPCT'],
                    'wins': row['WINS'],
                    'losses': row['LOSSES']
                }
        
        return None
    
    def calculate_mvp_ladder(self, top_n=20, totals_df=None, pergame_df=None, standings_df=None):
        """Calculate MVP ladder using DNA Production Index.
        
        Args:
            top_n: Number of players to return
            totals_df: Player season totals (optional, uses stored if not provided)
            pergame_df: Player per-game averages (optional, uses stored if not provided)
            standings_df: Team standings with win % (optional, uses stored if not provided)
            
        Returns:
            DataFrame: MVP ladder with rankings and component scores
        """
        # Use stored dataframes if not provided
        if totals_df is None:
            totals_df = self.players_df
        if pergame_df is None:
            pergame_df = self.players_df  # Leaders df already has per-game stats
        if standings_df is None:
            standings_df = self.standings_df
        
        if totals_df is None or standings_df is None:
            return pd.DataFrame()
        
        # Calculate scarcity weights from current data
        weights = self.calculate_scarcity_weights(totals_df)
        
        # Work with per-game stats for the calculation
        df = pergame_df.copy()
        
        # Calculate weighted production components
        df['SCORING_VALUE'] = df['PTS'] * weights['PTS']
        df['REBOUNDING_VALUE'] = df['REB'] * weights['REB']
        df['PLAYMAKING_VALUE'] = df['AST'] * weights['AST']
        df['STEALS_VALUE'] = df['STL'] * weights['STL']
        df['BLOCKS_VALUE'] = df['BLK'] * weights['BLK']
        df['TURNOVER_PENALTY'] = df['TOV'] * weights['TOV']
        
        # Group for visualization
        df['DEFENSIVE_VALUE'] = df['STEALS_VALUE'] + df['BLOCKS_VALUE']
        
        # Calculate Raw Production Value
        df['RAW_VALUE'] = (
            df['SCORING_VALUE'] + 
            df['REBOUNDING_VALUE'] + 
            df['PLAYMAKING_VALUE'] + 
            df['STEALS_VALUE'] + 
            df['BLOCKS_VALUE'] - 
            df['TURNOVER_PENALTY']
        )
        
        # Match teams to standings using TEAM_ID and get win %, wins, losses
        def get_team_stats(team_id):
            result = self._match_team_to_standings(team_id, standings_df)
            if result:
                return pd.Series([result['win_pct'], result['wins'], result['losses']])
            return pd.Series([None, None, None])
        
        team_stats = df['TEAM_ID'].apply(get_team_stats)
        df['WIN_PCT'] = team_stats[0]
        df['WINS'] = team_stats[1]
        df['LOSSES'] = team_stats[2]
        
        # Count successful matches for debugging
        matched_count = df['WIN_PCT'].notna().sum()
        total_count = len(df)
        
        # Only use fallback if most matches failed (indicates data issue)
        if matched_count < total_count * 0.5:
            print(f"WARNING: Only {matched_count}/{total_count} teams matched to standings. Check TEAM_ID matching.")
        
        # Fill missing win % with league average (shouldn't happen with TEAM_ID matching)
        avg_win_pct = df['WIN_PCT'].dropna().mean()
        df['WIN_PCT'] = df['WIN_PCT'].fillna(avg_win_pct if not pd.isna(avg_win_pct) else 0.5)
        df['WINS'] = df['WINS'].fillna(0).astype(int)
        df['LOSSES'] = df['LOSSES'].fillna(0).astype(int)
        
        # Create Record column (e.g., "45-20")
        df['RECORD'] = df['WINS'].astype(str) + '-' + df['LOSSES'].astype(str)
        
        # Calculate final MVP Score with square root win % adjustment
        # This softens the penalty for players on losing teams
        df['MVP_SCORE'] = df['RAW_VALUE'] * (df['WIN_PCT'] ** 0.5)
        
        # Sort by MVP Score and get top N
        mvp_ladder = df.nlargest(top_n, 'MVP_SCORE').copy()
        
        # Add rank
        mvp_ladder['RANK'] = range(1, len(mvp_ladder) + 1)
        
        # Normalize MVP Score to 0-100 scale for display
        max_score = mvp_ladder['MVP_SCORE'].max()
        mvp_ladder['MVP_SCORE_NORMALIZED'] = (mvp_ladder['MVP_SCORE'] / max_score * 100) if max_score > 0 else 0
        
        # Add Rank column for display
        mvp_ladder['Rank'] = range(1, len(mvp_ladder) + 1)
        
        return mvp_ladder
    
    def get_display_ladder(self, mvp_ladder):
        """Format MVP ladder for display.
        
        Args:
            mvp_ladder: Full MVP ladder DataFrame
            
        Returns:
            DataFrame: Formatted for Streamlit display
        """
        display_df = mvp_ladder[[
            'RANK', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 
            'PTS', 'REB', 'AST', 'WIN_PCT', 'MVP_SCORE_NORMALIZED'
        ]].copy()
        
        display_df.columns = [
            'Rank', 'Player', 'Team', 
            'PPG', 'RPG', 'APG', 'Win%', 'MVP Score'
        ]
        
        # Format columns
        display_df['Win%'] = display_df['Win%'].apply(lambda x: f"{x:.1%}")
        display_df['MVP Score'] = display_df['MVP Score'].apply(lambda x: f"{x:.1f}")
        display_df['PPG'] = display_df['PPG'].apply(lambda x: f"{x:.1f}")
        display_df['RPG'] = display_df['RPG'].apply(lambda x: f"{x:.1f}")
        display_df['APG'] = display_df['APG'].apply(lambda x: f"{x:.1f}")
        
        return display_df
