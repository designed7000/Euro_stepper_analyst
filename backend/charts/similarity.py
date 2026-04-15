"""
Similarity section visualizations - clean, intuitive charts for player comparison.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Creamy white background for all charts
PLOT_BG = 'rgba(255, 253, 248, 0.9)'


def create_similarity_bar_chart(similar_players, player_name):
    """Create a horizontal bar chart showing similarity scores.
    
    Args:
        similar_players: List of similar player dicts with 'Player' and 'Similarity'
        player_name: Name of the reference player
        
    Returns:
        Plotly figure object
    """
    players = [p['Player'] for p in similar_players]
    # Extract numeric value from "85.2%" format
    similarities = [float(p['Similarity'].replace('%', '')) for p in similar_players]
    
    # Create gradient colors from dark to light blue
    colors = ['#1a5276', '#2874a6', '#3498db', '#5dade2', '#85c1e9']
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=players[::-1],  # Reverse for top-to-bottom ranking
        x=similarities[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f"{s:.1f}%" for s in similarities[::-1]],
        textposition='inside',
        textfont=dict(color='white', size=14, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Similarity: %{x:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text=f"Similarity Scores to {player_name}",
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Similarity %",
            range=[0, 100],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=12)
        ),
        height=300,
        margin=dict(l=120, r=40, t=60, b=40),
        plot_bgcolor=PLOT_BG
    )
    
    return fig


def create_stat_comparison_bars(selected_player, top_matches, league_stats):
    """Create grouped bar chart comparing key stats.
    
    Args:
        selected_player: Series with selected player data
        top_matches: List of top match player data (Series)
        league_stats: Full league stats DataFrame for percentile calc
        
    Returns:
        Plotly figure object
    """
    # Key stats to compare
    stats = ['PTS', 'AST', 'REB', 'STL', 'BLK']
    stat_labels = ['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks']
    
    # Get values
    selected_vals = [selected_player.get(s, 0) for s in stats]
    match_vals = [top_matches[0].get(s, 0) for s in stats] if top_matches else [0] * len(stats)
    
    fig = go.Figure()
    
    # Selected player bars
    fig.add_trace(go.Bar(
        name=selected_player['PLAYER_NAME'],
        x=stat_labels,
        y=selected_vals,
        marker_color='#3498db',
        text=[f"{v:.1f}" for v in selected_vals],
        textposition='outside',
        textfont=dict(size=11)
    ))
    
    # Top match bars
    if top_matches:
        fig.add_trace(go.Bar(
            name=top_matches[0]['PLAYER_NAME'],
            x=stat_labels,
            y=match_vals,
            marker_color='#e74c3c',
            text=[f"{v:.1f}" for v in match_vals],
            textposition='outside',
            textfont=dict(size=11)
        ))
    
    fig.update_layout(
        title=dict(
            text="Per Game Stats Comparison",
            font=dict(size=16)
        ),
        barmode='group',
        xaxis=dict(title=""),
        yaxis=dict(title="Per Game", showgrid=True, gridcolor='rgba(0,0,0,0.1)'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=350,
        margin=dict(t=80, b=40),
        plot_bgcolor=PLOT_BG
    )
    
    return fig


def create_style_profile_chart(selected_player, top_matches, player_name):
    """Create a clean dot chart comparing playing style features.
    
    Args:
        selected_player: Series with selected player data
        top_matches: List of dicts with match player data
        player_name: Name of selected player
        
    Returns:
        Plotly figure object
    """
    # Style features - values are in decimal format (0-1 for percentages)
    features = ['USG_PCT', 'TS_PCT', 'AST_PCT', 'REB_PCT', '3P_AR']
    labels = ['Usage Rate', 'True Shooting', 'Assist Rate', 'Rebound Rate', '3PT Rate']
    
    # Get raw values and convert to display percentages
    def get_pct(player, feat):
        val = player.get(feat, 0)
        if pd.isna(val):
            return 0
        return float(val) * 100  # Convert to percentage
    
    selected_vals = [get_pct(selected_player, f) for f in features]
    match_vals = [get_pct(top_matches[0], f) for f in features] if top_matches else [0] * len(features)
    
    fig = go.Figure()
    
    # Add connecting lines for visual clarity
    for i, label in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[selected_vals[i], match_vals[i]],
            y=[label, label],
            mode='lines',
            line=dict(color='rgba(150,150,150,0.4)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Selected player dots
    fig.add_trace(go.Scatter(
        x=selected_vals,
        y=labels,
        mode='markers+text',
        name=player_name,
        marker=dict(size=16, color='#3498db', symbol='circle', 
                    line=dict(color='white', width=2)),
        text=[f"{v:.1f}%" for v in selected_vals],
        textposition='top center',
        textfont=dict(size=10, color='#3498db'),
        hovertemplate='<b>%{y}</b>: %{x:.1f}%<extra>' + player_name + '</extra>'
    ))
    
    # Top match dots
    if top_matches:
        fig.add_trace(go.Scatter(
            x=match_vals,
            y=labels,
            mode='markers+text',
            name=top_matches[0]['PLAYER_NAME'],
            marker=dict(size=16, color='#e74c3c', symbol='diamond',
                        line=dict(color='white', width=2)),
            text=[f"{v:.1f}%" for v in match_vals],
            textposition='bottom center',
            textfont=dict(size=10, color='#e74c3c'),
            hovertemplate='<b>%{y}</b>: %{x:.1f}%<extra>' + top_matches[0]['PLAYER_NAME'] + '</extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text="Style Comparison",
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Percentage",
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.2)'
        ),
        yaxis=dict(title="", categoryorder='array', categoryarray=labels[::-1]),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=320,
        margin=dict(l=100, t=70, b=40, r=40),
        plot_bgcolor=PLOT_BG
    )
    
    return fig


def create_percentile_chart(selected_player, league_stats, player_name):
    """Create a lollipop chart showing league-wide percentile rankings.
    
    Args:
        selected_player: Series with selected player data
        league_stats: Full league DataFrame
        player_name: Name of selected player
        
    Returns:
        Plotly figure object
    """
    # Stats to show percentiles for (use available columns)
    stats = ['PTS', 'AST', 'REB', 'TS_PCT', 'USG_PCT']
    labels = ['Scoring', 'Playmaking', 'Rebounding', 'Efficiency', 'Usage']
    
    percentiles = []
    for stat in stats:
        try:
            if stat in league_stats.columns:
                # Get player value - handle both Series and dict-like access
                if hasattr(selected_player, 'get'):
                    val = selected_player.get(stat, None)
                else:
                    val = selected_player[stat] if stat in selected_player.index else None
                
                if val is not None and pd.notna(val):
                    # Calculate percentile
                    valid_stats = league_stats[stat].dropna()
                    pct = (valid_stats < val).sum() / len(valid_stats) * 100
                    percentiles.append(pct)
                else:
                    percentiles.append(50)  # Default to median if value missing
            else:
                percentiles.append(50)
        except Exception:
            percentiles.append(50)  # Default on any error
    
    # Color based on percentile
    colors = ['#27ae60' if p >= 75 else '#f39c12' if p >= 50 else '#e74c3c' for p in percentiles]
    
    fig = go.Figure()
    
    # Lollipop stems
    for i, (label, pct, color) in enumerate(zip(labels, percentiles, colors)):
        fig.add_trace(go.Scatter(
            x=[0, pct],
            y=[label, label],
            mode='lines',
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Lollipop heads
    fig.add_trace(go.Scatter(
        x=percentiles,
        y=labels,
        mode='markers+text',
        marker=dict(size=20, color=colors, line=dict(color='white', width=2)),
        text=[f"{p:.0f}" for p in percentiles],
        textposition='middle center',
        textfont=dict(color='white', size=10, family='Arial Black'),
        hovertemplate='<b>%{y}</b><br>Percentile: %{x:.0f}%<extra></extra>',
        showlegend=False
    ))
    
    # Add 50th percentile line
    fig.add_vline(x=50, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_annotation(x=50, y=-0.3, text="League Avg", showarrow=False, 
                       font=dict(size=10, color='gray'), yref='paper')
    
    fig.update_layout(
        title=dict(
            text=f"{player_name}'s League Percentile Rankings",
            font=dict(size=16)
        ),
        xaxis=dict(
            title="Percentile",
            range=[0, 100],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.1)',
            tickvals=[0, 25, 50, 75, 100],
            ticktext=['0%', '25%', '50%', '75%', '100%']
        ),
        yaxis=dict(title=""),
        height=300,
        margin=dict(l=100, t=60, b=60),
        plot_bgcolor=PLOT_BG
    )
    
    return fig
