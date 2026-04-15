"""
Award visualization charts - MVP ladder visualizations.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from config import POSITION_COLORS


def create_mvp_value_breakdown_chart(mvp_ladder, top_n=10):
    """Create a stacked horizontal bar chart showing value breakdown.
    
    Shows WHY each player is valuable (scoring vs playmaking vs defense).
    
    Args:
        mvp_ladder: MVP ladder DataFrame with component values
        top_n: Number of players to show
        
    Returns:
        Plotly figure object
    """
    # Get top N players (reverse for horizontal bar order)
    df = mvp_ladder.head(top_n).copy()
    df = df.sort_values('MVP_SCORE', ascending=True)  # Reverse for horizontal bars
    
    fig = go.Figure()
    
    # Scoring Value (Blue)
    fig.add_trace(go.Bar(
        name='Scoring',
        y=df['PLAYER_NAME'],
        x=df['SCORING_VALUE'],
        orientation='h',
        marker_color='#1f77b4',
        hovertemplate='<b>%{y}</b><br>Scoring Value: %{x:.1f}<extra></extra>'
    ))
    
    # Playmaking Value (Green)
    fig.add_trace(go.Bar(
        name='Playmaking',
        y=df['PLAYER_NAME'],
        x=df['PLAYMAKING_VALUE'],
        orientation='h',
        marker_color='#2ca02c',
        hovertemplate='<b>%{y}</b><br>Playmaking Value: %{x:.1f}<extra></extra>'
    ))
    
    # Rebounding Value (Orange)
    fig.add_trace(go.Bar(
        name='Rebounding',
        y=df['PLAYER_NAME'],
        x=df['REBOUNDING_VALUE'],
        orientation='h',
        marker_color='#ff7f0e',
        hovertemplate='<b>%{y}</b><br>Rebounding Value: %{x:.1f}<extra></extra>'
    ))
    
    # Defensive Value (Purple) - STL + BLK combined
    fig.add_trace(go.Bar(
        name='Defense (STL+BLK)',
        y=df['PLAYER_NAME'],
        x=df['DEFENSIVE_VALUE'],
        orientation='h',
        marker_color='#9467bd',
        hovertemplate='<b>%{y}</b><br>Defensive Value: %{x:.1f}<extra></extra>'
    ))
    
    # Turnover Penalty (Red, negative)
    fig.add_trace(go.Bar(
        name='Turnover Penalty',
        y=df['PLAYER_NAME'],
        x=-df['TURNOVER_PENALTY'],  # Negative to show as penalty
        orientation='h',
        marker_color='#d62728',
        hovertemplate='<b>%{y}</b><br>TO Penalty: -%{customdata:.1f}<extra></extra>',
        customdata=df['TURNOVER_PENALTY']
    ))
    
    fig.update_layout(
        barmode='relative',
        title=dict(
            text='MVP Value Breakdown<br><sup>What makes each player valuable</sup>',
            font=dict(size=16)
        ),
        xaxis_title='Production Value (Scarcity-Weighted)',
        yaxis_title='',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5
        ),
        height=450,
        template='plotly_dark',
        margin=dict(l=120, r=20, t=80, b=40)
    )
    
    return fig


def create_winning_vs_stats_scatter(mvp_ladder, top_n=20):
    """Create a scatter plot of Team Win% vs Raw Production Value.
    
    Shows the balance between individual production and team success.
    
    Args:
        mvp_ladder: MVP ladder DataFrame
        top_n: Number of players to show
        
    Returns:
        Plotly figure object
    """
    df = mvp_ladder.head(top_n).copy()
    
    fig = px.scatter(
        df,
        x='WIN_PCT',
        y='RAW_VALUE',
        color='MVP_SCORE_NORMALIZED',
        size='MVP_SCORE_NORMALIZED',
        hover_name='PLAYER_NAME',
        hover_data={
            'WIN_PCT': ':.1%',
            'RAW_VALUE': ':.1f',
            'MVP_SCORE_NORMALIZED': ':.1f',
            'TEAM_ABBREVIATION': True,
            'PTS': ':.1f',
            'REB': ':.1f',
            'AST': ':.1f'
        },
        color_continuous_scale='Viridis',
        title='Winning vs. Individual Production'
    )
    
    # Calculate averages for reference lines
    avg_win = df['WIN_PCT'].mean()
    avg_raw = df['RAW_VALUE'].mean()
    
    # Add reference lines at averages
    fig.add_vline(
        x=avg_win, 
        line_dash="dash", 
        line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Avg Win%",
        annotation_position="top"
    )
    fig.add_hline(
        y=avg_raw, 
        line_dash="dash", 
        line_color="rgba(255,255,255,0.4)",
        annotation_text=f"Avg Production",
        annotation_position="right"
    )
    
    # Add quadrant labels
    x_range = df['WIN_PCT'].max() - df['WIN_PCT'].min()
    y_range = df['RAW_VALUE'].max() - df['RAW_VALUE'].min()
    
    # Top-Right: MVP Territory
    fig.add_annotation(
        x=avg_win + x_range * 0.25,
        y=avg_raw + y_range * 0.35,
        text="🏆 MVP Territory",
        showarrow=False,
        font=dict(size=12, color='#FFD700'),
        bgcolor='rgba(0,0,0,0.6)'
    )
    
    # Bottom-Right: Good Teams, Role Players
    fig.add_annotation(
        x=avg_win + x_range * 0.25,
        y=avg_raw - y_range * 0.25,
        text="Team Success",
        showarrow=False,
        font=dict(size=10, color='rgba(255,255,255,0.6)'),
        bgcolor='rgba(0,0,0,0.4)'
    )
    
    # Top-Left: Stats on Bad Teams
    fig.add_annotation(
        x=avg_win - x_range * 0.2,
        y=avg_raw + y_range * 0.35,
        text="Empty Stats?",
        showarrow=False,
        font=dict(size=10, color='rgba(255,255,255,0.6)'),
        bgcolor='rgba(0,0,0,0.4)'
    )
    
    fig.update_layout(
        template='plotly_dark',
        height=450,
        xaxis_title='Team Win Percentage',
        yaxis_title='Raw Production Value',
        xaxis=dict(tickformat='.0%'),
        coloraxis_colorbar=dict(
            title='MVP Score',
            tickformat='.0f'
        ),
        margin=dict(l=60, r=20, t=60, b=40)
    )
    
    fig.update_traces(marker=dict(line=dict(width=1, color='white'), sizemin=8))
    
    return fig


def create_scarcity_weights_chart(weights):
    """Create a bar chart showing the scarcity weights.
    
    Helps users understand how rare each stat is.
    
    Args:
        weights: Dictionary of scarcity weights
        
    Returns:
        Plotly figure object
    """
    # Prepare data
    stats = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV']
    values = [weights.get(s, 1.0) for s in stats]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#9467bd', '#d62728']
    
    fig = go.Figure(go.Bar(
        x=stats,
        y=values,
        marker_color=colors,
        text=[f'{v:.1f}x' for v in values],
        textposition='outside',
        hovertemplate='<b>%{x}</b><br>Weight: %{y:.2f}x<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Stat Scarcity Weights<br><sup>How rare each stat is relative to points</sup>',
            font=dict(size=14)
        ),
        xaxis_title='Stat Category',
        yaxis_title='Scarcity Multiplier',
        template='plotly_dark',
        height=300,
        showlegend=False,
        margin=dict(l=40, r=20, t=60, b=40)
    )
    
    # Add reference line at 1.0 (points baseline)
    fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.5)")
    
    return fig


def create_mvp_race_trend_chart(mvp_ladder, top_n=5):
    """Create a simple ranking visualization for top MVP candidates.
    
    Args:
        mvp_ladder: MVP ladder DataFrame
        top_n: Number of players to show
        
    Returns:
        Plotly figure object
    """
    df = mvp_ladder.head(top_n).copy()
    
    # Create horizontal bar for MVP scores
    fig = go.Figure()
    
    # Reverse order for display (1st at top)
    df = df.sort_values('RANK', ascending=False)
    
    fig.add_trace(go.Bar(
        y=[f"#{row['RANK']} {row['PLAYER_NAME']}" for _, row in df.iterrows()],
        x=df['MVP_SCORE_NORMALIZED'],
        orientation='h',
        marker=dict(
            color=df['MVP_SCORE_NORMALIZED'],
            colorscale='Viridis',
            line=dict(width=1, color='white')
        ),
        text=[f"{row['MVP_SCORE_NORMALIZED']:.1f} ({row['TEAM_ABBREVIATION']})" for _, row in df.iterrows()],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>MVP Score: %{x:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='MVP Race - Current Standings',
            font=dict(size=16)
        ),
        xaxis_title='MVP Score (Normalized)',
        yaxis_title='',
        template='plotly_dark',
        height=280,
        margin=dict(l=160, r=60, t=50, b=40),
        showlegend=False
    )
    
    return fig
