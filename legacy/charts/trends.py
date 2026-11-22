"""
Historical trends and league leader visualizations.
"""

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from config import POSITION_COLORS, PLAYER_COLORS, MIN_USAGE_LEADERS, MIN_ASSISTS_PLAYMAKING, MIN_MINUTES_TWOWAY


def create_historical_efg_chart(hist_a, hist_b, player_a_name, player_b_name):
    """Create eFG% trend chart for two players.
    
    Args:
        hist_a: Historical data DataFrame for player A
        hist_b: Historical data DataFrame for player B
        player_a_name: First player's name
        player_b_name: Second player's name
        
    Returns:
        Plotly figure object
    """
    efg_trend = go.Figure()
    efg_trend.add_trace(go.Scatter(
        x=hist_a['season'], y=hist_a['efg'],
        mode='lines+markers', name=player_a_name,
        line=dict(color=PLAYER_COLORS['player_a'])
    ))
    efg_trend.add_trace(go.Scatter(
        x=hist_b['season'], y=hist_b['efg'],
        mode='lines+markers', name=player_b_name,
        line=dict(color=PLAYER_COLORS['player_b'])
    ))
    efg_trend.update_layout(
        title="eFG% Trend",
        yaxis_title="eFG%",
        yaxis=dict(ticksuffix='%'),
        height=300
    )
    return efg_trend


def create_historical_gsaa_chart(hist_a, hist_b, player_a_name, player_b_name):
    """Create GSAA trend chart for two players.
    
    Args:
        hist_a: Historical data DataFrame for player A
        hist_b: Historical data DataFrame for player B
        player_a_name: First player's name
        player_b_name: Second player's name
        
    Returns:
        Plotly figure object
    """
    gsaa_trend = go.Figure()
    gsaa_trend.add_trace(go.Scatter(
        x=hist_a['season'], y=hist_a['gsaa'],
        mode='lines+markers', name=player_a_name,
        line=dict(color=PLAYER_COLORS['player_a'])
    ))
    gsaa_trend.add_trace(go.Scatter(
        x=hist_b['season'], y=hist_b['gsaa'],
        mode='lines+markers', name=player_b_name,
        line=dict(color=PLAYER_COLORS['player_b'])
    ))
    gsaa_trend.update_layout(
        title="Points Added (GSAA) Trend",
        yaxis_title="GSAA",
        height=300
    )
    return gsaa_trend


def create_scoring_efficiency_chart(leaders_df):
    """Create Scoring Load vs Efficiency scatter chart.
    
    Args:
        leaders_df: League leaders DataFrame
        
    Returns:
        Plotly figure object or None if insufficient data
    """
    scoring_df = leaders_df[
        leaders_df['USG_PCT'].notna() & 
        (leaders_df['USG_PCT'] >= MIN_USAGE_LEADERS)
    ].copy()
    
    if scoring_df.empty:
        return None
    
    # Convert to display percentages
    scoring_df['USG_DISPLAY'] = scoring_df['USG_PCT'] * 100
    scoring_df['TS_DISPLAY'] = scoring_df['TS_PCT'] * 100
    
    # Calculate league average TS%
    league_avg_ts = scoring_df['TS_DISPLAY'].mean()
    
    fig_helio = px.scatter(
        scoring_df,
        x='USG_DISPLAY',
        y='TS_DISPLAY',
        size='PTS',
        color='POSITION_GROUP',
        hover_name='PLAYER_NAME',
        hover_data={
            'USG_DISPLAY': ':.1f',
            'TS_DISPLAY': ':.1f',
            'PTS': ':.1f',
            'POSITION_GROUP': False
        },
        title='Scoring Load vs Efficiency',
        color_discrete_map=POSITION_COLORS,
        template='plotly_dark'
    )
    
    # Add league average TS% line
    fig_helio.add_hline(
        y=league_avg_ts, 
        line_dash="dash", 
        line_color="rgba(255,255,255,0.5)",
        annotation_text=f"League Avg ({league_avg_ts:.1f}%)",
        annotation_position="top right",
        annotation_font_color="rgba(255,255,255,0.7)"
    )
    
    # Add zone annotations
    x_max = scoring_df['USG_DISPLAY'].max()
    x_min = scoring_df['USG_DISPLAY'].min()
    y_max = scoring_df['TS_DISPLAY'].max()
    
    fig_helio.add_annotation(
        x=x_max - 3, y=y_max - 2,
        text="ELITE",
        showarrow=False,
        font=dict(size=11, color='#FFD700'),
        bgcolor='rgba(0,0,0,0.5)'
    )
    
    fig_helio.add_annotation(
        x=x_max - 3, y=league_avg_ts - 3,
        text="Volume",
        showarrow=False,
        font=dict(size=10, color='rgba(255,255,255,0.6)'),
        bgcolor='rgba(0,0,0,0.4)'
    )
    
    fig_helio.add_annotation(
        x=x_min + 5, y=y_max - 2,
        text="Efficient",
        showarrow=False,
        font=dict(size=10, color='rgba(255,255,255,0.6)'),
        bgcolor='rgba(0,0,0,0.4)'
    )
    
    fig_helio.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Usage Rate %",
        yaxis_title="True Shooting %"
    )
    fig_helio.update_traces(marker=dict(line=dict(width=1, color='white'), sizemin=5))
    
    return fig_helio


def create_playmaking_chart(leaders_df):
    """Create Playmaking scatter chart (AST vs TOV per 100).
    
    Args:
        leaders_df: League leaders DataFrame
        
    Returns:
        Plotly figure object or None if insufficient data
    """
    play_df = leaders_df[
        leaders_df['AST_PER100'].notna() & 
        leaders_df['TOV_PER100'].notna() &
        (leaders_df['AST'] >= MIN_ASSISTS_PLAYMAKING)
    ].copy()
    
    if play_df.empty:
        return None
    
    # Use AST_TO ratio for bubble size
    play_df['AST_TO_SAFE'] = play_df['AST_TO'].fillna(1).clip(lower=0.5)
    
    fig_play = px.scatter(
        play_df,
        x='AST_PER100',
        y='TOV_PER100',
        size='AST_TO_SAFE',
        color='POSITION_GROUP',
        hover_name='PLAYER_NAME',
        hover_data={
            'AST_PER100': ':.1f',
            'TOV_PER100': ':.1f',
            'AST_TO_SAFE': False,
            'AST_TO': ':.2f',
            'POSITION_GROUP': False
        },
        title='Playmaking: Creation vs Security',
        color_discrete_map=POSITION_COLORS,
        template='plotly_dark',
        labels={'AST_PER100': 'AST per 100', 'TOV_PER100': 'TOV per 100'}
    )
    
    # INVERT Y-axis so low turnovers are at top
    y_max = play_df['TOV_PER100'].max() + 1
    y_min = max(0, play_df['TOV_PER100'].min() - 1)
    fig_play.update_yaxes(range=[y_max, y_min], autorange=False)
    
    # Add "Elite Zone" annotation
    fig_play.add_annotation(
        x=play_df['AST_PER100'].max() * 0.9,
        y=y_min + 0.5,
        text="ELITE",
        showarrow=False,
        font=dict(size=12, color='#FFD700'),
        bgcolor='rgba(0,0,0,0.5)'
    )
    
    fig_play.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Assists per 100 Poss",
        yaxis_title="Turnovers per 100 Poss"
    )
    fig_play.update_traces(marker=dict(line=dict(width=1, color='white'), sizemin=6))
    
    return fig_play


def create_twoway_quadrant_chart(leaders_df):
    """Create Two-Way Impact quadrant chart (OFF vs DEF Rating).
    
    Args:
        leaders_df: League leaders DataFrame
        
    Returns:
        Plotly figure object or None if insufficient data
    """
    twoway_df = leaders_df[
        leaders_df['OFF_RATING'].notna() & 
        leaders_df['DEF_RATING'].notna() &
        (leaders_df['MIN'] >= MIN_MINUTES_TWOWAY)
    ].copy()
    
    if twoway_df.empty:
        return None
    
    # Calculate averages for reference lines
    avg_off = twoway_df['OFF_RATING'].mean()
    avg_def = twoway_df['DEF_RATING'].mean()
    
    fig_quadrant = px.scatter(
        twoway_df,
        x='OFF_RATING',
        y='DEF_RATING',
        color='POSITION_GROUP',
        hover_name='PLAYER_NAME',
        hover_data={
            'OFF_RATING': ':.1f',
            'DEF_RATING': ':.1f',
            'NET_RATING': ':.1f',
            'PTS': ':.1f',
            'POSITION_GROUP': False
        },
        title='Offensive vs Defensive Rating',
        color_discrete_map=POSITION_COLORS,
        template='plotly_dark'
    )
    
    # INVERT Y-axis for defense (lower DEF_RATING = better defense)
    y_max = twoway_df['DEF_RATING'].max() + 2
    y_min = twoway_df['DEF_RATING'].min() - 2
    fig_quadrant.update_yaxes(range=[y_max, y_min], autorange=False)
    
    # Add crosshair reference lines
    fig_quadrant.add_vline(x=avg_off, line_dash="solid", line_color="rgba(255,255,255,0.3)", line_width=2)
    fig_quadrant.add_hline(y=avg_def, line_dash="solid", line_color="rgba(255,255,255,0.3)", line_width=2)
    
    # Add quadrant labels
    x_range = twoway_df['OFF_RATING'].max() - twoway_df['OFF_RATING'].min()
    y_range = y_max - y_min
    
    fig_quadrant.add_annotation(
        x=avg_off + x_range * 0.3, y=avg_def - y_range * 0.35,
        text="MVP Candidates",
        showarrow=False, font=dict(size=11, color='#FFD700'),
        bgcolor='rgba(0,0,0,0.6)'
    )
    fig_quadrant.add_annotation(
        x=avg_off + x_range * 0.3, y=avg_def + y_range * 0.35,
        text="Offense Only",
        showarrow=False, font=dict(size=10, color='#FF6B6B'),
        bgcolor='rgba(0,0,0,0.5)'
    )
    fig_quadrant.add_annotation(
        x=avg_off - x_range * 0.3, y=avg_def - y_range * 0.35,
        text="Defense Only",
        showarrow=False, font=dict(size=10, color='#98D8C8'),
        bgcolor='rgba(0,0,0,0.5)'
    )
    
    fig_quadrant.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Offensive Rating",
        yaxis_title="Defensive Rating (lower = better)"
    )
    fig_quadrant.update_traces(marker=dict(size=10, line=dict(width=1, color='white')))
    
    return fig_quadrant


def create_top_scorers_bar(leaders_df):
    """Create horizontal bar chart of top 10 scorers.
    
    Args:
        leaders_df: League leaders DataFrame
        
    Returns:
        Plotly figure object
    """
    top_scorers = leaders_df.nlargest(10, 'PTS')[['PLAYER_NAME', 'PTS', 'POSITION_GROUP', 'TEAM_ABBREVIATION']].copy()
    top_scorers = top_scorers.sort_values('PTS', ascending=True)
    
    bar_colors = [POSITION_COLORS.get(pos, '#888888') for pos in top_scorers['POSITION_GROUP']]
    
    fig_top = go.Figure(go.Bar(
        x=top_scorers['PTS'],
        y=top_scorers['PLAYER_NAME'],
        orientation='h',
        marker_color=bar_colors,
        text=top_scorers.apply(lambda r: f"{r['PTS']:.1f} ({r['TEAM_ABBREVIATION']})", axis=1),
        textposition='outside'
    ))
    fig_top.update_layout(
        template='plotly_dark',
        height=400,
        margin=dict(l=20, r=80, t=20, b=20),
        xaxis_title="Points Per Game",
        yaxis_title="",
        showlegend=False
    )
    
    return fig_top
