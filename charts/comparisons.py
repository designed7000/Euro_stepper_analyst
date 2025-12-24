"""
Comparison charts - radar charts, bar charts, zone breakdowns.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import ZONE_ORDER, RADAR_RANGES, PLAYER_COLORS
from utils.helpers import normalize_metric


def create_radar_comparison(player_a_name, player_b_name, metrics_a, metrics_b, 
                            rim_rate_a, rim_rate_b, three_rate_a, three_rate_b):
    """Create a radar chart comparing two players.
    
    Args:
        player_a_name: First player's name
        player_b_name: Second player's name
        metrics_a: Metrics dict for player A
        metrics_b: Metrics dict for player B
        rim_rate_a, rim_rate_b: Rim shot rates
        three_rate_a, three_rate_b: Three-point shot rates
        
    Returns:
        Plotly figure object
    """
    # Calculate GSAA per 100
    gsaa_per_100_a = (metrics_a['gsaa'] / metrics_a['attempts'] * 100) if metrics_a['attempts'] > 0 else 0
    gsaa_per_100_b = (metrics_b['gsaa'] / metrics_b['attempts'] * 100) if metrics_b['attempts'] > 0 else 0
    
    # Radar categories
    radar_categories = [
        'Shot Volume<br>(Total FGA)',
        'Accuracy<br>(FG%)',
        'Efficiency<br>(eFG%)',
        '3PT Frequency<br>(% of shots)',
        'Rim Frequency<br>(% of shots)',
        'Value Added<br>(GSAA per 100)'
    ]
    
    # Normalize values
    radar_a = [
        normalize_metric(metrics_a['attempts'], *RADAR_RANGES['attempts']),
        normalize_metric(metrics_a['fg_pct'] * 100, *RADAR_RANGES['fg_pct']),
        normalize_metric(metrics_a['efg'] * 100, *RADAR_RANGES['efg']),
        normalize_metric(three_rate_a, *RADAR_RANGES['three_rate']),
        normalize_metric(rim_rate_a, *RADAR_RANGES['rim_rate']),
        normalize_metric(gsaa_per_100_a, *RADAR_RANGES['gsaa_per_100'])
    ]
    radar_b = [
        normalize_metric(metrics_b['attempts'], *RADAR_RANGES['attempts']),
        normalize_metric(metrics_b['fg_pct'] * 100, *RADAR_RANGES['fg_pct']),
        normalize_metric(metrics_b['efg'] * 100, *RADAR_RANGES['efg']),
        normalize_metric(three_rate_b, *RADAR_RANGES['three_rate']),
        normalize_metric(rim_rate_b, *RADAR_RANGES['rim_rate']),
        normalize_metric(gsaa_per_100_b, *RADAR_RANGES['gsaa_per_100'])
    ]
    
    # Actual values for hover text
    actual_a = [
        f"{metrics_a['attempts']} shots",
        f"{metrics_a['fg_pct']:.1%}",
        f"{metrics_a['efg']:.1%}",
        f"{three_rate_a:.1f}%",
        f"{rim_rate_a:.1f}%",
        f"{gsaa_per_100_a:+.1f} pts"
    ]
    actual_b = [
        f"{metrics_b['attempts']} shots",
        f"{metrics_b['fg_pct']:.1%}",
        f"{metrics_b['efg']:.1%}",
        f"{three_rate_b:.1f}%",
        f"{rim_rate_b:.1f}%",
        f"{gsaa_per_100_b:+.1f} pts"
    ]
    
    radar_fig = go.Figure()
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_a + [radar_a[0]],
        theta=radar_categories + [radar_categories[0]],
        fill='toself',
        name=player_a_name,
        line_color=PLAYER_COLORS['player_a'],
        fillcolor='rgba(31, 119, 180, 0.3)',
        line_width=3,
        customdata=actual_a + [actual_a[0]],
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.0f}/100<br>Actual: %{customdata}<extra>' + player_a_name + '</extra>'
    ))
    radar_fig.add_trace(go.Scatterpolar(
        r=radar_b + [radar_b[0]],
        theta=radar_categories + [radar_categories[0]],
        fill='toself',
        name=player_b_name,
        line_color=PLAYER_COLORS['player_b'],
        fillcolor='rgba(255, 127, 14, 0.3)',
        line_width=3,
        customdata=actual_b + [actual_b[0]],
        hovertemplate='<b>%{theta}</b><br>Score: %{r:.0f}/100<br>Actual: %{customdata}<extra>' + player_b_name + '</extra>'
    ))
    radar_fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True, 
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                ticktext=['25', '50', '75', '100'],
                tickfont=dict(size=10),
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#333'),
                rotation=90,
                direction='clockwise'
            ),
            bgcolor='rgba(248,248,248,0.8)'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=14)
        ),
        height=550,
        title=dict(
            text="Player Skill Profile<br><sup>Normalized 0-100 scale (hover for actual values)</sup>",
            font=dict(size=16)
        ),
        margin=dict(t=80, b=80, l=80, r=80)
    )
    
    return radar_fig


def create_zone_frequency_comparison(stats_a, stats_b, player_a_name, player_b_name):
    """Create a grouped bar chart comparing shot distribution between two players.
    
    Args:
        stats_a: Zone stats DataFrame for player A
        stats_b: Zone stats DataFrame for player B
        player_a_name: First player's name
        player_b_name: Second player's name
        
    Returns:
        Plotly figure object
    """
    freq_bar_fig = go.Figure()
    
    # Player A bars
    freq_bar_fig.add_trace(go.Bar(
        name=player_a_name,
        x=[z + ' ' for z in ZONE_ORDER],
        y=stats_a['freq_pct'],
        marker_color=PLAYER_COLORS['player_a'],
        text=[f"<b>{row['fg_pct']:.1f}%</b><br>({int(row['makes'])}/{int(row['count'])})" 
              for _, row in stats_a.iterrows()],
        textposition='inside',
        textfont=dict(color='white', size=12),
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>Frequency: %{y:.1f}%<extra>' + player_a_name + '</extra>',
        offsetgroup=0
    ))
    
    # Player B bars
    freq_bar_fig.add_trace(go.Bar(
        name=player_b_name,
        x=[z + ' ' for z in ZONE_ORDER],
        y=stats_b['freq_pct'],
        marker_color=PLAYER_COLORS['player_b'],
        text=[f"<b>{row['fg_pct']:.1f}%</b><br>({int(row['makes'])}/{int(row['count'])})" 
              for _, row in stats_b.iterrows()],
        textposition='inside',
        textfont=dict(color='white', size=12),
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>Frequency: %{y:.1f}%<extra>' + player_b_name + '</extra>',
        offsetgroup=1
    ))
    
    freq_bar_fig.update_layout(
        barmode='group',
        title=dict(
            text="Shot Frequency by Zone<br><sup>Bar height = frequency | Inside label = accuracy (makes/attempts)</sup>",
            font=dict(size=14)
        ),
        xaxis=dict(
            title='Zone',
            ticktext=ZONE_ORDER,
            tickvals=[z + ' ' for z in ZONE_ORDER],
            categoryorder='array',
            categoryarray=[z + ' ' for z in ZONE_ORDER]
        ),
        yaxis=dict(
            title='Frequency (%)',
            ticksuffix='%',
            range=[0, max(stats_a['freq_pct'].max(), stats_b['freq_pct'].max()) * 1.15]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='center',
            x=0.5,
            font=dict(size=13)
        ),
        height=420,
        bargap=0.15,
        bargroupgap=0.1
    )
    
    return freq_bar_fig


def create_single_player_zone_chart(player_stats_df, league_zone_dist, player_name):
    """Create a zone frequency chart for a single player vs league average.
    
    Args:
        player_stats_df: Player zone stats DataFrame
        league_zone_dist: League zone distribution DataFrame
        player_name: Player name
        
    Returns:
        Plotly figure object
    """
    freq_fig = go.Figure()
    
    # Player bars
    freq_fig.add_trace(go.Bar(
        name=player_name,
        x=player_stats_df['zone'],
        y=player_stats_df['freq_pct'],
        marker_color=PLAYER_COLORS['player_a'],
        text=[f"<b>{row['fg_pct']:.1f}%</b><br>({int(row['makes'])}/{int(row['count'])})" 
              for _, row in player_stats_df.iterrows()],
        textposition='inside',
        textfont=dict(color='white', size=11),
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>Frequency: %{y:.1f}%<extra>' + player_name + '</extra>'
    ))
    
    # League average bars
    freq_fig.add_trace(go.Bar(
        name='League Avg',
        x=league_zone_dist['zone_group'],
        y=league_zone_dist['league_freq_pct'],
        marker_color=PLAYER_COLORS['league_avg'],
        text=[f"<b>{row['league_fg_pct']:.1f}%</b>" 
              for _, row in league_zone_dist.iterrows()],
        textposition='inside',
        textfont=dict(color='white', size=11),
        insidetextanchor='middle',
        hovertemplate='<b>%{x}</b><br>Frequency: %{y:.1f}%<extra>League Avg</extra>'
    ))
    
    # Add difference annotations
    for _, p_row in player_stats_df.iterrows():
        l_row = league_zone_dist[league_zone_dist['zone_group'] == p_row['zone']]
        if not l_row.empty:
            diff = p_row['freq_pct'] - l_row['league_freq_pct'].values[0]
            if diff > 0:
                text = f"▲ +{diff:.1f}%"
                color = 'green'
            else:
                text = f"▼ {diff:.1f}%"
                color = 'red'
            freq_fig.add_annotation(
                x=p_row['zone'],
                y=max(p_row['freq_pct'], l_row['league_freq_pct'].values[0]) + 3,
                text=text,
                showarrow=False,
                font=dict(color=color, size=11)
            )
    
    freq_fig.update_layout(
        barmode='group',
        title=dict(
            text=f"Shot Frequency: {player_name} vs League<br><sup>Bar height = frequency | Inside label = accuracy (FG%)</sup>",
            font=dict(size=14)
        ),
        xaxis=dict(categoryorder='array', categoryarray=ZONE_ORDER),
        yaxis=dict(title='Frequency (%)', ticksuffix='%'),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        height=420,
        bargap=0.15,
        bargroupgap=0.1
    )
    
    return freq_fig


def create_doppelganger_radar(player_name, match_name, similarity_pct,
                               selected_vals, match_vals):
    """Create a radar chart comparing a player to their statistical doppelganger.
    
    Args:
        player_name: Selected player's name
        match_name: Matched player's name
        similarity_pct: Similarity percentage string
        selected_vals: List of 6 values for selected player
        match_vals: List of 6 values for matched player
        
    Returns:
        Plotly figure object
    """
    from config import SIMILARITY_RANGES
    
    radar_cats = ['Usage Rate', 'True Shooting', 'Assist Rate', 'Rebound Rate', 'Pace', '3PT Rate']
    
    # Normalize for radar
    def norm_radar(vals):
        ranges = [
            SIMILARITY_RANGES['usg'],
            SIMILARITY_RANGES['ts'],
            SIMILARITY_RANGES['ast'],
            SIMILARITY_RANGES['reb'],
            SIMILARITY_RANGES['pace'],
            SIMILARITY_RANGES['3p_rate']
        ]
        normed = []
        for v, r in zip(vals, ranges):
            if r[1] - r[0] == 0:
                normed.append(50)
            else:
                normalized = (v - r[0]) / (r[1] - r[0]) * 100
                normed.append(max(5, min(100, normalized)))
        return normed
    
    selected_normed = norm_radar(selected_vals)
    match_normed = norm_radar(match_vals)
    
    # Format values for display
    def format_vals_display(vals):
        return [
            f"{vals[0]:.1f}%",
            f"{vals[1]:.1f}%",
            f"{vals[2]:.1f}%",
            f"{vals[3]:.1f}%",
            f"{vals[4]:.1f}",
            f"{vals[5]:.1f}%"
        ]
    
    selected_display = format_vals_display(selected_vals)
    match_display = format_vals_display(match_vals)
    
    doppel_radar = go.Figure()
    
    doppel_radar.add_trace(go.Scatterpolar(
        r=selected_normed + [selected_normed[0]],
        theta=radar_cats + [radar_cats[0]],
        fill='toself',
        name=player_name,
        line_color=PLAYER_COLORS['player_a'],
        fillcolor='rgba(31, 119, 180, 0.3)',
        line_width=3,
        customdata=selected_display + [selected_display[0]],
        hovertemplate='<b>%{theta}</b><br>Value: %{customdata}<extra>' + player_name + '</extra>'
    ))
    
    doppel_radar.add_trace(go.Scatterpolar(
        r=match_normed + [match_normed[0]],
        theta=radar_cats + [radar_cats[0]],
        fill='toself',
        name=match_name,
        line_color='#2ca02c',
        fillcolor='rgba(44, 160, 44, 0.3)',
        line_width=3,
        customdata=match_display + [match_display[0]],
        hovertemplate='<b>%{theta}</b><br>Value: %{customdata}<extra>' + match_name + '</extra>'
    ))
    
    doppel_radar.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[25, 50, 75, 100],
                tickfont=dict(size=9),
                gridcolor='rgba(0,0,0,0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='#333'),
                rotation=90,
                direction='clockwise'
            ),
            bgcolor='rgba(248,248,248,0.8)'
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.15,
            xanchor='center',
            x=0.5,
            font=dict(size=13)
        ),
        height=480,
        title=dict(
            text=f"Style Profile: {player_name} vs {match_name}<br><sup>Similarity: {similarity_pct}</sup>",
            font=dict(size=15)
        ),
        margin=dict(t=80, b=60, l=60, r=60)
    )
    
    return doppel_radar
