"""
Shot chart visualizations - scatter and hexbin court charts.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import EFFICIENCY_COLOR_RANGE


def get_court_shapes():
    """Get court line shapes for Plotly figures.
    
    Returns:
        list: List of shape dictionaries for court lines
    """
    return [
        dict(type="rect", x0=-25, y0=-5, x1=25, y1=42, line=dict(color="black", width=2)),
        dict(type="rect", x0=-8, y0=-5, x1=8, y1=14, line=dict(color="black", width=2)),
        dict(type="circle", x0=-6, y0=8, x1=6, y1=20, line=dict(color="black", width=2)),
        dict(type="circle", x0=-4, y0=-5, x1=4, y1=3, line=dict(color="black", width=2)),
        dict(type="circle", x0=-0.75, y0=-0.75, x1=0.75, y1=0.75, line=dict(color="orange", width=3)),
        dict(type="line", x0=-3, y0=-1, x1=3, y1=-1, line=dict(color="black", width=3)),
        dict(type="line", x0=-22, y0=-5, x1=-22, y1=9, line=dict(color="black", width=2)),
        dict(type="line", x0=22, y0=-5, x1=22, y1=9, line=dict(color="black", width=2)),
    ]


def add_three_point_arc(fig):
    """Add 3-point arc to a Plotly figure.
    
    Args:
        fig: Plotly figure object
    """
    theta = np.linspace(np.arccos(22/23.75), np.pi - np.arccos(22/23.75), 100)
    arc_x = 23.75 * np.cos(theta)
    arc_y = 23.75 * np.sin(theta)
    fig.add_trace(go.Scatter(
        x=arc_x, y=arc_y, 
        mode='lines', 
        line=dict(color='black', width=2), 
        showlegend=False
    ))


def create_shot_chart(df, player_name, season, color_range=EFFICIENCY_COLOR_RANGE):
    """Create a scatter shot chart figure for a player.
    
    Args:
        df: Processed player shot data
        player_name: Player name for title
        season: Season string for title
        color_range: Tuple of (min, max) for color scale
        
    Returns:
        Plotly figure object
    """
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        color='RELATIVE_EFFICIENCY',
        color_continuous_scale='RdBu_r',
        range_color=color_range,
        hover_data={
            'SHOT_ZONE_BASIC': True,
            'SHOT_DISTANCE': True,
            'ACTION_TYPE': True,
            'made': True,
            'LEAGUE_FG_PCT': ':.1%',
            'RELATIVE_EFFICIENCY': ':.1%'
        },
        opacity=0.7,
        title=f"{player_name} - {season}"
    )
    
    fig.update_coloraxes(
        colorbar_title="Rel. Eff.",
        colorbar_tickformat=".0%"
    )
    
    # Add court lines and arc
    fig.update_layout(
        shapes=get_court_shapes(),
        xaxis=dict(range=[-28, 28], showgrid=False, zeroline=False, title=""),
        yaxis=dict(range=[-7, 45], showgrid=False, zeroline=False, title="", scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        height=500,
        margin=dict(t=50, b=20)
    )
    
    add_three_point_arc(fig)
    
    return fig


def create_hexbin_chart(df, player_name, season):
    """Create a hexbin heatmap shot chart for a player.
    
    Args:
        df: Processed player shot data
        player_name: Player name for title
        season: Season string for title
        
    Returns:
        Plotly figure object or None if insufficient data
    """
    # Filter to valid court locations
    df_valid = df[(df['x'].between(-25, 25)) & (df['y'].between(-5, 40))].copy()
    
    if len(df_valid) < 10:
        return None
    
    x = df_valid['x'].values
    y = df_valid['y'].values
    efficiency = df_valid['RELATIVE_EFFICIENCY'].values
    made = df_valid['SHOT_MADE_FLAG'].values
    
    # Create grid-based binning
    gridsize = 12
    xmin, xmax = -25, 25
    ymin, ymax = -5, 40
    
    hex_data = []
    
    x_bins = np.linspace(xmin, xmax, gridsize + 1)
    y_bins = np.linspace(ymin, ymax, int(gridsize * 0.9) + 1)
    
    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            mask = (x >= x_bins[i]) & (x < x_bins[i+1]) & (y >= y_bins[j]) & (y < y_bins[j+1])
            count = mask.sum()
            if count >= 2:  # Minimum shots to show
                avg_eff = efficiency[mask].mean()
                fg_pct = made[mask].mean() * 100
                hex_data.append({
                    'x': (x_bins[i] + x_bins[i+1]) / 2,
                    'y': (y_bins[j] + y_bins[j+1]) / 2,
                    'count': count,
                    'efficiency': avg_eff,
                    'fg_pct': fg_pct,
                    'size': min(count * 2, 35)  # Cap size
                })
    
    if not hex_data:
        return None
    
    hex_df = pd.DataFrame(hex_data)
    
    # Create the figure with sized markers
    fig = go.Figure()
    
    # Add hexbin-style markers
    fig.add_trace(go.Scatter(
        x=hex_df['x'],
        y=hex_df['y'],
        mode='markers',
        marker=dict(
            size=hex_df['size'],
            color=hex_df['efficiency'],
            colorscale='RdBu_r',
            cmin=-0.15,
            cmax=0.15,
            colorbar=dict(
                title="Rel. Eff.",
                tickformat=".0%",
                thickness=15
            ),
            symbol='hexagon',
            line=dict(width=1, color='white')
        ),
        customdata=np.stack([hex_df['count'], hex_df['fg_pct'], hex_df['efficiency']], axis=1),
        hovertemplate=(
            '<b>Zone</b><br>'
            'Shots: %{customdata[0]:.0f}<br>'
            'FG%%: %{customdata[1]:.1f}%%<br>'
            'vs League: %{customdata[2]:+.1%}<extra></extra>'
        ),
        showlegend=False
    ))
    
    # Add court lines and arc
    fig.update_layout(
        shapes=get_court_shapes(),
        xaxis=dict(range=[-28, 28], showgrid=False, zeroline=False, title="", showticklabels=False),
        yaxis=dict(range=[-7, 45], showgrid=False, zeroline=False, title="", showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='#f8f8f8',
        height=500,
        margin=dict(t=50, b=20),
        title=f"{player_name} - {season} (Heatmap)"
    )
    
    add_three_point_arc(fig)
    
    return fig
