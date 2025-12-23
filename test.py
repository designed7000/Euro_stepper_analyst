import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from nba_api.stats.static import players
from nba_api.stats.endpoints import shotchartdetail, playercareerstats, leaguedashplayerstats
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from difflib import SequenceMatcher, get_close_matches

# --- APP CONFIGURATION ---
st.set_page_config(page_title="NBA Shot DNA", layout="wide")

# --- HELPER FUNCTIONS (Cached for Performance) ---
@st.cache_data
def get_all_player_names():
    """Get all NBA player names for fuzzy matching."""
    nba_players = players.get_players()
    return {p['full_name'].lower(): p for p in nba_players}

@st.cache_data
def get_player_id(name):
    """Find player ID with fuzzy matching support."""
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

@st.cache_data
def get_player_shots(player_id, season, clutch_only=False):
    # Fetch raw shot data
    # If clutch_only, use API's clutch_time and ahead_behind filters
    clutch_time = 'Last 5 Minutes' if clutch_only else None
    ahead_behind = 'Ahead or Behind' if clutch_only else None
    
    shot_data = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=player_id,
        context_measure_simple='FGA',
        season_nullable=season,
        clutch_time_nullable=clutch_time,
        ahead_behind_nullable=ahead_behind
    )
    frames = shot_data.get_data_frames()
    player_shots = frames[0]
    league_averages = frames[1]  # League averages by zone
    return player_shots, league_averages

@st.cache_data
def get_league_averages(season):
    """Fetch league-wide shot data to calculate zone averages."""
    shot_data = shotchartdetail.ShotChartDetail(
        team_id=0,
        player_id=0,  # 0 = all players
        context_measure_simple='FGA',
        season_nullable=season
    )
    league_df = shot_data.get_data_frames()[0]
    
    # Calculate FG% by zone
    total_league_attempts = len(league_df)
    zone_avgs = league_df.groupby('SHOT_ZONE_BASIC').agg(
        league_makes=('SHOT_MADE_FLAG', 'sum'),
        league_attempts=('SHOT_MADE_FLAG', 'count')
    ).reset_index()
    zone_avgs['league_fg_pct'] = zone_avgs['league_makes'] / zone_avgs['league_attempts']
    zone_avgs['league_freq_pct'] = zone_avgs['league_attempts'] / total_league_attempts
    
    return zone_avgs[['SHOT_ZONE_BASIC', 'league_fg_pct', 'league_freq_pct']]

def process_player_data(df, league_avg_df):
    """Process player shot data: merge with league averages, calculate metrics."""
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
    """Calculate summary metrics for a player."""
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

def create_shot_chart(df, player_name, season, color_range=(-0.15, 0.15)):
    """Create a shot chart figure for a player."""
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
    
    # Draw court lines
    court_shapes = [
        dict(type="rect", x0=-25, y0=-5, x1=25, y1=42, line=dict(color="black", width=2)),
        dict(type="rect", x0=-8, y0=-5, x1=8, y1=14, line=dict(color="black", width=2)),
        dict(type="circle", x0=-6, y0=8, x1=6, y1=20, line=dict(color="black", width=2)),
        dict(type="circle", x0=-4, y0=-5, x1=4, y1=3, line=dict(color="black", width=2)),
        dict(type="circle", x0=-0.75, y0=-0.75, x1=0.75, y1=0.75, line=dict(color="orange", width=3)),
        dict(type="line", x0=-3, y0=-1, x1=3, y1=-1, line=dict(color="black", width=3)),
        dict(type="line", x0=-22, y0=-5, x1=-22, y1=9, line=dict(color="black", width=2)),
        dict(type="line", x0=22, y0=-5, x1=22, y1=9, line=dict(color="black", width=2)),
    ]
    
    # Add 3-point arc
    theta = np.linspace(np.arccos(22/23.75), np.pi - np.arccos(22/23.75), 100)
    arc_x = 23.75 * np.cos(theta)
    arc_y = 23.75 * np.sin(theta)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines', line=dict(color='black', width=2), showlegend=False))
    
    fig.update_layout(
        shapes=court_shapes,
        xaxis=dict(range=[-28, 28], showgrid=False, zeroline=False, title=""),
        yaxis=dict(range=[-7, 45], showgrid=False, zeroline=False, title="", scaleanchor="x", scaleratio=1),
        plot_bgcolor='white',
        height=500,
        margin=dict(t=50, b=20)
    )
    
    return fig

def create_hexbin_chart(df, player_name, season):
    """Create a hexbin heatmap shot chart for a player."""
    import plotly.figure_factory as ff
    
    # Filter to valid court locations
    df_valid = df[(df['x'].between(-25, 25)) & (df['y'].between(-5, 40))].copy()
    
    if len(df_valid) < 10:
        return None
    
    # Create hexbin using numpy for binning
    from scipy import stats
    
    x = df_valid['x'].values
    y = df_valid['y'].values
    efficiency = df_valid['RELATIVE_EFFICIENCY'].values
    made = df_valid['SHOT_MADE_FLAG'].values
    
    # Create hexagonal grid
    gridsize = 12
    
    # Use scipy hexbin-like calculation
    xmin, xmax = -25, 25
    ymin, ymax = -5, 40
    
    # Create hex grid centers
    hex_width = (xmax - xmin) / gridsize
    hex_height = hex_width * np.sqrt(3) / 2
    
    hex_data = []
    
    # Simple grid-based binning (approximating hexbins with squares for Plotly)
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
    
    # Draw court lines
    court_shapes = [
        dict(type="rect", x0=-25, y0=-5, x1=25, y1=42, line=dict(color="black", width=2)),
        dict(type="rect", x0=-8, y0=-5, x1=8, y1=14, line=dict(color="black", width=2)),
        dict(type="circle", x0=-6, y0=8, x1=6, y1=20, line=dict(color="black", width=2)),
        dict(type="circle", x0=-4, y0=-5, x1=4, y1=3, line=dict(color="black", width=2)),
        dict(type="circle", x0=-0.75, y0=-0.75, x1=0.75, y1=0.75, line=dict(color="orange", width=3)),
        dict(type="line", x0=-3, y0=-1, x1=3, y1=-1, line=dict(color="black", width=3)),
        dict(type="line", x0=-22, y0=-5, x1=-22, y1=9, line=dict(color="black", width=2)),
        dict(type="line", x0=22, y0=-5, x1=22, y1=9, line=dict(color="black", width=2)),
    ]
    
    # Add 3-point arc
    theta = np.linspace(np.arccos(22/23.75), np.pi - np.arccos(22/23.75), 100)
    arc_x = 23.75 * np.cos(theta)
    arc_y = 23.75 * np.sin(theta)
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines', line=dict(color='black', width=2), showlegend=False))
    
    fig.update_layout(
        shapes=court_shapes,
        xaxis=dict(range=[-28, 28], showgrid=False, zeroline=False, title="", showticklabels=False),
        yaxis=dict(range=[-7, 45], showgrid=False, zeroline=False, title="", showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='#f8f8f8',
        height=500,
        margin=dict(t=50, b=20),
        title=f"{player_name} - {season} (Heatmap)"
    )
    
    return fig

# --- DOPPELGÄNGER FINDER (Player Similarity Engine) ---
@st.cache_data
def get_advanced_stats(season):
    """Fetch and merge base + advanced stats for all players in a season."""
    import time
    
    # Call 1: Base stats (for FG3A, FGA)
    base_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Base',
        per_mode_detailed='PerGame'
    )
    base_df = base_stats.get_data_frames()[0]
    time.sleep(0.6)  # Respect API rate limits
    
    # Call 2: Advanced stats (for USG_PCT, TS_PCT, AST_PCT, etc.)
    adv_stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        measure_type_detailed_defense='Advanced',
        per_mode_detailed='PerGame'
    )
    adv_df = adv_stats.get_data_frames()[0]
    
    # Merge on PLAYER_ID and PLAYER_NAME
    merged = base_df.merge(
        adv_df[['PLAYER_ID', 'PLAYER_NAME', 'USG_PCT', 'TS_PCT', 'AST_PCT', 'REB_PCT', 'PACE']],
        on=['PLAYER_ID', 'PLAYER_NAME'],
        how='left'
    )
    
    # Feature Engineering: 3-Point Attempt Rate
    merged['3P_AR'] = merged['FG3A'] / merged['FGA']
    merged['3P_AR'] = merged['3P_AR'].fillna(0)
    
    # Filtering: Remove low-sample players (< 15 games OR < 10 min avg)
    merged = merged[(merged['GP'] >= 15) & (merged['MIN'] >= 10)]
    
    return merged

def build_similarity_model(stats_df):
    """Build the ML pipeline for player similarity."""
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
    """Find the 5 most similar players to the given player."""
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

# --- MAIN DASHBOARD INTERFACE ---
st.title("🏀 NBA Player DNA: Spatial Efficiency Engine")

# Sidebar for Inputs
with st.sidebar:
    # Logo at the top
    st.image("logo.png", use_container_width=True)
    
    st.header("Analyst Controls")
    player_name_a = st.text_input("Player A Name", "Stephen Curry")
    
    st.subheader("Comparison Mode")
    compare_mode = st.checkbox("Compare Players", value=False)
    if compare_mode:
        player_name_b = st.text_input("Player B Name", "LeBron James")
    
    season = st.selectbox("Season", ["2024-25", "2023-24", "2022-23", "2021-22", "2015-16"])
    
    st.subheader("Game Context")
    clutch_only = st.checkbox("Clutch Time Only", value=False, 
                               help="Last 5 min, score within 5 pts")
    
    st.subheader("🔍 Player Discovery")
    show_doppelganger = st.checkbox("Find Similar Players", value=False,
                                     help="ML-powered player similarity engine")

# --- ANALYTICS ENGINE ---
try:
    # Get Player A data with fuzzy matching
    p_id_a, corrected_name_a, match_msg_a = get_player_id(player_name_a)
    
    if not p_id_a:
        st.error(f"Player A '{player_name_a}' not found. Check spelling.")
        st.stop()
    
    # Show correction message if name was fuzzy matched
    if match_msg_a:
        st.info(f"🔍 {match_msg_a}")
    
    # Use corrected name for display
    player_name_a = corrected_name_a
    
    with st.spinner(f"Loading data for {player_name_a}..."):
        df_a, league_avg_df = get_player_shots(p_id_a, season, clutch_only)
    
    if df_a.empty:
        st.warning(f"No shot data found for {player_name_a} in {season}.")
        st.stop()
    
    # Process Player A data
    df_a = process_player_data(df_a, league_avg_df)
    metrics_a = calculate_player_metrics(df_a)
    
    # Get Player B data if in compare mode
    if compare_mode:
        p_id_b, corrected_name_b, match_msg_b = get_player_id(player_name_b)
        
        if not p_id_b:
            st.error(f"Player B '{player_name_b}' not found. Check spelling.")
            st.stop()
        
        # Show correction message if name was fuzzy matched
        if match_msg_b:
            st.info(f"🔍 {match_msg_b}")
        
        # Use corrected name for display
        player_name_b = corrected_name_b
        
        with st.spinner(f"Loading data for {player_name_b}..."):
            df_b, _ = get_player_shots(p_id_b, season, clutch_only)
        
        if df_b.empty:
            st.warning(f"No shot data found for {player_name_b} in {season}.")
            st.stop()
        
        # Process Player B data
        df_b = process_player_data(df_b, league_avg_df)
        metrics_b = calculate_player_metrics(df_b)
    
    # Show clutch indicator if enabled
    if clutch_only:
        if compare_mode:
            st.info(f"🔥 Clutch Time: {player_name_a} ({len(df_a)} shots) vs {player_name_b} ({len(df_b)} shots)")
        else:
            st.info(f"🔥 Clutch Time: {len(df_a)} shots (Last 5 min, score within 5 pts)")
    
    # --- METRICS DISPLAY ---
    if compare_mode:
        # Comparison metrics table
        st.subheader("📊 Head-to-Head Comparison")
        
        # Calculate deltas
        delta_attempts = metrics_a['attempts'] - metrics_b['attempts']
        delta_fg = (metrics_a['fg_pct'] - metrics_b['fg_pct']) * 100
        delta_efg = (metrics_a['efg'] - metrics_b['efg']) * 100
        delta_gsaa = metrics_a['gsaa'] - metrics_b['gsaa']
        delta_3pt = metrics_a['threes_attempted'] - metrics_b['threes_attempted']
        
        comparison_data = {
            'Metric': ['Volume (FGA)', 'FG%', 'eFG%', 'Points Added (GSAA)', '3PT Attempts'],
            player_name_a: [
                metrics_a['attempts'],
                f"{metrics_a['fg_pct']:.1%}",
                f"{metrics_a['efg']:.1%}",
                f"{metrics_a['gsaa']:+.1f}",
                metrics_a['threes_attempted']
            ],
            player_name_b: [
                metrics_b['attempts'],
                f"{metrics_b['fg_pct']:.1%}",
                f"{metrics_b['efg']:.1%}",
                f"{metrics_b['gsaa']:+.1f}",
                metrics_b['threes_attempted']
            ],
            'Delta': [
                f"{delta_attempts:+d}",
                f"{delta_fg:+.1f}%",
                f"{delta_efg:+.1f}%",
                f"{delta_gsaa:+.1f}",
                f"{delta_3pt:+d}"
            ],
            '_delta_raw': [delta_attempts, delta_fg, delta_efg, delta_gsaa, delta_3pt]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Style function to color Delta column
        def style_delta(row):
            delta_val = row['_delta_raw']
            if delta_val > 0:
                return ['', '', '', 'background-color: #90EE90', '']  # Light green
            elif delta_val < 0:
                return ['', '', '', 'background-color: #FFB6C1', '']  # Light red
            else:
                return ['', '', '', '', '']
        
        # Apply styling and hide the helper column
        styled_df = comparison_df.drop(columns=['_delta_raw']).style.apply(
            lambda x: ['', '', '', 
                       'background-color: #90EE90' if comparison_df.loc[x.name, '_delta_raw'] > 0 
                       else ('background-color: #FFB6C1' if comparison_df.loc[x.name, '_delta_raw'] < 0 else '')
                      ], axis=1
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.caption("💡 Delta shows Player A minus Player B. 🟢 Positive = Player A higher | 🔴 Negative = Player B higher")
        
    else:
        # Single player metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Volume (FGA)", f"{metrics_a['attempts']}")
        col2.metric("Raw FG%", f"{metrics_a['fg_pct']:.1%}")
        col3.metric("Effective FG% (eFG)", f"{metrics_a['efg']:.1%}")
        col4.metric("3PT Volume", f"{metrics_a['threes_attempted']}")
        col5.metric("Points Added (GSAA)", f"{metrics_a['gsaa']:+.1f}")
        
        st.caption("💡 **Points Added (GSAA)**: Points generated solely by shooting skill above league average expectation.")
    
    # --- SHOT CHARTS ---
    st.subheader("Relative Efficiency Maps")
    
    # Chart view toggle
    chart_view = st.radio(
        "Chart Style",
        ["Scatter (Individual Shots)", "Heatmap (Zone Aggregated)"],
        horizontal=True,
        help="Scatter shows each shot. Heatmap groups shots into zones showing hot/cold areas."
    )
    use_hexbin = chart_view == "Heatmap (Zone Aggregated)"
    
    # Add explanation based on view
    if use_hexbin:
        st.caption("🔥 **Heatmap View**: Larger hexagons = more shots from that area. Color shows efficiency vs league average (🔵 blue = above avg, 🔴 red = below avg)")
    else:
        st.caption("📍 **Scatter View**: Each dot is a shot attempt. Color shows efficiency vs league average (🔵 blue = above avg, 🔴 red = below avg)")
    
    # Use consistent color scale for comparison
    color_range = (-0.15, 0.15)
    
    if compare_mode:
        # Side-by-side charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            if use_hexbin:
                fig_a = create_hexbin_chart(df_a, player_name_a, season)
                if fig_a is None:
                    st.warning("Not enough data for heatmap view")
                    fig_a = create_shot_chart(df_a, player_name_a, season, color_range)
            else:
                fig_a = create_shot_chart(df_a, player_name_a, season, color_range)
            st.plotly_chart(fig_a, use_container_width=True)
        
        with chart_col2:
            if use_hexbin:
                fig_b = create_hexbin_chart(df_b, player_name_b, season)
                if fig_b is None:
                    st.warning("Not enough data for heatmap view")
                    fig_b = create_shot_chart(df_b, player_name_b, season, color_range)
            else:
                fig_b = create_shot_chart(df_b, player_name_b, season, color_range)
            st.plotly_chart(fig_b, use_container_width=True)
        
        # --- RADAR CHART COMPARISON ---
        st.subheader("📊 Player Profile Comparison")
        
        # Calculate zone rates first
        zone_mapping = {
            'Restricted Area': 'Rim', 'In The Paint (Non-RA)': 'Rim',
            'Mid-Range': 'Mid-Range',
            'Left Corner 3': '3-Point', 'Right Corner 3': '3-Point', 'Above the Break 3': '3-Point',
            'Backcourt': 'Other'
        }
        
        df_a['zone_group'] = df_a['SHOT_ZONE_BASIC'].map(zone_mapping)
        df_b['zone_group'] = df_b['SHOT_ZONE_BASIC'].map(zone_mapping)
        
        # Raw values for display
        rim_rate_a = len(df_a[df_a['zone_group'] == 'Rim']) / len(df_a) * 100 if len(df_a) > 0 else 0
        rim_rate_b = len(df_b[df_b['zone_group'] == 'Rim']) / len(df_b) * 100 if len(df_b) > 0 else 0
        three_rate_a = len(df_a[df_a['zone_group'] == '3-Point']) / len(df_a) * 100 if len(df_a) > 0 else 0
        three_rate_b = len(df_b[df_b['zone_group'] == '3-Point']) / len(df_b) * 100 if len(df_b) > 0 else 0
        gsaa_per_100_a = (metrics_a['gsaa'] / metrics_a['attempts'] * 100) if metrics_a['attempts'] > 0 else 0
        gsaa_per_100_b = (metrics_b['gsaa'] / metrics_b['attempts'] * 100) if metrics_b['attempts'] > 0 else 0
        
        # Normalize metrics for radar chart (0-100 scale)
        def normalize_metric(val, min_val, max_val):
            return max(0, min(100, ((val - min_val) / (max_val - min_val)) * 100))
        
        # More descriptive categories with actual values in hover
        radar_categories = [
            'Shot Volume<br>(Total FGA)',
            'Accuracy<br>(FG%)',
            'Efficiency<br>(eFG%)',
            '3PT Frequency<br>(% of shots)',
            'Rim Frequency<br>(% of shots)',
            'Value Added<br>(GSAA per 100)'
        ]
        
        # Normalize values (using typical NBA ranges)
        radar_a = [
            normalize_metric(metrics_a['attempts'], 0, 1500),
            normalize_metric(metrics_a['fg_pct'] * 100, 35, 55),
            normalize_metric(metrics_a['efg'] * 100, 45, 65),
            normalize_metric(three_rate_a, 20, 60),
            normalize_metric(rim_rate_a, 15, 50),
            normalize_metric(gsaa_per_100_a, -10, 15)
        ]
        radar_b = [
            normalize_metric(metrics_b['attempts'], 0, 1500),
            normalize_metric(metrics_b['fg_pct'] * 100, 35, 55),
            normalize_metric(metrics_b['efg'] * 100, 45, 65),
            normalize_metric(three_rate_b, 20, 60),
            normalize_metric(rim_rate_b, 15, 50),
            normalize_metric(gsaa_per_100_b, -10, 15)
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
            name=player_name_a,
            line_color='#1f77b4',
            fillcolor='rgba(31, 119, 180, 0.3)',
            line_width=3,
            customdata=actual_a + [actual_a[0]],
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.0f}/100<br>Actual: %{customdata}<extra>' + player_name_a + '</extra>'
        ))
        radar_fig.add_trace(go.Scatterpolar(
            r=radar_b + [radar_b[0]],
            theta=radar_categories + [radar_categories[0]],
            fill='toself',
            name=player_name_b,
            line_color='#ff7f0e',
            fillcolor='rgba(255, 127, 14, 0.3)',
            line_width=3,
            customdata=actual_b + [actual_b[0]],
            hovertemplate='<b>%{theta}</b><br>Score: %{r:.0f}/100<br>Actual: %{customdata}<extra>' + player_name_b + '</extra>'
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
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Add interpretation guide
        with st.expander("📖 How to Read This Chart"):
            st.markdown("""
            **Each axis represents a different skill dimension, normalized to a 0-100 scale:**
            - **Shot Volume**: Total field goal attempts (0 = low usage, 100 = high volume scorer)
            - **Accuracy (FG%)**: Raw shooting percentage (scaled: 35% = 0, 55% = 100)
            - **Efficiency (eFG%)**: Effective FG% accounting for 3-pointers (scaled: 45% = 0, 65% = 100)
            - **3PT Frequency**: Percentage of shots from 3-point range (scaled: 20% = 0, 60% = 100)
            - **Rim Frequency**: Percentage of shots at the rim (scaled: 15% = 0, 50% = 100)
            - **Value Added (GSAA/100)**: Points generated per 100 shots above league average
            
            *Larger area = more well-rounded/dominant player. Hover over points to see actual values.*
            """)
        
        # --- SHOT FREQUENCY COMPARISON ---
        st.subheader("🎯 Shot Distribution & Accuracy")
        
        # Calculate distributions AND accuracy for both players by zone
        zones = ['Rim', 'Mid-Range', '3-Point']
        
        def calc_zone_stats(df, player_name):
            stats = []
            for zone in zones:
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
        
        stats_a = calc_zone_stats(df_a, player_name_a)
        stats_b = calc_zone_stats(df_b, player_name_b)
        
        # Create grouped bar chart with accuracy annotations inside bars
        freq_bar_fig = go.Figure()
        
        # Bar width and positioning
        bar_width = 0.35
        x_positions = np.arange(len(zones))
        
        # Player A bars
        freq_bar_fig.add_trace(go.Bar(
            name=player_name_a,
            x=[z + f' ' for z in zones],  # Slight offset trick for grouping
            y=stats_a['freq_pct'],
            marker_color='#1f77b4',
            text=[f"<b>{row['fg_pct']:.1f}%</b><br>({int(row['makes'])}/{int(row['count'])})" 
                  for _, row in stats_a.iterrows()],
            textposition='inside',
            textfont=dict(color='white', size=12),
            insidetextanchor='middle',
            hovertemplate='<b>%{x}</b><br>Frequency: %{y:.1f}%<br>FG%: %{text}<extra>' + player_name_a + '</extra>',
            offsetgroup=0
        ))
        
        # Player B bars
        freq_bar_fig.add_trace(go.Bar(
            name=player_name_b,
            x=[z + f' ' for z in zones],
            y=stats_b['freq_pct'],
            marker_color='#ff7f0e',
            text=[f"<b>{row['fg_pct']:.1f}%</b><br>({int(row['makes'])}/{int(row['count'])})" 
                  for _, row in stats_b.iterrows()],
            textposition='inside',
            textfont=dict(color='white', size=12),
            insidetextanchor='middle',
            hovertemplate='<b>%{x}</b><br>Frequency: %{y:.1f}%<br>FG%: %{text}<extra>' + player_name_b + '</extra>',
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
                ticktext=['Rim', 'Mid-Range', '3-Point'],
                tickvals=[z + ' ' for z in zones],
                categoryorder='array',
                categoryarray=[z + ' ' for z in zones]
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
        
        st.plotly_chart(freq_bar_fig, use_container_width=True)
        
        st.caption("📊 **Reading the chart**: Bar height shows how often each player shoots from that zone. The percentage inside shows their accuracy (FG%) from that zone.")
        
        # --- HISTORICAL SEASON TREND ---
        st.subheader("📈 Historical Season Comparison")
        
        historical_seasons = ["2024-25", "2023-24", "2022-23", "2021-22"]
        
        @st.cache_data
        def get_historical_metrics(player_id, seasons, clutch):
            """Fetch metrics across multiple seasons."""
            history = []
            for s in seasons:
                try:
                    shots, league_avg = get_player_shots(player_id, s, clutch)
                    if not shots.empty:
                        shots = process_player_data(shots, league_avg)
                        m = calculate_player_metrics(shots)
                        history.append({
                            'season': s,
                            'efg': m['efg'] * 100,
                            'gsaa': m['gsaa'],
                            'attempts': m['attempts']
                        })
                except:
                    pass
            return pd.DataFrame(history)
        
        with st.spinner("Loading historical data..."):
            hist_a = get_historical_metrics(p_id_a, historical_seasons, clutch_only)
            hist_b = get_historical_metrics(p_id_b, historical_seasons, clutch_only)
        
        if not hist_a.empty and not hist_b.empty:
            hist_col1, hist_col2 = st.columns(2)
            
            with hist_col1:
                # eFG% trend
                efg_trend = go.Figure()
                efg_trend.add_trace(go.Scatter(
                    x=hist_a['season'], y=hist_a['efg'],
                    mode='lines+markers', name=player_name_a,
                    line=dict(color='#1f77b4')
                ))
                efg_trend.add_trace(go.Scatter(
                    x=hist_b['season'], y=hist_b['efg'],
                    mode='lines+markers', name=player_name_b,
                    line=dict(color='#ff7f0e')
                ))
                efg_trend.update_layout(
                    title="eFG% Trend",
                    yaxis_title="eFG%",
                    yaxis=dict(ticksuffix='%'),
                    height=300
                )
                st.plotly_chart(efg_trend, use_container_width=True)
            
            with hist_col2:
                # GSAA trend
                gsaa_trend = go.Figure()
                gsaa_trend.add_trace(go.Scatter(
                    x=hist_a['season'], y=hist_a['gsaa'],
                    mode='lines+markers', name=player_name_a,
                    line=dict(color='#1f77b4')
                ))
                gsaa_trend.add_trace(go.Scatter(
                    x=hist_b['season'], y=hist_b['gsaa'],
                    mode='lines+markers', name=player_name_b,
                    line=dict(color='#ff7f0e')
                ))
                gsaa_trend.update_layout(
                    title="Points Added (GSAA) Trend",
                    yaxis_title="GSAA",
                    height=300
                )
                st.plotly_chart(gsaa_trend, use_container_width=True)
        else:
            st.caption("Historical data not available for one or both players.")
        
    else:
        # Single player chart (larger)
        if use_hexbin:
            fig_a = create_hexbin_chart(df_a, player_name_a, season)
            if fig_a is None:
                st.warning("Not enough data for heatmap view, showing scatter instead")
                fig_a = create_shot_chart(df_a, player_name_a, season, color_range)
            else:
                fig_a.update_layout(height=700)
        else:
            fig_a = create_shot_chart(df_a, player_name_a, season, color_range)
            fig_a.update_layout(height=700)
        st.plotly_chart(fig_a, use_container_width=True)
    
    # --- ZONE BREAKDOWN (Single player only for now) ---
    if not compare_mode:
        # Fetch league averages for zone breakdown
        with st.spinner("Loading league averages..."):
            league_avgs = get_league_averages(season)
        
        # Calculate player FG% by zone
        total_player_attempts = len(df_a)
        player_zone_stats = df_a.groupby('SHOT_ZONE_BASIC').agg(
            player_makes=('SHOT_MADE_FLAG', 'sum'),
            player_attempts=('SHOT_MADE_FLAG', 'count')
        ).reset_index()
        player_zone_stats['player_fg_pct'] = player_zone_stats['player_makes'] / player_zone_stats['player_attempts']
        player_zone_stats['player_freq_pct'] = player_zone_stats['player_attempts'] / total_player_attempts
        
        # Merge with league averages
        zone_comparison = player_zone_stats.merge(league_avgs, on='SHOT_ZONE_BASIC', how='left')
        zone_comparison['relative_fg_pct'] = (zone_comparison['player_fg_pct'] - zone_comparison['league_fg_pct']) * 100
        
        # Zone breakdown table
        st.subheader("Zone Breakdown vs League Average")
        zone_display = zone_comparison[['SHOT_ZONE_BASIC', 'player_attempts', 'player_fg_pct', 'league_fg_pct', 'relative_fg_pct']].copy()
        zone_display.columns = ['Zone', 'Attempts', 'Player FG%', 'League FG%', 'Relative FG%']
        zone_display['Player FG%'] = zone_display['Player FG%'].apply(lambda x: f"{x:.1%}")
        zone_display['League FG%'] = zone_display['League FG%'].apply(lambda x: f"{x:.1%}")
        zone_display['Relative FG%'] = zone_display['Relative FG%'].apply(lambda x: f"{x:+.1f}%")
        st.dataframe(zone_display, use_container_width=True)
        
        # Shot Frequency Profile with Accuracy
        st.subheader("Shot Frequency Profile")
        
        zone_mapping = {
            'Restricted Area': 'Rim',
            'In The Paint (Non-RA)': 'Rim',
            'Mid-Range': 'Mid-Range',
            'Left Corner 3': '3-Point',
            'Right Corner 3': '3-Point',
            'Above the Break 3': '3-Point',
            'Backcourt': 'Other'
        }
        
        # Calculate player zone stats with accuracy
        df_a['zone_group'] = df_a['SHOT_ZONE_BASIC'].map(zone_mapping)
        zones = ['Rim', 'Mid-Range', '3-Point']
        
        player_zone_stats = []
        for zone in zones:
            zone_df = df_a[df_a['zone_group'] == zone]
            count = len(zone_df)
            makes = zone_df['SHOT_MADE_FLAG'].sum()
            fg_pct = (makes / count * 100) if count > 0 else 0
            freq_pct = (count / len(df_a) * 100) if len(df_a) > 0 else 0
            player_zone_stats.append({
                'zone': zone,
                'count': count,
                'makes': makes,
                'fg_pct': fg_pct,
                'freq_pct': freq_pct
            })
        player_stats_df = pd.DataFrame(player_zone_stats)
        
        # Calculate league zone stats
        league_avg_df_copy = league_avg_df.copy()
        league_avg_df_copy['zone_group'] = league_avg_df_copy['SHOT_ZONE_BASIC'].map(zone_mapping)
        league_zone_dist = league_avg_df_copy.groupby('zone_group').agg(
            league_fga=('FGA', 'sum'),
            league_fgm=('FGM', 'sum')
        ).reset_index()
        total_league_fga = league_zone_dist['league_fga'].sum()
        league_zone_dist['league_freq_pct'] = league_zone_dist['league_fga'] / total_league_fga * 100
        league_zone_dist['league_fg_pct'] = league_zone_dist['league_fgm'] / league_zone_dist['league_fga'] * 100
        league_zone_dist = league_zone_dist[league_zone_dist['zone_group'].isin(zones)]
        
        # Create grouped bar chart with accuracy inside bars
        freq_fig = go.Figure()
        
        # Player bars
        freq_fig.add_trace(go.Bar(
            name=player_name_a,
            x=player_stats_df['zone'],
            y=player_stats_df['freq_pct'],
            marker_color='#1f77b4',
            text=[f"<b>{row['fg_pct']:.1f}%</b><br>({int(row['makes'])}/{int(row['count'])})" 
                  for _, row in player_stats_df.iterrows()],
            textposition='inside',
            textfont=dict(color='white', size=11),
            insidetextanchor='middle',
            hovertemplate='<b>%{x}</b><br>Frequency: %{y:.1f}%<extra>' + player_name_a + '</extra>'
        ))
        
        # League average bars
        freq_fig.add_trace(go.Bar(
            name='League Avg',
            x=league_zone_dist['zone_group'],
            y=league_zone_dist['league_freq_pct'],
            marker_color='#7f7f7f',
            text=[f"<b>{row['league_fg_pct']:.1f}%</b>" 
                  for _, row in league_zone_dist.iterrows()],
            textposition='inside',
            textfont=dict(color='white', size=11),
            insidetextanchor='middle',
            hovertemplate='<b>%{x}</b><br>Frequency: %{y:.1f}%<extra>League Avg</extra>'
        ))
        
        # Add difference annotations above player bars
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
                text=f"Shot Frequency: {player_name_a} vs League<br><sup>Bar height = frequency | Inside label = accuracy (FG%)</sup>",
                font=dict(size=14)
            ),
            xaxis=dict(categoryorder='array', categoryarray=['Rim', 'Mid-Range', '3-Point']),
            yaxis=dict(title='Frequency (%)', ticksuffix='%'),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
            height=420,
            bargap=0.15,
            bargroupgap=0.1
        )
        
        st.plotly_chart(freq_fig, use_container_width=True)
        st.caption("📊 **Reading the chart**: Bar height = shot frequency from zone | Inside % = shooting accuracy (FG%) | ▲▼ = difference vs league")
    
    # --- DOPPELGÄNGER FINDER SECTION ---
    if show_doppelganger:
        st.markdown("---")
        st.subheader("🔍 Statistical Doppelgängers")
        st.caption("Finding players with similar playing styles using machine learning")
        
        with st.spinner("Building similarity model..."):
            try:
                # Get league-wide stats for the season
                league_stats = get_advanced_stats(season)
                
                if league_stats.empty:
                    st.warning("Could not load league stats for this season.")
                else:
                    # Build the ML model
                    nn_model, scaler, feature_cols = build_similarity_model(league_stats)
                    
                    # Find similar players
                    similar_players, selected_player_data, error = find_similar_players(
                        player_name_a, league_stats, nn_model, scaler, feature_cols
                    )
                    
                    if error:
                        st.warning(f"⚠️ {error}. Player may not have enough games this season.")
                    elif similar_players:
                        # Display results
                        st.markdown(f"**Top 5 players most similar to {player_name_a}:**")
                        
                        # Create display dataframe
                        display_df = pd.DataFrame(similar_players)[['Rank', 'Player', 'Team', 'Similarity', 'USG%', 'TS%', '3P Rate']]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # --- RADAR CHART: Selected Player vs #1 Match ---
                        st.markdown("#### Style Comparison: Selected Player vs Top Match")
                        
                        top_match_idx = similar_players[0]['_idx']
                        top_match_data = league_stats.iloc[top_match_idx]
                        top_match_name = top_match_data['PLAYER_NAME']
                        
                        # Radar chart categories and values
                        radar_cats = ['Usage Rate', 'True Shooting', 'Assist Rate', 'Rebound Rate', 'Pace', '3PT Rate']
                        
                        # Helper to safely get values (handle NaN)
                        def safe_val(val, default=0):
                            if pd.isna(val):
                                return default
                            return float(val)
                        
                        # Get raw values for both players
                        # Note: NBA API returns USG_PCT, TS_PCT, AST_PCT, REB_PCT as percentages (e.g., 25.5 for 25.5%)
                        # PACE is raw number (~100), 3P_AR is decimal (0.35 = 35%)
                        selected_vals = [
                            safe_val(selected_player_data['USG_PCT']),
                            safe_val(selected_player_data['TS_PCT']),  # Already percentage from API
                            safe_val(selected_player_data['AST_PCT']),
                            safe_val(selected_player_data['REB_PCT']),
                            safe_val(selected_player_data['PACE']),
                            safe_val(selected_player_data['3P_AR']) * 100  # Convert decimal to percentage
                        ]
                        
                        match_vals = [
                            safe_val(top_match_data['USG_PCT']),
                            safe_val(top_match_data['TS_PCT']),
                            safe_val(top_match_data['AST_PCT']),
                            safe_val(top_match_data['REB_PCT']),
                            safe_val(top_match_data['PACE']),
                            safe_val(top_match_data['3P_AR']) * 100
                        ]
                        
                        # Normalize for radar (0-100 scale based on typical NBA ranges)
                        # Adjusted ranges based on actual NBA API percentage values
                        def norm_radar(vals):
                            ranges = [
                                (15, 35),   # USG% (typically 15-35%)
                                (50, 70),   # TS% (typically 50-70%)
                                (5, 35),    # AST% (typically 5-35%)
                                (3, 18),    # REB% (typically 3-18%)
                                (95, 105),  # PACE (typically 95-105)
                                (15, 55)    # 3P Rate (typically 15-55%)
                            ]
                            normed = []
                            for v, r in zip(vals, ranges):
                                if r[1] - r[0] == 0:
                                    normed.append(50)  # Avoid division by zero
                                else:
                                    normalized = (v - r[0]) / (r[1] - r[0]) * 100
                                    normed.append(max(5, min(100, normalized)))  # Min 5 so it's visible
                            return normed
                        
                        selected_normed = norm_radar(selected_vals)
                        match_normed = norm_radar(match_vals)
                        
                        doppel_radar = go.Figure()
                        
                        # Format values for display with proper units
                        def format_vals_display(vals):
                            return [
                                f"{vals[0]:.1f}%",   # USG
                                f"{vals[1]:.1f}%",   # TS
                                f"{vals[2]:.1f}%",   # AST
                                f"{vals[3]:.1f}%",   # REB
                                f"{vals[4]:.1f}",    # PACE (no %)
                                f"{vals[5]:.1f}%"    # 3P Rate
                            ]
                        
                        selected_display = format_vals_display(selected_vals)
                        match_display = format_vals_display(match_vals)
                        
                        doppel_radar.add_trace(go.Scatterpolar(
                            r=selected_normed + [selected_normed[0]],
                            theta=radar_cats + [radar_cats[0]],
                            fill='toself',
                            name=player_name_a,
                            line_color='#1f77b4',
                            fillcolor='rgba(31, 119, 180, 0.3)',
                            line_width=3,
                            customdata=selected_display + [selected_display[0]],
                            hovertemplate='<b>%{theta}</b><br>Value: %{customdata}<extra>' + player_name_a + '</extra>'
                        ))
                        
                        doppel_radar.add_trace(go.Scatterpolar(
                            r=match_normed + [match_normed[0]],
                            theta=radar_cats + [radar_cats[0]],
                            fill='toself',
                            name=top_match_name,
                            line_color='#2ca02c',
                            fillcolor='rgba(44, 160, 44, 0.3)',
                            line_width=3,
                            customdata=match_display + [match_display[0]],
                            hovertemplate='<b>%{theta}</b><br>Value: %{customdata}<extra>' + top_match_name + '</extra>'
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
                                text=f"Style Profile: {player_name_a} vs {top_match_name}<br><sup>Similarity: {similar_players[0]['Similarity']}</sup>",
                                font=dict(size=15)
                            ),
                            margin=dict(t=80, b=60, l=60, r=60)
                        )
                        
                        st.plotly_chart(doppel_radar, use_container_width=True)
                        
                        # Interpretation
                        with st.expander("📖 About the Similarity Model"):
                            st.markdown("""
                            **How it works:**
                            
                            The Doppelgänger Finder uses a **K-Nearest Neighbors** algorithm to find players with similar "style vectors."
                            
                            **Features used for matching:**
                            | Metric | What it measures |
                            |--------|-----------------|
                            | **Usage Rate (USG%)** | How often the player uses possessions |
                            | **True Shooting (TS%)** | Overall scoring efficiency |
                            | **Assist Rate (AST%)** | Percentage of teammate FGs assisted |
                            | **Rebound Rate (REB%)** | Percentage of available rebounds grabbed |
                            | **Pace** | Team's possessions per 48 minutes |
                            | **3-Point Rate (3P_AR)** | Percentage of shots from 3-point range |
                            
                            *All features are normalized before comparison so no single stat dominates.*
                            
                            **Filters applied:** Only players with 15+ games and 10+ minutes per game are included.
                            """)
                    else:
                        st.info("No similar players found.")
            except Exception as e:
                st.error(f"Error loading similarity data: {str(e)}")
    
    # --- RAW DATA INSPECTOR ---
    with st.expander("Inspect Raw Possession Data"):
        if compare_mode:
            tab1, tab2 = st.tabs([player_name_a, player_name_b])
            with tab1:
                st.dataframe(df_a)
            with tab2:
                st.dataframe(df_b)
        else:
            st.dataframe(df_a)

except Exception as e:
    st.error(f"An error occurred: {e}")