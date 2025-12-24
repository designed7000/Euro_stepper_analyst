"""
NBA Player DNA: Spatial Efficiency Engine
Main Streamlit application entry point.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Local imports
from config import (
    SEASONS, HISTORICAL_SEASONS, ZONE_MAPPING, ZONE_ORDER,
    EFFICIENCY_COLOR_RANGE, CHART_HEIGHT
)
from utils.helpers import get_player_id
from data.api import (
    get_player_shots, get_league_averages, get_advanced_stats, get_league_leaders, get_standings
)
from data.processing import (
    process_player_data, calculate_player_metrics, calculate_zone_stats,
    add_zone_groups, get_top_players_by_position_smart
)
from analysis.awards import AwardTracker
from charts.awards import create_mvp_value_breakdown_chart, create_winning_vs_stats_scatter
from charts.court import create_shot_chart, create_hexbin_chart
from charts.comparisons import (
    create_radar_comparison, create_zone_frequency_comparison,
    create_single_player_zone_chart, create_doppelganger_radar
)
from charts.trends import (
    create_historical_efg_chart, create_historical_gsaa_chart,
    create_scoring_efficiency_chart, create_playmaking_chart,
    create_twoway_quadrant_chart, create_top_scorers_bar
)
from analysis.similarity import (
    build_similarity_model, find_similar_players, get_player_style_values
)


# --- APP CONFIGURATION ---
st.set_page_config(page_title="NBA Shot DNA", layout="wide")


# --- CACHED HELPER FOR HISTORICAL DATA ---
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


# --- MAIN DASHBOARD INTERFACE ---
st.title("NBA Player DNA: Spatial Efficiency Engine")

# Sidebar for Inputs
with st.sidebar:
    # Logo at the top
    st.image("logo.png", use_container_width=True)
    
    st.header("Analyst Controls")
    
    # Season selector at top
    season = st.selectbox("Season", SEASONS)
    
    st.markdown("---")
    
    # --- LEAGUE LEADERS ---
    st.subheader("League Leaders")
    show_leaders = st.checkbox("Show Leaders by Position", value=False,
                               help="View top performers across positions")
    
    # --- MVP TRACKER ---
    st.subheader("🏆 Awards Tracker")
    show_mvp_tracker = st.checkbox("MVP Ranking Projection", value=False,
                                    help="DNA Production Index MVP rankings")
    
    st.markdown("---")
    
    # --- PLAYER ANALYSIS ---
    st.subheader("Player Analysis")
    show_player_analysis = st.checkbox("Analyze Player", value=False,
                                        help="Shot charts and efficiency analysis")
    
    player_name_a = st.text_input("Player A Name", "Stephen Curry")
    
    compare_mode = st.checkbox("Compare Players", value=False)
    if compare_mode:
        player_name_b = st.text_input("Player B Name", "LeBron James")
    
    st.subheader("Game Context")
    clutch_only = st.checkbox("Clutch Time Only", value=False, 
                               help="Last 5 min, score within 5 pts")
    
    st.subheader("Player Discovery")
    show_doppelganger = st.checkbox("Find Similar Players", value=False,
                                     help="ML-powered player similarity engine")


# --- ANALYTICS ENGINE ---
try:
    # --- WELCOME PAGE (no mode selected) ---
    if not show_leaders and not show_mvp_tracker and not show_player_analysis and not show_doppelganger:
        st.markdown("""        
        <div style="text-align: center; padding: 100px 20px;">
            <h1 style="font-size: 3em; margin-bottom: 10px;">🏀 NBA Player DNA</h1>
            <h2 style="font-size: 1.5em; color: #888; font-weight: normal;">Spatial Efficiency Engine</h2>
            <p style="font-size: 1.2em; color: #aaa; margin-top: 40px;">
                Select an option from the sidebar to begin your analysis.
            </p>
            <p style="font-size: 2em; color: #666; margin-top: 20px;">
                 <b>League Leaders</b> - View top performers by position<br>
                 <b>MVP Ladder</b> - DNA Production Index rankings<br>
                 <b>Player Analysis</b> - Deep dive into shot charts<br>
                 <b>Similar Players</b> - ML-powered player matching
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # --- LEAGUE LEADERS MODE ---
    if show_leaders:
        st.subheader("League Leaders by Position")
        
        leader_tab1, leader_tab2, leader_tab3 = st.tabs(["Scoring Impact", "Playmaking", "Two-Way Impact"])
        
        with st.spinner("Loading league leaders..."):
            try:
                leaders_df = get_league_leaders(season)
                
                if leaders_df.empty or leaders_df['POSITION_GROUP'].isna().all():
                    st.warning("Could not load position data for this season. Try a different season.")
                else:
                    # Helper function to display position tables
                    def display_position_tables_with_selector(df, category, description, key_suffix):
                        tables_container = st.container()
                        
                        _, sel_col, _ = st.columns([2, 1, 2])
                        with sel_col:
                            num_leaders = st.selectbox("Show", ["Top 5", "Top 10", "Top 20"], index=0, 
                                                       label_visibility="collapsed", key=f"leaders_{key_suffix}")
                            num_leaders = int(num_leaders.split()[1])
                        
                        with tables_container:
                            table_height = 35 + (num_leaders * 35)
                            top_players = get_top_players_by_position_smart(df, category, n=num_leaders)
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("**Guards**")
                                if 'Guard' in top_players and not top_players['Guard'].empty:
                                    guard_df = top_players['Guard'].reset_index(drop=True)
                                    guard_df.index = guard_df.index + 1
                                    st.dataframe(guard_df, use_container_width=True, hide_index=False, height=table_height)
                                else:
                                    st.caption("No guard data")
                            
                            with col2:
                                st.markdown("**Forwards**")
                                if 'Forward' in top_players and not top_players['Forward'].empty:
                                    forward_df = top_players['Forward'].reset_index(drop=True)
                                    forward_df.index = forward_df.index + 1
                                    st.dataframe(forward_df, use_container_width=True, hide_index=False, height=table_height)
                                else:
                                    st.caption("No forward data")
                            
                            with col3:
                                st.markdown("**Centers**")
                                if 'Center' in top_players and not top_players['Center'].empty:
                                    center_df = top_players['Center'].reset_index(drop=True)
                                    center_df.index = center_df.index + 1
                                    st.dataframe(center_df, use_container_width=True, hide_index=False, height=table_height)
                                else:
                                    st.caption("No center data")
                    
                    # Display each tab
                    with leader_tab1:
                        st.caption(f"Ranked by USG% × TS% (load × efficiency) | {season}")
                        display_position_tables_with_selector(leaders_df, "scoring", "PTS × TS%", "scoring")
                    
                    with leader_tab2:
                        st.caption(f"Ranked by AST/100 poss weighted by AST/TO ratio | {season}")
                        display_position_tables_with_selector(leaders_df, "playmaking", "AST Impact", "playmaking")
                    
                    with leader_tab3:
                        st.caption(f"Ranked by Net Rating (Offense - Defense) | {season}")
                        display_position_tables_with_selector(leaders_df, "impact", "Two-Way", "impact")
                    
                    # --- VISUALIZATIONS ---
                    st.markdown("---")
                    st.subheader("Advanced Analytics")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        fig_helio = create_scoring_efficiency_chart(leaders_df)
                        if fig_helio:
                            st.plotly_chart(fig_helio, use_container_width=True)
                            st.caption("Bubble size = PPG • Above line = efficient scorers")
                        else:
                            st.warning("No scoring data available for this filter")
                    
                    with viz_col2:
                        fig_play = create_playmaking_chart(leaders_df)
                        if fig_play:
                            st.plotly_chart(fig_play, use_container_width=True)
                            st.caption("Bubble size = AST/TO ratio | Top-right = elite playmakers")
                    
                    # Two-Way Quadrant
                    st.markdown("##### Two-Way Impact Quadrant")
                    fig_quadrant = create_twoway_quadrant_chart(leaders_df)
                    if fig_quadrant:
                        st.plotly_chart(fig_quadrant, use_container_width=True)
                        st.caption("Crosshairs = league average | Top-right quadrant = elite two-way players")
                    
                    # Top Scorers
                    st.markdown("##### Top 10 Scorers Across All Positions")
                    fig_top = create_top_scorers_bar(leaders_df)
                    st.plotly_chart(fig_top, use_container_width=True)
                    st.caption("Guard (cyan) | Forward (red) | Center (green)")
                            
            except Exception as e:
                st.error(f"Error loading leaders: {str(e)}")
        
        st.stop()
    
    # --- MVP TRACKER MODE ---
    if show_mvp_tracker:
        st.subheader("🏆 MVP Ladder - DNA Production Index")
        st.caption("Scarcity-weighted statistics × team success = player value")
        
        with st.spinner("Calculating MVP ladder..."):
            try:
                # Get data
                leaders_df = get_league_leaders(season)
                standings_df = get_standings(season)
                
                if leaders_df.empty:
                    st.warning("Could not load player stats for this season.")
                    st.stop()
                
                if standings_df.empty:
                    st.warning("Could not load standings for this season.")
                    st.stop()
                
                # Calculate MVP ladder
                tracker = AwardTracker(leaders_df, standings_df)
                mvp_ladder = tracker.calculate_mvp_ladder(top_n=20)  # Get 20, display based on selector
                
                if mvp_ladder.empty:
                    st.warning("Could not calculate MVP ladder.")
                    st.stop()
                
                # --- MVP LADDER TABLE ---
                # Use container pattern like leaders section
                mvp_table_container = st.container()
                
                # Selector below the table
                _, sel_col, _ = st.columns([2, 1, 2])
                with sel_col:
                    mvp_top_n = st.selectbox("Show", ["Top 5", "Top 10", "Top 20"], index=1, 
                                              label_visibility="collapsed", key="mvp_top_n")
                    mvp_top_n = int(mvp_top_n.split()[1])
                
                with mvp_table_container:
                    st.markdown(f"##### Top {mvp_top_n} MVP Candidates")
                    
                    # Format display columns (include Record)
                    display_cols = ['Rank', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'MVP_SCORE', 'RAW_VALUE', 'RECORD', 'WIN_PCT']
                    mvp_display = mvp_ladder[display_cols].head(mvp_top_n).copy()
                    mvp_display.columns = ['Rank', 'Player', 'Team', 'MVP Score', 'Raw Value', 'Record', 'Win %']
                    mvp_display['MVP Score'] = mvp_display['MVP Score'].apply(lambda x: f"{x:.1f}")
                    mvp_display['Raw Value'] = mvp_display['Raw Value'].apply(lambda x: f"{x:.1f}")
                    mvp_display['Win %'] = mvp_display['Win %'].apply(lambda x: f"{x:.1%}")
                    
                    # Dynamic table height based on selection
                    table_height = 35 + (mvp_top_n * 35)
                    st.dataframe(mvp_display, use_container_width=True, hide_index=True, height=table_height)
                
                st.caption("**MVP Score** = Raw Production Value × √(Team Win %) | Higher score = stronger MVP case")
                
                # --- VISUALIZATIONS ---
                st.markdown("---")
                st.subheader("DNA Production Analysis")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    fig_breakdown = create_mvp_value_breakdown_chart(mvp_ladder.head(10))
                    if fig_breakdown:
                        st.plotly_chart(fig_breakdown, use_container_width=True)
                        st.caption("Stacked bars show contribution from each stat category")
                
                with viz_col2:
                    fig_scatter = create_winning_vs_stats_scatter(mvp_ladder.head(15))
                    if fig_scatter:
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        st.caption("Top-right quadrant = elite production + winning")
                
                # --- METHODOLOGY EXPANDER ---
                with st.expander("📖 About DNA Production Index"):
                    st.markdown("""
                    **How the MVP Ladder is calculated:**
                    
                    The DNA Production Index uses **scarcity-weighted statistics** to measure player value:
                    
                    **Step 1: Calculate Scarcity Weights**
                    | Stat | Formula |
                    |------|---------|
                    | Points | Total League Points / League PTS Sum |
                    | Assists | Total League Points / League AST Sum |
                    | Rebounds | Total League Points / League REB Sum |
                    | Steals | Total League Points / League STL Sum |
                    | Blocks | Total League Points / League BLK Sum |
                    
                    *Rare stats (like blocks) get higher weights because they're harder to accumulate.*
                    
                    **Step 2: Calculate Raw Production Value**
                    ```
                    Raw Value = (PTS × w_pts) + (AST × w_ast) + (REB × w_reb) + (STL × w_stl) + (BLK × w_blk)
                    ```
                    
                    **Step 3: Apply Team Success Modifier**
                    ```
                    MVP Score = Raw Value × √(Team Win Percentage)
                    ```
                    
                    *This rewards players on winning teams while not punishing those on mid-tier teams too heavily.*
                    
                    **Filters Applied:**
                    - Minimum 15 games played
                    - Minimum 20 minutes per game
                    """)
                
            except Exception as e:
                st.error(f"Error calculating MVP ladder: {str(e)}")
        
        st.stop()
    
    # --- PLAYER ANALYSIS MODE ---
    if not show_player_analysis and not show_doppelganger:
        st.stop()
    
    # Get Player A data with fuzzy matching
    p_id_a, corrected_name_a, match_msg_a = get_player_id(player_name_a)
    
    if not p_id_a:
        st.error(f"Player A '{player_name_a}' not found. Check spelling.")
        st.stop()
    
    if match_msg_a:
        st.info(f"{match_msg_a}")
    
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
        
        if match_msg_b:
            st.info(f"{match_msg_b}")
        
        player_name_b = corrected_name_b
        
        with st.spinner(f"Loading data for {player_name_b}..."):
            df_b, _ = get_player_shots(p_id_b, season, clutch_only)
        
        if df_b.empty:
            st.warning(f"No shot data found for {player_name_b} in {season}.")
            st.stop()
        
        df_b = process_player_data(df_b, league_avg_df)
        metrics_b = calculate_player_metrics(df_b)
    
    # Show clutch indicator
    if clutch_only:
        if compare_mode:
            st.info(f"Clutch Time: {player_name_a} ({len(df_a)} shots) vs {player_name_b} ({len(df_b)} shots)")
        else:
            st.info(f"Clutch Time: {len(df_a)} shots (Last 5 min, score within 5 pts)")
    
    # --- METRICS DISPLAY ---
    if compare_mode:
        st.subheader("Head-to-Head Comparison")
        
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
        
        styled_df = comparison_df.drop(columns=['_delta_raw']).style.apply(
            lambda x: ['', '', '', 
                       'background-color: #90EE90' if comparison_df.loc[x.name, '_delta_raw'] > 0 
                       else ('background-color: #FFB6C1' if comparison_df.loc[x.name, '_delta_raw'] < 0 else '')
                      ], axis=1
        )
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.caption("Delta shows Player A minus Player B. Positive = Player A higher | Negative = Player B higher")
        
    else:
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Volume (FGA)", f"{metrics_a['attempts']}")
        col2.metric("Raw FG%", f"{metrics_a['fg_pct']:.1%}")
        col3.metric("Effective FG% (eFG)", f"{metrics_a['efg']:.1%}")
        col4.metric("3PT Volume", f"{metrics_a['threes_attempted']}")
        col5.metric("Points Added (GSAA)", f"{metrics_a['gsaa']:+.1f}")
        
        st.caption("**Points Added (GSAA)**: Points generated solely by shooting skill above league average expectation.")
    
    # --- SHOT CHARTS ---
    st.subheader("Relative Efficiency Maps")
    
    chart_view = st.radio(
        "Chart Style",
        ["Scatter (Individual Shots)", "Heatmap (Zone Aggregated)"],
        horizontal=True,
        help="Scatter shows each shot. Heatmap groups shots into zones showing hot/cold areas."
    )
    use_hexbin = chart_view == "Heatmap (Zone Aggregated)"
    
    if use_hexbin:
        st.caption("**Heatmap View**: Larger hexagons = more shots from that area. Color shows efficiency vs league average (blue = above avg, red = below avg)")
    else:
        st.caption("**Scatter View**: Each dot is a shot attempt. Color shows efficiency vs league average (blue = above avg, red = below avg)")
    
    if compare_mode:
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            if use_hexbin:
                fig_a = create_hexbin_chart(df_a, player_name_a, season)
                if fig_a is None:
                    st.warning("Not enough data for heatmap view")
                    fig_a = create_shot_chart(df_a, player_name_a, season, EFFICIENCY_COLOR_RANGE)
            else:
                fig_a = create_shot_chart(df_a, player_name_a, season, EFFICIENCY_COLOR_RANGE)
            st.plotly_chart(fig_a, use_container_width=True)
        
        with chart_col2:
            if use_hexbin:
                fig_b = create_hexbin_chart(df_b, player_name_b, season)
                if fig_b is None:
                    st.warning("Not enough data for heatmap view")
                    fig_b = create_shot_chart(df_b, player_name_b, season, EFFICIENCY_COLOR_RANGE)
            else:
                fig_b = create_shot_chart(df_b, player_name_b, season, EFFICIENCY_COLOR_RANGE)
            st.plotly_chart(fig_b, use_container_width=True)
        
        # --- RADAR CHART COMPARISON ---
        st.subheader("Player Profile Comparison")
        
        df_a = add_zone_groups(df_a)
        df_b = add_zone_groups(df_b)
        
        rim_rate_a = len(df_a[df_a['zone_group'] == 'Rim']) / len(df_a) * 100 if len(df_a) > 0 else 0
        rim_rate_b = len(df_b[df_b['zone_group'] == 'Rim']) / len(df_b) * 100 if len(df_b) > 0 else 0
        three_rate_a = len(df_a[df_a['zone_group'] == '3-Point']) / len(df_a) * 100 if len(df_a) > 0 else 0
        three_rate_b = len(df_b[df_b['zone_group'] == '3-Point']) / len(df_b) * 100 if len(df_b) > 0 else 0
        
        radar_fig = create_radar_comparison(
            player_name_a, player_name_b, metrics_a, metrics_b,
            rim_rate_a, rim_rate_b, three_rate_a, three_rate_b
        )
        st.plotly_chart(radar_fig, use_container_width=True)
        
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
        st.subheader("Shot Distribution & Accuracy")
        
        stats_a = calculate_zone_stats(df_a, player_name_a)
        stats_b = calculate_zone_stats(df_b, player_name_b)
        
        freq_bar_fig = create_zone_frequency_comparison(stats_a, stats_b, player_name_a, player_name_b)
        st.plotly_chart(freq_bar_fig, use_container_width=True)
        st.caption("**Reading the chart**: Bar height shows how often each player shoots from that zone. The percentage inside shows their accuracy (FG%) from that zone.")
        
        # --- HISTORICAL SEASON TREND ---
        st.subheader("Historical Season Comparison")
        
        with st.spinner("Loading historical data..."):
            hist_a = get_historical_metrics(p_id_a, HISTORICAL_SEASONS, clutch_only)
            hist_b = get_historical_metrics(p_id_b, HISTORICAL_SEASONS, clutch_only)
        
        if not hist_a.empty and not hist_b.empty:
            hist_col1, hist_col2 = st.columns(2)
            
            with hist_col1:
                efg_trend = create_historical_efg_chart(hist_a, hist_b, player_name_a, player_name_b)
                st.plotly_chart(efg_trend, use_container_width=True)
            
            with hist_col2:
                gsaa_trend = create_historical_gsaa_chart(hist_a, hist_b, player_name_a, player_name_b)
                st.plotly_chart(gsaa_trend, use_container_width=True)
        else:
            st.caption("Historical data not available for one or both players.")
        
    else:
        # Single player chart
        if use_hexbin:
            fig_a = create_hexbin_chart(df_a, player_name_a, season)
            if fig_a is None:
                st.warning("Not enough data for heatmap view, showing scatter instead")
                fig_a = create_shot_chart(df_a, player_name_a, season, EFFICIENCY_COLOR_RANGE)
            else:
                fig_a.update_layout(height=CHART_HEIGHT['single_player'])
        else:
            fig_a = create_shot_chart(df_a, player_name_a, season, EFFICIENCY_COLOR_RANGE)
            fig_a.update_layout(height=CHART_HEIGHT['single_player'])
        st.plotly_chart(fig_a, use_container_width=True)
    
    # --- ZONE BREAKDOWN (Single player only) ---
    if not compare_mode:
        with st.spinner("Loading league averages..."):
            league_avgs = get_league_averages(season)
        
        total_player_attempts = len(df_a)
        player_zone_stats = df_a.groupby('SHOT_ZONE_BASIC').agg(
            player_makes=('SHOT_MADE_FLAG', 'sum'),
            player_attempts=('SHOT_MADE_FLAG', 'count')
        ).reset_index()
        player_zone_stats['player_fg_pct'] = player_zone_stats['player_makes'] / player_zone_stats['player_attempts']
        player_zone_stats['player_freq_pct'] = player_zone_stats['player_attempts'] / total_player_attempts
        
        zone_comparison = player_zone_stats.merge(league_avgs, on='SHOT_ZONE_BASIC', how='left')
        zone_comparison['relative_fg_pct'] = (zone_comparison['player_fg_pct'] - zone_comparison['league_fg_pct']) * 100
        
        st.subheader("Zone Breakdown vs League Average")
        zone_display = zone_comparison[['SHOT_ZONE_BASIC', 'player_attempts', 'player_fg_pct', 'league_fg_pct', 'relative_fg_pct']].copy()
        zone_display.columns = ['Zone', 'Attempts', 'Player FG%', 'League FG%', 'Relative FG%']
        zone_display['Player FG%'] = zone_display['Player FG%'].apply(lambda x: f"{x:.1%}")
        zone_display['League FG%'] = zone_display['League FG%'].apply(lambda x: f"{x:.1%}")
        zone_display['Relative FG%'] = zone_display['Relative FG%'].apply(lambda x: f"{x:+.1f}%")
        st.dataframe(zone_display, use_container_width=True)
        
        # Shot Frequency Profile
        st.subheader("Shot Frequency Profile")
        
        df_a = add_zone_groups(df_a)
        
        player_zone_stats_list = []
        for zone in ZONE_ORDER:
            zone_df = df_a[df_a['zone_group'] == zone]
            count = len(zone_df)
            makes = zone_df['SHOT_MADE_FLAG'].sum()
            fg_pct = (makes / count * 100) if count > 0 else 0
            freq_pct = (count / len(df_a) * 100) if len(df_a) > 0 else 0
            player_zone_stats_list.append({
                'zone': zone,
                'count': count,
                'makes': makes,
                'fg_pct': fg_pct,
                'freq_pct': freq_pct
            })
        player_stats_df = pd.DataFrame(player_zone_stats_list)
        
        # Calculate league zone stats
        league_avg_df_copy = league_avg_df.copy()
        league_avg_df_copy['zone_group'] = league_avg_df_copy['SHOT_ZONE_BASIC'].map(ZONE_MAPPING)
        league_zone_dist = league_avg_df_copy.groupby('zone_group').agg(
            league_fga=('FGA', 'sum'),
            league_fgm=('FGM', 'sum')
        ).reset_index()
        total_league_fga = league_zone_dist['league_fga'].sum()
        league_zone_dist['league_freq_pct'] = league_zone_dist['league_fga'] / total_league_fga * 100
        league_zone_dist['league_fg_pct'] = league_zone_dist['league_fgm'] / league_zone_dist['league_fga'] * 100
        league_zone_dist = league_zone_dist[league_zone_dist['zone_group'].isin(ZONE_ORDER)]
        
        freq_fig = create_single_player_zone_chart(player_stats_df, league_zone_dist, player_name_a)
        st.plotly_chart(freq_fig, use_container_width=True)
        st.caption("**Reading the chart**: Bar height = shot frequency from zone | Inside % = shooting accuracy (FG%) | Up/Down arrows = difference vs league")
    
    # --- DOPPELGÄNGER FINDER ---
    if show_doppelganger:
        st.markdown("---")
        st.subheader("Statistical Doppelgangers")
        st.caption("Finding players with similar playing styles using machine learning")
        
        with st.spinner("Building similarity model..."):
            try:
                league_stats = get_advanced_stats(season)
                
                if league_stats.empty:
                    st.warning("Could not load league stats for this season.")
                else:
                    nn_model, scaler, feature_cols = build_similarity_model(league_stats)
                    
                    similar_players, selected_player_data, error = find_similar_players(
                        player_name_a, league_stats, nn_model, scaler, feature_cols
                    )
                    
                    if error:
                        st.warning(f"{error}. Player may not have enough games this season.")
                    elif similar_players:
                        st.markdown(f"**Top 5 players most similar to {player_name_a}:**")
                        
                        display_df = pd.DataFrame(similar_players)[['Rank', 'Player', 'Team', 'Similarity', 'USG%', 'TS%', '3P Rate']]
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Radar Chart
                        st.markdown("#### Style Comparison: Selected Player vs Top Match")
                        
                        top_match_idx = similar_players[0]['_idx']
                        top_match_data = league_stats.iloc[top_match_idx]
                        top_match_name = top_match_data['PLAYER_NAME']
                        
                        selected_vals = get_player_style_values(selected_player_data)
                        match_vals = get_player_style_values(top_match_data)
                        
                        doppel_radar = create_doppelganger_radar(
                            player_name_a, top_match_name, similar_players[0]['Similarity'],
                            selected_vals, match_vals
                        )
                        st.plotly_chart(doppel_radar, use_container_width=True)
                        
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
