import json
import pandas as pd
from fastapi import APIRouter, Query, HTTPException

from data.api import get_player_shots, get_league_averages
from data.processing import (
    process_player_data, calculate_player_metrics,
    calculate_zone_stats, add_zone_groups
)
from charts.court import create_shot_chart, create_hexbin_chart
from charts.comparisons import (
    create_radar_comparison, create_zone_frequency_comparison,
    create_single_player_zone_chart
)
from charts.trends import create_historical_efg_chart, create_historical_gsaa_chart
from utils.helpers import get_player_id, get_all_player_names
from config import EFFICIENCY_COLOR_RANGE, ZONE_ORDER, ZONE_MAPPING, HISTORICAL_SEASONS

router = APIRouter()


@router.get("/players/autocomplete")
async def player_autocomplete():
    """All NBA player names for search autocomplete."""
    players = get_all_player_names()
    return {"players": sorted(p['full_name'] for p in players.values())}


@router.get("/player/shots")
async def player_shots(
    name: str = Query(...),
    season: str = Query("2024-25"),
    clutch: bool = Query(False)
):
    player_id, corrected_name, match_msg = get_player_id(name)

    if not player_id:
        raise HTTPException(status_code=404, detail=f"Player '{name}' not found. Check spelling.")

    shots_df, league_avg_df = get_player_shots(player_id, season, clutch)

    if shots_df.empty:
        raise HTTPException(status_code=404, detail=f"No shot data for {corrected_name} in {season}.")

    shots_df = process_player_data(shots_df, league_avg_df)
    metrics = calculate_player_metrics(shots_df)
    shots_df = add_zone_groups(shots_df)

    # Zone stats
    zone_stats = []
    total = len(shots_df)
    for zone in ZONE_ORDER:
        zone_df = shots_df[shots_df['zone_group'] == zone]
        count = len(zone_df)
        makes = int(zone_df['SHOT_MADE_FLAG'].sum())
        zone_stats.append({
            "zone": zone,
            "attempts": count,
            "makes": makes,
            "fg_pct": round(makes / count * 100, 1) if count > 0 else 0,
            "freq_pct": round(count / total * 100, 1) if total > 0 else 0
        })

    # Zone vs league comparison
    league_avgs = get_league_averages(season)
    total_player_attempts = len(shots_df)
    player_zone_stats = shots_df.groupby('SHOT_ZONE_BASIC').agg(
        player_makes=('SHOT_MADE_FLAG', 'sum'),
        player_attempts=('SHOT_MADE_FLAG', 'count')
    ).reset_index()
    player_zone_stats['player_fg_pct'] = player_zone_stats['player_makes'] / player_zone_stats['player_attempts']
    player_zone_stats['player_freq_pct'] = player_zone_stats['player_attempts'] / total_player_attempts
    zone_comparison = player_zone_stats.merge(league_avgs, on='SHOT_ZONE_BASIC', how='left')
    zone_comparison['relative_fg_pct'] = (zone_comparison['player_fg_pct'] - zone_comparison['league_fg_pct']) * 100
    zone_breakdown = zone_comparison[['SHOT_ZONE_BASIC', 'player_attempts', 'player_fg_pct',
                                      'league_fg_pct', 'relative_fg_pct']].fillna(0).to_dict(orient='records')

    # Charts
    scatter_fig = create_shot_chart(shots_df, corrected_name, season, EFFICIENCY_COLOR_RANGE)
    hexbin_fig = create_hexbin_chart(shots_df, corrected_name, season)

    charts = {"scatter": json.loads(scatter_fig.to_json())}
    if hexbin_fig:
        charts["heatmap"] = json.loads(hexbin_fig.to_json())

    # Zone frequency profile chart (non-fatal if it fails)
    try:
        player_stats_df, league_zone_dist = _build_zone_profile(shots_df, league_avg_df)
        zone_profile_fig = create_single_player_zone_chart(player_stats_df, league_zone_dist, corrected_name)
        if zone_profile_fig:
            charts["zone_profile"] = json.loads(zone_profile_fig.to_json())
    except Exception:
        pass

    return {
        "player_name": corrected_name,
        "match_message": match_msg,
        "season": season,
        "clutch": clutch,
        "metrics": {
            "attempts": int(metrics["attempts"]),
            "fg_pct": round(float(metrics["fg_pct"]) * 100, 1),
            "efg": round(float(metrics["efg"]) * 100, 1),
            "gsaa": round(float(metrics["gsaa"]), 1),
            "threes_attempted": int(metrics["threes_attempted"])
        },
        "zone_stats": zone_stats,
        "zone_breakdown": zone_breakdown,
        "charts": charts
    }


@router.get("/player/compare")
async def player_compare(
    name_a: str = Query(...),
    name_b: str = Query(...),
    season: str = Query("2024-25"),
    clutch: bool = Query(False)
):
    id_a, name_a_c, msg_a = get_player_id(name_a)
    id_b, name_b_c, msg_b = get_player_id(name_b)

    if not id_a:
        raise HTTPException(status_code=404, detail=f"Player A '{name_a}' not found.")
    if not id_b:
        raise HTTPException(status_code=404, detail=f"Player B '{name_b}' not found.")

    shots_a, league_avg = get_player_shots(id_a, season, clutch)
    shots_b, _ = get_player_shots(id_b, season, clutch)

    if shots_a.empty:
        raise HTTPException(status_code=404, detail=f"No shot data for {name_a_c} in {season}.")
    if shots_b.empty:
        raise HTTPException(status_code=404, detail=f"No shot data for {name_b_c} in {season}.")

    shots_a = process_player_data(shots_a, league_avg)
    shots_b = process_player_data(shots_b, league_avg)
    shots_a = add_zone_groups(shots_a)
    shots_b = add_zone_groups(shots_b)
    metrics_a = calculate_player_metrics(shots_a)
    metrics_b = calculate_player_metrics(shots_b)

    rim_rate_a = len(shots_a[shots_a['zone_group'] == 'Rim']) / len(shots_a) * 100 if len(shots_a) > 0 else 0
    rim_rate_b = len(shots_b[shots_b['zone_group'] == 'Rim']) / len(shots_b) * 100 if len(shots_b) > 0 else 0
    three_rate_a = len(shots_a[shots_a['zone_group'] == '3-Point']) / len(shots_a) * 100 if len(shots_a) > 0 else 0
    three_rate_b = len(shots_b[shots_b['zone_group'] == '3-Point']) / len(shots_b) * 100 if len(shots_b) > 0 else 0

    scatter_a = create_shot_chart(shots_a, name_a_c, season, EFFICIENCY_COLOR_RANGE)
    scatter_b = create_shot_chart(shots_b, name_b_c, season, EFFICIENCY_COLOR_RANGE)
    radar = create_radar_comparison(
        name_a_c, name_b_c, metrics_a, metrics_b,
        rim_rate_a, rim_rate_b, three_rate_a, three_rate_b
    )
    stats_a = calculate_zone_stats(shots_a, name_a_c)
    stats_b = calculate_zone_stats(shots_b, name_b_c)
    zone_freq = create_zone_frequency_comparison(stats_a, stats_b, name_a_c, name_b_c)

    charts = {
        "scatter_a": json.loads(scatter_a.to_json()),
        "scatter_b": json.loads(scatter_b.to_json()),
        "radar": json.loads(radar.to_json()),
        "zone_freq": json.loads(zone_freq.to_json())
    }

    # Historical trends
    try:
        hist_a = _get_historical_metrics(id_a, HISTORICAL_SEASONS, clutch)
        hist_b = _get_historical_metrics(id_b, HISTORICAL_SEASONS, clutch)
        if not hist_a.empty and not hist_b.empty:
            charts["efg_trend"] = json.loads(create_historical_efg_chart(hist_a, hist_b, name_a_c, name_b_c).to_json())
            charts["gsaa_trend"] = json.loads(create_historical_gsaa_chart(hist_a, hist_b, name_a_c, name_b_c).to_json())
    except Exception:
        pass

    def fmt_metrics(m):
        return {
            "attempts": int(m["attempts"]),
            "fg_pct": round(float(m["fg_pct"]) * 100, 1),
            "efg": round(float(m["efg"]) * 100, 1),
            "gsaa": round(float(m["gsaa"]), 1),
            "threes_attempted": int(m["threes_attempted"])
        }

    return {
        "player_a": name_a_c,
        "player_b": name_b_c,
        "match_message_a": msg_a,
        "match_message_b": msg_b,
        "season": season,
        "clutch": clutch,
        "metrics_a": fmt_metrics(metrics_a),
        "metrics_b": fmt_metrics(metrics_b),
        "charts": charts
    }


def _get_historical_metrics(player_id, seasons, clutch):
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
        except Exception:
            pass
    return pd.DataFrame(history)


def _build_zone_profile(shots_df, league_avg_df):
    player_stats_list = []
    for zone in ZONE_ORDER:
        zone_df = shots_df[shots_df['zone_group'] == zone]
        count = len(zone_df)
        makes = zone_df['SHOT_MADE_FLAG'].sum()
        player_stats_list.append({
            'zone': zone,
            'count': count,
            'makes': makes,
            'fg_pct': (makes / count * 100) if count > 0 else 0,
            'freq_pct': (count / len(shots_df) * 100) if len(shots_df) > 0 else 0
        })

    # frames[1] from ShotChartDetail has FGA, FGM, FG_PCT columns (not league_freq_pct/league_fg_pct)
    league_avg_copy = league_avg_df.copy()
    league_avg_copy['zone_group'] = league_avg_copy['SHOT_ZONE_BASIC'].map(ZONE_MAPPING)
    league_zone_dist = league_avg_copy.groupby('zone_group').agg(
        league_fga=('FGA', 'sum'),
        league_fgm=('FGM', 'sum')
    ).reset_index()
    league_zone_dist['league_fg_pct'] = (
        league_zone_dist['league_fgm'] / league_zone_dist['league_fga']
    ).fillna(0)
    total = league_zone_dist['league_fga'].sum()
    league_zone_dist['league_freq_pct'] = (
        league_zone_dist['league_fga'] / total * 100 if total > 0 else 0
    )
    league_zone_dist = league_zone_dist[league_zone_dist['zone_group'].isin(ZONE_ORDER)]

    return pd.DataFrame(player_stats_list), league_zone_dist
