import json
from fastapi import APIRouter, Query, HTTPException

from data.api import get_advanced_stats
from analysis.similarity import build_similarity_model, find_similar_players, get_player_style_values
from charts.similarity import (
    create_similarity_bar_chart, create_stat_comparison_bars,
    create_style_profile_chart, create_percentile_chart
)
from charts.comparisons import create_doppelganger_radar

router = APIRouter()


@router.get("/similarity")
async def player_similarity(
    name: str = Query(...),
    season: str = Query("2024-25")
):
    league_stats = get_advanced_stats(season)

    if league_stats.empty:
        raise HTTPException(status_code=404, detail="No league stats available for this season")

    nn_model, scaler, feature_cols = build_similarity_model(league_stats)
    similar_players, selected_data, error = find_similar_players(name, league_stats, nn_model, scaler, feature_cols)

    if error:
        raise HTTPException(status_code=404, detail=error)

    top_match_idx = similar_players[0]['_idx']
    top_match_data = league_stats.iloc[top_match_idx]
    top_match_name = top_match_data['PLAYER_NAME']
    top_matches_data = [league_stats.iloc[p['_idx']] for p in similar_players[:3]]

    charts = {}
    try:
        charts["similarity_bar"] = json.loads(create_similarity_bar_chart(similar_players, name).to_json())
    except Exception:
        pass
    try:
        charts["percentile"] = json.loads(create_percentile_chart(selected_data, league_stats, name).to_json())
    except Exception:
        pass
    try:
        charts["stat_comparison"] = json.loads(create_stat_comparison_bars(selected_data, top_matches_data, league_stats).to_json())
    except Exception:
        pass
    try:
        charts["style_profile"] = json.loads(create_style_profile_chart(selected_data, top_matches_data, name).to_json())
    except Exception:
        pass
    try:
        selected_vals = get_player_style_values(selected_data)
        match_vals = get_player_style_values(top_match_data)
        charts["radar"] = json.loads(create_doppelganger_radar(
            name, top_match_name, similar_players[0]['Similarity'],
            selected_vals, match_vals
        ).to_json())
    except Exception:
        pass

    # Strip internal fields before returning
    clean_similar = [{k: v for k, v in p.items() if not k.startswith('_')} for p in similar_players]

    return {
        "player_name": name,
        "season": season,
        "top_match": top_match_name,
        "similar_players": clean_similar,
        "charts": charts
    }
