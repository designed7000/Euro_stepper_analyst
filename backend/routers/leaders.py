from fastapi import APIRouter, Query
from data.api import get_league_leaders
from data.processing import get_top_players_by_position_smart
from charts.trends import (
    create_scoring_efficiency_chart, create_playmaking_chart,
    create_twoway_quadrant_chart, create_top_scorers_bar
)
import json

router = APIRouter()


@router.get("/leaders")
async def league_leaders(season: str = Query("2024-25")):
    try:
        leaders_df = get_league_leaders(season)

        if leaders_df.empty:
            return {"error": "No data available for this season"}

        # Top players by position for each category
        result = {}
        for category in ["scoring", "playmaking", "impact"]:
            top = get_top_players_by_position_smart(leaders_df, category, n=20)
            result[category] = {}
            for position, df in top.items():
                result[category][position] = df.to_dict(orient="records") if not df.empty else []

        # Charts
        charts = {}
        for name, fn in [
            ("scoring", create_scoring_efficiency_chart),
            ("playmaking", create_playmaking_chart),
            ("twoway_quadrant", create_twoway_quadrant_chart),
            ("top_scorers", create_top_scorers_bar),
        ]:
            try:
                fig = fn(leaders_df)
                if fig:
                    charts[name] = json.loads(fig.to_json())
            except Exception:
                pass

        return {
            "season": season,
            "leaders": result,
            "charts": charts
        }
    except Exception as e:
        return {"error": str(e)}
