from fastapi import APIRouter, Query
from data.api import get_league_leaders, get_standings
from analysis.awards import AwardTracker
from charts.awards import create_mvp_value_breakdown_chart, create_winning_vs_stats_scatter
import json

router = APIRouter()


@router.get("/mvp")
async def mvp_ladder(season: str = Query("2024-25"), top_n: int = Query(20)):
    try:
        leaders_df = get_league_leaders(season)
        standings_df = get_standings(season)

        if leaders_df.empty or standings_df.empty:
            return {"error": "No data available for this season"}

        tracker = AwardTracker(leaders_df, standings_df)
        mvp_ladder_df = tracker.calculate_mvp_ladder(top_n=top_n)

        if mvp_ladder_df.empty:
            return {"error": "Could not calculate MVP ladder"}

        display_cols = ['Rank', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'MVP_SCORE',
                        'RAW_VALUE', 'RECORD', 'WIN_PCT', 'PTS', 'REB', 'AST']
        available_cols = [c for c in display_cols if c in mvp_ladder_df.columns]
        ladder_data = mvp_ladder_df[available_cols].to_dict(orient="records")

        # Format numeric fields for JSON serialization
        for row in ladder_data:
            for k, v in row.items():
                try:
                    import math
                    if isinstance(v, float) and math.isnan(v):
                        row[k] = None
                except Exception:
                    pass

        charts = {}
        try:
            fig = create_mvp_value_breakdown_chart(mvp_ladder_df.head(10))
            if fig:
                charts["breakdown"] = json.loads(fig.to_json())
        except Exception:
            pass
        try:
            fig = create_winning_vs_stats_scatter(mvp_ladder_df.head(15))
            if fig:
                charts["scatter"] = json.loads(fig.to_json())
        except Exception:
            pass

        return {
            "season": season,
            "ladder": ladder_data,
            "charts": charts
        }
    except Exception as e:
        return {"error": str(e)}
