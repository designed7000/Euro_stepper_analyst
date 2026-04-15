from fastapi import APIRouter, Query
from data.api import get_league_leaders, get_standings
from utils.home_page import generate_quiz_question, get_fun_fact
import plotly.graph_objects as go
import json

router = APIRouter()


@router.get("/home")
async def home_page(season: str = Query("2024-25")):
    try:
        leaders_df = get_league_leaders(season)
        standings_df = get_standings(season)

        quiz = generate_quiz_question(leaders_df, season)
        fun_fact = get_fun_fact(leaders_df, season)

        # Key leaders summary
        leaders_data = []
        for col, label, fmt in [
            ('PTS', 'Points', '{:.1f}'),
            ('AST', 'Assists', '{:.1f}'),
            ('REB', 'Rebounds', '{:.1f}'),
            ('TS_PCT', 'True Shooting %', '{:.1%}'),
        ]:
            if col in leaders_df.columns and not leaders_df.empty:
                top = leaders_df.nlargest(1, col).iloc[0]
                leaders_data.append({
                    "metric": label,
                    "player": top['PLAYER_NAME'],
                    "team": top.get('TEAM_ABBREVIATION', ''),
                    "value": fmt.format(top[col])
                })

        # Season progress
        season_progress = None
        if not standings_df.empty and 'WINS' in standings_df.columns and 'LOSSES' in standings_df.columns:
            avg_games = (standings_df['WINS'] + standings_df['LOSSES']).mean()
            season_progress = round((avg_games / 82) * 100, 1)

        # Volume vs Efficiency scatter (top 30 by usage)
        charts = {}
        if not leaders_df.empty and 'TS_PCT' in leaders_df.columns and 'USG_PCT' in leaders_df.columns:
            try:
                plot_data = leaders_df[['PLAYER_NAME', 'TS_PCT', 'USG_PCT', 'PTS']].dropna().nlargest(30, 'USG_PCT')
                avg_ts = leaders_df['TS_PCT'].mean()
                colors = ['#00D26A' if ts >= avg_ts else '#FF6B6B' for ts in plot_data['TS_PCT']]
                fig = go.Figure()
                fig.add_hline(
                    y=avg_ts, line_dash='dash', line_color='rgba(255,255,255,0.4)',
                    annotation_text=f'League Avg: {avg_ts:.1%}',
                    annotation_position='top left',
                    annotation_font_size=12, annotation_font_color='#aaa'
                )
                fig.add_trace(go.Scatter(
                    x=plot_data['USG_PCT'], y=plot_data['TS_PCT'],
                    mode='markers+text',
                    marker=dict(size=18, color=colors, line=dict(color='white', width=1.5), opacity=0.85),
                    text=plot_data['PLAYER_NAME'].apply(lambda x: x.split()[-1][:7]),
                    textposition='top center',
                    textfont=dict(size=11, color='#ddd'),
                    hovertemplate='<b>%{customdata[0]}</b><br>Usage: %{x:.1%}<br>TS%: %{y:.1%}<br>PPG: %{customdata[1]:.1f}<extra></extra>',
                    customdata=list(zip(plot_data['PLAYER_NAME'], plot_data['PTS'])),
                    showlegend=False
                ))
                fig.update_layout(
                    title=dict(text='<b>Volume vs Efficiency</b><br><sup>🟢 Above Avg TS%  🔴 Below Avg TS%</sup>', font=dict(size=18)),
                    xaxis=dict(title='Usage Rate %', tickformat='.0%', gridcolor='rgba(255,255,255,0.08)'),
                    yaxis=dict(title='True Shooting %', tickformat='.0%', gridcolor='rgba(255,255,255,0.08)'),
                    height=500, margin=dict(l=70, r=30, t=80, b=60),
                    template='plotly_dark', hovermode='closest'
                )
                charts['volume_efficiency'] = json.loads(fig.to_json())
            except Exception:
                pass

        # Win distribution histogram
        if not standings_df.empty and 'WINS' in standings_df.columns:
            try:
                import pandas as pd
                wins = standings_df['WINS'].astype(float)
                max_wins = int(wins.max())
                bins = list(range(0, max_wins + 6, 5))
                labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
                dist = pd.cut(wins, bins=bins, labels=labels, right=False).value_counts().sort_index()
                n_bars = len(dist)
                color_stops = [(91, 141, 190), (255, 158, 100), (232, 132, 92)]
                colors = []
                for i in range(n_bars):
                    ratio = i / max(n_bars - 1, 1)
                    if ratio < 0.5:
                        progress = ratio * 2
                        s, e = color_stops[0], color_stops[1]
                    else:
                        progress = (ratio - 0.5) * 2
                        s, e = color_stops[1], color_stops[2]
                    r = int(s[0] + (e[0] - s[0]) * progress)
                    g = int(s[1] + (e[1] - s[1]) * progress)
                    b = int(s[2] + (e[2] - s[2]) * progress)
                    colors.append(f'rgb({r},{g},{b})')
                fig = go.Figure(data=[go.Bar(
                    x=dist.index.astype(str), y=dist.values,
                    marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.2)', width=1)),
                    text=dist.values, textposition='outside',
                    textfont=dict(size=11, color='#fff'),
                    hovertemplate='<b>%{x}</b><br>Teams: %{y}<extra></extra>',
                    width=0.5
                )])
                fig.update_layout(
                    title=dict(text='<b>Win Distribution</b><br><sup>How teams are spread across win ranges</sup>', font=dict(size=16)),
                    xaxis_title='Win Range', yaxis_title='Number of Teams',
                    yaxis=dict(range=[0, max(dist.values) * 1.15]),
                    height=380, margin=dict(l=70, r=30, t=80, b=60),
                    showlegend=False, template='plotly_dark', hovermode='x unified'
                )
                charts['win_distribution'] = json.loads(fig.to_json())
            except Exception:
                pass

        return {
            "season": season,
            "quiz": quiz,
            "fun_fact": fun_fact,
            "leaders": leaders_data,
            "season_progress": season_progress,
            "charts": charts
        }
    except Exception as e:
        return {"error": str(e)}
