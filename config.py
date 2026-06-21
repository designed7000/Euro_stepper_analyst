"""
Configuration settings and constants for NBA Shot DNA app.
"""

import os
from datetime import date
from pathlib import Path


# --- SEASON DERIVATION ---------------------------------------------------

def current_season(today=None):
    """Return the current NBA season string (e.g. '2025-26').

    NBA seasons start in October. From October through December the season
    is (year, year+1); from January through September it is (year-1, year).

    Args:
        today: Optional date to evaluate against (defaults to today).

    Returns:
        str: Season in 'YYYY-YY' format.
    """
    today = today or date.today()
    start_year = today.year if today.month >= 10 else today.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_string(start_year):
    """Build a 'YYYY-YY' season string from its starting year."""
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def recent_seasons(n=6, today=None):
    """Return the current season plus the previous n-1 seasons, newest first."""
    start = int(current_season(today).split("-")[0])
    return [season_string(start - i) for i in range(n)]


def historical_seasons(n=4, today=None):
    """Return the n completed seasons before the current one, newest first."""
    start = int(current_season(today).split("-")[0])
    return [season_string(start - i) for i in range(1, n + 1)]


# Available seasons for selection (current + previous 5)
SEASONS = recent_seasons(6)

# Historical seasons for trend analysis (4 completed seasons before current)
HISTORICAL_SEASONS = historical_seasons(4)


# --- SNAPSHOT STORE ------------------------------------------------------

# Where pre-fetched snapshots live. Override with NBA_STORE_DIR (used by tests).
STORE_DIR = os.environ.get(
    "NBA_STORE_DIR",
    str(Path(__file__).resolve().parent / "data_store"),
)


# --- CHART COLOR SCHEMES -------------------------------------------------

POSITION_COLORS = {
    'Guard': '#00CED1',
    'Forward': '#FF6B6B',
    'Center': '#98D8C8'
}

PLAYER_COLORS = {
    'player_a': '#1f77b4',
    'player_b': '#ff7f0e',
    'league_avg': '#7f7f7f'
}

# Shot zone mapping
ZONE_MAPPING = {
    'Restricted Area': 'Rim',
    'In The Paint (Non-RA)': 'Rim',
    'Mid-Range': 'Mid-Range',
    'Left Corner 3': '3-Point',
    'Right Corner 3': '3-Point',
    'Above the Break 3': '3-Point',
    'Backcourt': 'Other'
}

# Zone order for charts
ZONE_ORDER = ['Rim', 'Mid-Range', '3-Point']

# Normalization ranges for radar charts (typical NBA ranges)
RADAR_RANGES = {
    'attempts': (0, 1500),
    'fg_pct': (35, 55),
    'efg': (45, 65),
    'three_rate': (20, 60),
    'rim_rate': (15, 50),
    'gsaa_per_100': (-10, 15)
}

# Similarity model normalization ranges
SIMILARITY_RANGES = {
    'usg': (15, 35),
    'ts': (50, 70),
    'ast': (5, 35),
    'reb': (3, 18),
    'pace': (95, 105),
    '3p_rate': (15, 55)
}

# Chart settings
CHART_HEIGHT = {
    'single_player': 700,
    'comparison': 500,
    'radar': 550,
    'bar': 420,
    'trend': 300,
    'leaders': 380,
    'quadrant': 420,
    'top_scorers': 400
}

# Color ranges for efficiency charts
EFFICIENCY_COLOR_RANGE = (-0.15, 0.15)

# Minimum thresholds
MIN_GAMES = 15
MIN_MINUTES = 10
MIN_MINUTES_LEADERS = 20
MIN_MINUTES_TWOWAY = 25
MIN_USAGE_LEADERS = 0.15
MIN_ASSISTS_PLAYMAKING = 3
