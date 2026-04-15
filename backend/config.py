"""
Configuration settings and constants for NBA Shot DNA app.
"""

# Available seasons for selection
SEASONS = ["2025-26", "2024-25", "2023-24", "2022-23", "2021-22", "2015-16"]

# Historical seasons for trend analysis
HISTORICAL_SEASONS = ["2024-25", "2023-24", "2022-23", "2021-22"]

# Chart color schemes
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
