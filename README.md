# NBA Player DNA: Spatial Efficiency Engine

A comprehensive NBA analytics platform built with Streamlit that provides advanced shooting analysis, player comparisons, MVP tracking, and machine learning-powered player similarity matching.

## Overview

NBA Player DNA is an interactive web application that transforms raw NBA statistics into actionable insights through spatial efficiency analysis, scarcity-weighted valuation, and comparative performance metrics. The platform combines shot-level data visualization with advanced analytics to help analysts, coaches, and fans understand player performance at a granular level.

## Key Features

### 1. League Leaders by Position
Analyze top performers across three key dimensions:
- **Scoring Impact**: Players ranked by usage load multiplied by efficiency (USG% × TS%)
- **Playmaking**: Facilitators ranked by assists per 100 possessions, weighted by assist-to-turnover ratio
- **Two-Way Impact**: Elite performers ranked by net rating (offensive rating minus defensive rating)

Features:
- Position-specific breakdowns (Guards, Forwards, Centers)
- Configurable display options (Top 5, Top 10, Top 20)
- Advanced visualization suite including scoring efficiency scatter plots, playmaking creation charts, and two-way quadrant analysis

### 2. MVP Ladder - DNA Production Index
A proprietary MVP ranking system using scarcity-weighted statistics combined with team success metrics.

**Algorithm Overview:**
- Calculates scarcity weights for each statistical category based on league-wide rarity
- Applies impact modifiers to balance position bias (dampens blocks, rewards assists)
- Multiplies raw production value by square root of team win percentage
- Produces a comprehensive MVP score that balances individual excellence with team success

**Features:**
- Configurable rankings (Top 5, Top 10, Top 20)
- Detailed breakdowns showing value contributions by category (scoring, playmaking, rebounding, defense)
- Win percentage and team record integration
- Visual analytics including value breakdown charts and production vs. winning scatter plots

### 3. Shot Chart Analysis
Interactive spatial efficiency visualizations with two display modes:

**Scatter View:**
- Individual shot attempts plotted by court location
- Color-coded by efficiency relative to league average
- Filterable by clutch situations (last 5 minutes, within 5 points)

**Heatmap View:**
- Zone-aggregated shooting performance
- Hexagonal binning for pattern recognition
- Hot/cold zone identification

### 4. Player Comparison Engine
Side-by-side performance analysis with:
- Dual shot charts with synchronized filtering
- Head-to-head metrics comparison with delta analysis
- Radar chart profile overlays
- Shot distribution and accuracy breakdowns by zone
- Historical season trend analysis (eFG%, GSAA)

### 5. Doppelganger Finder
Machine learning-powered player similarity engine using K-Nearest Neighbors algorithm.

**Features Analyzed:**
- Usage Rate (USG%)
- True Shooting Percentage (TS%)
- Assist Rate (AST%)
- Rebound Rate (REB%)
- Team Pace
- Three-Point Attempt Rate (3P_AR)

All features are normalized using StandardScaler for balanced comparison across different statistical ranges.

### 6. Advanced Metrics
- **Effective Field Goal Percentage (eFG%)**: Weighted shooting percentage accounting for three-point value
- **Goals Saved Above Average (GSAA)**: Points generated above league average expectation based on shot location
- **Relative Efficiency**: Zone-specific performance compared to league benchmarks
- **Position-Adjusted Statistics**: Normalized metrics for fair cross-position comparison

## Installation

### Requirements
- Python 3.10 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/designed7000/Euro_stepper_analyst.git
cd Euro_stepper_analyst
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Launch the application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Usage Guide

### Getting Started
When you first launch the application, you'll see a welcome screen. Select any analysis mode from the sidebar to begin.

### League Leaders Analysis
1. Check "Show Leaders by Position" in the sidebar
2. Select your desired season
3. Navigate between tabs: Scoring Impact, Playmaking, Two-Way Impact
4. Use the dropdown selector to adjust the number of leaders displayed (Top 5/10/20)
5. Scroll down to view advanced analytics visualizations

### MVP Ladder
1. Check "Show MVP Ladder" in the sidebar
2. Select your desired season
3. Use the dropdown to view Top 5, Top 10, or Top 20 candidates
4. Analyze the value breakdown charts to understand each player's contribution profile
5. Expand the methodology section to learn about the DNA Production Index algorithm

### Player Analysis
1. Check "Analyze Player" in the sidebar
2. Enter a player name (fuzzy matching supported - handles typos and partial names)
3. Select a season
4. Optional: Enable "Compare Players" for side-by-side analysis
5. Optional: Enable "Clutch Time Only" to filter to high-leverage situations
6. Optional: Enable "Find Similar Players" to activate the ML similarity engine
7. Toggle between Scatter and Heatmap chart views

## Data Sources

All data is sourced from official NBA statistics via the `nba_api` Python package:
- **Shot Chart Detail**: Individual shot locations, outcomes, and game context
- **League Dashboard Player Stats**: Base statistics and advanced metrics
- **League Standings**: Team records and win percentages for MVP calculations
- **Per 100 Possessions**: Rate-based statistics for normalized comparisons

## Technical Architecture

### Stack
- **Frontend Framework**: Streamlit
- **Data Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn (StandardScaler, NearestNeighbors)
- **Data API**: nba_api

### Project Structure
```
Euro_stepper_analyst/
├── app.py                    # Main application entry point
├── config.py                 # Configuration and constants
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
│
├── data/                     # Data access layer
│   ├── api.py                # NBA API calls with caching
│   └── processing.py         # Data transformations and aggregations
│
├── charts/                   # Visualization layer
│   ├── court.py              # Shot charts (scatter, hexbin)
│   ├── comparisons.py        # Radar charts, zone comparisons
│   ├── trends.py             # Historical trends, leader charts
│   └── awards.py             # MVP ladder visualizations
│
├── analysis/                 # Analytics engines
│   ├── similarity.py         # ML player similarity matching
│   └── awards.py             # MVP ladder calculations (DNA Production Index)
│
└── utils/                    # Helper functions
    └── helpers.py            # Player name matching, data normalization
```

## Key Metrics Reference

| Metric | Formula | Description |
|--------|---------|-------------|
| **FG%** | FGM / FGA | Field Goal Percentage |
| **eFG%** | (FGM + 0.5 × 3PM) / FGA | Effective FG% (weights three-pointers) |
| **GSAA** | Actual Points - Expected Points | Points generated above league average by shot location |
| **Relative Efficiency** | Player FG% - League Avg FG% (same zone) | Zone-specific performance differential |
| **USG%** | Player Possessions / Team Possessions | Percentage of team plays used by player |
| **TS%** | PTS / (2 × (FGA + 0.44 × FTA)) | True Shooting Percentage |
| **Net Rating** | Offensive Rating - Defensive Rating | Two-way impact metric |
| **MVP Score** | Raw Value × √(Win %) | DNA Production Index MVP ranking |

## DNA Production Index Methodology

The MVP ladder uses a scarcity-weighted approach to value production:

1. **Scarcity Weight Calculation**: `Weight = (Total League Points) / (Total League Stat)`
2. **Impact Modifiers**: Applied to balance position bias
   - Assists: ×1.5 (reward creation)
   - Rebounds: ×0.7 (dampen raw volume)
   - Blocks: ×0.6 (prevent center dominance)
   - Steals: ×1.0 (balanced)
3. **Raw Production**: Sum of weighted stat contributions minus turnover penalty
4. **Team Success Modifier**: `MVP Score = Raw Value × √(Team Win Percentage)`

This approach rewards individual excellence while accounting for team success, using a square root function to avoid over-penalizing players on mid-tier teams.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a Pull Request

Please ensure your code follows existing style conventions and includes appropriate documentation.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- NBA API development team for the `nba_api` package
- Streamlit team for the web application framework
- NBA for providing comprehensive basketball statistics
- Open source community for various supporting libraries

## Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the repository owner.

---

**Note**: This application is for educational and analytical purposes. All NBA data is property of the NBA and used in accordance with their terms of service.