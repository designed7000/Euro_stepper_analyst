# NBA Shot DNA: Spatial Efficiency Engine

A Streamlit-powered NBA analytics dashboard that visualizes player shooting patterns, compares performance metrics, and uses machine learning to find statistically similar players.

## Features

### League Leaders by Position
- **Scoring Impact**: Top scorers ranked by USG% × TS% (usage load × efficiency)
- **Playmaking**: Best facilitators ranked by AST/100 possessions weighted by AST/TO ratio
- **Two-Way Impact**: Elite players ranked by Net Rating (Offense - Defense)
- **Position Breakdown**: Guards, Forwards, and Centers displayed in separate columns
- **Flexible Display**: Toggle between Top 5, Top 10, and Top 20 players per position

### Advanced Analytics Visualizations
- **Scoring Load vs Efficiency**: Scatter plot of USG% vs TS% with league average reference line
- **Playmaking Creation vs Security**: AST per 100 possessions vs TOV per 100 (inverted axis for elite zone in top-right)
- **Two-Way Quadrant Chart**: Offensive Rating vs Defensive Rating with quadrant zones
- **Top Scorers Bar Chart**: Horizontal bar chart of top 10 PPG leaders by position

### Shot Chart Visualization
- **Relative Efficiency Maps**: Interactive shot charts showing each shot attempt colored by performance vs league average
- **Heatmap View**: Aggregated zone-based visualization showing hot/cold shooting areas
- **Toggle Between Views**: Switch between individual shot scatter plots and zone heatmaps

### Player Comparison Mode
- Side-by-side shot chart comparison
- Head-to-head metrics table with delta analysis
- Radar chart player profile overlay
- Shot distribution comparison with accuracy metrics
- Historical season trend analysis (eFG% and GSAA)

### Advanced Metrics
- **Effective Field Goal Percentage (eFG%)**: Accounts for 3-point shot value
- **Points Added (GSAA)**: Goals Saved Above Average - points generated above league expectation
- **Relative Efficiency**: Per-shot performance compared to league average from same location
- **Zone Breakdown**: Detailed FG% by court zone vs league benchmarks

### Doppelganger Finder (ML-Powered Player Similarity)
- Uses K-Nearest Neighbors algorithm to find players with similar playing styles
- Features analyzed: Usage Rate, True Shooting %, Assist Rate, Rebound Rate, Pace, 3PT Attempt Rate
- Normalized feature scaling for balanced comparison
- Radar chart visualization comparing selected player vs top match

### Additional Features
- **Clutch Time Filter**: Analyze performance in last 5 minutes with score within 5 points
- **Fuzzy Player Search**: Handles typos and partial name matches
- **Multi-Season Support**: Compare across different NBA seasons
- **Raw Data Inspector**: Access underlying shot-level data

## Installation

### Prerequisites
- Python 3.10+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nba-shot-dna.git
cd nba-shot-dna
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit pandas numpy plotly nba-api scikit-learn
```

4. Run the application:
```bash
streamlit run test.py
```

## Usage

1. **League Leaders** (enabled by default):
   - View top performers across Guards, Forwards, and Centers
   - Switch between Scoring Impact, Playmaking, and Two-Way Impact tabs
   - Use the dropdown below tables to show Top 5, Top 10, or Top 20 players
   - Explore advanced analytics charts below the tables

2. **Player Analysis** (un-check League Leaders to focus):
   - Enter a player name in the sidebar (fuzzy matching supported)
   - Select a season
   - Toggle options:
     - **Compare Players**: Enable side-by-side comparison mode
     - **Clutch Time Only**: Filter to clutch situations
     - **Find Similar Players**: Activate the ML similarity engine
   - Use the chart style toggle to switch between Scatter and Heatmap views

## Data Sources

- **NBA API**: All data sourced from official NBA statistics via the `nba_api` Python package
- **Shot Chart Detail**: Individual shot locations and outcomes
- **League Dash Player Stats**: Base stats (PTS, AST, REB, etc.) and Advanced metrics (USG%, TS%, OFF/DEF Rating, AST/TO)
- **Per 100 Possessions Stats**: Rate-based metrics for fair cross-player comparison

## Tech Stack

- **Frontend**: Streamlit
- **Visualization**: Plotly (interactive charts)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: scikit-learn (StandardScaler, NearestNeighbors)
- **Data Source**: nba_api

## Project Structure

```
nba-shot-dna/
├── test.py              # Main Streamlit application
├── logo.png             # Application logo
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── cache/               # Cached API responses
```

## Metrics Explained

| Metric | Description |
|--------|-------------|
| FG% | Field Goal Percentage (makes / attempts) |
| eFG% | Effective FG% = (FGM + 0.5 * 3PM) / FGA |
| GSAA | Points scored minus expected points based on shot location |
| Relative Efficiency | Player FG% minus league average FG% from same zone |
| Usage Rate | Percentage of team possessions used by player |
| True Shooting % | Points / (2 * (FGA + 0.44 * FTA)) |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- NBA API maintainers for the `nba_api` package
- Streamlit team for the web framework
- NBA for the underlying statistics