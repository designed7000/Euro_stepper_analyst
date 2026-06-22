<p align="center">
  <img src="images/logo.png" alt="NBA Player DNA Logo" width="140"/>
</p>

<h1 align="center">NBA Player DNA</h1>
<h3 align="center">Spatial Efficiency Engine</h3>

<p align="center">
  <em>Advanced NBA analytics platform for shooting analysis, player comparisons, MVP tracking, and ML-powered similarity matching</em>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/">
    <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  </a>
  <a href="https://streamlit.io/">
    <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  </a>
  <a href="https://plotly.com/">
    <img src="https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white" alt="Plotly"/>
  </a>
  <a href="https://pandas.pydata.org/">
    <img src="https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/>
  </a>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#installation">Installation</a> •
  <a href="#methodology">Methodology</a> •
  <a href="#contributing">Contributing</a>
</p>

---

## Features

### MVP Ladder - DNA Production Index
A proprietary MVP ranking system using **scarcity-weighted statistics** combined with team success metrics.

- Calculates scarcity weights based on league-wide rarity
- Applies impact modifiers to balance position bias
- Multiplies production by √(win percentage) for team success
- Configurable Top 5/10/20 rankings with visual breakdowns

### League Leaders by Position
Analyze top performers across three dimensions:

| Category | Metric | Description |
|----------|--------|-------------|
| **Scoring Impact** | USG% × TS% | Usage load multiplied by efficiency |
| **Playmaking** | AST/100 × AST:TO | Creation rate weighted by turnover ratio |
| **Two-Way Impact** | Net Rating | Offensive rating minus defensive rating |

### Shot Chart Analysis
Interactive spatial efficiency visualizations:

- **Scatter View**: Individual shots with color-coded efficiency
- **Heatmap View**: Zone-aggregated hexagonal binning
- **Clutch Filter**: Last 5 minutes, within 5 points

### Player Comparison Engine
Side-by-side performance analysis:

- Dual synchronized shot charts
- Head-to-head metrics with delta analysis
- Radar chart profile overlays
- Historical season trends (eFG%, GSAA)

### Statistical Doppelgangers
ML-powered player similarity using **K-Nearest Neighbors**:

- Analyzes 6 style dimensions (USG%, TS%, AST%, REB%, Pace, 3P Rate)
- Visual similarity rankings with match scores
- League percentile comparisons
- Style profile dot charts

---

## Installation

### Prerequisites
- Python 3.13+
- [uv](https://docs.astral.sh/uv/) for environment management

> **Note:** Do not place this repo inside an iCloud-synced folder (e.g. `~/Desktop`
> or `~/Documents` with "Desktop & Documents" sync on). Cloud-placeholder files
> break native extension loading in the virtualenv. Use a path like `~/code/`.

### Quick Start

```bash
# Clone the repository
git clone https://github.com/designed7000/Euro_stepper_analyst.git
cd Euro_stepper_analyst

# Install dependencies (creates .venv)
uv sync

# Fetch a season's data into the local snapshot store (required before first run)
uv run python -m data.refresh --season 2024-25

# Launch the app
uv run streamlit run app.py
```

The app will open at `http://localhost:8501`

### Data refresh (snapshot-first)

The app **never calls the NBA API at request time** — it reads pre-fetched
snapshots from a local `data_store/` directory. This keeps the UI fast and
decoupled from the NBA API's rate limits and cloud-IP blocking.

```bash
# Refresh the current season (default)
uv run python -m data.refresh

# Refresh a specific season
uv run python -m data.refresh --season 2023-24

# Refresh a subset of datasets
uv run python -m data.refresh --datasets leaders standings
```

A season only shows data in the app **after** it has been refreshed. League-wide
data (leaders, standings, advanced stats, zone averages, MVP) is served entirely
from snapshots; per-player shot charts are fetched once on first view and then
cached. Run the refresh on a schedule (cron / launchd / CI) whenever new games
are played. The `data_store/` directory is gitignored.

### Running tests

```bash
uv run pytest -q
```

## 📖 Usage

### Getting Started
Launch the app and select any mode from the sidebar:

| Mode | Checkbox | Description |
|------|----------|-------------|
| League Leaders | Show Leaders by Position | Top performers by category |
| MVP Ladder | Show MVP Ladder | DNA Production Index rankings |
| Player Analysis | Analyze Player | Shot charts & metrics |
| Doppelgangers | Find Similar Players | ML similarity matching |

### Player Analysis Tips
- **Fuzzy matching**: Handles typos (e.g., "Lebron" → "LeBron James")
- **Compare mode**: Enable for side-by-side analysis
- **Clutch filter**: Focus on high-leverage situations
- **Chart toggle**: Switch between scatter and heatmap views

---

## Methodology

### DNA Production Index (MVP Score)

```
MVP Score = Raw Value × √(Win Percentage)
```

**Step 1: Scarcity Weights**
```
Weight = Total League Points / Total League Stat
```

**Step 2: Impact Modifiers**
| Stat | Modifier | Rationale |
|------|----------|-----------|
| Assists | ×1.5 | Reward creation |
| Rebounds | ×0.7 | Dampen raw volume |
| Blocks | ×0.6 | Prevent center bias |
| Steals | ×1.0 | Balanced |

**Step 3: Team Success**
- Uses √(Win%) to avoid over-penalizing players on mid-tier teams
- Rewards winning while maintaining individual excellence weight

### Key Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **eFG%** | (FGM + 0.5×3PM) / FGA | Effective FG% (weights threes) |
| **GSAA** | Actual - Expected Points | Points above average by location |
| **TS%** | PTS / (2×(FGA + 0.44×FTA)) | True Shooting Percentage |
| **Net Rating** | ORtg - DRtg | Two-way impact metric |

---

## Project Structure

```
Euro_stepper_analyst/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration & constants
├── requirements.txt       # Dependencies
│
├── data/
│   ├── api.py                # NBA API calls with caching
│   └── processing.py         # Data transformations
│
├── charts/
│   ├── court.py              # Shot charts (scatter, hexbin)
│   ├── comparisons.py        # Radar charts, zone breakdowns
│   ├── similarity.py         # Doppelganger visualizations
│   ├── trends.py             # Historical trends
│   └── awards.py             # MVP ladder charts
│
├── analysis/
│   ├── similarity.py         # KNN player matching
│   └── awards.py             # DNA Production Index
│
├── utils/
│   └── helpers.py            # Player matching, normalization
│
└── images/                # App icons & screenshots
```

---

## Tech Stack

| Category | Technology |
|----------|------------|
| **Frontend** | Streamlit |
| **Visualization** | Plotly |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn |
| **Data Source** | nba_api |

---

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Commit** your changes: `git commit -m 'Add amazing feature'`
4. **Push** to the branch: `git push origin feature/amazing-feature`
5. **Open** a Pull Request

---

## License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [nba_api](https://github.com/swar/nba_api) - NBA statistics API
- [Streamlit](https://streamlit.io) - Web application framework
- [Plotly](https://plotly.com) - Interactive visualizations
- [scikit-learn](https://scikit-learn.org) - Machine learning library

---

<p align="center">
  <sub>Built for the ❤️ of the game</sub>
</p>

<p align="center">
  <sub><strong>Note:</strong> This application is for educational and analytical purposes. All NBA data is property of the NBA.</sub>
</p>
