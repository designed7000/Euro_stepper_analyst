# Design: platform skeleton and Shot DNA module (part 1)

Date: 2026-11-15. Status: waits for approval.

Lines with **[call]** are decisions that Claude made. The owner can change them.

## 1. Intent

Build a public NBA analysis website that replaces the Streamlit prototype.
The primary reader is one expert on an NBA data team. The public is the second reader.
The product is the method: unusual models, with correct statistics and visible validation.

Part 1 proves the full path from the daily load to a public page, with one real module.

## 2. Goals, non-goals, anti-goals

Goals of part 1:

- A daily load on the owner's computer fills a local warehouse.
- A module contract lets us add one analysis idea as one unit.
- The Shot DNA module for players is complete, with validation.
- A public site shows the player page, the compare page, the leaderboard, and the methods page.
- The site continues to operate when the owner's computer is off.

Non-goals of part 1:

- Team pages, RAPM, contracts. Each gets its own design.
- Shot allocation. **[call]** It moves to the team part, because a correct method (lineup points lost) needs lineup context.
- Accounts, comments, and saved views.
- Raw tracking methods, clutch filters, live game data.

Anti-goals:

- The site becomes a mirror of raw NBA stats.
- A metric shows a number with no uncertainty.
- A new module forces a change in a different module.

## 3. Constraints

- stats.nba.com blocks cloud servers. All calls to it come from the owner's computer.
- NBA.com terms do not permit a public stats database. The cloud holds and serves only our result tables and aggregates.
- Budget: maximum $100 each month. Target for part 1: less than $25 each month. The builder must verify prices.
- Refresh: one time each day.
- No raw data in the Git repo. The repo is public.

## 4. Approaches

| | A: Baked site | B: API server | C: Baked site + small API |
|---|---|---|---|
| Delivery | Result files on a CDN | Cloud database + API | Files, plus an API for special queries |
| Queries | Calculated in advance | Any query, live | Mixed |
| Cost and work | Lowest | Highest | Low |

The owner selected **B**. Claude recommended A for cost and low risk.
This design uses the strength of B: live filters on shot data and a clean path to accounts.

## 5. Architecture

```
OWNER'S COMPUTER (one run each day)
  load ──► raw (Parquet) ──► warehouse (DuckDB, dbt) ──► modules (Python) ──► publish
                                                                                │
CLOUD                                                                           ▼
  visitor ──► web (Next.js) ──► API (FastAPI, read-only) ──► Postgres (result tables only)
```

### 5.1 Load

- `nba_api` gets the daily data. **[call]** One league-wide `ShotChartDetail` call for each season gets all the shots.
- `PlayerIndex` gives the official position, height, and team. This replaces the position guess.
- `LeagueGameLog` gives the games.
- **[call]** History comes from the `shufinskiy/nba_data` archives (1996-97 to now). The daily load adds new games.
- Each run loads the last 3 days again, because the NBA corrects stats.
- If days are missing, the run starts from the last good game date in `run_log`.
- **[call]** Regular season games only in part 1.
- Risk: nobody verified the archive content or the endpoints from the owner's computer. The first build step must do this.
- Each call has a timeout, retries with backoff, and a pause between calls.
- The load writes each response to Parquet as received, in `raw/<source>/<season>/<load_date>/`. Raw files do not change.

### 5.2 Warehouse

- **[call]** DuckDB is the engine. dbt (`dbt-duckdb`) builds the tables in SQL.
- Layers: `staging` (clean, typed, one row for each source row) and `core` (shared tables).
- Core tables of part 1: `dim_player`, `dim_team`, `dim_game`, `fct_shot`.
- dbt tests check keys, nulls, relations, and accepted values.

### 5.3 Module contract

A module is one unit with these parts:

| Part | Content |
|---|---|
| Manifest | Name, version, input tables, output tables with columns and keys |
| Model | Python code that reads inputs and writes outputs |
| Validation | Code that writes metrics of model quality to an output table |
| Tests | Unit tests with small fixture data |
| API | Optional custom routes |
| Page | React components for the module section and the methods page |

Rules:

- A module reads only `core` tables and the outputs of modules that its manifest names.
- A module writes only its own output tables.
- The runner reads all manifests and runs the modules in dependency order.
- **[call]** The API makes a typed read endpoint for each output table from the manifest. Custom routes are necessary only for live aggregates.
- To add a module, you add its unit and one registry line in the web app. You change no other module.

The builder decides the physical folder layout across Python and TypeScript.

### 5.4 Run and schedule

- **[call]** One CLI (`esa`) has the commands `load`, `build`, `run`, `publish`, and `daily`. `daily` does the four in sequence.
- **[call]** `launchd` starts `esa daily` one time each day. No orchestrator service in part 1.
- Each command is idempotent. A second run on the same day gives the same result.
- A `run_log` table records each run, its status, and the last game date.
- If a step fails, the run stops and `publish` does not start. The site keeps the data of the last good run.

### 5.5 Publish

- `publish` copies only module output tables and the necessary `dim` columns to Postgres.
- The copy goes to a new schema. One transaction then replaces the live schema. Visitors do not see a partial state.
- `publish` uses a write role. The API uses a different, read-only role.

### 5.6 API

- FastAPI, read-only, version prefix `/v1`.
- Pydantic models type each response. The OpenAPI file generates the TypeScript client.
- Responses give aggregates and model results. No endpoint gives raw shot rows or a bulk export.
- Each response has cache headers, because data changes one time each day.
- Rate limit for each IP address. CORS permits only the site domain.

### 5.7 Web

- **[call]** Next.js with React and TypeScript. Server rendering lets search engines find the pages.
- Charts are SVG with D3. One shared half-court component.
- Each page shows "Data through <date>" from `run_log`.
- The footer says that the site is not affiliated with the NBA.

### 5.8 Hosting

**[call]** Managed Postgres, one small container for the API, and a host with a free tier for the web app.
The builder selects the vendors after a price check and stays below the part 1 target.
Secrets stay in environment variables. They are not in the repo.

## 6. Shot DNA module

Input: `fct_shot` with location, action type, shot value, period, and result.

**[call]** Scope rules:

- Published seasons: 2015-16 to now. The module runs for each season.
- A player who changes team has one player-season across his teams.
- Each player has a page. The leaderboard and the similarity list include only player-seasons with 200 shots or more.
- The daily run fits the current season only. Past seasons run again only when the module version changes.

### 6.1 Shot diet (where a player shoots)

1. Divide the half court into cells. Count the shots of each player-season in each cell and smooth the counts.
2. Apply non-negative matrix factorization (NMF) to the player-by-cell matrix. This follows Miller et al. (2014), with kernel smoothing in place of a point-process model.
3. The result is K basis maps (for example "corner three", "restricted area") and K weights for each player.
4. The weights are the Shot DNA vector of the player.

**[call]** Fit the bases one time on all published seasons, then keep them fixed. Seasons are then comparable.
Select K by reconstruction error on held-out data and by the stability of the bases across random starts.
For a new day of the current season, calculate the weights with non-negative least squares on the fixed bases.

### 6.2 Shot making (how well a player shoots)

1. A league model estimates the make probability of each shot from location and action type. **[call]** Gradient boosting, one model for each season. Each shot gets its estimate from a model that did not see its game.
2. A hierarchical Bayesian logistic model then estimates player effects. The league estimate is the offset. Each player has one effect for each basis, with partial pooling.
3. **[call]** NumPyro with NUTS, on counts aggregated by player, basis, and probability bin.
4. Output: points above expected for each 100 shots, for each player and basis, with a 90% credible interval.

Partial pooling pulls small samples to the league mean. This replaces GSAA.

### 6.3 Similarity

- **[call]** Diet distance is the Wasserstein distance between two DNA vectors. The cost between two bases is the Wasserstein distance between their maps, calculated one time. This distance knows that near court areas are similar. A bin-by-bin distance does not.
- The builder can use a cheap distance first to decrease the number of pairs.
- Making distance is the distance between posterior means, scaled by posterior uncertainty.
- Output: the 10 nearest player-seasons for each player-season, with the two distances shown separately.

### 6.4 Table for live filters

The hex map needs one output table for live aggregates.
Its grain is player, game, court cell, shot family, and period.
Its values are attempts, makes, and the sum of expected makes.
The table has no exact shot coordinates. The API gives only sums from it.
A shot family is a group of action types (for example jump shot, layup, dunk). The module holds the mapping.

### 6.5 Validation (shown on the methods page)

| Check | Question |
|---|---|
| Log loss and Brier score against the zone-average baseline | Does the league model predict better than the old method? |
| Calibration curve | Are the probabilities correct? |
| Split-half reliability | Is the DNA vector stable in one season? |
| Year-to-year correlation | Is the making effect a skill or noise? |
| Posterior coverage | Do the 90% intervals contain held-out results 90% of the time? |

All checks use games that the model did not see.

### 6.6 Limits (shown on the methods page)

- Public shot data has no defender distance. The league model cannot see how open a shot is.
- Action type labels come from scorers and change between seasons.
- Shot selection depends on teammates and scheme. Part 1 describes it and does not judge it.

## 7. Pages of part 1

```
/players/<id>-<name>
  [Name · Team · Position]                [Season ▼]
  ── Shot DNA ──────────────────────────────────────
  [Hex map: size = frequency,   ]  [DNA weights against league ]
  [color = making over expected ]  [Making by basis ± interval ]
  Filters: shot family · period · date range
  Similar players: 1 … 10
  [Link: how this works]

/compare?a=<name>&b=<name>     two players side by side, same charts
/modules/shot-dna              leaderboard with intervals, map of all players
/modules/shot-dna/methods      model, validation tables, limits
/                              player search, list of modules
```

The hex map filters are live API aggregates.

## 8. Tests and quality

- Python: `pytest` for load parsers, module models, and API routes. Fixtures replace NBA calls.
- Data: dbt tests on each build.
- Web: component tests and one browser smoke test for each page.
- CI on GitHub: lint, type checks, and all tests. CI makes no call to stats.nba.com.

## 9. Old code

**[call]** Move the Streamlit app to `legacy/`. Port the zone mapping and the court geometry. Delete `legacy/` when the Shot DNA pages are live.

## 10. Left to the builder

- Folder layout, Python version (3.12 or later), and package tools.
- Hosting vendors, after a price check.
- Cell size, smoothing width, prior scales, and K, selected by the validation in 6.5.
- Visual style of the site.
- The time of the daily run, set in configuration.

The owner makes the accounts at the hosting vendors. The builder does not make accounts.

## 11. Later parts

The owner selected Shot DNA as the first module on 2026-11-15. Claude recommended spillover value first.
**[call]** The order below is a proposal. Each part gets its own design.

| Part | Content | Base |
|---|---|---|
| 2 | Possession table, our own RAPM, and spillover value (own production against effect on teammates) | Arcidiacono, Kinsler, Price 2017 |
| 3 | Team pages: hypergraph plus-minus, team connectome, allocation efficiency, lineup niche overlap | Josephs and Upton 2025. Connectome and niche overlap are new ideas with no published test. |
| 4 | Contracts: surplus value with uncertainty, with spillover included | Standard method plus part 2 |
| 5 | Game-flow physics: the score as a random walk | Clauset, Kogan, Redner 2015 |

Part 1 must not block these parts. The `core` layer must accept a possession table later with no change to `fct_shot`.

## 12. Open, does not block part 1

- The site name and the domain. The repo says "Euro Stepper Analyst" and the README says "NBA Player DNA".
- If the budget is $100 each month or $100 in total.
