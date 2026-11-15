-- One row for each game. The game log has one row for each team:
-- 'BOS vs. NYK' is the home row of BOS, and 'NYK @ BOS' is the away row of NYK.
select
    game_id,
    any_value(season) as season,
    min(game_date) as game_date,
    max(case when matchup like '% vs. %' then team_id end) as home_team_id,
    max(case when matchup like '% @ %' then team_id end) as away_team_id
from {{ ref('stg_game_log') }}
group by game_id
