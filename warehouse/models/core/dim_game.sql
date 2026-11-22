-- One row for each game. The game log has one row for each team:
-- 'BOS vs. NYK' is the home row of BOS, and 'NYK @ BOS' is the away row of NYK.
-- At a neutral site the two rows show '@'. The shot data then gives the team
-- that the NBA designates as the home team.
with designated as (
    select
        game_id,
        any_value(home_abbreviation) as home_abbreviation
    from {{ ref('stg_shots') }}
    group by game_id
),

per_game as (
    select
        l.game_id,
        any_value(l.season) as season,
        min(l.game_date) as game_date,
        max(case when l.matchup like '% vs. %' then l.team_id end) as listed_home_team_id,
        max(case when l.abbreviation = d.home_abbreviation then l.team_id end)
            as designated_home_team_id,
        min(l.team_id) as team_id_low,
        max(l.team_id) as team_id_high
    from {{ ref('stg_game_log') }} as l
    left join designated as d on l.game_id = d.game_id
    group by l.game_id
),

with_home as (
    select
        *,
        coalesce(listed_home_team_id, designated_home_team_id) as home_team_id
    from per_game
)

select
    game_id,
    season,
    game_date,
    home_team_id,
    -- A game with one team in the game log has no away team. The not_null test
    -- then fails the build, so the home team is not also given as the away team.
    case
        when home_team_id is null then null
        when team_id_low = team_id_high then null
        when home_team_id = team_id_low then team_id_high
        else team_id_low
    end as away_team_id,
    listed_home_team_id is null as is_neutral_site
from with_home
