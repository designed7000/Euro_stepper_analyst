-- One row for each team, with its latest abbreviation and name.
select
    team_id,
    arg_max(abbreviation, game_date) as abbreviation,
    arg_max(name, game_date) as name
from {{ ref('stg_game_log') }}
group by team_id
