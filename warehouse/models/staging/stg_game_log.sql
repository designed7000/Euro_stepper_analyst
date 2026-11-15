-- One row for each team in each game. The latest load wins.
with typed as (
    select
        {{ game_id('GAME_ID') }} as game_id,
        {{ to_bigint('TEAM_ID') }} as team_id,
        TEAM_ABBREVIATION as abbreviation,
        TEAM_NAME as name,
        _season as season,
        cast(GAME_DATE as date) as game_date,
        MATCHUP as matchup,
        _loaded_at
    from {{ source('raw', 'nba_game_log') }}
)

select * exclude (_loaded_at)
from typed
qualify row_number() over (partition by game_id, team_id order by _loaded_at desc) = 1
