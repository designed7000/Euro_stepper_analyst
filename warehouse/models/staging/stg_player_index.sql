-- One row for each player. The latest load wins.
with typed as (
    select
        {{ to_bigint('PERSON_ID') }} as player_id,
        trim(coalesce(PLAYER_FIRST_NAME, '') || ' ' || coalesce(PLAYER_LAST_NAME, '')) as full_name,
        PLAYER_SLUG as slug,
        nullif(trim(POSITION), '') as position,
        -- HEIGHT is feet-inches, for example '6-7'
        try_cast(split_part(HEIGHT, '-', 1) as integer) * 12
            + try_cast(split_part(HEIGHT, '-', 2) as integer) as height_in,
        try_cast(FROM_YEAR as integer) as from_year,
        try_cast(TO_YEAR as integer) as to_year,
        try_cast(try_cast(TEAM_ID as double) as bigint) as team_id,
        _loaded_at
    from {{ source('raw', 'nba_player_index') }}
)

select * exclude (_loaded_at)
from typed
qualify row_number() over (partition by player_id order by _loaded_at desc) = 1
