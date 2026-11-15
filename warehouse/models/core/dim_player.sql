-- One row for each player. Players of the index, plus each player who has
-- shots and is not in the index (name from the shot data, other columns null).
with indexed as (
    select player_id, full_name, slug, position, height_in, from_year, to_year, team_id
    from {{ ref('stg_player_index') }}
),

shot_only as (
    select
        player_id,
        arg_max(player_name, game_date) as full_name
    from {{ ref('stg_shots') }}
    where player_id not in (select player_id from indexed)
    group by player_id
)

select * from indexed
union all
select
    player_id,
    full_name,
    cast(null as varchar) as slug,
    cast(null as varchar) as position,
    cast(null as integer) as height_in,
    cast(null as integer) as from_year,
    cast(null as integer) as to_year,
    cast(null as bigint) as team_id
from shot_only
