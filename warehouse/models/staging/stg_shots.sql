-- One row for each shot event. The latest load of each game wins for the full game:
-- a load holds all the shots of each game in its date window, so a shot that the NBA
-- removed or numbered again is not in the latest load, and it goes from this model.
with typed as (
    select
        {{ game_id('GAME_ID') }} as game_id,
        {{ to_int('GAME_EVENT_ID') }} as game_event_id,
        {{ to_bigint('PLAYER_ID') }} as player_id,
        PLAYER_NAME as player_name,
        {{ to_bigint('TEAM_ID') }} as team_id,
        _season as season,
        cast(strptime(GAME_DATE, '%Y%m%d') as date) as game_date,
        {{ to_int('PERIOD') }} as period,
        {{ to_int('MINUTES_REMAINING') }} * 60 + {{ to_int('SECONDS_REMAINING') }} as seconds_remaining,
        ACTION_TYPE as action_type,
        case when SHOT_TYPE like '3PT%' then 3 else 2 end as shot_value,
        SHOT_ZONE_BASIC as zone_basic,
        SHOT_ZONE_AREA as zone_area,
        SHOT_ZONE_RANGE as zone_range,
        {{ to_int('SHOT_DISTANCE') }} as shot_distance_ft,
        cast(LOC_X as double) / 10.0 as loc_x_ft,
        cast(LOC_Y as double) / 10.0 as loc_y_ft,
        {{ to_int('SHOT_ATTEMPTED_FLAG') }} = 1 as attempted,
        {{ to_int('SHOT_MADE_FLAG') }} = 1 as made,
        HTM as home_abbreviation,
        VTM as away_abbreviation,
        _origin,
        _loaded_at
    from {{ source('raw', 'nba_shots') }}
),

latest_load_of_game as (
    select *
    from typed
    qualify _loaded_at = max(_loaded_at) over (partition by game_id)
)

-- A load can have the same shot event two times. Keep one.
select * exclude (_origin, _loaded_at)
from latest_load_of_game
qualify row_number() over (
    partition by game_id, game_event_id
    order by _loaded_at desc, _origin
) = 1
