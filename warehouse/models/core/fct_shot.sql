-- One row for each field goal attempt of the regular season.
select
    game_id || '-' || cast(game_event_id as varchar) as shot_id,
    game_id,
    game_event_id,
    player_id,
    team_id,
    season,
    game_date,
    period,
    seconds_remaining,
    action_type,
    shot_value,
    zone_basic,
    zone_area,
    zone_range,
    shot_distance_ft,
    loc_x_ft,
    loc_y_ft,
    made
from {{ ref('stg_shots') }}
where attempted
