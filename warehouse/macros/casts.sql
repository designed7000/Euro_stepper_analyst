{# Raw columns are strings. An integer can arrive as '12' or as '12.0'. #}
{% macro to_bigint(column) -%}
    cast(cast({{ column }} as double) as bigint)
{%- endmacro %}

{% macro to_int(column) -%}
    cast(cast({{ column }} as double) as integer)
{%- endmacro %}

{# NBA game ids have 10 characters with leading zeros. Some sources drop the zeros. #}
{% macro game_id(column) -%}
    lpad(cast({{ to_bigint(column) }} as varchar), 10, '0')
{%- endmacro %}
