begin;

truncate table
    public.data_quality_checks,
    public.metrics_by_station,
    public.predictions,
    public.model_versions,
    public.pipeline_runs,
    public.station_time_features,
    public.observations,
    public.context_observations,
    public.time_slots,
    public.stations
restart identity cascade;

with raw_csv as (
    select (extensions.http_get('https://pulso-transmi.72-60-245-2.sslip.io/v1/downloads/stations.csv')).content::text as body
),
lines as (
    select trim(trailing E'\r' from line) as line, ordinality
    from raw_csv, regexp_split_to_table(body, E'\n') with ordinality as csv(line, ordinality)
),
parsed as (
    select string_to_array(line, ',') as cols
    from lines
    where ordinality > 1 and line <> ''
)
insert into public.stations (station_id, station_name, corridor, latitude, longitude)
select
    cols[1],
    cols[2],
    cols[3],
    cols[4]::double precision,
    cols[5]::double precision
from parsed
on conflict (station_id) do update set
    station_name = excluded.station_name,
    corridor = excluded.corridor,
    latitude = excluded.latitude,
    longitude = excluded.longitude;

with raw_csv as (
    select (extensions.http_get('https://pulso-transmi.72-60-245-2.sslip.io/v1/downloads/context.csv')).content::text as body
),
lines as (
    select trim(trailing E'\r' from line) as line, ordinality
    from raw_csv, regexp_split_to_table(body, E'\n') with ordinality as csv(line, ordinality)
),
parsed as (
    select string_to_array(line, ',') as cols
    from lines
    where ordinality > 1 and line <> ''
),
context_source as (
    select
        cols[1]::timestamptz as observed_at,
        cols[2]::double precision as rain_mm,
        cols[3]::double precision as rain_forecast,
        cols[4]::double precision as temperature_c,
        cols[5]::double precision as temperature_forecast,
        cols[6]::double precision as event_intensity
    from parsed
),
slot_source as (
    select
        observed_at,
        observed_at at time zone 'America/Bogota' as local_time
    from context_source
)
insert into public.time_slots (
    observed_at,
    local_date,
    hour,
    minute,
    day_of_week,
    day_name,
    is_weekend,
    quarter_hour_block
)
select
    observed_at,
    local_time::date,
    extract(hour from local_time)::integer,
    extract(minute from local_time)::integer,
    (extract(isodow from local_time)::integer - 1),
    case extract(isodow from local_time)::integer
        when 1 then 'lunes'
        when 2 then 'martes'
        when 3 then 'miercoles'
        when 4 then 'jueves'
        when 5 then 'viernes'
        when 6 then 'sabado'
        when 7 then 'domingo'
    end,
    extract(isodow from local_time)::integer in (6, 7),
    extract(hour from local_time)::integer * 4 + (extract(minute from local_time)::integer / 15)
from slot_source
on conflict (observed_at) do nothing;

with raw_csv as (
    select (extensions.http_get('https://pulso-transmi.72-60-245-2.sslip.io/v1/downloads/context.csv')).content::text as body
),
lines as (
    select trim(trailing E'\r' from line) as line, ordinality
    from raw_csv, regexp_split_to_table(body, E'\n') with ordinality as csv(line, ordinality)
),
parsed as (
    select string_to_array(line, ',') as cols
    from lines
    where ordinality > 1 and line <> ''
)
insert into public.context_observations (
    observed_at,
    rain_mm,
    rain_forecast,
    temperature_c,
    temperature_forecast,
    event_intensity
)
select
    cols[1]::timestamptz,
    cols[2]::double precision,
    cols[3]::double precision,
    cols[4]::double precision,
    cols[5]::double precision,
    cols[6]::double precision
from parsed
on conflict (observed_at) do update set
    rain_mm = excluded.rain_mm,
    rain_forecast = excluded.rain_forecast,
    temperature_c = excluded.temperature_c,
    temperature_forecast = excluded.temperature_forecast,
    event_intensity = excluded.event_intensity;

with raw_csv as (
    select (extensions.http_get('https://pulso-transmi.72-60-245-2.sslip.io/v1/downloads/observations.csv')).content::text as body
),
lines as (
    select trim(trailing E'\r' from line) as line, ordinality
    from raw_csv, regexp_split_to_table(body, E'\n') with ordinality as csv(line, ordinality)
),
parsed as (
    select string_to_array(line, ',') as cols
    from lines
    where ordinality > 1 and line <> ''
)
insert into public.observations (observed_at, station_id, demand)
select
    cols[1]::timestamptz,
    cols[2],
    cols[3]::integer
from parsed
on conflict (station_id, observed_at) do update set
    demand = excluded.demand;

commit;
