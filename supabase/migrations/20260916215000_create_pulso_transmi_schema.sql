create extension if not exists pgcrypto;

create table if not exists public.stations (
    station_id text primary key,
    station_name text not null,
    corridor text not null,
    latitude double precision not null,
    longitude double precision not null,
    constraint stations_station_id_format check (station_id ~ '^[0-9]{5}$')
);

create table if not exists public.time_slots (
    observed_at timestamptz primary key,
    local_date date not null,
    hour integer not null check (hour between 0 and 23),
    minute integer not null check (minute in (0, 15, 30, 45)),
    day_of_week integer not null check (day_of_week between 0 and 6),
    day_name text not null,
    is_weekend boolean not null,
    quarter_hour_block integer not null check (quarter_hour_block between 0 and 95)
);

create table if not exists public.observations (
    station_id text not null references public.stations (station_id) on update cascade on delete restrict,
    observed_at timestamptz not null references public.time_slots (observed_at) on update cascade on delete restrict,
    demand integer not null check (demand >= 0),
    ingested_at timestamptz not null default now(),
    primary key (station_id, observed_at)
);

create table if not exists public.context_observations (
    observed_at timestamptz primary key references public.time_slots (observed_at) on update cascade on delete restrict,
    rain_mm double precision not null check (rain_mm >= 0),
    rain_forecast double precision not null check (rain_forecast >= 0),
    temperature_c double precision not null,
    temperature_forecast double precision not null,
    event_intensity double precision not null check (event_intensity >= 0)
);

create table if not exists public.station_time_features (
    station_id text not null references public.stations (station_id) on update cascade on delete restrict,
    observed_at timestamptz not null references public.time_slots (observed_at) on update cascade on delete restrict,
    demand_lag_24h double precision,
    demand_lag_7d double precision,
    rolling_mean_24h double precision,
    rolling_mean_7d double precision,
    rolling_std_24h double precision,
    primary key (station_id, observed_at)
);

create table if not exists public.pipeline_runs (
    run_id uuid primary key default gen_random_uuid(),
    started_at timestamptz not null default now(),
    finished_at timestamptz,
    status text not null check (status in ('running', 'success', 'failed')),
    git_commit text,
    data_cutoff timestamptz,
    trigger_type text not null default 'manual',
    error_message text
);

create table if not exists public.model_versions (
    model_id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.pipeline_runs (run_id) on update cascade on delete restrict,
    model_name text not null,
    model_type text not null,
    artifact_uri text,
    trained_until timestamptz not null,
    feature_config jsonb not null default '{}'::jsonb,
    params jsonb not null default '{}'::jsonb,
    is_active boolean not null default false
);

create table if not exists public.predictions (
    prediction_id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.pipeline_runs (run_id) on update cascade on delete restrict,
    model_id uuid not null references public.model_versions (model_id) on update cascade on delete restrict,
    station_id text not null references public.stations (station_id) on update cascade on delete restrict,
    target_observed_at timestamptz not null references public.time_slots (observed_at) on update cascade on delete restrict,
    horizon_minutes integer not null check (horizon_minutes > 0),
    prediction double precision not null check (prediction >= 0),
    created_at timestamptz not null default now()
);

create table if not exists public.metrics_by_station (
    metric_id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.pipeline_runs (run_id) on update cascade on delete restrict,
    model_id uuid references public.model_versions (model_id) on update cascade on delete set null,
    station_id text not null references public.stations (station_id) on update cascade on delete restrict,
    split_name text not null,
    evaluation_start timestamptz not null,
    evaluation_end timestamptz not null,
    wape double precision not null check (wape >= 0),
    accuracy double precision not null check (accuracy >= 0),
    constraint metrics_valid_window check (evaluation_end >= evaluation_start)
);

create table if not exists public.data_quality_checks (
    check_id uuid primary key default gen_random_uuid(),
    run_id uuid not null references public.pipeline_runs (run_id) on update cascade on delete cascade,
    check_name text not null,
    status text not null check (status in ('passed', 'failed', 'warning')),
    actual_value integer,
    expected_value integer,
    details text,
    checked_at timestamptz not null default now()
);

create index if not exists observations_observed_at_idx on public.observations (observed_at);
create index if not exists observations_station_observed_at_idx on public.observations (station_id, observed_at);
create index if not exists predictions_target_idx on public.predictions (target_observed_at, station_id);
create index if not exists metrics_run_station_idx on public.metrics_by_station (run_id, station_id);
create index if not exists model_versions_active_idx on public.model_versions (is_active) where is_active;

alter table public.stations enable row level security;
alter table public.time_slots enable row level security;
alter table public.observations enable row level security;
alter table public.context_observations enable row level security;
alter table public.station_time_features enable row level security;
alter table public.pipeline_runs enable row level security;
alter table public.model_versions enable row level security;
alter table public.predictions enable row level security;
alter table public.metrics_by_station enable row level security;
alter table public.data_quality_checks enable row level security;

revoke all on table public.stations from anon, authenticated;
revoke all on table public.time_slots from anon, authenticated;
revoke all on table public.observations from anon, authenticated;
revoke all on table public.context_observations from anon, authenticated;
revoke all on table public.station_time_features from anon, authenticated;
revoke all on table public.pipeline_runs from anon, authenticated;
revoke all on table public.model_versions from anon, authenticated;
revoke all on table public.predictions from anon, authenticated;
revoke all on table public.metrics_by_station from anon, authenticated;
revoke all on table public.data_quality_checks from anon, authenticated;
