create table if not exists public.prediction_traces (
    trace_id uuid primary key default gen_random_uuid(),
    model_id uuid not null references public.model_versions (model_id) on update cascade on delete cascade,
    cycle_id text not null references public.forecast_cycles (cycle_id) on update cascade on delete cascade,
    station_id text not null references public.stations (station_id) on update cascade on delete restrict,
    target_at timestamptz not null,
    horizon_minutes integer not null check (horizon_minutes > 0),
    base_prediction double precision,
    calibration_factor double precision,
    final_prediction double precision not null check (final_prediction >= 0),
    feedback_samples integer not null default 0 check (feedback_samples >= 0),
    data_cutoff timestamptz not null,
    created_at timestamptz not null default now(),
    constraint prediction_traces_cycle_target_fkey
        foreign key (cycle_id, station_id, target_at)
        references public.forecast_cycle_targets (cycle_id, station_id, target_at)
        on update cascade
        on delete cascade
);

create unique index if not exists prediction_traces_cycle_model_target_uidx
    on public.prediction_traces (cycle_id, model_id, station_id, target_at);

create index if not exists prediction_traces_model_created_idx
    on public.prediction_traces (model_id, created_at desc);

create table if not exists public.monitoring_snapshots (
    snapshot_id uuid primary key default gen_random_uuid(),
    generated_at timestamptz not null default now(),
    model_id uuid references public.model_versions (model_id) on update cascade on delete set null,
    overall_status text not null check (overall_status in ('HEALTHY', 'WATCH', 'DRIFT', 'FAILED', 'INSUFFICIENT_DATA')),
    drift_status text not null check (drift_status in ('HEALTHY', 'WATCH', 'DRIFT', 'INSUFFICIENT_DATA')),
    retrain_recommended boolean not null default false,
    report jsonb not null default '{}'::jsonb
);

create index if not exists monitoring_snapshots_generated_idx
    on public.monitoring_snapshots (generated_at desc);

create index if not exists monitoring_snapshots_status_idx
    on public.monitoring_snapshots (overall_status, drift_status);

alter table public.prediction_traces enable row level security;
alter table public.monitoring_snapshots enable row level security;

revoke all on table public.prediction_traces from anon, authenticated;
revoke all on table public.monitoring_snapshots from anon, authenticated;

comment on table public.prediction_traces is
    'Read-only observability trace for adaptive predictions. Does not affect official submission payloads.';

comment on table public.monitoring_snapshots is
    'Canonical MLOps dashboard snapshot generated from Supabase state and drift_monitor.build_drift_report.';
