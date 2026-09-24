alter table public.data_quality_checks
    add column if not exists severity text not null default 'pass',
    add column if not exists details_json jsonb not null default '{}'::jsonb,
    add column if not exists idempotency_key text;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'data_quality_checks_severity_check'
          and conrelid = 'public.data_quality_checks'::regclass
    ) then
        alter table public.data_quality_checks
            add constraint data_quality_checks_severity_check
            check (severity in ('pass', 'warning', 'quarantine', 'critical'));
    end if;
end $$;

create unique index if not exists data_quality_checks_idempotency_key_uidx
    on public.data_quality_checks (idempotency_key);

create index if not exists data_quality_checks_severity_idx
    on public.data_quality_checks (severity);

create table if not exists public.runtime_ingestion_events (
    event_id uuid primary key default gen_random_uuid(),
    run_id uuid references public.pipeline_runs (run_id) on update cascade on delete set null,
    stream_name text not null default 'observations',
    cursor text,
    batch_hash text not null check (length(batch_hash) > 0),
    row_hash text,
    idempotency_key text not null check (length(idempotency_key) > 0),
    event_type text not null check (event_type in ('batch', 'row', 'api_error', 'schema', 'model_compatibility')),
    severity text not null check (severity in ('pass', 'warning', 'quarantine', 'critical')),
    station_id text,
    observed_at timestamptz,
    raw_payload jsonb not null default '{}'::jsonb,
    normalized_payload jsonb not null default '{}'::jsonb,
    details jsonb not null default '{}'::jsonb,
    reason_codes text[] not null default '{}'::text[],
    created_at timestamptz not null default now()
);

create unique index if not exists runtime_ingestion_events_idempotency_key_uidx
    on public.runtime_ingestion_events (idempotency_key);

create index if not exists runtime_ingestion_events_stream_cursor_idx
    on public.runtime_ingestion_events (stream_name, cursor);

create index if not exists runtime_ingestion_events_severity_idx
    on public.runtime_ingestion_events (severity);

create index if not exists runtime_ingestion_events_station_observed_idx
    on public.runtime_ingestion_events (station_id, observed_at);

alter table public.runtime_ingestion_events enable row level security;

revoke all on table public.runtime_ingestion_events from anon, authenticated;

comment on table public.runtime_ingestion_events is
    'Runtime stream validation audit log. Rollback via DROP TABLE public.runtime_ingestion_events removes accumulated audit history.';
