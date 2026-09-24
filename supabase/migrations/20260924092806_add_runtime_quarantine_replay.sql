alter table public.runtime_ingestion_events
    add column if not exists resolution_status text,
    add column if not exists resolved_at timestamptz,
    add column if not exists replay_count integer not null default 0,
    add column if not exists resolution_details jsonb not null default '{}'::jsonb;

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'runtime_ingestion_events_resolution_status_check'
          and conrelid = 'public.runtime_ingestion_events'::regclass
    ) then
        alter table public.runtime_ingestion_events
            add constraint runtime_ingestion_events_resolution_status_check
            check (resolution_status is null or resolution_status in ('pending', 'resolved'));
    end if;
end $$;

create index if not exists runtime_ingestion_events_quarantine_replay_idx
    on public.runtime_ingestion_events (resolution_status, created_at)
    where severity = 'quarantine';

comment on column public.runtime_ingestion_events.resolution_status is
    'Replay status for quarantined runtime events. Original raw event remains immutable; resolved only means a later replay accepted or safely reconciled it.';

comment on column public.runtime_ingestion_events.replay_count is
    'Number of replay attempts for a quarantined runtime event.';
