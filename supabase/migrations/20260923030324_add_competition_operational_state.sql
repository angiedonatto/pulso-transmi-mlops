create table if not exists public.stream_cursors (
    stream_name text primary key,
    confirmed_cursor text,
    confirmed_at timestamptz not null default now(),
    last_observed_at timestamptz,
    rows_written integer not null default 0 check (rows_written >= 0),
    client_run_id uuid references public.pipeline_runs (run_id) on update cascade on delete set null
);

create table if not exists public.forecast_cycles (
    cycle_id text primary key,
    data_cutoff timestamptz not null,
    closes_at timestamptz not null,
    expected_predictions integer not null check (expected_predictions > 0),
    raw_cycle jsonb not null default '{}'::jsonb,
    first_seen_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    constraint forecast_cycles_valid_window check (closes_at > data_cutoff)
);

create table if not exists public.forecast_cycle_targets (
    cycle_id text not null references public.forecast_cycles (cycle_id) on update cascade on delete cascade,
    station_id text not null references public.stations (station_id) on update cascade on delete restrict,
    target_at timestamptz not null,
    horizon_minutes integer not null check (horizon_minutes > 0),
    target_order integer check (target_order >= 0),
    raw_target jsonb not null default '{}'::jsonb,
    primary key (cycle_id, station_id, target_at)
);

alter table public.predictions
    add column if not exists cycle_id text;

do $$
begin
    if not exists (select 1 from pg_constraint where conname = 'predictions_cycle_id_fkey') then
        alter table public.predictions
            add constraint predictions_cycle_id_fkey
            foreign key (cycle_id)
            references public.forecast_cycles (cycle_id)
            on update cascade
            on delete restrict;
    end if;

    if not exists (select 1 from pg_constraint where conname = 'predictions_cycle_target_fkey') then
        alter table public.predictions
            add constraint predictions_cycle_target_fkey
            foreign key (cycle_id, station_id, target_observed_at)
            references public.forecast_cycle_targets (cycle_id, station_id, target_at)
            on update cascade
            on delete restrict;
    end if;
end $$;

create table if not exists public.submission_receipts (
    receipt_id uuid primary key default gen_random_uuid(),
    submission_id text,
    cycle_id text not null references public.forecast_cycles (cycle_id) on update cascade on delete restrict,
    model_id uuid not null references public.model_versions (model_id) on update cascade on delete restrict,
    payload_hash text not null check (length(payload_hash) > 0),
    idempotency_key text not null check (length(idempotency_key) > 0),
    attempt integer check (attempt between 1 and 3),
    accepted_at timestamptz,
    client_run_id uuid references public.pipeline_runs (run_id) on update cascade on delete set null,
    status text not null check (status in ('pending', 'accepted', 'idempotent_replay', 'rejected', 'failed')),
    is_official boolean not null default false,
    receipt_payload jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint submission_receipts_success_requires_attempt check (
        status not in ('accepted', 'idempotent_replay')
        or attempt is not null
    ),
    constraint submission_receipts_success_has_receipt check (
        status not in ('accepted', 'idempotent_replay')
        or (submission_id is not null and accepted_at is not null)
    ),
    constraint submission_receipts_official_must_be_success check (
        not is_official
        or status in ('accepted', 'idempotent_replay')
    )
);

create unique index if not exists submission_receipts_submission_id_uidx
    on public.submission_receipts (submission_id)
    where submission_id is not null;

create unique index if not exists submission_receipts_idempotency_key_uidx
    on public.submission_receipts (idempotency_key);

create unique index if not exists submission_receipts_cycle_attempt_uidx
    on public.submission_receipts (cycle_id, attempt)
    where attempt is not null;

create unique index if not exists submission_receipts_one_official_per_cycle_uidx
    on public.submission_receipts (cycle_id)
    where is_official;

create index if not exists forecast_cycle_targets_station_target_idx
    on public.forecast_cycle_targets (station_id, target_at);

create unique index if not exists forecast_cycle_targets_order_uidx
    on public.forecast_cycle_targets (cycle_id, target_order)
    where target_order is not null;

create index if not exists predictions_cycle_idx
    on public.predictions (cycle_id);

create unique index if not exists predictions_cycle_model_target_uidx
    on public.predictions (cycle_id, model_id, station_id, target_observed_at)
    where cycle_id is not null;

create index if not exists submission_receipts_cycle_model_idx
    on public.submission_receipts (cycle_id, model_id);

create index if not exists submission_receipts_delivered_cycle_idx
    on public.submission_receipts (cycle_id)
    where status in ('accepted', 'idempotent_replay');

alter table public.stream_cursors enable row level security;
alter table public.forecast_cycles enable row level security;
alter table public.forecast_cycle_targets enable row level security;
alter table public.submission_receipts enable row level security;

revoke all on table public.stream_cursors from anon, authenticated;
revoke all on table public.forecast_cycles from anon, authenticated;
revoke all on table public.forecast_cycle_targets from anon, authenticated;
revoke all on table public.submission_receipts from anon, authenticated;
