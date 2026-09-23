alter table public.stream_cursors
  add column if not exists is_exhausted boolean not null default false;
