-- trend_annotations — bitácora de notas del analista sobre la tendencia.
-- Idempotente. Correr en el SQL editor de Supabase (editor 60s / ANALYZE tras índice).
create table if not exists public.trend_annotations (
  id          bigint generated always as identity primary key,
  instance_id text        not null,
  ts          timestamptz not null,
  text        text        not null,
  author      text        default '',
  created_at  timestamptz not null default now()
);

create index if not exists idx_trend_annotations_instance_ts
  on public.trend_annotations (instance_id, ts);

analyze public.trend_annotations;
