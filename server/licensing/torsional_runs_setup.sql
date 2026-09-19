-- Watermelon Torsional — corridas en la nube (tabla + RLS anon + bucket cruda)
-- ============================================================================
-- El .exe de campo usa la ANON key (no service key) para subir corridas. Estas
-- policies le dan EXACTO lo que necesita (subir metadata + data cruda) y nada más.
-- La WEB lee/analiza con service key (bypass RLS). Homólogo a rls_modal_anon.sql.
--
-- APLICAR una vez en el SQL editor de Supabase (proyecto Watermelon Cloud).
-- Idempotente (create if not exists / drop-create policies).

-- ---- Tabla torsional_runs (metadata de la corrida; la cruda va en Storage) ----
create table if not exists public.torsional_runs (
  id         text primary key,
  name       text,
  metadata   jsonb,
  updated_at text,
  account    text default '',
  client     text default '',
  tag        text default '',
  hostname   text default '',
  ip         text default '',
  geo        text default '',
  module     text default 'Torsional',
  created_at timestamptz default now()
);
-- Idempotente para tablas ya creadas antes de estas columnas:
alter table if exists public.torsional_runs add column if not exists ip         text default '';
alter table if exists public.torsional_runs add column if not exists geo        text default '';
alter table if exists public.torsional_runs add column if not exists module     text default 'Torsional';
alter table if exists public.torsional_runs add column if not exists created_at timestamptz default now();

alter table if exists public.torsional_runs enable row level security;
drop policy if exists "anon read torsional_runs"   on public.torsional_runs;
drop policy if exists "anon write torsional_runs"  on public.torsional_runs;
drop policy if exists "anon update torsional_runs" on public.torsional_runs;
create policy "anon read torsional_runs"   on public.torsional_runs for select to anon using (true);
create policy "anon write torsional_runs"  on public.torsional_runs for insert to anon with check (true);
create policy "anon update torsional_runs" on public.torsional_runs for update to anon using (true) with check (true);

-- ---- Storage: bucket privado `torsional-raw` (data cruda gzip .npy.gz) ----
insert into storage.buckets (id, name, public)
values ('torsional-raw', 'torsional-raw', false)
on conflict (id) do nothing;

drop policy if exists "torsional-raw insert" on storage.objects;
drop policy if exists "torsional-raw update" on storage.objects;
create policy "torsional-raw insert" on storage.objects for insert
  with check (bucket_id = 'torsional-raw');
create policy "torsional-raw update" on storage.objects for update
  using (bucket_id = 'torsional-raw') with check (bucket_id = 'torsional-raw');

-- NOTA: sin policy de SELECT en el bucket → nadie descarga con la anon key;
-- la web descarga la cruda con su service key (server-side).
