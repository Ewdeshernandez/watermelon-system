-- Watermelon Balance — corridas en la nube (tabla + RLS anon + bucket cruda)
-- ============================================================================
-- El .exe de campo usa la ANON key (no service key) para subir corridas. Estas
-- policies le dan EXACTO lo que necesita (subir metadata + data cruda) y nada más.
-- La WEB lee/analiza con service key (bypass RLS). Homólogo a rls_modal_anon.sql.
--
-- APLICAR una vez en el SQL editor de Supabase (proyecto Watermelon Cloud).
-- Idempotente (create if not exists / drop-create policies).

-- ---- Tabla balance_runs (metadata de la corrida; la cruda va en Storage) ----
create table if not exists public.balance_runs (
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
  module     text default 'Balance',
  created_at timestamptz default now()
);
-- Idempotente para tablas ya creadas antes de estas columnas:
alter table if exists public.balance_runs add column if not exists ip         text default '';
alter table if exists public.balance_runs add column if not exists geo        text default '';
alter table if exists public.balance_runs add column if not exists module     text default 'Balance';
alter table if exists public.balance_runs add column if not exists created_at timestamptz default now();

alter table if exists public.balance_runs enable row level security;
drop policy if exists "anon read balance_runs"   on public.balance_runs;
drop policy if exists "anon write balance_runs"  on public.balance_runs;
drop policy if exists "anon update balance_runs" on public.balance_runs;
create policy "anon read balance_runs"   on public.balance_runs for select to anon using (true);
create policy "anon write balance_runs"  on public.balance_runs for insert to anon with check (true);
create policy "anon update balance_runs" on public.balance_runs for update to anon using (true) with check (true);

-- ---- Storage: bucket privado `balance-raw` (data cruda gzip .npy.gz) ----
insert into storage.buckets (id, name, public)
values ('balance-raw', 'balance-raw', false)
on conflict (id) do nothing;

drop policy if exists "balance-raw insert" on storage.objects;
drop policy if exists "balance-raw update" on storage.objects;
create policy "balance-raw insert" on storage.objects for insert
  with check (bucket_id = 'balance-raw');
create policy "balance-raw update" on storage.objects for update
  using (bucket_id = 'balance-raw') with check (bucket_id = 'balance-raw');

-- NOTA: sin policy de SELECT en el bucket → nadie descarga con la anon key;
-- la web descarga la cruda con su service key (server-side).
