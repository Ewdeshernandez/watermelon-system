-- =====================================================================
-- Watermelon System — Historial de conexiones de licencias (Nivel B)
-- Tabla append-only: una fila por CADA arranque/activación de un cliente.
-- La escribe la Edge Function `activate` (service role) en cada gate_check.
-- Da la traza de auditoría: qué módulo, desde qué PC/IP/ubicación y cuándo.
-- IDEMPOTENTE. Aplica en Supabase → SQL Editor → Run.
-- =====================================================================

create table if not exists public.license_events (
  id          uuid primary key default gen_random_uuid(),
  license_id  uuid references public.licenses(id) on delete cascade,
  account     text,
  machine_fp  text,                         -- huella del PC
  hostname    text,                         -- nombre del PC + usuario
  app         text,                         -- módulo: Modal / Torsional / Balanceo
  ip          text,                         -- IP pública del arranque
  ip_geo      text,                         -- "Ciudad, PAÍS"
  created_at  timestamptz not null default now()
);

create index if not exists license_events_lic_idx  on public.license_events(license_id, created_at desc);
create index if not exists license_events_time_idx on public.license_events(created_at desc);

alter table public.license_events enable row level security;

-- El usuario autenticado ve SOLO sus eventos (jwt email = account).
do $$ begin
  if not exists (select 1 from pg_policies
                 where tablename='license_events' and policyname='own license events') then
    execute 'create policy "own license events" on public.license_events
             for select using (account = auth.jwt() ->> ''email'')';
  end if;
end $$;
-- NADIE inserta desde el cliente: solo la Edge Function (service role) escribe.
