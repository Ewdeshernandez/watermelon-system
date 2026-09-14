-- Watermelon Modal — licenciamiento (Supabase / Postgres)
-- Aplica en el SQL editor o con `supabase db push`.

-- Licencias por cuenta (una fila por licencia comprada)
create table if not exists public.licenses (
  id           uuid primary key default gen_random_uuid(),
  account      text not null,                 -- email / id de la cuenta
  seats        int  not null default 1,       -- cuántas máquinas puede activar
  features     text[] not null default '{oma,ema,report}',
  expires_at   timestamptz not null,          -- vencimiento de la licencia (fuente de verdad)
  status       text not null default 'active',-- active | suspended | revoked
  created_at   timestamptz not null default now()
);

-- Activaciones (una fila por máquina activada; hace el binding + kill-switch)
create table if not exists public.activations (
  id            uuid primary key default gen_random_uuid(),
  license_id    uuid not null references public.licenses(id) on delete cascade,
  account       text not null,
  machine_fp    text not null,                -- huella de hardware del cliente
  is_vm         boolean not null default false,
  revoked       boolean not null default false, -- kill-switch por máquina
  last_seen     timestamptz not null default now(),
  created_at    timestamptz not null default now(),
  unique (license_id, machine_fp)
);

alter table public.licenses    enable row level security;
alter table public.activations enable row level security;

-- El usuario autenticado SOLO ve/gestiona lo suyo (jwt email = account).
create policy "own licenses"    on public.licenses
  for select using (account = auth.jwt() ->> 'email');
create policy "own activations" on public.activations
  for select using (account = auth.jwt() ->> 'email');
-- NADIE inserta licencias/activaciones/tokens desde el cliente: sólo la Edge Function
-- (service role) escribe. Sin policy de insert para 'authenticated' => bloqueado por RLS.

-- ---- Policies para las corridas modales (subida desde el .exe con anon+auth) ----
-- (ajusta a tus tablas reales: modal_runs + storage bucket modal-raw)
alter table if exists public.modal_runs enable row level security;
do $$ begin
  if exists (select 1 from information_schema.columns
             where table_name='modal_runs' and column_name='owner') then
    execute 'create policy "insert own runs" on public.modal_runs for insert
             with check (owner = auth.jwt() ->> ''email'')';
    execute 'create policy "read own runs"  on public.modal_runs for select
             using (owner = auth.jwt() ->> ''email'')';
  end if;
end $$;
