-- =====================================================================
-- Watermelon System — Consola de Licencias (campo: Modal/Torsional/Balanceo/Field)
-- Migración ADITIVA e IDEMPOTENTE. Aplica en Supabase → SQL Editor → Run.
-- Amplía las tablas `licenses` y `activations` (server/licensing/schema.sql) para
-- que la consola web de Administración muestre PC, IP, ubicación y última
-- conexión, y para el estado comercial (suspensión por falta de pago).
-- No borra datos. Seguro de correr varias veces.
-- =====================================================================

-- --- licenses: datos comerciales para la consola ---------------------
alter table public.licenses add column if not exists key             text;      -- clave WM-XXXX-XXXX-XXXX (ya existe en prod)
alter table public.licenses add column if not exists customer        text;      -- nombre legible del cliente/empresa
alter table public.licenses add column if not exists plan            text;      -- paquete comercial (informativo)
alter table public.licenses add column if not exists notes           text;      -- notas internas SIGA
alter table public.licenses add column if not exists suspended_reason text;     -- motivo visible al cliente al bloquear
alter table public.licenses add column if not exists updated_at      timestamptz not null default now();

-- Índice único de la clave (si aún no existe). Ignora el error si ya está.
do $$ begin
  if not exists (select 1 from pg_indexes where indexname = 'licenses_key_uidx') then
    execute 'create unique index licenses_key_uidx on public.licenses(key)';
  end if;
exception when others then null; end $$;

create index if not exists licenses_status_idx on public.licenses(status);

-- --- activations: identidad del equipo + red -------------------------
alter table public.activations add column if not exists hostname   text;         -- nombre del PC + usuario (machine_label)
alter table public.activations add column if not exists ip         text;         -- IP pública desde donde activó/heartbeat
alter table public.activations add column if not exists ip_geo     text;         -- "Ciudad, PAÍS" (best-effort)
alter table public.activations add column if not exists app        text;         -- módulo que activó (modal/torsional/…)
alter table public.activations add column if not exists updated_at timestamptz not null default now();

create index if not exists activations_license_idx on public.activations(license_id);

-- --- Estados válidos de licenses.status ------------------------------
--   active     → funciona
--   suspended  → BLOQUEADA por falta de pago (cliente ve el motivo, se puede reactivar)
--   revoked    → BLOQUEADA definitiva
-- (status es text; no se fuerza CHECK para no romper filas existentes.)
