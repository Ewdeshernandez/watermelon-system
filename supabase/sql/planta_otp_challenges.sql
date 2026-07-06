-- =============================================================
-- planta_otp_challenges — challenges OTP de la Edge Function planta-auth
-- =============================================================
-- Guarda el reto de un solo uso para el login OTP de Watermelon Planta:
-- un código de 6 dígitos (solo su HMAC, nunca en claro), su expiración y el
-- contador de intentos. Un challenge activo por email (PK = email; el
-- "request" hace upsert y reemplaza el anterior).
--
-- Acceso: SOLO service_role (la Edge Function). RLS habilitada y SIN policies
-- públicas → anon/authenticated no pueden leer ni escribir. El service_role
-- bypassa RLS por diseño.
--
-- Correr en: Supabase Dashboard → SQL Editor (proyecto watermelon-prod).

create table if not exists public.planta_otp_challenges (
  email       text primary key,
  code_hash   text        not null,
  expires_at  timestamptz not null,
  attempts    integer     not null default 0,
  created_at  timestamptz not null default now()
);

alter table public.planta_otp_challenges enable row level security;

-- (Sin policies a propósito: nadie salvo service_role puede tocar la tabla.)

-- Índice para limpiar expirados eficientemente.
create index if not exists idx_planta_otp_expires
  on public.planta_otp_challenges (expires_at);
