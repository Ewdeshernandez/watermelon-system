-- Watermelon Modal — trazabilidad y control multi-tenant de las corridas subidas.
-- Añade a modal_runs quién/de-qué-cliente subió cada corrida, para (1) saber quién
-- subió qué, (2) filtrar por cliente (cada cliente ve solo lo suyo), (3) notificar.
-- Aplica en el SQL editor de Supabase. Idempotente.

alter table public.modal_runs add column if not exists account   text;         -- cuenta/licencia que subió
alter table public.modal_runs add column if not exists client    text;         -- cliente dueño del activo
alter table public.modal_runs add column if not exists tag       text;         -- tag del activo (para scope multi-tenant)
alter table public.modal_runs add column if not exists hostname  text;         -- PC de campo que subió
alter table public.modal_runs add column if not exists created_at timestamptz default now();  -- para "corridas nuevas"

-- Índice para listar rápido por fecha.
create index if not exists idx_modal_runs_created on public.modal_runs (created_at desc);
