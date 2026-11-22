-- =====================================================================
-- rm_setup_versions — historial VERSIONADO de la config de máquina
-- (Machines / AcqSetup). Fase 2 migración Modbus → NI.
--
-- rm_setups sigue siendo el puntero a la versión PUBLICADA (lo que leen el
-- collector, el nodo en línea y el campo). Aquí vive cada versión:
--   draft      → borrador (web o campo); en máquina EN LÍNEA un cambio de
--                campo llega así y espera aceptación en la web
--   published  → vigente (una por máquina)
--   superseded → publicada antes, reemplazada
--   rejected   → borrador descartado
-- Sin sobrescritura silenciosa: publicar exige base_version = publicada actual.
-- =====================================================================
create table if not exists public.rm_setup_versions (
  id              bigint generated always as identity primary key,
  setup_id        text        not null,
  version         int         not null,
  status          text        not null default 'draft'
                  check (status in ('draft', 'published', 'superseded', 'rejected')),
  origin          text        not null default 'web'
                  check (origin in ('web', 'field', 'import', 'api')),
  author          text,
  base_version    int,
  note            text,
  spec            jsonb       not null,
  created_at      timestamptz not null default now(),
  published_at    timestamptz,
  published_by    text,
  node_status     text,          -- applied | rejected (confirmación del nodo en línea)
  node_detail     text,
  node_at         timestamptz,
  unique (setup_id, version)
);

create index if not exists rm_setup_versions_setup_idx
  on public.rm_setup_versions (setup_id, version desc);

alter table public.rm_setup_versions enable row level security;

-- (Tabla nueva: sin DROP — el editor de Supabase los bloquea.)
-- El .exe de campo usa la anon key embebida: puede LEER y subir BORRADORES de
-- campo; publicar / rechazar / ack del nodo = service role (web / API).
create policy rm_setup_versions_read on public.rm_setup_versions
  for select to anon, authenticated using (true);

create policy rm_setup_versions_field_draft on public.rm_setup_versions
  for insert to anon, authenticated
  with check (status = 'draft' and origin = 'field');
