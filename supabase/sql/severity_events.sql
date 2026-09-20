-- =====================================================================
-- severity_events — Event log PERSISTENTE (cruces de umbral) estilo System1
-- =====================================================================
-- Historial de transiciones de severidad por canal: cuándo entró en
-- Alarma/Danger, cuánto duró, cuándo volvió a Normal, y quién lo reconoció.
-- Lo escribe core/severity_events.record_events (idempotente: solo cambios);
-- lo lee list_recent (con duración calculada) y ack_event marca el ack.
-- La app usa service_key → bypassa RLS.
-- =====================================================================
create table if not exists severity_events (
    id           bigserial primary key,
    instance_id  text not null,
    sensor_label text,
    variable     text,
    from_status  text,
    to_status    text,
    value        double precision,
    unit         text,
    alarm        double precision,
    danger       double precision,
    crossed_at   timestamptz not null default now(),
    ack_by       text,
    ack_at       timestamptz
);

create index if not exists idx_severity_events_inst_time
    on severity_events (instance_id, crossed_at desc);

alter table severity_events enable row level security;

grant select, insert, update on severity_events to service_role;
grant usage, select on sequence severity_events_id_seq to service_role;
