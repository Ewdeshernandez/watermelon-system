-- =====================================================================
-- trend_bucketed_v2 — tendencia downsampled server-side (Live Monitoring)
-- =====================================================================
-- FIX (2026-09-20): el rango "30 días / 1 año" hacía statement timeout
-- (57014) por falta de índice → seq scan de toda live_readings. La app caía
-- en silencio al crudo de 1000 filas y mostraba solo ~3 h aunque el usuario
-- pidiera 30 días. Error visible al cliente.
--
-- Estado APLICADO en producción (watermelon-prod):
--   1) índice idx_live_readings_trend            → creado
--   2) función trend_bucketed_v2 (p_bucket text) → creada, la app la usa
--      (core/live_readings.history_bucketed → rpc "trend_bucketed_v2")
--
-- NOTA: la función vieja `trend_bucketed` quedó con DOS overloads
-- (p_bucket interval + text) → PostgREST no podía resolver cuál usar
-- (PGRST203). Por eso la app apunta a trend_bucketed_v2 (nombre único, sin
-- ambigüedad). Limpieza opcional al final (requiere DROP manual).
-- =====================================================================

-- 1) Índice compuesto — corre SOLO (CONCURRENTLY no va dentro de la
--    transacción que envuelve el SQL Editor; si falla por 25001, quita
--    CONCURRENTLY y córrelo así, plano):
create index concurrently if not exists idx_live_readings_trend
    on live_readings (instance_id, variable, metric, captured_at);

-- 2) Función (nombre único v2 → sin ambigüedad de overload).
create or replace function trend_bucketed_v2(
    p_instance text,
    p_variable text,
    p_metric   text,
    p_from     timestamptz,
    p_bucket   text
)
returns table (
    bucket   timestamptz,
    avg_val  double precision,
    min_val  double precision,
    max_val  double precision,
    n        bigint
)
language sql
stable
set statement_timeout to '30s'
as $$
    select
        date_bin(p_bucket::interval, captured_at, timestamptz '1970-01-01 00:00:00+00') as bucket,
        avg(value)::float8  as avg_val,
        min(value)::float8  as min_val,
        max(value)::float8  as max_val,
        count(*)::bigint    as n
    from live_readings
    where instance_id = p_instance
      and variable    = p_variable
      and metric      = p_metric
      and captured_at >= p_from
    group by 1
    order by 1
$$;

grant execute on function trend_bucketed_v2(text, text, text, timestamptz, text)
    to anon, authenticated, service_role;

-- 3) LIMPIEZA OPCIONAL — eliminar los overloads viejos y ambiguos de
--    trend_bucketed (ya nadie los llama). Correr manualmente si se desea:
-- drop function if exists trend_bucketed(text, text, text, timestamptz, interval);
-- drop function if exists trend_bucketed(text, text, text, timestamptz, text);
