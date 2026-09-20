-- =====================================================================
-- trend_bucketed — tendencia downsampled server-side para Live Monitoring
-- =====================================================================
-- FIX (2026-09-20): el rango "30 días / 1 año" hacía statement timeout
-- (código 57014) porque faltaba un índice compuesto → seq scan de toda la
-- tabla live_readings. La app caía en silencio al crudo de 1000 filas y
-- mostraba solo ~3 h aunque el usuario pidiera 30 días. Error visible al
-- cliente.
--
-- Aplicar TODO este archivo en Supabase → SQL Editor → Run.
-- El CREATE INDEX CONCURRENTLY debe ir SOLO (no dentro de una transacción);
-- si el editor lo envuelve en BEGIN/COMMIT, córrelo aparte.
-- =====================================================================

-- 1) Índice compuesto: convierte el filtro (instance, variable, metric,
--    captured_at >= from) en un index range scan. Sin esto, 30 d agrega
--    millones de filas por seq scan y revienta el timeout.
create index concurrently if not exists idx_live_readings_trend
    on live_readings (instance_id, variable, metric, captured_at);

-- 2) Función optimizada (misma firma y mismas columnas que consume la app:
--    bucket, avg_val, min_val, max_val, n). date_bin agrupa por balde de
--    tiempo (p_bucket = '6 hours', '1 day', etc.). statement_timeout propio
--    de 30 s como red de seguridad para el rango de 1 año.
create or replace function trend_bucketed(
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

-- 3) Permisos (la app usa service_key, pero dejamos anon/auth por si acaso).
grant execute on function trend_bucketed(text, text, text, timestamptz, text)
    to anon, authenticated, service_role;
