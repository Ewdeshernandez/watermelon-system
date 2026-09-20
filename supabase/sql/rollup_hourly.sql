-- =====================================================================
-- ROLLUP HORARIO — tendencias de largo plazo instantáneas (clase System1)
-- =====================================================================
-- Problema: agregar live_readings crudo en 30d/1a tarda ~10 s (millones de
-- filas). System1 responde al instante porque guarda agregados
-- pre-calculados. Replicamos con una tabla horaria + refresh por pg_cron.
--
-- ESTADO APLICADO en watermelon-prod (2026-09-20):
--   · tabla live_readings_hourly (RLS on; la app lee con service_key)
--   · trend_rollup(...)              → lectura re-bucketizada (instantánea)
--   · refresh_live_readings_hourly() → upsert incremental por instancia
--   · backfill_instance_hourly()     → recomputa histórico de una instancia
--   · pg_cron: wm_refresh_sgt300b (*/10), wm_backfill_sgt300b (*/3, one-shot)
--
-- IMPORTANTE (límite del SQL Editor de Supabase = 60 s de gateway): el
-- backfill de una instancia completa NO cabe en el editor → se corre por
-- pg_cron (server-side, sin ese límite; statement_timeout 1200 s en la
-- función). Tras poblar, desagendar el backfill:
--   select cron.unschedule('wm_backfill_sgt300b');
-- Para sumar otra instancia (nuevo cliente real), agendar su refresh + un
-- backfill one-shot igual que SGT300B.
--
-- La app usa trend_rollup para 7D/30D/1Y y trend_bucketed_v2 (crudo) para
-- 1H/6H/24H (sub-hora). Si el rollup está vacío, cae a crudo (v2).
-- =====================================================================

create table if not exists live_readings_hourly (
    instance_id text        not null,
    variable    text        not null,
    metric      text        not null,
    bucket      timestamptz not null,
    avg_val     double precision,
    min_val     double precision,
    max_val     double precision,
    n           bigint,
    primary key (instance_id, variable, metric, bucket)
);
alter table live_readings_hourly enable row level security;

-- Refresh incremental (últimas p_hours horas de UNA instancia). El filtro por
-- instance_id usa el índice idx_live_readings_trend (instance_id primero).
create or replace function refresh_live_readings_hourly(p_instance text, p_hours int default 3)
returns void language sql set statement_timeout to '55s' as $$
    insert into live_readings_hourly (instance_id, variable, metric, bucket, avg_val, min_val, max_val, n)
    select instance_id, variable, metric,
        date_bin(interval '1 hour', captured_at, timestamptz '1970-01-01 00:00:00+00'),
        avg(value)::float8, min(value)::float8, max(value)::float8, count(*)::bigint
    from live_readings
    where instance_id = p_instance
      and captured_at >= now() - make_interval(hours => p_hours)
    group by 1, 2, 3, 4
    on conflict (instance_id, variable, metric, bucket) do update
        set avg_val = excluded.avg_val, min_val = excluded.min_val,
            max_val = excluded.max_val, n = excluded.n;
$$;

-- Backfill del histórico completo de UNA instancia. timeout 1200 s → correr
-- SOLO por pg_cron (no cabe en el editor).
create or replace function backfill_instance_hourly(p_instance text)
returns bigint language plpgsql set statement_timeout to '1200s' as $$
declare _n bigint;
begin
    insert into live_readings_hourly (instance_id, variable, metric, bucket, avg_val, min_val, max_val, n)
    select instance_id, variable, metric,
        date_bin(interval '1 hour', captured_at, timestamptz '1970-01-01 00:00:00+00'),
        avg(value)::float8, min(value)::float8, max(value)::float8, count(*)::bigint
    from live_readings
    where instance_id = p_instance
    group by 1, 2, 3, 4
    on conflict (instance_id, variable, metric, bucket) do update
        set avg_val = excluded.avg_val, min_val = excluded.min_val,
            max_val = excluded.max_val, n = excluded.n;
    get diagnostics _n = row_count;
    return _n;
end;
$$;

-- Lectura re-bucketizada desde la horaria (avg ponderado por n).
create or replace function trend_rollup(
    p_instance text, p_variable text, p_metric text, p_from timestamptz, p_bucket text)
returns table (bucket timestamptz, avg_val double precision, min_val double precision, max_val double precision, n bigint)
language sql stable as $$
    select date_bin(p_bucket::interval, bucket, timestamptz '1970-01-01 00:00:00+00') as bucket,
        (sum(avg_val * n) / nullif(sum(n), 0))::float8 as avg_val,
        min(min_val)::float8 as min_val, max(max_val)::float8 as max_val, sum(n)::bigint as n
    from live_readings_hourly
    where instance_id = p_instance and variable = p_variable and metric = p_metric and bucket >= p_from
    group by 1 order by 1
$$;

grant execute on function refresh_live_readings_hourly(text, int) to service_role;
grant execute on function backfill_instance_hourly(text)          to service_role;
grant execute on function trend_rollup(text, text, text, timestamptz, text)
    to anon, authenticated, service_role;

-- pg_cron (server-side, sin límite de gateway).
create extension if not exists pg_cron;
select cron.schedule('wm_refresh_sgt300b', '*/10 * * * *',
    $cron$ select refresh_live_readings_hourly('turbina_sgt300_b', 3) $cron$);
-- One-shot: agendar, dejar correr una vez, y desagendar:
select cron.schedule('wm_backfill_sgt300b', '*/3 * * * *',
    $cron$ select backfill_instance_hourly('turbina_sgt300_b') $cron$);
-- select cron.unschedule('wm_backfill_sgt300b');   -- tras poblar
