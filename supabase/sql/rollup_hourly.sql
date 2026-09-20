-- =====================================================================
-- ROLLUP HORARIO — tendencias de largo plazo instantáneas (clase System1)
-- =====================================================================
-- Problema: agregar live_readings crudo en 30d/1a tardaba ~10 s (millones de
-- filas). System1 responde al instante porque guarda agregados
-- pre-calculados. Replicado con una tabla horaria + refresh por pg_cron.
--
-- RESULTADO: trend_rollup 30d/1XD = 121 baldes en ~180 ms (antes ~10 s).
--
-- ESTADO APLICADO en watermelon-prod (2026-09-20):
--   · tabla live_readings_hourly (RLS on; la app lee con service_key)
--   · trend_rollup(...)              → lectura re-bucketizada (instantánea)
--   · refresh_live_readings_hourly() → upsert incremental (deriva pares de la
--     propia tabla horaria; se mantiene solo)
--   · backfill_instance_chunked()    → recomputa histórico por (variable,metric)
--     con COMMIT por par → resumible
--   · pg_cron: wm_refresh_sgt300b (*/10)   [el backfill one-shot ya se desagendó]
--
-- LECCIONES (límites de Supabase que costaron sangre):
--   1) El SQL Editor tiene gateway de ~60 s. El worker de pg_cron mata cualquier
--      statement a los ~2 min. NINGÚN agregado que escanee toda la tabla cabe.
--   2) Tras crear un índice hay que correr ANALYZE <tabla> o el planner hace
--      seq scan (cada par tardaba ~2 min → 1 par por corrida → nunca avanzaba).
--      Con ANALYZE live_readings el planner usa idx_live_readings_trend y cada
--      par se agrega en segundos.
--   3) El backfill itera una lista EXPLÍCITA de pares (no `select distinct`, que
--      escanea toda la instancia) y sólo el metric 'Direct' (lo único que grafica
--      el trend). Es resumible: salta pares ya presentes en la horaria.
--
-- Para sumar un cliente real nuevo: correr ANALYZE, agendar su refresh, y correr
-- backfill_instance_chunked(su_instance) por cron/CALL hasta cubrir sus canales.
--
-- La app usa trend_rollup para 7D/30D/1Y y trend_bucketed_v2 (crudo) para
-- 1H/6H/24H. Si el rollup no cubre el rango (backfill incompleto), cae a crudo.
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

-- Lectura re-bucketizada desde la horaria (avg ponderado por n). ~180 ms/30d.
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

-- Refresh incremental: deriva pares de la horaria (tabla chica → distinct rápido)
-- y reagrega las últimas p_hours horas por par (índice → tight).
create or replace function refresh_live_readings_hourly(p_instance text, p_hours int default 3)
returns void language plpgsql as $$
declare r record;
begin
    for r in select distinct variable, metric from live_readings_hourly where instance_id = p_instance loop
        insert into live_readings_hourly (instance_id, variable, metric, bucket, avg_val, min_val, max_val, n)
        select instance_id, variable, metric,
            date_bin(interval '1 hour', captured_at, timestamptz '1970-01-01 00:00:00+00'),
            avg(value)::float8, min(value)::float8, max(value)::float8, count(*)::bigint
        from live_readings
        where instance_id = p_instance and variable = r.variable and metric = r.metric
          and captured_at >= now() - make_interval(hours => p_hours)
        group by 1, 2, 3, 4
        on conflict (instance_id, variable, metric, bucket) do update
            set avg_val = excluded.avg_val, min_val = excluded.min_val,
                max_val = excluded.max_val, n = excluded.n;
    end loop;
end;
$$;

-- Backfill resumible por (variable,metric) — lista explícita de canales Direct.
-- Ajustar la lista de VALUES al agregar/renombrar sensores.
create or replace procedure backfill_instance_chunked(p_instance text)
language plpgsql as $body$
declare r record;
begin
    for r in select * from (values
        ('1XD Turbina DE','Direct'),('1YD Turbina DE','Direct'),
        ('2XD Turbina NDE','Direct'),('2YD Turbina NDE','Direct'),
        ('3YA ACCEL Gearbox','Direct'),('3YV VEL Gearbox','Direct'),
        ('4XA ACCEL starter','Direct'),('4XD Gearbox','Direct'),('4XV VEL starter','Direct'),
        ('4YA ACCEL bomba','Direct'),('4YD Gearbox','Direct'),('4YV VEL bomba','Direct'),
        ('5XD GEN DE','Direct'),('5YD GEN DE','Direct'),
        ('6XD GEN NDE','Direct'),('6YD GEN NDE','Direct'),
        ('Velocidad Generador','Direct'),('Velocidad Turbina','Direct')
    ) as t(variable, metric) loop
        if exists (select 1 from live_readings_hourly h
                   where h.instance_id = p_instance and h.variable = r.variable and h.metric = r.metric) then
            continue;  -- resumible: par ya backfilleado
        end if;
        insert into live_readings_hourly (instance_id, variable, metric, bucket, avg_val, min_val, max_val, n)
        select instance_id, variable, metric,
            date_bin(interval '1 hour', captured_at, timestamptz '1970-01-01 00:00:00+00'),
            avg(value)::float8, min(value)::float8, max(value)::float8, count(*)::bigint
        from live_readings
        where instance_id = p_instance and variable = r.variable and metric = r.metric
        group by 1, 2, 3, 4
        on conflict (instance_id, variable, metric, bucket) do update
            set avg_val = excluded.avg_val, min_val = excluded.min_val,
                max_val = excluded.max_val, n = excluded.n;
        commit;
    end loop;
end;
$body$;

grant execute on function refresh_live_readings_hourly(text, int) to service_role;
grant execute on function trend_rollup(text, text, text, timestamptz, text)
    to anon, authenticated, service_role;

-- Puesta en marcha (una vez por instancia real):
--   analyze live_readings;                                  -- CLAVE: planner usa el índice
--   create extension if not exists pg_cron;
--   select cron.schedule('wm_refresh_sgt300b','*/10 * * * *',
--       $cron$ select refresh_live_readings_hourly('turbina_sgt300_b',3) $cron$);
--   -- backfill: correr backfill_instance_chunked('turbina_sgt300_b') por cron/CALL
--   -- (resumible) hasta cubrir los 18 canales; luego desagendar.
