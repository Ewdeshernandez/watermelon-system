-- ============================================================
-- Watermelon System — Downsampling de tendencias (Ciclo 23.146)
-- ============================================================
-- Resuelve el límite de 1000 filas de PostgREST: en vez de traer lecturas
-- crudas (millones para 1 año), agrega los datos en "baldes" de tiempo y
-- devuelve avg/min/max por balde. La app llama a esta función según el rango.
--
-- Correr UNA VEZ en: Supabase → SQL Editor → New query → pegar todo → Run.
-- ============================================================

-- 1) Índice para que las consultas de tendencia vuelen
create index if not exists idx_live_readings_trend
  on public.live_readings (instance_id, variable, metric, captured_at);

-- 2) Función de downsampling por baldes de tiempo
create or replace function public.trend_bucketed(
    p_instance text,
    p_variable text,
    p_metric   text,
    p_from     timestamptz,
    p_bucket   interval
)
returns table (
    bucket  timestamptz,
    avg_val double precision,
    min_val double precision,
    max_val double precision,
    n       bigint
)
language sql
stable
security definer
set search_path = public
as $$
    select
        date_bin(p_bucket, captured_at, p_from) as bucket,
        avg(value)::double precision  as avg_val,
        min(value)::double precision  as min_val,
        max(value)::double precision  as max_val,
        count(*)                      as n
    from public.live_readings
    where instance_id = p_instance
      and variable    = p_variable
      and metric      = p_metric
      and captured_at >= p_from
    group by 1
    order by 1
$$;

-- 3) Permisos de ejecución
grant execute on function public.trend_bucketed(text, text, text, timestamptz, interval)
    to anon, authenticated, service_role;

-- ============================================================
-- Verificación rápida (opcional): debería devolver ~168 filas (7 días / 1h)
-- ============================================================
-- select * from public.trend_bucketed(
--     'tes1', '1YA ACCEL CRF', 'Direct', now() - interval '7 days', interval '1 hour'
-- ) order by bucket;
