-- ===========================================================================
-- Migration: live_readings (Tier 0 A — Live Data Ingestion)
-- ---------------------------------------------------------------------------
-- Tabla append-only que almacena cada lectura individual de un sensor
-- enviada por un wm-collector en planta. Soporta:
--
--   * Direct overall (Direct)        → magnitud RMS / 0-pk del overall
--   * Vectores 1X / 2X (Ampl + Phase) → para Polar Plot, Bode, Shaft Centerline
--   * Diagnostic (Gap, BiasVoltage)  → health del transducer
--   * Speed (rpm)                     → referencia para órdenes
--
-- Diseño:
--   - append-only (nunca UPDATE) — historial inmutable, ideal para trends.
--   - índice por (instance_id, variable, metric, captured_at DESC) para
--     queries "último valor" rápidos.
--   - índice por (instance_id, captured_at DESC) para queries por rango temporal.
--   - metadata JSONB para info extra del collector sin alterar schema.
--
-- Operación esperada:
--   * 30-50 readings cada 10s × 24h × 30 días ≈ 13M filas/mes/máquina.
--   * Vamos a particionar por mes en una migración futura cuando crezca,
--     por ahora una sola tabla aguanta.
-- ===========================================================================

CREATE TABLE IF NOT EXISTS public.live_readings (
    id              BIGSERIAL PRIMARY KEY,
    instance_id     TEXT        NOT NULL,
    sensor_label    TEXT,                 -- '1Y_V', '3X_D', etc. (puede ser NULL si es speed/diagnostic)
    variable        TEXT        NOT NULL, -- '1YV VEL CRF', 'Velocidad Generador', etc.
    metric          TEXT        NOT NULL, -- 'Direct' | 'Gap' | 'BiasVoltage' | '1X_Ampl' | '1X_Phase' | '2X_Ampl' | '2X_Phase'
    value           DOUBLE PRECISION,
    unit            TEXT,                 -- 'mm/s pk', 'g pk', 'mil pp', 'V DC', 'deg', 'rpm'
    captured_at     TIMESTAMPTZ NOT NULL, -- timestamp del collector cuando leyó del 3500/92
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    register        INTEGER,              -- Modbus register address (debug)
    quality         TEXT        NOT NULL DEFAULT 'good', -- 'good' | 'stale' | 'overrange' | 'comm_fail'
    metadata        JSONB       NOT NULL DEFAULT '{}'::jsonb
);

-- Índice principal: query "último valor de cada métrica de un activo"
CREATE INDEX IF NOT EXISTS idx_live_readings_lookup
    ON public.live_readings (instance_id, variable, metric, captured_at DESC);

-- Índice secundario: query "todo lo del activo en un rango temporal"
CREATE INDEX IF NOT EXISTS idx_live_readings_instance_time
    ON public.live_readings (instance_id, captured_at DESC);

-- Índice por sensor para vistas de Polar/Bode (todos los puntos 1X de un sensor)
CREATE INDEX IF NOT EXISTS idx_live_readings_sensor_metric_time
    ON public.live_readings (instance_id, sensor_label, metric, captured_at DESC)
    WHERE sensor_label IS NOT NULL;

-- ===========================================================================
-- View: latest_live_reading
-- ---------------------------------------------------------------------------
-- DISTINCT ON sobre (instance_id, variable, metric) ordenado por captured_at
-- DESC devuelve la fila más reciente de cada combinación. Es el "current
-- values" del dashboard.
-- ===========================================================================

CREATE OR REPLACE VIEW public.latest_live_reading AS
    SELECT DISTINCT ON (instance_id, variable, metric)
        instance_id,
        sensor_label,
        variable,
        metric,
        value,
        unit,
        captured_at,
        ingested_at,
        quality,
        metadata
    FROM public.live_readings
    ORDER BY instance_id, variable, metric, captured_at DESC;

-- ===========================================================================
-- RLS — por ahora abierto para service_key, cerrado para el resto.
-- Cuando el frontend lea directo (sin pasar por API), agregamos políticas
-- por instance_id + tenant.
-- ===========================================================================

ALTER TABLE public.live_readings ENABLE ROW LEVEL SECURITY;

-- Default deny — sólo el service_role (API) puede escribir/leer.
CREATE POLICY live_readings_service_role_all
    ON public.live_readings
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

COMMENT ON TABLE public.live_readings IS
'Append-only de lecturas en tiempo real desde wm-collector. Cada fila = un (instance, variable, metric, captured_at).';
