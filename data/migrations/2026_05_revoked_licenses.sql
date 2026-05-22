-- ============================================================
-- 2026_05_revoked_licenses.sql — FASE J1 (v3.31.221)
-- ============================================================
-- Tabla de control de licencias revocadas para el sistema de
-- heartbeat de Watermelon Planta Edition.
--
-- Flujo:
--   1. SIGA emite licencia (license_id = JTI del JWT) con tools/license_issue.py
--   2. Si SIGA quiere revocar: INSERT en esta tabla
--   3. Planta arranca → hace HTTP GET al endpoint license-check?jti=<license_id>
--   4. La Edge Function lee esta tabla y devuelve {status, reason}
--   5. Si revocada → Planta bloquea la app
--
-- Importante:
--   - El license_id es público (está en el token del cliente, no es secreto)
--   - Esta tabla solo lista los REVOCADOS — las activas no se guardan acá
--     (la fuente de verdad de las activas es tools/licenses_issued/)
--   - Esquema simple: si está en esta tabla, está revocada. Punto.
--
-- Aplicar:
--   psql -h <host> -U postgres < data/migrations/2026_05_revoked_licenses.sql
--   o copiar/pegar en Supabase Dashboard → SQL Editor
-- ============================================================

CREATE TABLE IF NOT EXISTS public.revoked_licenses (
    license_id   uuid PRIMARY KEY,
    revoked_at   timestamptz NOT NULL DEFAULT now(),
    revoked_by   text,                          -- email del admin SIGA
    reason       text NOT NULL,                 -- motivo visible al cliente
    customer     text,                          -- nombre del cliente (debug)
    customer_email text,                        -- email del cliente (debug)
    metadata     jsonb DEFAULT '{}'::jsonb      -- info extra opcional
);

COMMENT ON TABLE public.revoked_licenses IS
    'Licencias de Watermelon Planta revocadas explícitamente por SIGA. '
    'Planta consulta esta tabla via Edge Function license-check al arrancar.';

COMMENT ON COLUMN public.revoked_licenses.license_id IS
    'JWT ID (jti) de la licencia revocada. Coincide con el claim jti del license.token.';

COMMENT ON COLUMN public.revoked_licenses.reason IS
    'Motivo de revocación. Se muestra al cliente en la pantalla de bloqueo.';

-- ============================================================
-- INDEX para lookup rápido por license_id (ya es PK pero explicito)
-- ============================================================
-- Ya cubierto por PRIMARY KEY, no hace falta CREATE INDEX

-- ============================================================
-- RLS — IMPORTANTE
-- ============================================================
-- Esta tabla NO debe ser accesible directamente por el cliente
-- (Planta lee a través de la Edge Function, no del SDK Supabase).
-- Solo el admin SIGA puede leer y escribir.
--
-- La Edge Function usa el SERVICE_ROLE_KEY (bypass RLS) para leer.
ALTER TABLE public.revoked_licenses ENABLE ROW LEVEL SECURITY;

-- Policy: solo usuarios con email @sigasas.com pueden ver/editar
-- (la admin UI en Cloud usa esto)
DROP POLICY IF EXISTS "siga_admin_full_access" ON public.revoked_licenses;
CREATE POLICY "siga_admin_full_access"
    ON public.revoked_licenses
    FOR ALL
    TO authenticated
    USING ((auth.jwt() ->> 'email') LIKE '%@sigasas.com')
    WITH CHECK ((auth.jwt() ->> 'email') LIKE '%@sigasas.com');

-- ============================================================
-- Datos de ejemplo (comentado — solo para test)
-- ============================================================
-- INSERT INTO public.revoked_licenses (license_id, revoked_by, reason, customer)
-- VALUES (
--     '00000000-0000-0000-0000-000000000000',
--     'ehernandez@sigasas.com',
--     'Test de revocación',
--     'Cliente Demo'
-- );
