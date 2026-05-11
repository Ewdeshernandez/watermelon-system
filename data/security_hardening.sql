-- ============================================================
-- Watermelon System — Security hardening SQL
-- ============================================================
-- Correr este script en Supabase Dashboard → SQL Editor → New
-- query → pegar todo → Run.
--
-- Activa Row Level Security (RLS) en TODAS las tablas detectadas.
-- Como el backend Watermelon usa service_role key (que bypasea
-- RLS), esto NO rompe nada del flujo actual. Funciona como
-- DEFENSA EN PROFUNDIDAD: si en el futuro se expone una anon_key
-- o un bug usa la key equivocada, RLS evita que extraños lean
-- data de otros clientes.
--
-- Tiempo: ~30 segundos para correr.
-- Reversible: cada ALTER abajo tiene su DISABLE comentado si
-- necesitás rollback.
-- ============================================================


-- ------------------------------------------------------------
-- 1. Detectar todas las tablas en el schema public
-- ------------------------------------------------------------
-- Antes de aplicar el hardening, listá qué tablas tenés:
-- (Descomenta y corré solo esta query primero para ver el inventario)

-- SELECT tablename
--   FROM pg_catalog.pg_tables
--  WHERE schemaname = 'public'
--  ORDER BY tablename;


-- ------------------------------------------------------------
-- 2. Activar RLS en tablas conocidas
-- ------------------------------------------------------------
-- Si alguna tabla NO existe en tu base, el ALTER simplemente
-- tirará un NOTICE, no rompe el script. Usamos DO blocks para
-- que sea idempotente.

DO $$
DECLARE
    tbl text;
    target_tables text[] := ARRAY[
        'instances',
        'live_readings',
        'users',
        'clients',
        'reports',
        'reports_archive',
        'sensor_readings',
        'machinery',
        'documents',
        'audit_log',
        'password_resets'
    ];
BEGIN
    FOREACH tbl IN ARRAY target_tables LOOP
        IF EXISTS (
            SELECT 1 FROM pg_catalog.pg_tables
             WHERE schemaname = 'public' AND tablename = tbl
        ) THEN
            EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', tbl);
            RAISE NOTICE 'RLS activado: %', tbl;
        ELSE
            RAISE NOTICE 'Tabla NO existe (skip): %', tbl;
        END IF;
    END LOOP;
END $$;


-- ------------------------------------------------------------
-- 3. Policy default: service_role tiene acceso total
-- ------------------------------------------------------------
-- Esto NO afecta el backend Watermelon (que usa service_role,
-- el cual bypasea RLS por diseño). Pero documenta explícitamente
-- la intención. Si en el futuro creás roles más restringidos
-- (e.g. "client_read_only"), esto sirve de base.

-- Por tabla, una policy que permite todo al service_role
-- (que es el role que usa tu backend):

DO $$
DECLARE
    tbl text;
    target_tables text[] := ARRAY[
        'instances',
        'live_readings',
        'users',
        'clients',
        'reports',
        'reports_archive'
    ];
BEGIN
    FOREACH tbl IN ARRAY target_tables LOOP
        IF EXISTS (
            SELECT 1 FROM pg_catalog.pg_tables
             WHERE schemaname = 'public' AND tablename = tbl
        ) THEN
            -- Drop policy si existe (idempotente)
            EXECUTE format(
                'DROP POLICY IF EXISTS "service_role_full_access" ON public.%I',
                tbl
            );
            EXECUTE format(
                'CREATE POLICY "service_role_full_access" ON public.%I '
                'FOR ALL TO service_role USING (true) WITH CHECK (true)',
                tbl
            );
            RAISE NOTICE 'Policy service_role applied: %', tbl;
        END IF;
    END LOOP;
END $$;


-- ------------------------------------------------------------
-- 4. Audit log de DDL (cambios de schema)
-- ------------------------------------------------------------
-- Opcional pero recomendado: registra cualquier DROP TABLE,
-- ALTER TABLE, etc. Útil para detectar manipulación.
-- Activá la extensión pgaudit en el dashboard:
--   Database → Extensions → "pgaudit" → enable

-- Después podés correr:
-- ALTER ROLE postgres SET pgaudit.log = 'ddl,role';


-- ------------------------------------------------------------
-- 5. Verificación final
-- ------------------------------------------------------------
-- Confirmá qué tablas tienen RLS activado:

SELECT
    tablename,
    CASE WHEN rowsecurity THEN '✅ RLS ON' ELSE '❌ RLS OFF' END AS status
  FROM pg_catalog.pg_tables
 WHERE schemaname = 'public'
 ORDER BY rowsecurity DESC, tablename;


-- ============================================================
-- ROLLBACK (solo si algo se rompe)
-- ============================================================
-- Para desactivar RLS de una tabla específica:
--   ALTER TABLE public.<tablename> DISABLE ROW LEVEL SECURITY;
--
-- Para borrar la policy:
--   DROP POLICY "service_role_full_access" ON public.<tablename>;
-- ============================================================
