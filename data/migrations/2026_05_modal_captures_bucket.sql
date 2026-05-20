-- =============================================================
-- Migration · v3.31.209 · Bucket "modal-captures" + RLS policies
-- =============================================================
--
-- APLICAR MANUALMENTE en el Supabase dashboard del proyecto Watermelon Cloud:
--   1. Ir a Storage → Create new bucket → "modal-captures" (Public: OFF)
--   2. Ir a SQL Editor → New query → pegar este archivo → Run
--
-- Por qué un bucket nuevo (no reusar diagram-shares):
-- · Separación de responsabilidades — los TDMS modales son archivos de
--   tipo diferente (binario científico vs PNG/SVG de diagramas)
-- · Permite distintas policies (los modal-captures son privados por user,
--   los diagram-shares son compartidos públicamente vía link firmado)
-- · Permite tracking independiente de consumo de storage por feature
--
-- Estructura esperada de objetos en el bucket:
--   {user_email}/{año}/{mes}/{nombre_archivo}.tdms
--
-- Ejemplo:
--   ehernandez@sigasas.com/2026/05/planta_ema_20260520_134255.tdms
--   ehernandez@sigasas.com/2026/05/planta_oma_20260520_145812.tdms
--   linasanchez@sigasas.com/2026/05/test_real_001.tdms

-- =============================================================
-- 1. Verificar que el bucket existe (si no, créalo manualmente en UI)
-- =============================================================
-- Si querés crear el bucket vía SQL en vez de UI:
INSERT INTO storage.buckets (id, name, public)
VALUES ('modal-captures', 'modal-captures', false)
ON CONFLICT (id) DO NOTHING;

-- =============================================================
-- 2. RLS Policies del bucket modal-captures
-- =============================================================

-- Policy: INSERT — un user puede subir solo bajo SU folder (su email)
DROP POLICY IF EXISTS "modal_captures_insert_own" ON storage.objects;
CREATE POLICY "modal_captures_insert_own"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (
    bucket_id = 'modal-captures'
    AND (storage.foldername(name))[1] = auth.jwt()->>'email'
);

-- Policy: SELECT (read/list) — un user puede leer solo SUS objetos
DROP POLICY IF EXISTS "modal_captures_select_own" ON storage.objects;
CREATE POLICY "modal_captures_select_own"
ON storage.objects FOR SELECT
TO authenticated
USING (
    bucket_id = 'modal-captures'
    AND (storage.foldername(name))[1] = auth.jwt()->>'email'
);

-- Policy: UPDATE — un user puede sobreescribir solo SUS objetos
-- (necesario para el x-upsert: true que usa el sync_uploader)
DROP POLICY IF EXISTS "modal_captures_update_own" ON storage.objects;
CREATE POLICY "modal_captures_update_own"
ON storage.objects FOR UPDATE
TO authenticated
USING (
    bucket_id = 'modal-captures'
    AND (storage.foldername(name))[1] = auth.jwt()->>'email'
);

-- Policy: DELETE — un user puede borrar solo SUS objetos
DROP POLICY IF EXISTS "modal_captures_delete_own" ON storage.objects;
CREATE POLICY "modal_captures_delete_own"
ON storage.objects FOR DELETE
TO authenticated
USING (
    bucket_id = 'modal-captures'
    AND (storage.foldername(name))[1] = auth.jwt()->>'email'
);

-- =============================================================
-- 3. Verificación
-- =============================================================
-- Para verificar que se aplicaron bien las policies:
--   SELECT * FROM pg_policies WHERE tablename = 'objects'
--     AND policyname LIKE 'modal_captures%';
-- Deben aparecer 4 policies (insert, select, update, delete).

-- =============================================================
-- 4. Limitación de tamaño (opcional — recomendado)
-- =============================================================
-- Para evitar uploads accidentales gigantes (e.g. 32ch × 1h × 5120 Hz =
-- ~700 MB), limitar el tamaño máximo por archivo a 500 MB:
-- (esto se configura en Supabase Dashboard → Storage → bucket settings)
