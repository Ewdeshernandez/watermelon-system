-- =====================================================================
-- Watermelon System — Setup del bucket diagram-shares (v3.31.67 Fase 1)
-- =====================================================================
-- Corré este SQL UNA SOLA VEZ en Supabase Dashboard → SQL Editor.
-- Crea el bucket público + policies para que el componente JS del cliente
-- pueda subir PNGs del diagrama Live Monitoring y compartirlos por
-- WhatsApp / Email.
--
-- Modelo de seguridad:
--   • Bucket: público en LECTURA (cualquiera con el link ve la imagen).
--   • Bucket: INSERT permitido solo con anon key (no sirve para enumerar).
--   • Archivos: limit 5 MB (más que suficiente para PNG 4K del diagrama).
--   • TTL: 24h. Hay 2 opciones de cleanup, elegí UNA al final del script.
--
-- IMPORTANTE:
--   Si volvés a correr este script sobre un bucket que ya existe,
--   los `INSERT INTO buckets` darán error y `CREATE POLICY` también.
--   Eso está bien — significa que ya está configurado. Ignoralos.
-- =====================================================================

-- ----------------------------------------------------------------------
-- 1. Crear el bucket público "diagram-shares"
-- ----------------------------------------------------------------------
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'diagram-shares',
    'diagram-shares',
    true,                                            -- público (read sin auth)
    5242880,                                         -- 5 MB max por archivo
    ARRAY['image/png', 'image/svg+xml', 'image/jpeg']
)
ON CONFLICT (id) DO UPDATE
    SET public = EXCLUDED.public,
        file_size_limit = EXCLUDED.file_size_limit,
        allowed_mime_types = EXCLUDED.allowed_mime_types;


-- ----------------------------------------------------------------------
-- 2. Policy: cualquiera (incluido anon) puede LEER objetos del bucket
-- ----------------------------------------------------------------------
DROP POLICY IF EXISTS "diagram_shares_public_read" ON storage.objects;
CREATE POLICY "diagram_shares_public_read"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'diagram-shares');


-- ----------------------------------------------------------------------
-- 3. Policy: anon puede INSERTAR objetos en el bucket diagram-shares
-- ----------------------------------------------------------------------
DROP POLICY IF EXISTS "diagram_shares_anon_insert" ON storage.objects;
CREATE POLICY "diagram_shares_anon_insert"
    ON storage.objects FOR INSERT
    WITH CHECK (bucket_id = 'diagram-shares');


-- ----------------------------------------------------------------------
-- 4. Policy: el dueño del objeto puede borrarlo (cleanup desde la app)
-- ----------------------------------------------------------------------
DROP POLICY IF EXISTS "diagram_shares_owner_delete" ON storage.objects;
CREATE POLICY "diagram_shares_owner_delete"
    ON storage.objects FOR DELETE
    USING (bucket_id = 'diagram-shares');


-- ----------------------------------------------------------------------
-- 5. CLEANUP DE ARCHIVOS > 24h — Elegí UNA opción y ejecutala:
-- ----------------------------------------------------------------------

-- OPCIÓN A (recomendada): cron job nativo de Supabase (pg_cron).
-- Requiere que `pg_cron` esté habilitado en Database → Extensions.
-- Borra archivos del bucket con más de 24h cada hora en :00.
--
-- CREATE EXTENSION IF NOT EXISTS pg_cron;
-- SELECT cron.schedule(
--     'cleanup-diagram-shares-hourly',
--     '0 * * * *',
--     $$
--     DELETE FROM storage.objects
--     WHERE bucket_id = 'diagram-shares'
--       AND created_at < (now() - interval '24 hours');
--     $$
-- );

-- OPCIÓN B: cleanup manual. Corré este DELETE cada cuanto te acuerdes.
--
-- DELETE FROM storage.objects
-- WHERE bucket_id = 'diagram-shares'
--   AND created_at < (now() - interval '24 hours');


-- ----------------------------------------------------------------------
-- 6. Verificación rápida
-- ----------------------------------------------------------------------
SELECT id, name, public, file_size_limit, allowed_mime_types
FROM storage.buckets
WHERE id = 'diagram-shares';

SELECT polname, polcmd
FROM pg_policy
WHERE polrelid = 'storage.objects'::regclass
  AND polname LIKE 'diagram_shares_%';

-- Debería ver:
-- buckets: 1 fila con public=true, file_size_limit=5242880
-- policies: 3 filas (public_read SELECT, anon_insert INSERT, owner_delete DELETE)
