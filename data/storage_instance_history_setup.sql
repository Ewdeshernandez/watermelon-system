-- =====================================================================
-- Watermelon System — Setup del bucket instance-history (v3.31.76)
-- =====================================================================
-- Corré este SQL UNA SOLA VEZ en Supabase Dashboard → SQL Editor.
-- Crea el bucket público + policies para que el sistema pueda persistir
-- snapshots históricos de análisis (waveforms, spectrums, orbits, scl,
-- polar, bode, trend, tabular) y mostrarlos en Live Monitoring para
-- consumo del cliente.
--
-- Modelo de seguridad:
--   • Bucket: público en LECTURA (cualquiera con el link ve la imagen).
--     Esto está bien porque los paths son no-enumerables vía API REST
--     pública: incluyen instance_id + snapshot_id timestamped.
--   • Bucket: INSERT permitido con anon key (la app autentica al
--     usuario antes de cualquier write).
--   • Bucket: DELETE permitido con anon key (rotación LRU automática
--     cuando se excede el límite de 10 snapshots por tipo).
--   • Archivos: limit 5 MB (JSON gzipped — un waveform downsampleado
--     a 16k samples ocupa <500 KB después de compresión).
--
-- Layout en el bucket:
--   instance-history/
--     {instance_id}/
--       {snapshot_type}/        ← scl, polar, bode, trend, waveform,
--                                 spectrum, orbit, tabular
--         {snapshot_id}.json.gz ← snapshot_id codifica el timestamp
--                                 (ej. scl_20260511_153022.json.gz)
--
-- Naming convention asegura sorting lexicográfico = sorting cronológico.
-- Rotación LRU se hace listando objects en el prefijo y borrando el más
-- viejo (sorted asc) cuando el count > 10.
--
-- IMPORTANTE:
--   Si volvés a correr este script sobre un bucket que ya existe,
--   los INSERT INTO buckets darán conflict y CREATE POLICY error.
--   El ON CONFLICT y DROP POLICY IF EXISTS lo hacen idempotente.
-- =====================================================================

-- ----------------------------------------------------------------------
-- 1. Crear el bucket público "instance-history"
-- ----------------------------------------------------------------------
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'instance-history',
    'instance-history',
    true,                                      -- público (read sin auth)
    5242880,                                   -- 5 MB max por archivo
    ARRAY[
        'application/gzip',                    -- JSON gzipped (path principal)
        'application/json',                    -- JSON sin comprimir (fallback)
        'application/x-gzip',                  -- alias
        'application/octet-stream'             -- catch-all para cualquier binario
    ]
)
ON CONFLICT (id) DO UPDATE
    SET public = EXCLUDED.public,
        file_size_limit = EXCLUDED.file_size_limit,
        allowed_mime_types = EXCLUDED.allowed_mime_types;


-- ----------------------------------------------------------------------
-- 2. Policy: cualquiera (incluido anon) puede LEER objetos del bucket
--    El path no es enumerable trivialmente — incluye instance_id +
--    snapshot_id con timestamp. La auth real está en la app.
-- ----------------------------------------------------------------------
DROP POLICY IF EXISTS "instance_history_public_read" ON storage.objects;
CREATE POLICY "instance_history_public_read"
    ON storage.objects FOR SELECT
    USING (bucket_id = 'instance-history');


-- ----------------------------------------------------------------------
-- 3. Policy: anon puede INSERTAR objetos en el bucket
--    El bucket-level file_size_limit + mime_types ya restringen contenido.
-- ----------------------------------------------------------------------
DROP POLICY IF EXISTS "instance_history_anon_insert" ON storage.objects;
CREATE POLICY "instance_history_anon_insert"
    ON storage.objects FOR INSERT
    WITH CHECK (bucket_id = 'instance-history');


-- ----------------------------------------------------------------------
-- 4. Policy: anon puede BORRAR objetos del bucket
--    Necesario para la rotación LRU automática (al insertar el snapshot
--    #11, la app borra el más viejo). NO es problema de seguridad porque
--    la auth real está en la app — el cliente final nunca llega a
--    poder llamar este endpoint directamente.
-- ----------------------------------------------------------------------
DROP POLICY IF EXISTS "instance_history_anon_delete" ON storage.objects;
CREATE POLICY "instance_history_anon_delete"
    ON storage.objects FOR DELETE
    USING (bucket_id = 'instance-history');


-- ----------------------------------------------------------------------
-- 5. (Opcional) BACKUP BUCKET — para v3.31.83 / backup automático semanal
--    Crea ahora el segundo bucket para que esté listo cuando habilitemos
--    pg_cron weekly backup. Si no querés backups todavía, ignorá esto.
-- ----------------------------------------------------------------------
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'instance-history-backups',
    'instance-history-backups',
    false,                                     -- NO público — los backups
                                               -- son archivos sensibles
                                               -- (data completo del cliente)
    524288000,                                 -- 500 MB max por ZIP backup
    ARRAY[
        'application/zip',
        'application/gzip',
        'application/x-tar',
        'application/octet-stream'
    ]
)
ON CONFLICT (id) DO UPDATE
    SET public = EXCLUDED.public,
        file_size_limit = EXCLUDED.file_size_limit,
        allowed_mime_types = EXCLUDED.allowed_mime_types;

-- Policy backup: solo service_role puede acceder (NO anon)
-- Esto se aplica por DEFAULT en buckets privados — no hace falta policy
-- explícita, pero la dejamos comentada para documentación:
--
-- DROP POLICY IF EXISTS "instance_history_backups_service_role" ON storage.objects;
-- CREATE POLICY "instance_history_backups_service_role"
--     ON storage.objects FOR ALL
--     USING (bucket_id = 'instance-history-backups')
--     WITH CHECK (bucket_id = 'instance-history-backups');


-- ----------------------------------------------------------------------
-- 6. Verificación rápida
-- ----------------------------------------------------------------------
SELECT id, name, public, file_size_limit, allowed_mime_types
FROM storage.buckets
WHERE id IN ('instance-history', 'instance-history-backups');

SELECT polname, polcmd
FROM pg_policy
WHERE polrelid = 'storage.objects'::regclass
  AND polname LIKE 'instance_history_%';

-- Esperado:
-- buckets:  2 filas (instance-history público + instance-history-backups privado)
-- policies: 3 filas (public_read SELECT, anon_insert INSERT, anon_delete DELETE)
