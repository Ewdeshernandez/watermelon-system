-- =============================================================
-- Watermelon System — Supabase schema (Ciclo 9)
-- =============================================================
-- Pegá el contenido completo de este archivo en SQL Editor → New
-- query → Run desde tu dashboard de Supabase. Crea la tabla y los
-- índices necesarios para el storage persistente del Vault.
--
-- Después de ejecutar este SQL, andá a Storage → New bucket →
-- nombre: instance-documents, Public bucket: OFF.
-- =============================================================


-- ----------------------------------------------------------
-- Tabla: instances
-- ----------------------------------------------------------
-- Cada fila es una máquina física registrada (ej. brush_tes1,
-- siemens_sgt300_planta_b). El campo metadata guarda toda la info
-- de la instancia como JSONB: profile_key, tag, serial_number,
-- location, notes, captured_parameters (dict de parámetros
-- físicos del cojinete y del aceite), documents (lista de docs
-- subidos con sus IDs y storage_filenames).
--
-- El JSONB nos da flexibilidad de schema sin tener que migrar la
-- tabla cada vez que agreguemos un campo a captured_parameters.
-- Si en el futuro queremos queryability (ej. listar todas las
-- instancias con clearance < 0.5 mm), podemos crear índices GIN
-- sobre los campos del JSONB.
-- ----------------------------------------------------------

CREATE TABLE IF NOT EXISTS instances (
    id          text PRIMARY KEY,
    metadata    jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

-- Índice para query rápido por updated_at descendente (lista
-- ordenada por más recientes primero, default del UI)
CREATE INDEX IF NOT EXISTS idx_instances_updated_at
    ON instances (updated_at DESC);

-- Índice GIN sobre metadata para query por profile_key, tag, etc.
-- Útil cuando crezca a decenas/cientos de instancias.
CREATE INDEX IF NOT EXISTS idx_instances_metadata
    ON instances USING gin (metadata);


-- ----------------------------------------------------------
-- Trigger: actualizar updated_at automáticamente en cada UPDATE
-- ----------------------------------------------------------
CREATE OR REPLACE FUNCTION trg_instances_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS instances_updated_at ON instances;
CREATE TRIGGER instances_updated_at
    BEFORE UPDATE ON instances
    FOR EACH ROW
    EXECUTE FUNCTION trg_instances_updated_at();


-- ----------------------------------------------------------
-- Comentarios documentales
-- ----------------------------------------------------------
COMMENT ON TABLE instances IS
'Asset Instances de Watermelon System (una fila por máquina física).
El campo metadata contiene como JSONB el dataclass Instance completo:
profile_key, tag, serial_number, location, notes, captured_parameters,
documents. Storage de los binarios va en bucket instance-documents.';

COMMENT ON COLUMN instances.id IS
'Slug único de la instancia (ej. brush_tes1). Tiene que coincidir con
metadata->>instance_id. Se usa también como prefijo de path en el
bucket de Storage para los documentos asociados.';


-- =============================================================
-- VERIFICACIÓN
-- =============================================================
-- Tras ejecutar este script, podés validar que todo quedó OK con:
--
--   SELECT id, metadata->>'tag' AS tag, metadata->>'profile_key' AS profile,
--          metadata->'documents' AS docs, updated_at
--   FROM instances ORDER BY updated_at DESC;
--
-- Inicialmente la tabla está vacía. Las instancias se crean desde la UI
-- de la app (Asset Documents → "+ Crear nueva instancia") y aparecen
-- automáticamente acá una vez configurado el secret en Streamlit Cloud.
-- =============================================================
