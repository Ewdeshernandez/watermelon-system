-- =====================================================================
-- Watermelon · usuario READ-ONLY para el agente System1
-- Correr como superusuario (postgres) en el server Parex.
-- Reemplazar  <DB_SYSTEM1>  por el nombre real (lo muestra --discover)
-- y  CAMBIA_ESTA_CLAVE  por una clave fuerte.
-- El agente SOLO hace SELECT: este rol no puede escribir ni borrar.
-- =====================================================================

CREATE ROLE watermelon_ro LOGIN PASSWORD 'CAMBIA_ESTA_CLAVE';

GRANT CONNECT ON DATABASE "<DB_SYSTEM1>" TO watermelon_ro;

-- Conectarse a la DB de System1 antes de los GRANT de esquema:
\connect "<DB_SYSTEM1>"

-- System1 puede usar el esquema public u otro (lo dice --discover).
-- Ajustar el nombre de esquema si no es public:
GRANT USAGE ON SCHEMA public TO watermelon_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO watermelon_ro;

-- Que las tablas futuras también sean legibles por el agente:
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO watermelon_ro;

-- Verificación:
--   \du watermelon_ro
--   SET ROLE watermelon_ro; SELECT current_user;  RESET ROLE;
