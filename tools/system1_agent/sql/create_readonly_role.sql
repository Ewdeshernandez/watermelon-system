-- =====================================================================
-- Watermelon · login READ-ONLY para el agente System1 (SQL Server)
-- Backend de System1 = SQL Server (MSSQLSERVER), base BNC_Databases.
-- Producción normal usa Windows Auth (sin password) porque el agente corre
-- en el mismo server. Este login SQL es OPCIONAL, para endurecer.
-- Correr como sysadmin (sqlcmd -S localhost -E -i create_readonly_role.sql).
-- Reemplazar 'CAMBIA_ESTA_CLAVE' por una clave fuerte.
-- =====================================================================

-- 1) Login a nivel servidor
IF NOT EXISTS (SELECT 1 FROM sys.server_principals WHERE name = 'watermelon_ro')
    CREATE LOGIN watermelon_ro WITH PASSWORD = 'CAMBIA_ESTA_CLAVE',
        CHECK_POLICY = ON;
GO

-- 2) Usuario dentro de BNC_Databases con solo lectura
USE BNC_Databases;
GO
IF NOT EXISTS (SELECT 1 FROM sys.database_principals WHERE name = 'watermelon_ro')
    CREATE USER watermelon_ro FOR LOGIN watermelon_ro;
GO
ALTER ROLE db_datareader ADD MEMBER watermelon_ro;   -- SELECT en todas las tablas
GO

-- Verificación:
--   SELECT name, type_desc FROM sys.database_principals WHERE name='watermelon_ro';
-- En config.toml: trusted = false, user = "watermelon_ro", password = "..."
