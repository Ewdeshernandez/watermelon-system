================================================================================
  WATERMELON PLANTA — DRIVER DEL EQUIPO (para paquete de venta COMPLETO)
================================================================================

Para que el instalador sea TURNKEY (máquina nueva, sin nada preinstalado,
sin descargar nada de internet), el runtime del controlador del equipo de
adquisición debe quedar EMPACADO dentro del instalador.

Coloca aquí el instalador silencioso del driver, así:

    dependencies/
      driver-extracted/
        Install.exe        <-- el instalador del runtime del fabricante
        ... (archivos de soporte que acompañan a Install.exe)

El installer.iss lo recoge automáticamente (Source: dependencies\driver-extracted\*)
y lo ejecuta en silencio durante la instalación (/qb /norestart /ACCEPTEULAS),
SIN dejarlo en el disco del cliente (deleteafterinstall).

--------------------------------------------------------------------------------
  IMPORTANTE
--------------------------------------------------------------------------------
- Usa el RUNTIME del driver (no el paquete de desarrollo completo): es más
  liviano y es lo único que la app necesita para hablar con el equipo.
- Respeta los términos de redistribución del fabricante del runtime.
- build_installer.bat ahora AVISA y se detiene si esta carpeta falta, para
  no generar nunca un instalador de venta incompleto por accidente.
- Esta carpeta está gitignored (el binario es grande). Vive solo en el PC de
  build de SIGA.

================================================================================
