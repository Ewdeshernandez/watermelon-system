# SGT300A → Live Monitoring (clonar el setup de SGT300B)

Replica EXACTA de B. Arquitectura REAL de producción (la VM no tiene internet,
el host sí) → **2 tareas** que comparten `C:\WM_wave\csv`:

```
VM (System1)   WM_export  →  C:\WM_wave\export.bat
                            = python s1_rpa_export.py --expall C:\WM_wave\csv
                            (activa la hoja "Export csv1" 4×2 y exporta las 8 ondas)
                            C:\WM_wave\csv  = carpeta COMPARTIDA (share)
HOST (internet) WM_upload  →  C:\Watermelon\upload.bat
                            = python s1_agent.py --csv --once
                            (lee el share de la VM y sube a Supabase)
```

## Estado
- **Nube: LISTA** — activo `siemens_sgt300_gearbox_brush_turbogen` (tag "SGT300A"),
  in_service, Parex, bucket `dynamic_raw`. No hay nada que crear.
- **Host A: VACÍO** (instalación de cero). La VM ya tiene System1 tomando datos.
- Solo cambian vs B: el **asset** (arriba) y los **IP** (ZeroTier host A
  192.168.192.223 / gateway 192.168.1.163 / rack TDI 192.168.1.17).

## A) En la VM de SGT300A (donde está System1)
1. Copiar `system1_agent\` a la VM (ej. `C:\WM_wave\system1_agent\`).
2. Python 3.11+ (Add to PATH) → `pip install -r requirements.txt`.
3. En System1: crear/abrir la hoja **"Export csv1"** = parrilla **4×2** con las 8
   ondas (1xd,1yd,2xd,2yd,5xd,5yd,6xd,6yd) — IGUAL que B. Confirmar tags:
   `python s1_rpa_export.py --inspect`.
4. Crear carpeta `C:\WM_wave\csv` y **compartirla** (lectura para el host).
5. `C:\WM_wave\export.bat` = `python <ruta>\s1_rpa_export.py --expall C:\WM_wave\csv`.
6. Task Scheduler (en la VM) → tarea **WM_export** cada hora → corre `export.bat`
   (en sesión interactiva; trae System1 al frente aunque haya otra ventana).
7. Probar: `python s1_rpa_export.py --expall C:\WM_wave\csv` → deben salir 8 CSV.

## B) En el HOST de SGT300A (internet)
1. Copiar `system1_agent\` a `C:\Watermelon\system1_agent\`.
2. Python 3.11+ → `pip install -r requirements.txt`.
3. `config.toml` (desde `config.sgt300a.example.toml`):
   - `asset = siemens_sgt300_gearbox_brush_turbogen`.
   - `[supabase] service_key` (o env `SUPABASE_SERVICE_KEY`).
   - `[system1_csv] folder` = la ruta del **share de la VM** (ej. `\\VM\WM_wave\csv`
     o la letra mapeada).
4. `C:\Watermelon\upload.bat` = `python C:\Watermelon\system1_agent\s1_agent.py --csv --once`.
5. Task Scheduler (en el host) → tarea **WM_upload** cada hora → corre `upload.bat`.
6. Probar: `python s1_agent.py --csv --once` → sube los CSV del share.
7. Abrir Watermelon Live → **SGT300A** → *Dynamic analysis* → onda/espectro/órbita.

## Lo irreducible (manual on-site, igual que B)
- Iniciar sesión en la VM (contraseña) y dejar la hoja "Export csv1" armada.
- Compartir `C:\WM_wave\csv` entre VM↔host y mapearla.
- Confirmar los tags con `--inspect` (deben ser iguales a B).

## Verificación final
- Supabase Storage → bucket `dynamic_raw` → archivos bajo
  `siemens_sgt300_gearbox_brush_turbogen`.
- Live Monitoring: SGT300A deja de estar "fuera de línea".
