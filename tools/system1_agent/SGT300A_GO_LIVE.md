# SGT300A → Live Monitoring (clon EXACTO de SGT300B)

Arquitectura REAL verificada contra B (host de B, 2026-10-15). **2 máquinas /
2 tareas**. La VM exporta los CSV; el HOST (único con internet) lee el share de
la VM por UNC y sube a Supabase.

```
VM de A (System1)   WM_export  →  s1_rpa_export.py --expall C:\WM_wave\csv
  (sin internet)                  (+ caja/gearbox a C:\WM_wave\csv_gb)
                                  comparte C: (admin share C$)
HOST de A (internet) WM_upload  →  C:\watermelon\upload.bat
                                  cd C:\watermelon\wm\tools\system1_agent
                                  python s1_agent.py --csv --once
                                  python s1_agent.py --gearbox --once
                                  (lee \\<IP-VM>\C$\WM_wave\csv  por UNC)
```

## Lo único que cambia vs B
| cosa | B | A |
|---|---|---|
| `asset` | `turbina_sgt300_b` | `siemens_sgt300_gearbox_brush_turbogen` |
| UNC `folder` | `\\192.168.192.130\C$\WM_wave\csv` | `\\192.168.192.172\C$\WM_wave\csv` |
| UNC `gearbox_folder` | `\\192.168.192.130\C$\WM_wave\csv_gb` | `\\192.168.192.172\C$\WM_wave\csv_gb` |
| IPs host/gateway/rack | — | host 192.168.192.223 / gw 192.168.1.163 / rack 192.168.1.17 |

VM de A = `192.168.192.172` (ZeroTier). Todo lo demás (8 puntos RPA, plot_xy, backend SQL, service_key) = IGUAL que B.

## Estado
- **Nube: LISTA** — activo `siemens_sgt300_gearbox_brush_turbogen` (tag "SGT300A"),
  in_service, Parex, bucket `dynamic_raw`. Nada que crear.
- **Host A: VACÍO** (instalación de cero).
- **VM de A: System1 ya toma datos** (rack 192.168.1.17). Falta la hoja de export
  "Export csv1" + la tarea WM_export.

## A) En la VM de SGT300A (donde está System1) — hace el EXPORT
1. Copiar el repo `wm\` (o solo `tools\system1_agent\`) a la VM.
2. Python 3.11+ (Add to PATH) → `pip install -r requirements.txt`.
3. En System1: armar la hoja **"Export csv1"** = parrilla **4×2** con las 8 ondas
   (1xd,1yd,2xd,2yd,5xd,5yd,6xd,6yd) — IGUAL que B. Confirmar tags:
   `python s1_rpa_export.py --inspect`.
4. Crear `C:\WM_wave\csv` y `C:\WM_wave\csv_gb`. El host los lee por el admin
   share `C$` (ya existe) → no hace falta compartir manual; sí abrir el firewall
   SMB entre VM↔host en la red ZeroTier.
5. `C:\WM_wave\export.bat` = `python <ruta>\s1_rpa_export.py --expall C:\WM_wave\csv`.
6. Task Scheduler (VM) → **WM_export** cada hora (sesión interactiva: trae
   System1 al frente aunque haya otra ventana).
7. Probar: `python s1_rpa_export.py --expall C:\WM_wave\csv` → 8 CSV.

## B) En el HOST de SGT300A (internet) — hace el UPLOAD
1. Clonar el repo a `C:\watermelon\wm` (igual que B).
2. Python 3.11+ → `pip install -r requirements.txt`.
3. `C:\watermelon\wm\tools\system1_agent\config.toml` (desde
   `config.sgt300a.example.toml`):
   - `asset = "siemens_sgt300_gearbox_brush_turbogen"`.
   - `[supabase] service_key` (o env `SUPABASE_SERVICE_KEY`).
   - `[system1_csv] folder = '\\192.168.192.172\C$\WM_wave\csv'`
   - `[system1_csv] gearbox_folder = '\\192.168.192.172\C$\WM_wave\csv_gb'`
4. `C:\watermelon\upload.bat` (idéntico a B):
   ```
   cd /d C:\watermelon\wm\tools\system1_agent
   python s1_agent.py --csv --once     >> C:\watermelon\upload.log 2>&1
   python s1_agent.py --gearbox --once >> C:\watermelon\upload.log 2>&1
   ```
5. Task Scheduler (host) → **WM_upload** cada hora → corre `upload.bat`.
6. Probar: `python s1_agent.py --csv --once` → sube los CSV del share de la VM.
7. Abrir Watermelon Live → **SGT300A** → *Dynamic analysis* → onda/espectro/órbita.

## export.bat de B (VERIFICADO en la VM de B, 2026-10-15)
Agente de B en `C:\Watermelon\agent`. `C:\WM_wave\export.bat`:
```bat
cd /d C:\Watermelon\agent
python s1_rpa_export.py --expall C:\WM_wave\csv --sheet "Export csv1" --sheet-xy 272 165 > C:\WM_wave\export.log 2>&1
python s1_rpa_export.py --expall C:\WM_wave\csv_gb --sheet GearBox --sheet-xy 445 165 --names 4xa,4ya,3ya,4yd,4xd,4xv,4yv,3yv >> C:\WM_wave\export.log 2>&1
```
- 8 canales gearbox (orden parrilla 4×2): `4xa,4ya,3ya,4yd,4xd,4xv,4yv,3yv`
- Tabs: "Export csv1" @ (272,165) · "GearBox" @ (445,165)
- Tiles 4×2 = DEFAULT_TILES (384/742/1100/1458 × 405/680), calibrados a frame VM 1740×882.
- Para A: MISMO .bat; solo cambia la ruta del agente. ⚠️ Si la VM de A tiene
  otra resolución, recalibrar sheet-xy + tiles con `--inspect`/`--grab`/`--pick`.

## HOST A — HECHO (2026-10-15)
Agente en `C:\watermelon\wm\tools\system1_agent` (s1_agent.py + s1_csv.py + config.toml).
service_key en env `SUPABASE_SERVICE_KEY` (setx, usuario backup1). UNC a la VM
autenticado con `cmdkey /add:192.168.192.172 /user:192.168.192.172\Administrator`.
`C:\watermelon\upload.bat` (csv + gearbox) + tarea **WM_upload** horaria (backup1).
Internet IPv4 arreglado: `route -p add 0.0.0.0 mask 0.0.0.0 192.168.1.1 metric 1`
(Starlink daba solo IPv6; ZeroTier full-tunnel tragaba el IPv4 → supabase es IPv4-only).
Probado: WM_upload Last Result 0, lee share, conecta Supabase. Falta: CSV desde la VM.

## COLECTOR MODBUS A (valores estáticos → Live Monitoring) — HECHO (2026-10-15)
Alimenta Live Monitoring (overalls/RPM → online + alarmas). Es el feed PRINCIPAL.
Corre en el **HOST de A** (no la VM): host alcanza el gateway 192.168.1.163:502
(Ethernet4, local) **y** la API. La VM también podía, pero el host da resiliencia
(sobrevive reinicio de la VM). En `C:\watermelon\collector\`:
- `wm_collector.py` + `wm_collector.config.json` (asset `siemens_sgt300_gearbox_brush_turbogen`,
  api `watermelon-api-bpv4.onrender.com`) + `modbus_maps\sgt300a.json` (gateway 192.168.1.163:502, 72 regs).
- deps: `pymodbus==3.6.9 requests` (python de backup1).
- **Servicio nssm** `WatermelonCollectorSGT300A` (nssm-2.24\win64), AUTO_START + auto-restart,
  AppDirectory C:\watermelon\collector, logs en `logs_sgt300a\svc_out.log`.
- Verificado: SERVICE_RUNNING, `Cycle read 72/72 good` + `POST OK ingested=72` c/10s.
- ⚠️ Red: host→rack dio False al inicio y luego True (Ethernet4 192.168.1.101). Starlink
  del host e instrumentación son AMBAS 192.168.1.0/24 (choque de subred) — vigilar estabilidad.

## ONDA CRUDA SGT300A (Dynamic analysis) — HECHO (2026-10-16)
VM A: Python 3.12 + RPA en `C:\Watermelon\agent` (s1_rpa_export.py + s1_csv.py) +
pywinauto/opencv/pywin32. Calibrado a **1718×839**:
- `config.toml` (bearings): tiles 343/733/1113/1493 × 295/595, sheet_xy [357,116] (Export CSV1).
- `config_gb.toml` (gearbox): mismas tiles (7, falta 4YA cableándose), sheet_xy [470,116] (Export CSV2).
- ⚠️ Gearbox va a `C:\WM_wave\csvgb` **sin underscore** (el `_` congela el Save As, ver [[system1-rpa-saveas-underscore]]). Bearings a `csv`.
- `C:\WM_wave\export.bat` (CSV1 + gearbox) + tarea **WM_export** horaria (interactiva, Administrator).
- Verificado E2E: export 8 bearings + 7 gearbox → host WM_upload sube BRG1/2/5/6 + GB (6) → Supabase HTTP200.

## SUPERVIVENCIA A REINICIO (2026-10-16)
- **Live Monitoring (colector + upload): 100% AUTO.** Colector = servicio nssm AUTO_START (host).
  WM_upload = tarea "Interactive/Background" con clave (host). Vuelven solos tras reinicio.
- **Onda cruda: casi auto.** VM = **auto-login Administrator** (Sysinternals Autologon, cifrado;
  DisableCAD=1) + **System1.lnk en Startup** (abre solo). PERO System1 **no selecciona la base
  Parex solo** (limitación System1) → tras reinicio, 1 toque manual: abrir base Parex + dejar
  Export CSV1/CSV2. Luego WM_export + upload siguen solos. **Recomendación: VM prendida 24/7.**

## Lo irreducible (manual on-site, igual que B)
- Iniciar sesión en la VM y dejar la hoja "Export csv1" armada.
- Firewall SMB VM↔host en ZeroTier (para que el UNC `\\IP-VM\C$` resuelva).
- Confirmar los tags con `--inspect` (deben ser iguales a B).

## Verificación final
- Supabase Storage → bucket `dynamic_raw` → archivos bajo
  `siemens_sgt300_gearbox_brush_turbogen`.
- Live Monitoring: SGT300A deja de estar "fuera de línea".
