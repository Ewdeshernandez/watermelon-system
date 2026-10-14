# SGT300A → Live Monitoring (replicar lo de SGT300B)

Igual que SGT300B, el robot corre **en la VM de System1 de la turbina A** y sube
la onda cruda cada hora al Watermelon Cloud. SGT300A **ya existe** como activo:
`siemens_sgt300_gearbox_brush_turbogen` (tag web "SGT300A") → no hay que crearlo.

## Red (ya conectado)
- Host turbina A: **192.168.192.223** (usuario `backup1`) — la contraseña NO va aquí.
- Gateway: **192.168.1.163** · Rack TDI 3500: **192.168.1.17** (de ahí toma System1).
- La **VM** con System1 corre dentro de ese host. El agente va **dentro de la VM**.
- La VM necesita salida a internet para `*.supabase.co`.

## Pasos en la VM de SGT300A (≈20 min) — sigue INSTALL_NUEVA_VM.md con estos cambios
1. Copiar `system1_agent\` a `C:\Watermelon\system1_agent\`.
2. Python 3.11+ (Add to PATH) → `pip install -r requirements.txt`.
3. Config: copiar **`config.sgt300a.example.toml` → `config.toml`** (ya trae
   `asset = siemens_sgt300_gearbox_brush_turbogen` y los 8 puntos).
   - Poner `[supabase] service_key` (o env `SUPABASE_SERVICE_KEY`).
   - Ajustar `folder`/`out_folder` al usuario real del host (ej. `backup1`).
4. **Confirmar los tags** del árbol de System1 de A (pueden diferir en texto):
   ```
   python s1_rpa_export.py --inspect
   ```
   - Ajustar en `config.toml → [rpa]`: `plot_xy` (centro de la gráfica) y el
     `tag` EXACTO de cada punto (1xd,1yd,2xd,2yd,5xd,5yd,6xd,6yd).
5. Probar el ciclo:
   ```
   python s1_rpa_export.py --run
   python s1_agent.py --csv --once
   ```
   Abrir Watermelon Live → **SGT300A** → *Dynamic analysis* → onda/espectro/órbita.
6. Agendar cada hora: Task Scheduler → Import `WatermelonS1Agent.xml`
   (editar ruta si no es `C:\Watermelon\system1_agent\`).

## Lo único que puede diferir de B
- El **texto de los tags** en el árbol de System1 de A (paso 4). La estructura de
  sensores de A es la MISMA que B (4 cojinetes X/Y), así que son los mismos 8
  puntos; solo hay que confirmar cómo están rotulados en A.
- La **carpeta/usuario** del host (B usaba `Administrator`; A es `backup1`).

## Verificación final
- En Supabase Storage, bucket `dynamic_raw`, deben aparecer archivos bajo el
  asset `siemens_sgt300_gearbox_brush_turbogen`.
- En Live Monitoring, SGT300A deja de estar "fuera de línea" cuando llegan datos.
