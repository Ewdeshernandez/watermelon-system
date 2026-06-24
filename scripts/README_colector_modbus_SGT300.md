# Conectar SGT300 A/B a Live Monitoring (colector Modbus)

Igual que TES1/TES3 quedan "en vivo" con una tarea que empuja datos a Supabase,
SGT300 B se conecta con un **colector Modbus** que lee el gateway del Bently 3500
y empuja a `live_readings`. (SGT300 A: mismo procedimiento con su IP y su mapa.)

```
Bently 3500 → Gateway Modbus TCP (SGT300B: 192.168.1.228:502) → colector → Supabase → watermelonsystem.app
```

## 0) Datos del activo (SGT300 B)
- Gateway Modbus: **192.168.1.228:502**
- Mapa de registros: ya embebido en `wm_modbus_collector_sgt300b.py` (74 señales).
- `INSTANCE_ID`: el de la instancia SGT300B en la app (se ve en la URL de Live
  Monitoring del activo). **Confirmar y poner en el script.**

> El acceso RDP al servidor (192.168.192.206) lo hace Ewdes. El colector debe
> correr en una máquina que **alcance el gateway 192.168.1.228** (la red de
> instrumentación) — típicamente el mismo servidor del cliente.

## 1) Validar lectura primero (sin escribir nada)
En la máquina que alcanza el gateway:
```bash
pip install "pymodbus==3.*" supabase
python wm_modbus_collector_sgt300b.py --dry-run --once
```
Debe imprimir las ~74 lecturas con valores **coherentes** (Direct de proximidad
en µm, velocidades del gearbox en mm/s, RPM ~ velocidad real, etc.).

Si los valores salen **absurdos** (enormes/NaN), es el formato del float:
- En el script, cambiá `WORD_ORDER = "big"` → `"little"` y reintentá.
- Si están "corridos" (cada señal muestra el valor de la de al lado), probá
  `REGISTER_OFFSET = -1`.
- Confirmá `UNIT_ID` (slave Modbus; suele ser 1).
Repetí `--dry-run --once` hasta que los números cuadren con el panel del Bently.

## 2) Ajustar unidades (si hace falta)
En el script, `UNIT_PROX / UNIT_VEL / UNIT_ACCEL` deben coincidir con cómo está
configurado el 3500 (µm pp vs mil pp, mm/s rms, g). Que coincidan con la config
de Sondas del activo en la app para que las alarmas calcen.

## 3) Correr en vivo
```bash
export SUPABASE_URL="https://xxxx.supabase.co"
export SUPABASE_SERVICE_KEY="eyJ..."     # service key (la misma de la app)
python wm_modbus_collector_sgt300b.py
```
Empuja cada `POLL_SECONDS` (10 s por defecto). En la app, Live Monitoring del
SGT300B debe empezar a mostrar datos frescos.

## 4) Dejarlo permanente (como TES3)
**Windows (Programador de tareas):**
- Acción: `python C:\ruta\wm_modbus_collector_sgt300b.py`
- Disparador: al iniciar sesión / al arrancar el equipo; "reiniciar si falla".
- Variables `SUPABASE_URL` y `SUPABASE_SERVICE_KEY` a nivel de sistema (o un
  `.bat` que las exporte antes de llamar al script).

> Tip: poné el `.bat` con `:loop` + `timeout` o dejá que el propio script haga el
> bucle (ya lo hace). Para robustez, "reiniciar la tarea si falla" en el
> Programador.

## 5) App side (una sola vez)
- La instancia **SGT300B** debe existir con su mapa de sensores (planos 1–6 +
  gearbox) y unidades coherentes. Cliente **Parex** ya está en `clients.json`
  con `asset_tags: [SGT300A, SGT300B, C-200-C]`.
- Los `sensor_label` que empuja el colector (1YD, 1XD, 2YD, 2XD, 3YV, 3YA, 4XV,
  4XA, 4YV, 4YA, 4XD, 4YD, 5YD, 5XD, 6YD, 6XD) deben coincidir con los de la
  config de Sondas del activo.

## SGT300 A
Copiá `wm_modbus_collector_sgt300b.py` → `..._sgt300a.py`, cambiá `GATEWAY_IP`,
`INSTANCE_ID` y pegá **el mapa de registros de A** (cuando lo tengas). El resto
es idéntico.
