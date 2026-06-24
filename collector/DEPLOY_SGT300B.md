# Conectar SGT300 B (Parex) a datos en vivo — Guía de despliegue

Replica EXACTA de lo que ya corre en TES1/TES3. El colector es genérico: lee el
mapa Modbus de la instancia y empuja a la API → Supabase. **No hay que tocar
código** (igual que TES3): solo copiar `collector\`, su mapa y su config.

## Datos de SGT300 B

| Qué | Valor |
|-----|-------|
| Servidor de sitio (RDP) | lo hace Ewdes — la clave NO se guarda en repo ni memoria |
| Gateway Bently 3500 (Modbus TCP) | `192.168.1.228` puerto `502` |
| instance_id | `turbina_sgt300_b` |
| Mapa Modbus | `data/modbus_maps/sgt300b.json` (en el repo) |
| Config colector | `collector/wm_collector.config.sgt300b.example.json` |
| host_label | `SGT300B-WINSRV` |
| Conectividad ya verificada | server `192.168.1.102` → gateway `192.168.1.228:502` = True |

> Unidades Parex SGT300: proximidad `µm pp`, velocidad `mm/s rms`, aceleración
> `g rms` (ya quedaron así en `sgt300b.json`).

## Pasos

### 0. Pre-requisito en Watermelon (1 vez)
- El activo **SGT300 B** ya existe en Machinery Library (`turbina_sgt300_b`).
  Verificá que los `sensor_label` de sus Sondas coincidan con los del mapa:
  `1Y_D 1X_D 2Y_D 2X_D` (turbina), `3Y_V 3Y_A 4Y_V 4Y_A 4X_V 4X_A 4Y_D 4X_D`
  (gearbox), `5Y_D 5X_D 6Y_D 6X_D` (generador). Si no calzan, las lecturas
  llegan a Supabase pero no se mapean a la sonda → editá las Sondas del activo.

### 1. Entrar al servidor (RDP)  ← lo hace Ewdes
- El colector debe correr en una máquina que **alcance el gateway
  192.168.1.228** (la red de instrumentación). El server del sitio ya lo alcanza.

### 2. Verificar conectividad (PowerShell como Administrador)
```powershell
Test-NetConnection 192.168.1.228 -Port 502                        # Gateway Bently → True (ya confirmado)
Test-NetConnection watermelon-api-bpv4.onrender.com -Port 443     # Internet/API
python --version                                                  # 3.10+
```

### 3. Copiar el colector al servidor
- Copiá la carpeta `collector\` del repo a `C:\watermelon\collector\`.
- Copiá `data\modbus_maps\sgt300b.json` a `C:\watermelon\collector\modbus_maps\sgt300b.json`.
- Copiá `collector\wm_collector.config.sgt300b.example.json` y renombralo a
  `C:\watermelon\collector\wm_collector.config.json`.

### 3b. Dependencias
```powershell
cd C:\watermelon\collector
pip install -r requirements.txt
```
> El colector funciona con cualquier pymodbus 3.6+ (detecta solo `slave` vs
> `device_id`). Si ya instalaste `pymodbus 3.13`, no hay que desinstalar nada.

### 4. Probar SIN enviar (dry-run) — valida Modbus, byte order y escalado
```powershell
cd C:\watermelon\collector
python wm_collector.py --config wm_collector.config.json --dry-run
```
- Mirá los valores: velocidad ~3600–14000 rpm a régimen, gaps de proximidad
  razonables (V DC negativos típicos), Direct de proximidad en µm.
- Si ves números absurdos (1e-30 / 1e30) → cambiá `byte_order` en
  `sgt300b.json` de `ABCD` a `CDAB` y reintentá (lo más común).

### 5. Instalar como servicio Windows (auto-start + auto-restart)
```powershell
cd C:\watermelon\collector
.\install_windows.ps1 `
    -CollectorScript .\wm_collector.py `
    -ConfigFile .\wm_collector.config.json `
    -ModbusMap .\modbus_maps\sgt300b.json `
    -NssmPath "C:\nssm\nssm-2.24\win64\nssm.exe"
```
> Si el servicio `WatermelonCollector` YA existe en este servidor por otra
> instancia, instalalo con OTRO nombre de servicio o usá una segunda carpeta.

### 6. Verificar que fluye
```powershell
Get-Content C:\watermelon\collector\logs\wm_collector.log -Wait -Tail 50
```
- Debe loguear POST OK cada ~10 s.
- En Watermelon → **Live Monitoring** → **SGT300 B** → deben entrar lecturas en
  vivo (gauge de salud, tendencia, tabla 1X/2X).

## SGT300 A
Mismo procedimiento: cuando tengas su **gateway IP** y su **mapa de registros**,
creamos `data/modbus_maps/sgt300a.json` + `wm_collector.config.sgt300a.example.json`
y se despliega igual.
