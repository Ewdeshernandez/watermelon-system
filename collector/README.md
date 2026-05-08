# Watermelon Collector — Tier 0 A

Servicio de ingestión de datos en tiempo real desde gateways industriales
(Bently Nevada 3500/92 Modbus TCP por ahora, OPC UA y MQTT en versiones
futuras) hacia el API de Watermelon System.

## Arquitectura

```
[PC en planta + ZeroTier]                [Render]              [Supabase]
   ┌─────────────────┐                ┌────────┐          ┌──────────┐
   │  wm_collector   │  HTTPS POST    │  API   │  INSERT  │ live_    │
   │  (este script)  │ ─────────────▶ │ /v1/   │ ───────▶ │ readings │
   └────────┬────────┘  cada 10s      │ ingest │          │  (table) │
            │ Modbus TCP              └────────┘          └──────────┘
            ▼
   ┌─────────────────┐
   │ Bently 3500/92  │
   │  192.168.1.77   │
   └─────────────────┘
```

## Instalación rápida (Windows Server)

### Prerequisitos
1. **Python 3.10+** instalado y en PATH ([python.org](https://www.python.org/downloads/))
2. **NSSM** descargado: https://nssm.cc/release/nssm-2.24.zip — descomprimir en `C:\nssm`
3. PowerShell ejecutado como **Administrador**

### Pasos

```powershell
# 1. Editar wm_collector.config.example.json y guardarlo como wm_collector.config.json
#    Cambiá api_key, paths, host_label.

# 2. Correr el instalador
cd C:\path\to\collector
.\install_windows.ps1 `
    -CollectorScript .\wm_collector.py `
    -ConfigFile .\wm_collector.config.json `
    -ModbusMap .\tes1.json `
    -NssmPath "C:\nssm\nssm-2.24\win64\nssm.exe"
```

El instalador:
- Crea `C:\watermelon\collector\`
- Instala dependencias en venv aislado
- Registra el servicio Windows `WatermelonCollector` con auto-start + auto-restart
- Hace un dry-run de validación antes de arrancar
- Arranca el servicio

## Instalación manual (Linux / dev)

```bash
cd collector
pip install -r requirements.txt
python wm_collector.py --config wm_collector.config.json
```

Para probar **sin enviar datos** (solo lee Modbus y muestra payload):

```bash
python wm_collector.py --config wm_collector.config.json --dry-run
```

## Operación

### Ver logs en vivo

```powershell
Get-Content C:\watermelon\collector\logs\wm_collector.log -Wait -Tail 50
```

### Detener / arrancar / reiniciar

```powershell
& C:\nssm\nssm-2.24\win64\nssm.exe stop    WatermelonCollector
& C:\nssm\nssm-2.24\win64\nssm.exe start   WatermelonCollector
& C:\nssm\nssm-2.24\win64\nssm.exe restart WatermelonCollector
```

### Ver estado

```powershell
Get-Service WatermelonCollector
```

## Configuración

`wm_collector.config.json`:

| Campo | Descripción |
|-------|-------------|
| `api_url` | URL base del API. Default `https://watermelon-api-bpv4.onrender.com` |
| `api_key` | API key para Bearer auth. Pedila al admin. |
| `modbus_map` | Path al JSON con la traducción de Modbus addresses → variables. |
| `buffer_db` | SQLite local para batches no enviados (resiliencia). |
| `log_dir` | Carpeta de logs rotativos. |
| `host_label` | Etiqueta del PC (aparece en metadata, ayuda a debug). |
| `api_timeout_sec` | Timeout de cada POST. Default 15s. |

## Modbus Map

Los mapas viven en `data/modbus_maps/<instance_id>.json`. Estructura:

```json
{
  "instance_id": "tes1",
  "modbus": {
    "server_ip": "192.168.1.77",
    "port": 502,
    "byte_order": "ABCD"
  },
  "poll_interval_seconds": 10,
  "registers": {
    "6041": {
      "kind": "speed",
      "variable": "Velocidad Generador",
      "metric": "Direct",
      "unit": "rpm",
      "encoding": "float32"
    },
    "6031": {
      "kind": "sensor",
      "sensor_label": "1Y_V",
      "variable": "1YV VEL CRF",
      "metric": "Direct",
      "unit": "in/s pk"
    }
    ...
  }
}
```

### Byte order

El 3500/92 normalmente usa `ABCD` (big-endian). Si los valores que ves son
absurdos (e.g. 1e-30 o 1e30), probá `CDAB` (word-swapped). El collector
soporta los 4 órdenes: `ABCD`, `CDAB`, `BADC`, `DCBA`.

### Validación

Después de instalar, comparar los valores del Live Monitoring de Watermelon
contra `https://watermelonsys.net/monitoreo-estatico` (sistema legacy). Si
calzan ±0.5%, byte order y scaling están correctos.

## Troubleshooting

**Servicio no arranca:**
- `C:\watermelon\collector\logs\nssm_stderr.log` → errores de Python
- `C:\watermelon\collector\logs\wm_collector.log` → errores del script

**No conecta al Modbus:**
- Verificar que el PC puede pinguear `192.168.1.77`
- Verificar que el puerto 502 está abierto: `Test-NetConnection 192.168.1.77 -Port 502`
- Verificar que la tarjeta 3500/92 está habilitada para Modbus TCP

**No conecta al API:**
- Verificar que el PC tiene internet: `Test-NetConnection watermelon-api-bpv4.onrender.com -Port 443`
- Verificar API key con curl:
  ```
  curl -H "Authorization: Bearer YOUR_KEY" https://watermelon-api-bpv4.onrender.com/v1/health
  ```

## Versiones

- **v1.0.0** (Ciclo 23.1) — primera versión productiva. Modbus TCP, buffer SQLite, Windows service.
