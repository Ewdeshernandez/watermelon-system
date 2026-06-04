# Conectar TES3 a datos en vivo — Guía de despliegue

Replica EXACTA de lo que ya corre en TES1. El colector es genérico: lee el mapa
Modbus de la instancia y empuja a la API → Supabase. No hay que tocar código.

## Datos de TES3

| Qué | Valor |
|-----|-------|
| Servidor de sitio (RDP por ZeroTier) | `192.168.100.104` · usuario `Administrator` |
| Gateway Bently 3500/92 (Modbus TCP) | `192.168.1.70` puerto `502` |
| instance_id | `tes3` |
| Mapa Modbus | `data/modbus_maps/tes3.json` (en el repo) |
| Config colector | `collector/wm_collector.config.tes3.example.json` |
| host_label | `TES3-WINSRV` |

> El servidor está en la subred `192.168.100.x` y el gateway en `192.168.1.x`.
> Antes de instalar, confirmá conectividad (Paso 2). Si no pinguea, falta ruta
> o segunda NIC hacia la red del rack — eso lo resuelve el de planta.

## Pasos

### 0. Pre-requisito en Watermelon (1 vez)
- Asegurate de que el activo **TES3 exista en Machinery Library** (tag `TES3`,
  con su descripción de tren). Si no existe, créalo: sin la instancia, las
  lecturas llegan a Supabase pero el activo no aparece en el selector de Live
  Monitoring.

### 1. Entrar al servidor de TES3 (ZeroTier + RDP)
- Conectá ZeroTier (misma red que usás para TES1).
- Escritorio Remoto (RDP) → `192.168.100.104` → `Administrator` / (la clave que
  me pasaste). **Esa clave NO queda guardada en el repo ni en memoria.**

### 2. Verificar conectividad (PowerShell como Administrador)
```powershell
Test-NetConnection 192.168.1.70 -Port 502        # Gateway Bently → debe dar TcpTestSucceeded : True
Test-NetConnection watermelon-api-bpv4.onrender.com -Port 443   # Internet/API
python --version                                 # 3.10+  (si no, instalar de python.org y tildar "Add to PATH")
```

### 3. Copiar el colector al servidor
- Copiá la carpeta `collector\` del repo a `C:\watermelon\collector\` en el
  servidor (igual que en TES1).
- Copiá `data\modbus_maps\tes3.json` a `C:\watermelon\collector\modbus_maps\tes3.json`.
- Copiá `collector\wm_collector.config.tes3.example.json` y renombralo a
  `C:\watermelon\collector\wm_collector.config.json`.

### 4. Probar SIN enviar (dry-run) — valida Modbus, byte order y scaling
```powershell
cd C:\watermelon\collector
python wm_collector.py --config wm_collector.config.json --dry-run
```
- Mirá los valores en el payload: la velocidad (~3600 rpm a régimen), gaps de
  proximidad razonables (V DC negativos típicos), Direct de proximidad en mil.
- Si ves números absurdos (1e-30 / 1e30) → cambiá `byte_order` en `tes3.json`
  de `ABCD` a `CDAB` y reintentá (es lo más común en estos casos).

### 5. Instalar como servicio Windows (auto-start + auto-restart)
```powershell
cd C:\watermelon\collector
.\install_windows.ps1 `
    -CollectorScript .\wm_collector.py `
    -ConfigFile .\wm_collector.config.json `
    -ModbusMap .\modbus_maps\tes3.json `
    -NssmPath "C:\nssm\nssm-2.24\win64\nssm.exe"
```
Esto registra el servicio `WatermelonCollector` con arranque automático: si el
servidor se reinicia, el colector vuelve solo sin que nadie arranque nada.

> Si el servicio `WatermelonCollector` YA existe en este servidor por otra
> instancia, instalalo con OTRO nombre de servicio (parámetro del instalador) o
> usá una segunda carpeta. En TES3 (servidor dedicado) normalmente está libre.

### 6. Verificar que fluye
```powershell
Get-Content C:\watermelon\collector\logs\wm_collector.log -Wait -Tail 50
```
- Debe loguear POST OK cada ~10 s.
- En Watermelon → **Live Monitoring** → seleccioná **TES3** → deben entrar
  lecturas en vivo (gauge de salud, tendencia, tabla 1X/2X).

### 7. Validar contra el legacy
- Comparar valores del Live Monitoring de TES3 contra el sistema legacy
  (`https://watermelonsys.net/monitoreo-estatico` si TES3 está ahí). Si calzan
  ±0.5 %, byte order + unidades OK.

## ⚠️ Lo único a confirmar: UNIDADES
El mapa quedó con las unidades **espejadas de TES1** (velocidad `in/s pk`,
proximidad `mil pp`). Si en TES3 las proximidades están en **µm** o la velocidad
en **mm/s** (como en algunas máquinas), editá el campo `unit` de cada registro
en `tes3.json`. El dry-run del Paso 4 + la comparación del Paso 7 lo dejan claro.
