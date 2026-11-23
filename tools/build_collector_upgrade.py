"""
tools/build_collector_upgrade.py
================================

Genera un script PowerShell AUTOCONTENIDO para actualizar el collector de un
sitio a la versión del repo (collector/wm_collector.py embebido en base64 +
SHA-256). El script NO lleva secretos: reutiliza la config que ya está en el
servidor (api_key incluida) y solo le agrega remote_map / instance_id.

Uso:
    python tools/build_collector_upgrade.py turbina_sgt300_b collector/upgrade_sgt300b.ps1

En el servidor (PowerShell como Administrador):
    powershell -ExecutionPolicy Bypass -File .\\upgrade_sgt300b.ps1
    powershell -ExecutionPolicy Bypass -File .\\upgrade_sgt300b.ps1 -Rollback
"""

from __future__ import annotations

import base64
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TEMPLATE = r'''<#
  Watermelon Collector -> __VERSION__ (mapa Modbus REMOTO desde Machines) · instance __IID__
  Generado por tools/build_collector_upgrade.py — no editar a mano.

  Ejecutar en el servidor del sitio, PowerShell COMO ADMINISTRADOR:
      powershell -ExecutionPolicy Bypass -File .\__NAME__
  Volver atrás (restaura el último respaldo y reinicia el servicio):
      powershell -ExecutionPolicy Bypass -File .\__NAME__ -Rollback
  Si solo difieren AJUSTES del collector (no registros), confirmar con:
      powershell -ExecutionPolicy Bypass -File .\__NAME__ -AcceptSettings

  Qué hace (y aborta + restaura si algo falla):
    1. Verifica que la config/mapa del servidor sean de __IID__
    2. Respalda collector, config y mapa (*.bak_FECHA)
    3. Escribe el collector nuevo aparte (SHA-256 verificado)
    4. Compara el mapa local vs el que entrega la nube — si difieren NO activa
    5. Agrega remote_map=true + instance_id a la config
    6. Prueba en seco con el collector nuevo; si pasa, cambia el servicio y
       revisa que lleguen POST OK
#>
param(
    [switch]$Rollback,
    [switch]$AcceptSettings,
    [string]$Root = 'C:\watermelon\collector',
    [string]$Service = '__SERVICE__',
    [string]$Python = ''
)
$ErrorActionPreference = 'Stop'
$IID = '__IID__'
$SHA = '__SHA__'
$B64 = '__B64__'
$enc = New-Object System.Text.UTF8Encoding $false
function Step($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok($m)   { Write-Host "    OK  $m" -ForegroundColor Green }
function Warn($m) { Write-Host "    !!  $m" -ForegroundColor Yellow }

# Python: el MISMO del servicio (NSSM lo guarda en el registro); si no, venv / PATH.
$py = $Python
if (-not $py) {
    try { $py = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Services\$Service\Parameters" -ErrorAction Stop).Application } catch { $py = '' }
}
if (-not $py -or -not (Test-Path $py)) { $py = Join-Path $Root 'venv\Scripts\python.exe' }
if (-not (Test-Path $py)) { $py = (Get-Command python).Source }
# Config: la que usa el servicio (--config en AppParameters de NSSM); si no, la estándar.
$cfgPath = Join-Path $Root 'wm_collector.config.json'
try {
    $ap = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Services\$Service\Parameters" -ErrorAction Stop).AppParameters
    if ($ap -match '--config\s+"?([^"]+?\.json)"?') { $cfgPath = $Matches[1] }
} catch {}
if (-not [System.IO.Path]::IsPathRooted($cfgPath)) { $cfgPath = Join-Path $Root $cfgPath }
$colPath = Join-Path $Root 'wm_collector.py'
# Log: carpeta log_dir de la config (A usa logs_sgt300a, B usa logs)
$logDir = 'logs'
try { $c0 = Get-Content $cfgPath -Raw | ConvertFrom-Json; if ($c0.log_dir) { $logDir = $c0.log_dir } } catch {}
if (-not [System.IO.Path]::IsPathRooted($logDir)) { $logDir = Join-Path $Root $logDir }
$logPath = Join-Path $logDir 'wm_collector.log'

if ($Rollback) {
    Step 'Rollback: restaurando el último respaldo'
    $bc = Get-ChildItem "$colPath.bak_*" | Sort-Object Name -Descending | Select-Object -First 1
    $bg = Get-ChildItem "$cfgPath.bak_*" | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $bc -or -not $bg) { throw 'No hay respaldos (*.bak_*) en ' + $Root }
    Stop-Service $Service
    Copy-Item $bc.FullName $colPath -Force
    Copy-Item $bg.FullName $cfgPath -Force
    $cfg0 = Get-Content $cfgPath -Raw | ConvertFrom-Json
    $bm = Get-ChildItem "$($cfg0.modbus_map).bak_*" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
    if ($bm) { Copy-Item $bm.FullName $cfg0.modbus_map -Force }
    Start-Service $Service
    Ok "Restaurado $($bc.Name) + $($bg.Name). Servicio reiniciado."
    exit 0
}

Step '1/6 Verificando servidor'
if (-not (Test-Path $colPath)) { throw "No existe $colPath" }
$cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
$mapPath = $cfg.modbus_map
$map = Get-Content $mapPath -Raw | ConvertFrom-Json
if ($map.instance_id -ne $IID) { throw "Este script es para $IID; el mapa del servidor es $($map.instance_id)" }
Get-Service $Service | Out-Null
Ok "python: $py"
Ok "mapa: $mapPath ($IID)"

Step '2/6 Respaldo'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
Copy-Item $colPath "$colPath.bak_$stamp"
Copy-Item $cfgPath "$cfgPath.bak_$stamp"
Copy-Item $mapPath "$mapPath.bak_$stamp"
Ok "*.bak_$stamp"

function Restore-Config { Copy-Item "$cfgPath.bak_$stamp" $cfgPath -Force; Copy-Item "$mapPath.bak_$stamp" $mapPath -Force }

Step '3/6 Collector __VERSION__ (archivo aparte; el servicio sigue con el viejo)'
$newPath = Join-Path $Root 'wm_collector.__VERSION__.py'
[IO.File]::WriteAllBytes($newPath, [Convert]::FromBase64String($B64))
$got = (Get-FileHash $newPath -Algorithm SHA256).Hash.ToLower()
if ($got -ne $SHA) { Remove-Item $newPath; throw "SHA-256 no coincide ($got)" }
Ok "SHA-256 verificado"

Step '4/6 Mapa local vs mapa de la nube'
$cmp = Join-Path $env:TEMP 'wm_cmp_map.py'
@'
import json, sys, requests
cfg = json.load(open(sys.argv[1], encoding="utf-8"))
loc = json.load(open(cfg["modbus_map"], encoding="utf-8"))
local_ver = (loc.pop("_config", None) or {}).get("version")
url = cfg.get("api_url", "").rstrip("/") + "/v1/collector/map/" + loc["instance_id"]
r = requests.get(url, headers={"Authorization": "Bearer " + cfg["api_key"]}, timeout=20)
if r.status_code != 200:
    print("API status", r.status_code, r.text[:200]); sys.exit(2)
cl = r.json(); meta = cl.pop("_config", {})
print("nube: version", meta.get("version"), "-", meta.get("updated_at"), "| local:", local_ver)
if cl == loc:
    print("IGUALES"); sys.exit(0)
lr, cr = loc.get("registers", {}), cl.get("registers", {})
nreg = 0
for a in sorted(set(lr) | set(cr), key=lambda x: int(x) if x.isdigit() else 0):
    if lr.get(a) != cr.get(a):
        nreg += 1
        print("registro", a, "local:", lr.get(a), "nube:", cr.get(a))
for k in sorted(set(loc) | set(cl)):
    if k != "registers" and loc.get(k) != cl.get(k):
        print("ajuste", k, "local:", loc.get(k), "nube:", str(cl.get(k))[:90])
if nreg:
    print("REGISTROS DISTINTOS:", nreg); sys.exit(3)
print("REGISTROS IGUALES - solo cambian ajustes del collector"); sys.exit(4)
'@ | Set-Content -Path $cmp -Encoding ASCII
& $py $cmp $cfgPath
$rc = $LASTEXITCODE
if ($rc -eq 4 -and $AcceptSettings) {
    Warn 'Registros idénticos; se aceptan los AJUSTES de la nube (-AcceptSettings).'
} elseif ($rc -eq 4) {
    Remove-Item $newPath
    throw "Registros idénticos, pero cambian AJUSTES del collector (arriba). No se activó nada. Si son los esperados, corré de nuevo con -AcceptSettings."
} elseif ($rc -ne 0) {
    Remove-Item $newPath
    throw "El mapa de la nube NO coincide con el del servidor (codigo $rc). No se activó nada. Revisá las diferencias de arriba y publicá en Machines la versión correcta."
} else {
    Ok 'mapa idéntico: activar el remoto no cambia lo que se lee'
}

Step '5/6 Config: remote_map = true'
$cfg | Add-Member -NotePropertyName instance_id -NotePropertyValue $IID -Force
$cfg | Add-Member -NotePropertyName remote_map -NotePropertyValue $true -Force
$cfg | Add-Member -NotePropertyName map_refresh_sec -NotePropertyValue 600 -Force
[IO.File]::WriteAllText($cfgPath, ($cfg | ConvertTo-Json -Depth 10), $enc)
Ok 'config actualizada (sin BOM)'

Step '6/6 Prueba en seco con el collector nuevo'
# stderr de python NO debe cortar el script (ErrorActionPreference=Stop lo
# convertía en excepción y no se restauraba la config).
$eap = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
$out = & $py $newPath --config $cfgPath --dry-run 2>&1 | Out-String
$rc = $LASTEXITCODE
$ErrorActionPreference = $eap
if ($rc -ne 0 -or $out -notmatch '"source": "modbus"') {
    Write-Host $out
    Restore-Config; Remove-Item $newPath
    throw 'La prueba en seco falló. Config restaurada; el servicio sigue con el collector viejo.'
}
Ok 'lectura Modbus + payload OK'

Step 'Cambiando el servicio'
Stop-Service $Service
Copy-Item $newPath $colPath -Force
Start-Service $Service
Start-Sleep -Seconds 30
$tail = Get-Content $logPath -Tail 40 | Out-String
Write-Host $tail
if ($tail -match 'remote_map=True' -and $tail -match 'POST OK') {
    Ok "Collector __VERSION__ corriendo con mapa remoto. Respaldo: *.bak_$stamp"
} else {
    Warn 'No vi aún "remote_map=True" + "POST OK" en el log. Revisá unos segundos más:'
    Warn "  Get-Content $logPath -Wait -Tail 50"
    Warn "Si algo anda mal:  powershell -ExecutionPolicy Bypass -File .\__NAME__ -Rollback"
}
'''


def main() -> int:
    service = "WatermelonCollector"
    if len(sys.argv) == 4:
        service = sys.argv.pop(3)
    if len(sys.argv) != 3:
        print(__doc__)
        return 1
    iid, out = sys.argv[1], Path(sys.argv[2])
    src = (ROOT / "collector" / "wm_collector.py").read_bytes()
    ver = re.search(rb'COLLECTOR_VERSION = "([^"]+)"', src).group(1).decode()
    ps = (TEMPLATE.replace("__IID__", iid).replace("__VERSION__", ver)
          .replace("__NAME__", out.name).replace("__SERVICE__", service)
          .replace("__SHA__", hashlib.sha256(src).hexdigest())
          .replace("__B64__", base64.b64encode(src).decode()))
    out.write_text(ps, encoding="utf-8-sig")   # BOM: PowerShell 5 lee UTF-8 con acentos
    print(f"{out}  ·  collector {ver}  ·  {len(ps) // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
