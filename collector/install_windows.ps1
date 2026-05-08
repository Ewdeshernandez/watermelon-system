#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Instalador del Watermelon Collector como servicio Windows.

.DESCRIPTION
    Crea estructura en C:\watermelon\collector, copia wm_collector.py,
    instala dependencias Python en venv, y registra como servicio
    Windows usando NSSM (https://nssm.cc/).

.NOTES
    Antes de correr:
      1. Tener Python 3.10+ instalado en el sistema (https://www.python.org/downloads/).
      2. Descargar NSSM desde https://nssm.cc/release/nssm-2.24.zip
         y colocarlo en una carpeta accesible.
      3. Tener wm_collector.config.json LISTO con la api_key y rutas correctas.
      4. Tener el JSON del Modbus map (ej. tes1.json) listo para copiar.

.EXAMPLE
    PS> .\install_windows.ps1 `
            -CollectorScript .\wm_collector.py `
            -ConfigFile .\wm_collector.config.json `
            -ModbusMap .\tes1.json `
            -NssmPath "C:\nssm\nssm-2.24\win64\nssm.exe"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$CollectorScript,

    [Parameter(Mandatory=$true)]
    [string]$ConfigFile,

    [Parameter(Mandatory=$true)]
    [string]$ModbusMap,

    [Parameter(Mandatory=$true)]
    [string]$NssmPath,

    [string]$InstallRoot = "C:\watermelon\collector",
    [string]$ServiceName = "WatermelonCollector",
    [string]$ServiceDisplayName = "Watermelon Collector (Tier 0 A)",
    [string]$ServiceDescription = "Lee Bently 3500/92 via Modbus TCP y postea a Watermelon System."
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host ""
    Write-Host "==> $msg" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 1. Validaciones
# ---------------------------------------------------------------------------
Write-Step "Verificando prerequisitos"

if (-not (Test-Path $CollectorScript)) {
    throw "No se encuentra el script: $CollectorScript"
}
if (-not (Test-Path $ConfigFile)) {
    throw "No se encuentra el config: $ConfigFile"
}
if (-not (Test-Path $ModbusMap)) {
    throw "No se encuentra el Modbus map: $ModbusMap"
}
if (-not (Test-Path $NssmPath)) {
    throw "No se encuentra NSSM: $NssmPath"
}

$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    throw "Python no encontrado en PATH. Instalá Python 3.10+ desde python.org."
}
Write-Host "Python: $($pythonCmd.Source)" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 2. Estructura
# ---------------------------------------------------------------------------
Write-Step "Creando estructura en $InstallRoot"

$paths = @(
    $InstallRoot,
    "$InstallRoot\modbus_maps",
    "$InstallRoot\logs"
)
foreach ($p in $paths) {
    if (-not (Test-Path $p)) {
        New-Item -ItemType Directory -Path $p -Force | Out-Null
        Write-Host "Creado: $p" -ForegroundColor Green
    }
}

# ---------------------------------------------------------------------------
# 3. Copiar archivos
# ---------------------------------------------------------------------------
Write-Step "Copiando archivos"

Copy-Item -Path $CollectorScript -Destination "$InstallRoot\wm_collector.py" -Force
Copy-Item -Path $ConfigFile -Destination "$InstallRoot\wm_collector.config.json" -Force
$modbusFileName = Split-Path -Leaf $ModbusMap
Copy-Item -Path $ModbusMap -Destination "$InstallRoot\modbus_maps\$modbusFileName" -Force

Write-Host "Archivos copiados a $InstallRoot" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 4. Crear venv e instalar dependencias
# ---------------------------------------------------------------------------
Write-Step "Creando virtualenv e instalando dependencias"

$venvPath = "$InstallRoot\venv"
if (-not (Test-Path $venvPath)) {
    & python -m venv $venvPath
}
$venvPython = "$venvPath\Scripts\python.exe"
$venvPip = "$venvPath\Scripts\pip.exe"

& $venvPython -m pip install --upgrade pip
& $venvPip install "pymodbus==3.6.9" "requests>=2.31.0"
Write-Host "Dependencias instaladas" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 5. Registrar/actualizar servicio Windows con NSSM
# ---------------------------------------------------------------------------
Write-Step "Registrando servicio Windows: $ServiceName"

$existing = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Servicio ya existe — deteniendo y reconfigurando..." -ForegroundColor Yellow
    & $NssmPath stop $ServiceName confirm | Out-Null
    Start-Sleep -Seconds 2
    & $NssmPath remove $ServiceName confirm | Out-Null
    Start-Sleep -Seconds 2
}

& $NssmPath install $ServiceName $venvPython "$InstallRoot\wm_collector.py" "--config" "$InstallRoot\wm_collector.config.json"
& $NssmPath set $ServiceName DisplayName $ServiceDisplayName
& $NssmPath set $ServiceName Description $ServiceDescription
& $NssmPath set $ServiceName Start SERVICE_AUTO_START
& $NssmPath set $ServiceName AppStdout "$InstallRoot\logs\nssm_stdout.log"
& $NssmPath set $ServiceName AppStderr "$InstallRoot\logs\nssm_stderr.log"
& $NssmPath set $ServiceName AppRotateFiles 1
& $NssmPath set $ServiceName AppRotateBytes 5242880   # 5 MB
& $NssmPath set $ServiceName AppRestartDelay 5000     # esperar 5s antes de auto-restart

Write-Host "Servicio registrado" -ForegroundColor Green

# ---------------------------------------------------------------------------
# 6. Smoke test (dry-run) ANTES de arrancar el servicio
# ---------------------------------------------------------------------------
Write-Step "Smoke test (dry-run, sin POST)"

try {
    & $venvPython "$InstallRoot\wm_collector.py" --config "$InstallRoot\wm_collector.config.json" --dry-run
    Write-Host "Dry-run OK — el script lee y arma payload correctamente" -ForegroundColor Green
} catch {
    Write-Host "Dry-run falló: $_" -ForegroundColor Red
    Write-Host "Revisá la conectividad al 3500/92 antes de arrancar el servicio." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# 7. Arrancar el servicio
# ---------------------------------------------------------------------------
Write-Step "Arrancando servicio"

& $NssmPath start $ServiceName
Start-Sleep -Seconds 3

$state = (Get-Service -Name $ServiceName).Status
Write-Host "Estado actual: $state" -ForegroundColor Cyan

if ($state -eq "Running") {
    Write-Host ""
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host " ✓ Watermelon Collector instalado y corriendo" -ForegroundColor Green
    Write-Host "================================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Logs en: $InstallRoot\logs\wm_collector.log"
    Write-Host "Para ver logs en vivo: Get-Content $InstallRoot\logs\wm_collector.log -Wait -Tail 50"
    Write-Host "Para detener:          & '$NssmPath' stop $ServiceName"
    Write-Host "Para iniciar:          & '$NssmPath' start $ServiceName"
    Write-Host "Para desinstalar:      & '$NssmPath' remove $ServiceName confirm"
} else {
    Write-Host ""
    Write-Host "El servicio NO arrancó. Revisá los logs:" -ForegroundColor Red
    Write-Host "  $InstallRoot\logs\nssm_stderr.log"
    Write-Host "  $InstallRoot\logs\wm_collector.log"
}
