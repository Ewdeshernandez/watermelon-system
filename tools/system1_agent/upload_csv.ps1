<#
  upload_csv.ps1 — Uploader SIN Python (para arrancar hoy)
  =========================================================
  Sube los CSV que System1 exporta (carpeta Desktop\CSV) a Supabase Storage,
  bucket dynamic_raw, bajo {asset}/s1/{sensor}.csv. La web (Dynamic analysis)
  los lee, empareja X/Y por cojinete y reconstruye onda/espectro/orbita.

  Config: upload_csv.config.json  (al lado de este script) con:
    { "url": "https://xxxx.supabase.co",
      "service_key": "SERVICE_ROLE_KEY",
      "asset": "turbina_sgt300_b",
      "folder": "C:\\Users\\Administrator\\Desktop\\CSV" }
  (o variables de entorno SUPABASE_URL / SUPABASE_SERVICE_KEY)

  Correr:  powershell -ExecutionPolicy Bypass -File upload_csv.ps1
  Agendar cada hora con Task Scheduler (run_uploader.bat).
#>
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfgPath = Join-Path $here "upload_csv.config.json"

$url = $env:SUPABASE_URL
$key = $env:SUPABASE_SERVICE_KEY
$asset = "turbina_sgt300_b"
$folder = Join-Path $env:USERPROFILE "Desktop\CSV"

if (Test-Path $cfgPath) {
    $cfg = Get-Content $cfgPath -Raw | ConvertFrom-Json
    if ($cfg.url)         { $url = $cfg.url }
    if ($cfg.service_key) { $key = $cfg.service_key }
    if ($cfg.asset)       { $asset = $cfg.asset }
    if ($cfg.folder)      { $folder = $cfg.folder }
}

if (-not $url -or -not $key) {
    Write-Error "Falta url o service_key (config o variables de entorno)."
    exit 1
}
if (-not (Test-Path $folder)) {
    Write-Error "No existe la carpeta de CSV: $folder"
    exit 1
}

$hdr = @{ apikey = $key; Authorization = "Bearer $key"; "x-upsert" = "true" }
$files = Get-ChildItem -Path $folder -Filter *.csv | Where-Object {
    $_.BaseName -match '^\s*\d+\s*[xyXY]'   # 1xd, 1yd, 2xd, ... (X/Y por cojinete)
}
$ok = 0; $fail = 0
foreach ($f in $files) {
    $name = $f.Name.ToLower()
    $uri = "$url/storage/v1/object/dynamic_raw/$asset/s1/$name"
    try {
        Invoke-RestMethod -Method Put -Uri $uri -Headers $hdr `
            -ContentType "text/csv" -InFile $f.FullName | Out-Null
        Write-Host ("OK  {0}" -f $name)
        $ok++
    } catch {
        Write-Host ("FAIL {0} : {1}" -f $name, $_.Exception.Message)
        $fail++
    }
}
Write-Host ("Subidos: {0}  Fallidos: {1}  ({2})" -f $ok, $fail, $folder)
if ($fail -gt 0) { exit 1 }
