<#
  upload_shot.ps1 — sube un PNG a Supabase Storage y devuelve una URL firmada.
  =========================================================================
  Uso (en el HOST, que tiene internet y ve la VM por SMB):
     powershell -ExecutionPolicy Bypass -File upload_shot.ps1
  Lee url + service_key de config.toml (misma carpeta). Sube el PNG que la VM
  dejó por --shot y imprime una URL firmada (1h) para inspeccionarlo.

  Parametros opcionales:
     -Src   ruta del PNG (default \\192.168.192.130\C$\WM_wave\s1.png)
     -Dest  ruta destino en el bucket (default _rpa/s1.png)
#>
param(
  [string]$Src  = "\\192.168.192.130\C$\WM_wave\s1.png",
  [string]$Dest = "_rpa/s1.png",
  [string]$Bucket = "dynamic_raw"
)
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$cfgPath = Join-Path $here "config.toml"

$url = $env:SUPABASE_URL
$key = $env:SUPABASE_SERVICE_KEY
if (Test-Path $cfgPath) {
    foreach ($line in Get-Content $cfgPath) {
        if ($line -match '^\s*url\s*=\s*"([^"]+)"')          { $url = $Matches[1] }
        if ($line -match '^\s*service_key\s*=\s*"([^"]+)"')   { $key = $Matches[1] }
    }
}
if (-not $url -or -not $key) { Write-Error "Falta url/service_key (config.toml o env)"; exit 1 }
if (-not (Test-Path $Src))   { Write-Error "No existe el PNG: $Src"; exit 1 }

$hdr = @{ apikey = $key; Authorization = "Bearer $key"; "x-upsert" = "true" }

# 1) subir el PNG
$putUri = "$url/storage/v1/object/$Bucket/$Dest"
Invoke-RestMethod -Method Put -Uri $putUri -Headers $hdr -ContentType "image/png" -InFile $Src | Out-Null
Write-Host ("SUBIDO  {0}  ->  {1}/{2}" -f $Src, $Bucket, $Dest)

# 2) crear URL firmada (1 hora)
$signUri = "$url/storage/v1/object/sign/$Bucket/$Dest"
$body = @{ expiresIn = 3600 } | ConvertTo-Json
$resp = Invoke-RestMethod -Method Post -Uri $signUri -Headers $hdr -ContentType "application/json" -Body $body
$full = "$url/storage/v1$($resp.signedURL)"
Write-Host ""
Write-Host "URL FIRMADA (pegasela a Claude):"
Write-Host $full
