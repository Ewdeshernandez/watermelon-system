# =====================================================================
# force_enus.ps1 — fuerza English (US) como ÚNICO teclado, de forma que
# SOBREVIVA el reinicio de la VM. El guest (Server 2022 en VMware/RDP) revierte
# a Latin American en cada reinicio; con LAA el robot RPA teclea mal la ruta del
# "Save As" (\->}, :->ñ, _ inalcanzable) y se CONGELA. Set-WinUserLanguageList
# solo no aguanta → acá limpiamos el Preload del teclado en el registro.
# Se corre en cada login (ForceENUS.lnk → force_enus.bat → este .ps1) y a mano.
# =====================================================================
$ErrorActionPreference = "SilentlyContinue"

# 1) Lista de idiomas del usuario: solo en-US (quita el español de la lista).
Set-WinUserLanguageList en-US -Force

# 2) Preload del teclado del usuario = SOLO 00000409 (en-US). Borra cualquier
#    segundo layout (Latin American = 0000080A) y los Substitutes.
reg delete "HKCU\Keyboard Layout\Preload" /f        | Out-Null
reg add    "HKCU\Keyboard Layout\Preload" /v 1 /t REG_SZ /d 00000409 /f | Out-Null
reg delete "HKCU\Keyboard Layout\Substitutes" /f    | Out-Null

# 3) Activar en-US en la sesión ACTUAL (no esperar a re-login) — así el robot
#    que corre después ya usa en-US.
try {
    Add-Type -AssemblyName System.Windows.Forms
    $en = [System.Windows.Forms.InputLanguage]::InstalledInputLanguages |
          Where-Object { $_.Culture.Name -eq 'en-US' } | Select-Object -First 1
    if ($en) { [System.Windows.Forms.InputLanguage]::CurrentInputLanguage = $en }
} catch {}

Write-Output "force_enus: en-US fijado (lista + Preload del registro + sesion actual)"
