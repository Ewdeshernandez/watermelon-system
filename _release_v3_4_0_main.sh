#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.0 → MAIN
# =============================================================
# Promueve todo lo pendiente acumulado en dev a producción:
#
# - Ciclo 17.16   Recovery password por email + restricciones
#                 del role client + backend SMTP/Graph dual
# - Ciclo 17.16.1 Logging del envío de email a stderr
# - Ciclo 17.16.2 Hotfix CRÍTICO: leer secrets nested correctos
#                 en Streamlit Cloud (AttrDict fix). Sin esto el
#                 backend de email reportaba 'no_backend' aunque
#                 el TOML estuviera bien.
# - Ciclo 17.16.3 Rename pages/_reset_password.py → reset_password.py
#                 (Streamlit oculta páginas con underscore prefix)
#                 + auto-detect de base_url usando st.context.headers
#                 para que el link del email apunte a la app correcta
#
# Resultado: el flow completo de "olvidé mi contraseña" funciona
# end-to-end en producción. Validado en wm-test antes del release.
#
# RECORDATORIO IMPORTANTE — PEGAR EN STREAMLIT CLOUD WM-HOME-FINAL-2026:
# La sección [email.graph] que está en wm-test debe replicarse
# IDÉNTICA en los secrets de wm-home-final-2026, sino reset
# desde producción no va a funcionar. Hacelo ANTES de subir o
# inmediatamente después del redeploy de Cloud.
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚀 RELEASE v3.4.0 → MAIN"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.16   Recovery password + restricciones client + SMTP/Graph"
echo "  17.16.1 Logging del envío email a stderr"
echo "  17.16.2 Hotfix CRÍTICO secrets AttrDict en Streamlit Cloud"
echo "  17.16.3 Rename _reset_password.py + auto-detect URL base"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/6  Sincronizando dev con origin..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
if ! git diff --quiet || ! git diff --staged --quiet; then
    echo "✗ Hay cambios sin commitear en dev. Commiteá primero."
    exit 1
fi
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Pull dev falló."; exit 1; }
git push origin dev || { echo "✗ Push dev falló."; exit 1; }
echo "  ✓ dev sincronizado"
echo ""

echo "▶ 2/6  Cambiando a main..."
git checkout main || { echo "✗ No se pudo cambiar a main."; exit 1; }
git fetch origin main
git pull --rebase origin main || { echo "✗ Pull main falló."; exit 1; }
echo "  ✓ main actualizado"
echo ""

echo "▶ 3/6  Mergeando dev → main..."
MERGE_MSG="release(v3.4.0): merge dev -> main

Ciclo 17.16 + hotfixes 17.16.1/17.16.2/17.16.3.

Sistema de recovery de password completo end-to-end:
- Login: link 'Olvidaste tu contraseña' con form de email
- Email branded enviado vía Microsoft Graph API (Office 365)
  - Backend dual SMTP / Graph configurable via secrets
  - Anti-enumeration: no revela si el email existe
  - Token TTL 1 hora, one-shot use, storage en filesystem
- Pagina /reset_password con form de nueva contrasena
- Validacion + cambio en Supabase Auth + invalidacion de token

Restricciones del role client:
- Sin acceso a Machinery Library, Load Data, Diagnostics, Map
- Reports en modo solo-lectura (solo archivo histórico shared)
- Guards de defensa en profundidad en cada página restringida

Boton 'Cambiar mi password' en sidebar para users Supabase.

Hotfixes incluidos:
- 17.16.1: log a stderr cuando email_send falla (visible en logs)
- 17.16.2 CRÍTICO: _read_secret_section helper para extraer
  AttrDicts anidados de Streamlit Cloud como dicts puros.
  Sin esto el backend reportaba 'no_backend' aunque TOML bien.
- 17.16.3: rename pages/_reset_password.py -> reset_password.py
  (Streamlit esconde paginas con underscore prefix) + auto-detect
  de base_url usando st.context.headers Host para que link del
  email apunte a la app correcta (wm-test vs wm-home-final).

Validado end-to-end en wm-test antes del release.

PENDIENTE: pegar [email.graph] en secrets de wm-home-final-2026
para que reset desde produccion funcione. Es el mismo bloque
que ya esta en wm-test."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 4/6  Creando tag v3.4.0..."
TAG_EXISTS=$(git tag -l "v3.4.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.0 ya existe. Saltando creación."
else
    git tag -a v3.4.0 -m "Release v3.4.0 — Recovery password + restricciones client + SMTP/Graph"
    echo "  ✓ Tag v3.4.0 creado"
fi
echo ""

echo "▶ 5/6  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.0 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 6/6  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.4.0 COMPLETADO"
echo "================================================================"
echo ""
echo " ⚠️  ACCIÓN REQUERIDA YA — antes que tu equipo intente reset:"
echo ""
echo "  En Streamlit Cloud → Settings → Secrets de wm-home-final-2026:"
echo "  pegar EXACTAMENTE el mismo bloque [email.graph] que está en"
echo "  wm-test, sino el reset desde producción no va a funcionar."
echo ""
echo " ► PRÓXIMO PASO INMEDIATO (Ciclo 17.17):"
echo "  Atacar el bug de 'se cae con 10+ imágenes' separando las"
echo "  imágenes a archivos PNG del filesystem en lugar de base64"
echo "  inline en el JSON. Eso resuelve definitivamente:"
echo "    - Pérdida de reportes al colapso"
echo "    - Sistema cayéndose con muchas imágenes"
echo "    - Memory/timeout en Streamlit Cloud"
echo ""
echo "================================================================"
