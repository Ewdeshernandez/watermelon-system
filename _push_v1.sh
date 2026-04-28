#!/bin/bash
# =============================================================
# Watermelon System — Push v1.0 a GitHub (origin)
# =============================================================
# Empuja todo lo que _publish_v1.sh dejó listo localmente:
#   - feat/scl-cat-iv (snapshot de la rama de trabajo)
#   - dev (con todo el merge incorporado)
#   - main (con dev mergeado y tag v1.0)
#   - tags v1.0-pre-main y v1.0
#
# Te pedirá tus credenciales de GitHub si no están cacheadas.
#
# Ejecutar desde el root del repo:
#   bash _push_v1.sh
# =============================================================

set -e

cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " WATERMELON SYSTEM — Push v1.0 a GitHub"
echo "================================================================"
echo ""

echo "[1/4] Push de feat/scl-cat-iv (snapshot de la rama de trabajo)..."
git push origin feat/scl-cat-iv
echo "    OK"
echo ""

echo "[2/4] Push de dev (con todo el merge a v1.0)..."
git push origin dev
echo "    OK"
echo ""

echo "[3/4] Push de main (con dev v1.0 mergeado)..."
git push origin main
echo "    OK"
echo ""

echo "[4/4] Push de tags v1.0-pre-main y v1.0..."
# Borrar primero los tags remotos viejos por si están mal puestos
git push origin :refs/tags/v1.0 2>/dev/null || true
git push origin :refs/tags/v1.0-pre-main 2>/dev/null || true
# Empujar los nuevos
git push origin v1.0-pre-main
git push origin v1.0
echo "    OK"
echo ""

echo "================================================================"
echo " PUBLICACIÓN COMPLETA"
echo "================================================================"
echo ""
echo "main acaba de recibir todo el trabajo v1.0."
echo ""
echo "En 1-2 minutos Streamlit Cloud va a hacer pull automático y"
echo "watermelonsystem.app va a mostrar los cambios:"
echo "  - IBM Plex Sans en los reportes PDF"
echo "  - Resumen Ejecutivo con cinta de severidad"
echo "  - SCL Cat IV con convención API 670"
echo "  - Asset Profiles + Document Vault"
echo "  - Narrativas profesionales sin menciones de competencia"
echo ""
echo "Después podés borrar este script si querés:"
echo "  rm _publish_v1.sh _push_v1.sh"
echo "================================================================"
