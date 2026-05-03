#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.6 → DEV (no toca main)
# =============================================================
# Dos mejoras solicitadas:
#
# (1) Login estilo "internacional"
#     - Paleta sobria: azul corporativo profundo (#21478c) en
#       lugar de cyan brillante. Más empresarial, menos startup.
#     - Eliminado el RECUADRO ROJO que Streamlit pone al field
#       de password en focus ("Press Enter to submit"). Ahora
#       borde gris/azul sutil sin importar estado de validación.
#     - Hero tipográficamente más calmo: "Diagnóstico avanzado
#       para máquinas críticas" con accent gradiente azul.
#     - Tagline corporativa explícita (API 670 / 684 / ISO 20816).
#     - Trust chips: SSO-ready, API 670/684, ISO 20816,
#       Multi-instance.
#     - Card del login con glassmorphism muy sutil
#       (backdrop-filter: blur(14px)).
#     - Footer con build info compacto.
#     - Botón submit "Iniciar sesión" con gradient corporativo
#       y hover sutil.
#
# (2) Editor de Reports — 4 secciones unificadas
#     Antes:
#       - Resumen ejecutivo: full-width
#       - Objetivo: full-width
#       - Desarrollo + Recomendaciones: 2 columnas (chiquitas)
#       - 2 botones globales (Auto-redactar 1/2/3 + Resumen)
#     Ahora:
#       - Las 4 secciones full-width consistentes
#       - Cada sección tiene SU PROPIO botón "Auto-redactar"
#         arriba a la derecha
#       - Hint text descriptivo bajo cada título
#       - Helper _autodraft_single_section(section, meta, items)
#         para regenerar UNA sola sección en lugar de las 3
#
# Push solo a DEV — el usuario explícitamente pidió no tocar main.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando cambios 17.6..."
git add pages/00_Login.py
git add pages/16_Reports.py
git add _publish_ciclo17_6_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.6..."
    git commit -m "feat(login+reports): login premium internacional + editor unificado (17.6)

(1) Login pages/00_Login.py:
- Paleta corporativa sobria (#21478c) en lugar de cyan brillante
- Eliminado el recuadro rojo del field password en focus
  (override agresivo de cualquier border-color de validacion)
- Hero mas tipograficamente calmo con accent gradient azul
- Tagline corporativa: 'Diagnostico avanzado para maquinas criticas'
- Trust chips: SSO-ready, API 670/684, ISO 20816, Multi-instance
- Card con glassmorphism sutil (backdrop-filter blur 14px)
- Footer con build info compacto

(2) Editor de Reports pages/16_Reports.py:
- 4 secciones (Resumen ejecutivo, Objetivo, Desarrollo,
  Recomendaciones) ahora son full-width consistentes
- Cada seccion tiene su propio boton 'Auto-redactar' a la
  derecha del header
- Hint text descriptivo bajo cada titulo
- Helper _autodraft_single_section(section, meta, items)
  permite regenerar UNA sola seccion
- Eliminados los 2 botones globales redundantes
  (Auto-redactar 1/2/3 + Resumen Ejecutivo)" || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo ""
echo "▶ Reconciliando contra origin/dev..."
git fetch origin dev
git pull --rebase origin dev || {
    echo "✗ Rebase falló. Resolvelos a mano y re-ejecutá."
    exit 1
}

echo "▶ Pusheando dev..."
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.6 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Para probar:"
echo "   1. Login: paleta azul corporativa, sin recuadro rojo,"
echo "      trust chips visibles."
echo "   2. Reports → editor: las 4 secciones (Resumen, Objetivo,"
echo "      Desarrollo, Recomendaciones) full-width con boton"
echo "      'Auto-redactar' propio en cada una."
echo ""
echo " Cuando lo apruebes, publicamos a main con un script aparte."
echo "================================================================"
