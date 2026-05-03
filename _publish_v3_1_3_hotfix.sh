#!/bin/bash
# =============================================================
# Watermelon — v3.1.3 hotfix URGENTE: HD export aún sin traces
# =============================================================
# QUEJA POST-v3.1.2: "nada es nada urgente, main esta sin eso
# y el cliente opresiona". El PNG HD seguía saliendo SIN curvas
# dibujadas aunque axes y leyenda eran correctos.
#
# Causa NO RESUELTA en v3.1.2:
# Yo arreglé `_build_export_safe_figure` con clone via to_dict()
# preservando subplots. PERO había OTRA función,
# `_scale_export_figure`, que hacía exactamente el mismo error:
#
#     fig = go.Figure(data=new_data, layout=fig.layout)
#
# al final del loop de scaling. Esta línea recreaba la figura
# desde cero perdiendo la estructura subplots — el fix de
# v3.1.2 quedaba ANULADO inmediatamente porque _scale_export_
# figure se ejecuta DESPUÉS de _build_export_safe_figure.
#
# Pipeline completo:
#   build_export_png_bytes(fig)
#     ↓
#   _build_export_safe_figure(fig)   ← v3.1.2 lo arregló
#     ↓
#   _scale_export_figure(safe_fig)   ← AQUÍ lo rompía otra vez
#     ↓
#   .to_image(...)
#     ↓
#   PNG sin curvas
#
# FIX:
# `_scale_export_figure` ahora también trabaja sobre fig_dict
# (clone via to_dict) y solo modifica los campos line/marker en
# el dict directamente. La reconstrucción final
# `go.Figure(fig_dict)` preserva subplots + secondary_y + axis
# refs intactos.
#
# Confirmación de que esta era la causa: si v3.1.2 hubiera
# funcionado sin el fix de _scale_export_figure, el chart
# hubiera salido bien. El usuario reportó "nada cambió" después
# de v3.1.2, lo que confirma que algo ANULABA el fix.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.1.3"
RELEASE_TITLE="Hotfix urgente: _scale_export_figure también recreaba figura plana (subplots perdidos)"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo "▶ Stageando..."
git add pages/04_Trends.py
git add _publish_v3_1_3_hotfix.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commit..."
    git commit -m "fix(trend): HD export aun vacio — _scale_export_figure tambien recreaba figura plana (17.8.3)

Bug post-v3.1.2: usuario reporta 'nada es nada urgente'. PNG HD
sigue sin curvas. Razon: yo arregle _build_export_safe_figure
con clone via to_dict() preservando subplots, pero la siguiente
funcion del pipeline (_scale_export_figure) hacia
   fig = go.Figure(data=new_data, layout=fig.layout)
al final del loop, recreando la figura plana y PERDIENDO los
subplots otra vez. El fix v3.1.2 quedaba anulado inmediatamente.

Fix: _scale_export_figure ahora tambien trabaja sobre fig_dict
(clone via to_dict). Modifica line/marker dentro del dict y al
final hace go.Figure(fig_dict) preservando estructura entera." || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo "▶ Reconciliando..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Rebase falló."; exit 1; }

echo "▶ Push dev..."
git push origin dev

echo "▶ Switch a main..."
git checkout main
git pull origin main

echo "▶ Merge dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.1.2:

Bug del HD export 'sin curvas' resuelto definitivamente.
v3.1.2 arreglaba _build_export_safe_figure pero
_scale_export_figure (siguiente paso del pipeline) volvia a
recrear la figura plana. Ahora ambas funciones usan to_dict()
para preservar la estructura subplots + secondary_y entera."

echo "▶ Tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.1.2 (que era incompleto): ahora SÍ las curvas
del Trend HD export se dibujan correctamente en mixed mode."

echo "▶ Push main + tag..."
git push origin main
git push origin "${VERSION}"

echo "▶ Vuelta a dev..."
git checkout dev

echo ""
echo "================================================================"
echo " ✓ ${VERSION} publicado a main — esta vez SI deberia funcionar"
echo "================================================================"
echo ""
echo " Streamlit Cloud auto-redeploya en ~2 min. Refrescá Cmd+Shift+R."
echo " Probá Prepare PNG HD → Download — el PNG ahora DEBE traer las"
echo " curvas dibujadas (vibracion 5YV/8YV + presiones operacional)."
echo "================================================================"
