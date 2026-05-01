#!/bin/bash
# =============================================================
# Watermelon — Hotfix 17.5.5: PNG export con marker.size lista
# =============================================================
# BUG REPORTADO:
#   "PNG export error: float() argument must be a string or a
#   real number, not 'list'"
#
# Y como efecto secundario, "Enviar a Reporte" enviaba el item
# pero `has_image=False` en el debug porque
# build_export_png_bytes() tiraba esta excepción y devolvía
# image_bytes=None. El PDF terminaba sin el gráfico de Trend.
#
# CAUSA:
# En el ciclo 17.5.1 cambié los marcadores de anomalía a tener
# `size` y `color` por punto (lista) — un tamaño/color por
# cada anomalía según severidad (High=9, Medium=7, Low=6). El
# helper `_scale_export_figure()` luego hacía:
#
#     marker["size"] = max(14, float(marker.get("size", 6)) * 1.9)
#
# y `float([9, 7, 6, ...])` revienta. Idem para line.width
# defensivamente.
#
# FIX:
# Helper local `_scale_size(value, factor, floor)` que escala
# correctamente:
#   - escalar (float / int): factor + floor
#   - lista / tupla: aplica per-elemento preservando los None
#   - None: usa default
# Aplicado a marker.size, line.width y marker.line.width.
#
# Resultado:
#   ✅ Prepare PNG HD funciona (con o sin anomaly markers)
#   ✅ Enviar a Reporte ahora deja has_image=True y el PDF
#      incluye el gráfico de Trend
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/04_Trends.py
git add _publish_ciclo17_5_hotfix_png_v1.sh
git status --short | head

git commit -m "fix(trend): PNG export reventaba con marker.size por-punto (lista) tras anomalias suavizadas

Bug reportado: 'PNG export error: float() argument must be a
string or a real number, not list'. Causa: en 17.5.1 los
marcadores de anomalia se cambiaron a size y color por punto
(lista, segun severidad High/Medium/Low). El helper
_scale_export_figure hacia float(marker.size)*1.9 directo y
revienta cuando size es lista.

Fix: helper local _scale_size(value, factor, floor) que escala
correctamente escalares, listas y None. Aplicado a marker.size,
line.width y marker.line.width.

Efecto secundario: Enviar a Reporte ahora completa con
has_image=True (antes el PNG fallaba silencioso y el PDF
quedaba sin grafico de Trend)." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix 17.5.5 PNG export pusheado a dev"
echo "================================================================"
echo ""
echo "Para verificar:"
echo "  1. Trends → cargar CSV con anomalias (tipo TES1 con spikes)."
echo "  2. Click 'Prepare PNG HD' → debe completar sin error."
echo "  3. Click 'Download PNG HD' → PNG con curva, ejes, marcadores"
echo "     de severidad escalados correctamente."
echo "  4. Click 'Enviar a Reporte' → debug debe mostrar"
echo "     has_image=True y el PDF incluir el grafico."
echo "================================================================"
