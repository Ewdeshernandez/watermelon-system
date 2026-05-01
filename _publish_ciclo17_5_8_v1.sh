#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.5.8: Resumen Ejecutivo siempre live
# =============================================================
# Tras revisión del PDF post-17.5.7 detectamos que la severidad
# global del Resumen Ejecutivo seguía mostrando "CONDICIÓN
# ACEPTABLE" aunque la Figura 1 reportaba Strong change y
# variación 7088%.
#
# CAUSA RAÍZ:
# El executive_summary se cachea en meta tras la primera vez
# que se autoredacta. La lógica anterior solo regeneraba el
# draft cuando estaba VACÍO; si ya había prosa cached (de un
# run previo cuando todavía no había Trend en el reporte) la
# prosa quedaba pegada con "CONDICIÓN ACEPTABLE" para siempre.
# El badge de severidad se extraía del texto cached por regex,
# así que también quedaba pegado.
#
# FIX:
# (a) En cada generación de PDF se recomputa la severidad LIVE
#     con _extract_findings_from_items() + _global_severity()
#     sobre los items actuales.
# (b) Se compara la severidad cacheada (regex sobre la prosa)
#     contra la severidad live. Si difieren, el draft está
#     stale y se regenera automáticamente.
# (c) El badge de severidad usa SIEMPRE la live (live wins),
#     desacoplado del texto.
#
# PRESERVA edición manual: si el usuario editó la prosa y la
# severidad mencionada coincide con la live, no se regenera.
#
# CAMBIOS COSMÉTICOS adicionales:
# - Pluralización correcta "1 figura" / "N figuras" en la prosa
#   del Resumen Ejecutivo (antes salía "1 figuras" siempre).
# - Acuerdo gramatical "adquirida" / "adquiridas" según el
#   número de figuras.
# - "variación total —%" cuando el valor inicial es ~0
#   (división por cero) ahora se expresa como "variación
#   absoluta {delta} {unit}" en lugar del em-dash sin sentido.
#
# Ejecutar:
#   bash _publish_ciclo17_5_8_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/16_Reports.py
git add core/trend_diagnostics.py
git add _publish_ciclo17_5_8_v1.sh
git status --short | head

git commit -m "fix(reports): Resumen Ejecutivo siempre live + cosmetica (17.5.8)

Bug: tras 17.5.7 la severidad global del Resumen Ejecutivo
seguia mostrando 'CONDICION ACEPTABLE' aunque Figura 1 tenia
Strong change. Causa: executive_summary se cachea en meta;
la logica anterior solo autoredactaba cuando estaba VACIO. Si
habia prosa cached de un run previo con menos items, la
severidad quedaba pegada para siempre.

Fix:
(a) En cada generacion de PDF se recomputa severidad LIVE con
    _extract_findings_from_items + _global_severity sobre los
    items actuales.
(b) Si la severidad cached (regex sobre la prosa) difiere de
    la live, el draft esta stale y se regenera automaticamente.
(c) El badge de severidad usa SIEMPRE la live (live wins).

Preserva edicion manual: si el usuario edito la prosa y la
severidad mencionada coincide con la live, no se regenera.

Cosmetica adicional:
- Pluralizacion correcta '1 figura' / 'N figuras'.
- Acuerdo gramatical 'adquirida' / 'adquiridas'.
- 'variacion total -%' (division por cero cuando inicial es
  ~0) ahora muestra 'variacion absoluta {delta} {unit}'." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.5.8 pusheado a dev"
echo "================================================================"
echo ""
echo "Para verificar:"
echo "  1. Sesion limpia → enviar Trend con Strong change al reporte."
echo "  2. Generar PDF → Resumen Ejecutivo debe escalar a ATENCION."
echo "  3. Si tenias un PDF previo con CONDICION ACEPTABLE cached,"
echo "     re-generar PDF sin tocar el textarea — auto-regenera."
echo "  4. PDF dice '1 figura' (no '1 figuras') cuando hay solo 1."
echo "  5. Trend con valor inicial ~0 muestra 'variacion absoluta',"
echo "     no '—%'."
echo "================================================================"
