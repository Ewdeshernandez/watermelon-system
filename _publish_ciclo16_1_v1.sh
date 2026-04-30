#!/bin/bash
# =============================================================
# Watermelon — Ciclo 16.1 → DEV: Wizard auto-pattern Sensor Map
# =============================================================
# Soluciona el dolor del NDE que el usuario tuvo:
# Sensor Map del cliente con nomenclatura DCS (VE5807, VE5810)
# que no coincide con la convencion estandar API 670 (3X, 3Y,
# 4X, 4Y) → algunos sensores quedan sin csv_match_pattern y no
# matchean los CSVs cargados.
#
# Ahora en Machinery Library → Sensor Map aparece la seccion
# "🪄 Sugerir patterns desde CSVs cargados":
#
#   1. Detecta sensores SIN match definitivo (pattern vacio o
#      pattern explicito que no matchea ningun signal cargado).
#   2. Para cada uno, mira los signals en sesion compatibles
#      por tipo (proximity/velocity/accelerometer) y direccion
#      (X/Y/RAD/AXIAL).
#   3. Propone un pattern concreto basado en el numero unico
#      del Point name (ej. '*5810*' para Point 'VE5810 (X)').
#   4. Tabla con checkbox por fila + boton "Aplicar
#      seleccionados" que hace bulk update del Sensor Map.
#
# Confianza:
#   🟢 high   = unico candidato compatible
#   🟡 medium = varios candidatos, tomamos el primero
#   🔴 low    = pocos elementos discriminantes
#
# Bonus: nueva funcion detect_definitive_matches() que detecta
# matches solo cuando el pattern es no-vacio Y matchea
# explicitamente el Point. NO usa el fallback gracioso del
# resolver (que devuelve "el primer candidato del tipo
# correcto" aunque ningun pattern matchee). Asi el wizard sabe
# exactamente que sensores tienen un match real vs. cuales
# necesitan ayuda.
#
# Smoke validado contra los 8 CSVs reales del usuario (TES1):
#   * 6 matches definitivos: CRF/TRF VL+ACEL, 5807→3Y_D,
#     5808→3X_D.
#   * 4X_D (vacio) → propone *5810* matchea VE5810 (X) [high]
#   * 4Y_D (vacio) → propone *5809* matchea VE5809 (Y) [high]
#
# Archivos:
#   - core/sensor_map.py             (detect_definitive_matches,
#                                      suggest_pattern_for_sensor,
#                                      helpers de compatibilidad
#                                      tipo + direccion + extract
#                                      pattern token)
#   - pages/00_Machinery_Library.py  (seccion del wizard con UI
#                                      checkbox + bulk apply)
#
# Ejecutar:
#   bash _publish_ciclo16_1_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/sensor_map.py
git add pages/00_Machinery_Library.py
git status --short | head

git commit -m "feat(machinery-library): wizard auto-pattern desde CSVs (Ciclo 16.1)

Soluciona el dolor del NDE: cuando los Point names del DCS
(VE5807, VE5810) no coinciden con la convencion estandar
API 670 (3X, 3Y, 4X, 4Y), el ingeniero tenia que editar
csv_match_pattern manualmente sensor por sensor.

Ahora un boton en Machinery Library mira los CSVs en sesion,
detecta sensores sin match definitivo y propone patterns
concretos por tipo + direccion (ej. '*5810*' para 4X_D
basado en 'VE5810 (X)'). Tabla checkbox + bulk apply.

Nueva funcion detect_definitive_matches que NO usa el
fallback gracioso del resolver — solo cuenta como match
cuando el pattern es no-vacio Y matchea explicitamente el
Point name. Asi el wizard sabe exactamente que sensores
necesitan ayuda.

Smoke validado contra los 8 CSVs reales del usuario:
  - 5807 → 3Y_D (definitivo via *5807*)
  - 5808 → 3X_D (definitivo via *5808*)
  - 5809 → 4Y_D (sugiere *5809* con confidence high)
  - 5810 → 4X_D (sugiere *5810* con confidence high)
  - CRF/TRF VL+ACEL → matched via patterns existentes." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 16.1 pusheado a dev"
echo "================================================================"
echo ""
echo "Para usarlo (Streamlit Cloud redeploya en 1-2 min):"
echo "  1. Ir a Load Data, subir los CSVs."
echo "  2. Ir a Machinery Library, activar la instancia."
echo "  3. Sección 'Mapa de Sensores' → bajar hasta"
echo "     '🪄 Sugerir patterns desde CSVs cargados'."
echo "  4. El wizard muestra los sensores sin match con la"
echo "     sugerencia. Marcá los que querés aplicar y clic en"
echo "     'Aplicar X pattern(s) seleccionados'."
echo "  5. Vovlé a Tabular / Reports y vas a ver los nuevos"
echo "     matches reflejados."
echo "================================================================"
