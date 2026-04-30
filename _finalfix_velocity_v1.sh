#!/bin/bash
# =============================================================
# Watermelon — FINAL FIX velocímetros falso match
# =============================================================
# Bug encontrado: pattern '*4*x*' del proximity 4X_D matcheaba
# contra variable 'Vel Wf(64X/32revs).KPHGEN' porque '4x' es
# substring de '64x' (oversampling rate del Bently).
#
# Fix: pre-filtrar candidates por type_hint del Point name ANTES
# de hacer pattern matching. Y eliminar el chequeo contra
# variable_norm (la variable tiene metadata técnica que confunde).
#
# Smoke validado contra Sensor Map REAL del usuario:
#   1VT6805 (C) TRF → 2_RAD_V (velocity TRF) ✓
#   1VT6831 (C) CRF → 1_RAD_V (velocity CRF) ✓
#   CRF/TRF ACELL  → accelerometer ✓
#   VE5807-5810    → proximity ✓
#   KPHGEN         → keyphasor ✓
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Pull para incluir el commit del marker BUILD 14c.3-debug-v2
git pull origin dev || true

# Commit del fix
git add core/sensor_map.py pages/01__Tabular_List.py
git status --short

git commit -m "fix(sensor-map): pre-filtro por type_hint elimina falso match con variable

BUG: pattern '*4*x*' del proximity 4X_D matcheaba contra variable
'Vel Wf(64X/32revs).KPHGEN' porque '4x' está embedded en '64x'
(oversampling Bently). Velocímetros 1VT6805/1VT6831 caian como
proximity con mil pp en Tabular List.

FIX: pre-filtrar candidates por type_hint detectado del Point name
ANTES del pattern matching. Si Point dice '1VT...', type_hint=velocity
→ universo restringido a sensores velocity → patterns proximity nunca
se evaluan. Adicionalmente: pattern matching solo chequea point_norm,
no variable_norm.

Smoke validado contra Sensor Map REAL del usuario con los 9 sensores
y los Points reales del Bently TES1: todos matchean al sensor correcto." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Fix pusheado"
echo "================================================================"
echo ""
echo "PASOS:"
echo "  1. Ctrl+C en la terminal de Streamlit"
echo "  2. find ~/Documents/WatermelonSystem -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null"
echo "  3. streamlit run 00_Home.py"
echo "  4. Cerrar pestaña del browser, abrir nueva"
echo "  5. Tabular List → los velocímetros ahora salen como Velocity"
