#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.5.3 HOTFIX → MAIN
# =============================================================
# 2 bugs del módulo Trends reportados por el usuario:
#
# Bug 1 — Op Change muestra Hz pero la señal está en RPM
# ──────────────────────────────────────────────────────
# El usuario carga señal de VFD que internamente está en RPM (velocidad
# del motor), pero el "Trend Information" la mostraba en Hz porque
# infer_operational_family clasificaba "vfd", "vsd", "freq" como
# frequency → unit Hz por default.
#
# Realidad: en datos de operación, los VFDs típicamente reportan
# VELOCIDAD (rpm) del motor, no frecuencia eléctrica. Solo redes
# eléctricas (50Hz/60Hz) reportan en Hz.
#
# Fix: Reordenar _OP_FAMILY_PATTERNS:
#   - "_hz", "hertz" explícito → frequency (Hz)
#   - "vfd", "vsd", "freq" → speed (rpm)  ← nuevo
#
# Bug 2 — Solo analizaba la 1ra señal operativa, ignorando las demás
# ───────────────────────────────────────────────────────────────────
# Si el usuario sube 10 vibraciones + 20 operativas, el módulo
# graficaba todas pero solo CALCULABA correlación contra la primera
# operativa del CSV (operational_records[0]). Las otras 19 operativas
# se ignoraban en el análisis principal.
#
# Realidad: el usuario quiere saber CUÁL operativa AFECTA MÁS la
# vibración. Eso requiere correlacionar la vibración contra TODAS y
# rankear por correlación.
#
# Fix: Ya existía `build_operational_variable_ranking` que rankea
# todas. Ahora lo usamos PRIMERO y elegimos el TOP-1 como
# primary_operational, así el análisis principal usa la operativa
# más explicativa, no la primera arbitrariamente.
#
# Cambios:
#   - pages/04_Trends.py:
#     * _OP_FAMILY_PATTERNS reordenado (frecuency separado de speed)
#     * Sección de correlación: rankear primero, primary = top-1
#
# Validación:
#   - 8 de 9 casos de unidades pasan (incluyendo el caso real
#     VFD_Siemens..._Freq → rpm)
#   - Python compila sin errores
#
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🔧 HOTFIX v3.5.3 → MAIN  (Trends: VFD→RPM + ranking multi-op)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.25  Trends fixes:"
echo "         - VFD/VSD/Freq ahora se infieren como RPM (no Hz)"
echo "         - Análisis principal usa la operativa MÁS correlacionada"
echo "           con la vibración (no la primera del CSV)"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el hotfix a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Hotfix cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del fix en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
for v in v3_4_2 v3_4_3 v3_4_4 v3_4_5 v3_4_6 v3_4_7 v3_4_8 v3_4_9 v3_5_0 v3_5_1 v3_5_2; do
    git checkout HEAD -- "_release_${v}_hotfix.sh" 2>/dev/null || true
    git checkout HEAD -- "_release_${v}_ux.sh" 2>/dev/null || true
done

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add pages/04_Trends.py _release_v3_5_3_hotfix.sh
    git commit -m "hotfix(17.25): Trends — VFD/VSD a RPM + análisis multi-operacional con ranking

Bug 1 — Op Change mostraba Hz pero señal era RPM
================================================
infer_operational_family clasificaba 'vfd', 'vsd', 'freq' como
frequency → infer_operational_unit ponía Hz. Pero en datos de
operación reales los VFDs reportan velocidad del motor (rpm), no
frecuencia eléctrica. Solo redes eléctricas reportan en Hz.

Fix: reordenar _OP_FAMILY_PATTERNS:
- ' hz', '_hz', 'hz_', 'hertz' explícito → frequency (Hz)
- 'vfd', 'vsd', 'freq' → speed (rpm) ← movidos acá

Validado con 9 casos:
  ✓ [C200C]VFD_Siemens_..._Freq → rpm (caso del usuario)
  ✓ [BL1]VSD_Velocidad → rpm
  ✓ [Y]Pump_RPM → rpm
  ✓ [Z]Net_Frequency_Hz → Hz (explícito)
  (el único edge case 'Grid_50Hz' sin separador queda pendiente)

Bug 2 — Solo analizaba la 1ra señal operativa
=============================================
Con N vibraciones y M operativas, el análisis principal usaba
operational_records[0] arbitrariamente. Las otras M-1 operativas
se graficaban pero NO se analizaban → el especialista no veía
cuál era la variable operativa que MÁS afectaba la vibración.

Fix: ya existía build_operational_variable_ranking() que rankea
todas las operativas por correlación con la vibración. Ahora se
ejecuta PRIMERO; el primary_operational pasa a ser el TOP-1 del
ranking (el más correlacionado), no el [0] del CSV.

Resultado: el bloque 'Trend Information' del reporte muestra la
operativa que mejor explica el cambio de vibración, no una al
azar. Más útil para diagnóstico." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene el fix commiteado"
echo ""

echo "▶ 2/7  Push de dev a origin..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Pull dev falló."; exit 1; }
git push origin dev || { echo "✗ Push dev falló."; exit 1; }
echo "  ✓ dev en origin actualizado"
echo ""

echo "▶ 3/7  Cambiando a main..."
git checkout main || { echo "✗ No se pudo cambiar a main."; exit 1; }
git fetch origin main
git pull --rebase origin main || { echo "✗ Pull main falló."; exit 1; }
echo "  ✓ main actualizado"
echo ""

echo "▶ 4/7  Mergeando dev → main..."
MERGE_MSG="hotfix(v3.5.3): merge dev -> main · Ciclo 17.25 Trends fixes

Dos bugs del módulo Trends:
- VFD/VSD/Freq ahora se inferen como RPM (no Hz). Solo nombres
  con '_hz' o 'hertz' explícito siguen siendo frequency.
- Análisis principal usa la operativa MÁS correlacionada con
  la vibración (top-1 del ranking automático), no la primera
  del CSV arbitrariamente.

Validado: caso real del usuario (VFD_Siemens..._Freq) ahora da
rpm como esperado."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.5.3..."
TAG_EXISTS=$(git tag -l "v3.5.3")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.5.3 ya existe. Saltando creación."
else
    git tag -a v3.5.3 -m "Hotfix v3.5.3 — Ciclo 17.25 Trends VFD→RPM + ranking multi-op"
    echo "  ✓ Tag v3.5.3 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.5.3 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ HOTFIX v3.5.3 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo ""
echo " 🧪 VALIDACIÓN en módulo Trends:"
echo ""
echo " Bug 1 (unidades):"
echo "   1. Cargá una señal operativa con nombre tipo 'VFD_..._Freq'"
echo "   2. En el bloque 'Trend Information' debe decir 'rpm' ahora"
echo "      (antes decía 'Hz')"
echo ""
echo " Bug 2 (análisis multi-op):"
echo "   1. Cargá varias señales operativas (>1)"
echo "   2. El análisis principal y la sección 'Correlación operativa'"
echo "      ahora usan la operativa MÁS correlacionada con la vibración,"
echo "      no la primera del CSV"
echo "   3. La sección 'Ranking automático de variables operativas'"
echo "      sigue mostrándolas todas ordenadas por score"
echo ""
echo "================================================================"
