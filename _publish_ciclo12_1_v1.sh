#!/bin/bash
# =============================================================
# Watermelon — Ciclo 12.1: Time Waveform UX (dev only)
# =============================================================
# Pulida UX del módulo Time Waveforms en línea con Spectrum/Bode.
#
# Tres cambios concretos en pages/02_Time_Waveforms.py:
#
#   [1] AUTO-SELECT ALL al cargar:
#       Antes: si entraba sin selección persistida, mostraba sólo
#       el PRIMER waveform → cargabas 5/6 CSVs y veías 1 panel.
#       Ahora: muestra TODOS los waveforms cargados, mismo patrón
#       que ya tenías en Spectrum y Bode.
#
#       current_ids = [r.signal_id for r in records_all]   # all-by-default
#
#   [2] QUITADO bloque "Debug waveform":
#       Estaban impresos signals_count / metrics_count /
#       insights_count / impacts_count. Info técnica de developer
#       que no aporta a un reporte ingenieril. Eliminada junto con
#       sus contadores.
#
#   [3] DE-DUPLICACIÓN narrativa:
#       La sección "Análisis automático" tenía una narrativa
#       (insight_parts) que repetía con menor rigor lo que el
#       expander Cat IV ya dice con vocabulario de rotordinámica
#       (presesión reversa, deflexión térmica, normas ISO/API).
#
#       Ahora hay separación limpia:
#         · Sección "Análisis automático" = DATOS
#           (8 métricas + cantidad/threshold de transitorios)
#         · Expander Cat IV = INTERPRETACIÓN
#           (severidad + headline + detail + recomendaciones
#            numeradas + findings)
#
# Compatibilidad: NO toca core/. Sólo pages/02_Time_Waveforms.py.
# Compile clean (py_compile).
#
# Ejecutar:
#   bash _publish_ciclo12_1_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 12.1: Time Waveform UX (dev)"
echo "================================================================"
echo ""

[ -f .git/index.lock ] && rm -f .git/index.lock

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  echo "Cambiando a dev..."
  git checkout dev
fi
git pull origin dev

echo ""
echo "[1] Adoptando cambios..."
git add pages/02_Time_Waveforms.py
git status --short | head
echo ""

echo "[2] Commit..."
git commit -m "feat(waveform): Ciclo 12.1 — auto-select all + clean UX (dev only)

Pule la UX de Time Waveforms en linea con Spectrum/Bode.

[1] AUTO-SELECT ALL al cargar:
* Antes mostraba solo el primer waveform si no habia seleccion
  persistida. Ahora muestra TODOS los waveforms cargados,
  mismo patron de Spectrum/Bode.

[2] QUITADO bloque 'Debug waveform':
* Eliminados signals_count/metrics_count/insights_count/
  impacts_count y sus contadores. Info de developer, no de
  reporte ingenieril.

[3] DE-DUPLICACION narrativa:
* Removido insight_parts (kurtosis>4, CF>3, skewness, soporte
  cojinete vs rodamiento) que duplicaba con menor rigor lo que
  el expander Cat IV ya provee con vocabulario rotordinamica
  y normas citadas.
* Separacion limpia: 'Analisis automatico' = datos (metricas);
  expander Cat IV = interpretacion (severidad + recomendaciones).

Compatibilidad: NO toca core/. Solo pages/02_Time_Waveforms.py.
Compile clean."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 12.1 en dev"
echo "================================================================"
echo ""
echo "Validar en wm-test.streamlit.app:"
echo ""
echo "  1. Carga 4-6 CSVs de waveform desde Load Data."
echo ""
echo "  2. Andate a Time Waveforms → ahora deberian aparecer TODOS"
echo "     los paneles automaticamente (no solo el primero)."
echo ""
echo "  3. Bajo cada panel:"
echo "     - Diagnostico legacy (st.info corto)"
echo "     - Expander Cat IV (severidad + narrativa rotordinamica"
echo "       + recomendaciones numeradas con normas ISO/API)"
echo ""
echo "  4. Mas abajo, 'Analisis automatico de forma de onda':"
echo "     - 8 metricas (RMS, Peak, CF, Kurtosis, Mean, Std,"
echo "       Skewness, P2P) en cards"
echo "     - Cantidad de transitorios + Threshold dinamico"
echo "     - Indices detectados (caption)"
echo "     - YA NO aparece el bloque 'Debug waveform'"
echo "     - YA NO aparece la narrativa duplicada (esa info la da"
echo "       el expander Cat IV con mejor vocabulario)"
echo ""
echo "Cuando confirmes, podemos seguir con:"
echo "  - Ciclo 13 Orbit Cat IV (clasificador geometrico de orbitas)"
echo "  - Ciclo 10B Tabla 1 amplitudes en el reporte"
echo "================================================================"
