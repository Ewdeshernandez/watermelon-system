#!/bin/bash
# =============================================================
# Watermelon — Ciclo 12.2: Lenguaje cliente + fusión Análisis (dev)
# =============================================================
# Dos frentes pedidos por Ewdes:
#
# (A) "Cat IV / Categoría IV es jerga interna nuestra, no del cliente"
#     Para el cliente hablamos de NORMAS INTERNACIONALES y citamos
#     normas concretas (API 670, API 684, ISO 13373-1, ISO 7919,
#     ISO 20816, ISO 281, ISO 21940). 'Cat IV' queda SOLO como
#     comentario interno en .py — nunca en strings que terminan
#     en pantalla o en el PDF.
#
# (B) "El Análisis automático de forma de onda — para qué sirve?
#      Lo veo pero no va a ningún lado."
#     Razón: la sección mostraba 8 métricas + transitorios pero
#     era una vista huérfana — no llegaba al reporte ni se
#     correlacionaba con findings. Ahora esas métricas viven
#     DENTRO del expander avanzado por panel, junto a la narrativa
#     que SÍ las usa y SÍ va al reporte.
#
# CAMBIOS CONCRETOS:
#
# pages/16_Reports.py — OBJETIVO DEL SERVICIO (página 3 del PDF):
#   Antes:  "...e ISO 21940 para criterios de balanceo) bajo
#            prácticas de análisis Categoría IV del Vibration
#            Institute."
#   Ahora:  "...alineadas con las normas internacionales aplicables
#            al análisis avanzado de rotordinámica: API 670 para
#            instrumentación con sondas de proximidad, API 684 para
#            análisis rotodinámico, ISO 20816 para evaluación de
#            severidad de vibración mecánica e ISO 21940 para
#            criterios de balanceo."
#
# core/diagnostics.py — narrativa SCL al PDF:
#   Antes:  "...siguiendo prácticas Cat IV y los criterios de API 670"
#   Ahora:  "...bajo los criterios de API 670 para instrumentación
#            con sondas de proximidad y de API 684 para evaluación
#            rotodinámica avanzada"
#
# core/spectrum_diagnostics.py — narrativa Spectrum al PDF:
#   Antes:  "...firmas mecánicas Cat IV (1X / 2X / armónicos altos
#            / sub-sincrónicos / resonancia)"
#   Ahora:  "...firmas mecánicas (1X / 2X / armónicos altos /
#            sub-sincrónicos / resonancia) conforme a los criterios
#            de API 684 e ISO 13373-1"
#
# core/waveform_diagnostics.py — narrativa Waveform al PDF:
#   * Detail header:
#       Antes: "...aplica detectores Cat IV para identificar..."
#       Ahora: "...aplica detectores avanzados, conforme a los
#               lineamientos de ISO 13373-1 y API 670, para
#               identificar..."
#   * Crest factor finding:
#       Antes: "el bucket Cat IV {bucket}"
#       Ahora: "la zona {bucket}"
#   * Action intro:
#       Antes: "A partir del análisis Cat IV de la forma de onda..."
#       Ahora: "A partir del análisis avanzado de la forma de onda..."
#   * `structured` enriquecido: ahora expone `metrics` (rms, peak,
#     p2p, cf, kurt, skew, mean, std) e `impacts` (count, threshold)
#     como sub-objetos. Back-compat preservada (campos legacy en
#     nivel raíz).
#
# pages/03_Spectrum.py + pages/09_Shaft_Centerline.py: títulos de
#   expander cambiados de "Diagnóstico Cat IV (rotordynamics)" a
#   "Diagnóstico avanzado".
#
# pages/02_Time_Waveforms.py — fusión Análisis automático:
#   * Eliminada la sección huérfana "Análisis automático de forma
#     de onda" (~80 líneas con 8 métricas + transitorios sueltos).
#   * Las 8 métricas + transitorios ahora viven DENTRO del expander
#     "🔬 Diagnóstico avanzado" por panel, leídas de
#     cat_iv_wf_diag['structured']['metrics'] y ['impacts'] (sin
#     recálculo). Quedan al lado de la narrativa que las usa, y
#     todo lo que ve el ingeniero es lo que va al reporte.
#   * Título del expander: 'Cat IV (rotordynamics)' → 'avanzado'.
#
# Compatibilidad: NO toca compute_waveform_metrics, detect_impacts,
# generate_waveform_diagnostic legacy. Compile clean en los 7
# archivos modificados. Smoke runtime del wrapper enriquecido OK
# (severity, structured.metrics, structured.impacts validados).
#
# Auditoría AST final confirma 0 strings "Cat IV"/"Categoría IV"/
# "Vibration Institute" en código (excluyendo comentarios y
# docstrings) que puedan filtrarse a UI o PDF.
#
# Ejecutar:
#   bash _publish_ciclo12_2_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 12.2: Lenguaje cliente + fusión (dev)"
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
git add pages/16_Reports.py pages/03_Spectrum.py pages/09_Shaft_Centerline.py \
        pages/02_Time_Waveforms.py core/diagnostics.py \
        core/spectrum_diagnostics.py core/waveform_diagnostics.py
git status --short | head
echo ""

echo "[2] Commit..."
git commit -m "feat(language+ux): Ciclo 12.2 — lenguaje cliente + fusion analisis (dev)

(A) LENGUAJE CLIENTE — 'Cat IV / Categoria IV / Vibration Institute'
es jerga interna nuestra. Para el cliente hablamos de normas
internacionales aplicables y citamos normas concretas:

* pages/16_Reports.py: OBJETIVO DEL SERVICIO ahora dice
  'normas internacionales aplicables al analisis avanzado de
  rotordinamica' citando API 670, API 684, ISO 20816 e ISO 21940.
  La frase 'bajo practicas de analisis Categoria IV del Vibration
  Institute' fue eliminada.

* core/diagnostics.py: narrativa SCL del PDF cita 'criterios de
  API 670 + API 684' en lugar de 'practicas Cat IV'.

* core/spectrum_diagnostics.py: narrativa Spectrum del PDF cita
  'criterios de API 684 e ISO 13373-1' en lugar de 'firmas Cat IV'.

* core/waveform_diagnostics.py: narrativa Waveform del PDF cita
  'lineamientos de ISO 13373-1 y API 670' en lugar de 'detectores
  Cat IV'. Action intro y bucket de crest factor tambien limpios.

* pages/03_Spectrum.py + pages/09_Shaft_Centerline.py: titulos de
  expander en pantalla pasan de 'Diagnostico Cat IV' a
  'Diagnostico avanzado'.

(B) FUSION ANALISIS AUTOMATICO en pages/02_Time_Waveforms.py:
La seccion 'Analisis automatico de forma de onda' era huerfana —
mostraba 8 metricas + transitorios pero no llegaban al reporte ni
se correlacionaban con findings. Ahora esas metricas viven DENTRO
del expander 'Diagnostico avanzado' por panel, leidas de
cat_iv_wf_diag.structured.metrics e .impacts (sin recalculo).
Quedan al lado de la narrativa que las usa y SI van al reporte.

ENRIQUECIMIENTO de core/waveform_diagnostics.py: el dict
'structured' ahora expone 'metrics' (rms/peak/p2p/cf/kurt/skew/
mean/std) e 'impacts' (count/threshold) como sub-objetos.
Back-compat preservada (campos legacy en raiz).

VALIDADO: compile clean en 7 archivos; smoke runtime del wrapper
enriquecido OK; auditoria AST confirma 0 strings 'Cat IV' /
'Categoria IV' / 'Vibration Institute' en codigo (excluyendo
comentarios y docstrings) filtrables a UI o PDF."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 12.2 en dev"
echo "================================================================"
echo ""
echo "Validar en wm-test.streamlit.app:"
echo ""
echo "  1. Cargar 4-6 CSVs de waveform desde Load Data."
echo ""
echo "  2. Time Waveforms: deberian aparecer todos los paneles."
echo "     Bajo cada plot, el expander pasa a llamarse"
echo "     'Diagnostico avanzado' (sin 'Cat IV')."
echo "     Adentro del expander, ahora aparecen TAMBIEN las 8"
echo "     metricas (RMS, Peak, CF, Kurtosis, Mean, Std, Skewness,"
echo "     P2P) + cantidad de transitorios + threshold dinamico."
echo "     La seccion suelta 'Analisis automatico' al final ya no"
echo "     existe."
echo ""
echo "  3. Spectrum: el expander tambien dice 'Diagnostico avanzado'."
echo "     SCL: igual."
echo ""
echo "  4. Generar un reporte PDF — OBJETIVO DEL SERVICIO ya no"
echo "     menciona 'Categoria IV del Vibration Institute'. Cita"
echo "     normas: API 670, API 684, ISO 20816, ISO 21940."
echo "     Las narrativas por figura citan ISO 13373-1, API 670,"
echo "     API 684, ISO 281 sin etiquetar nada como 'Cat IV'."
echo ""
echo "Cuando confirmes:"
echo "  - Ciclo 13 Orbit avanzado (clasificador geometrico orbitas)"
echo "  - Ciclo 10B Tabla 1 amplitudes (NORMAL/ALARMA/DISPARO) en PDF"
echo "================================================================"
