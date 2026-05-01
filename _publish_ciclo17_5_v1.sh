#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.5 → DEV: Trend module clase mundial
# =============================================================
# Cierra el módulo de Trend con tres ejes:
#
#   ✅ Persistencia histórica de CSVs por instancia (corridas)
#   ✅ Polish visual del trend figure (zonas + health chip + slope)
#   ✅ Autodiagnóstico ejecutivo prosa Bently/ISO 20816
#
# == P1 — core/trend_history.py NUEVO ==
#
# A diferencia de Polar/Bode/SCL (snapshot de métricas derivadas),
# Trend persiste los CSV CRUDOS porque su valor está en la serie
# temporal completa de meses/años. Cada "corrida" = un upload
# (vibración + operacional) bajo:
#
#     {INSTANCES_DIR}/{instance_id}/trend_history/{corrida_id}/
#         metadata.json
#         files/
#             *.csv
#
# Funciones:
#   - save_trend_corrida(instance_id, files, label, notes,
#                        detected_time_range) -> corrida_id
#   - list_trend_corridas(instance_id) -> list[dict] (sin payload)
#   - load_trend_corrida_files(instance_id, corrida_id)
#         -> list[(file_name, csv_bytes)]
#   - delete_trend_corrida + get_corrida_metadata +
#     update_corrida_time_range + list_corridas_summary
#
# Auto-prune a 36 corridas (≈3 años de monitoreo mensual).
#
# == P2 — UI Histórico de Tendencias en pages/04_Trends.py ==
#
# Sidebar nueva sección "📚 Histórico de Tendencias":
#
#   - Selector de instancia activa (render_instance_selector)
#   - Resumen: N corridas archivadas + rango temporal global
#   - Expander "📸 Archivar corrida actual" con label/notas y
#     time_range auto-detectado
#   - Multiselect "Incluir corridas anteriores en el análisis"
#     que CONCATENA los CSVs históricos con la corrida actual
#     (parsers reciben _NamedBytesIO wrapping bytes archivados)
#   - Expander "Administrar corridas archivadas" con borrado
#
# Beneficio operativo: el ingeniero no conserva CSVs viejos
# localmente — el sistema los archiva. Para reportes anuales y
# post-mantenimiento el trend largo se reconstruye automáticamente.
#
# == P3 — Polish visual del trend figure ==
#
# build_trend_figure ahora dibuja:
#
#   - Bandas de severidad: zona Atención (warning→danger,
#     ámbar opacidad 0.08) + zona Acción (>danger, rojo opacidad
#     0.10) — gradiente instantáneo de severidad.
#
#   - Health chip top-right del strip con estado coloreado:
#       Normal (verde) / Vigilancia (azul, 85-100% Warning) /
#       Atención (ámbar, ≥Warning) / Acción Requerida
#       (rojo, ≥Danger).
#
#   - Línea bajo el chip con pendiente del último tramo
#     ({slope}/día) y forecast a Warning/Danger en días al
#     ritmo actual (regresión lineal sobre últimos 60 puntos).
#
# Margen top extendido a 120px para que el chip tenga aire.
#
# == P4 — Autodiagnóstico ejecutivo (build_trend_autodiagnostic) ==
#
# Síntesis Bently-style insertada ANTES de los detectores
# individuales (anomalías / drift / behavior change). Estructura:
#
#   - Headline ejecutivo (≤35 palabras) con status + cifra clave
#   - Párrafo 1: estado vs umbrales (% Warning + % Danger),
#     clasificación ISO 20816 (zona C/D)
#   - Párrafo 2: pendiente del último tramo + forecast a umbral
#     ("si la pendiente actual se mantiene...")
#   - Párrafo 3: anomalías puntuales (cantidad, severidad,
#     interpretación mecánica)
#   - Párrafo 4: drift progresivo (asociado a desgaste, drift
#     térmico, instrumentación, evolución de balance)
#   - Párrafo 5: cambio de régimen (transición clara vs ajuste
#     operacional)
#   - Párrafo 6: vínculo operacional cuando hay variables de
#     proceso (sugiere usar correlación con desfase para
#     distinguir cambio operacional vs degradación mecánica)
#   - Recomendaciones numeradas según status
#
# Vocabulario alineado con ISO 20816, API 670 §6.7 y los
# manuales Bently Nevada Technical Training.
#
# Ejecutar:
#   bash _publish_ciclo17_5_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/trend_history.py
git add pages/04_Trends.py
git add _publish_ciclo17_5_v1.sh
git status --short | head

git commit -m "feat(trend): Ciclo 17.5 — Trend module clase mundial (P1+P2+P3+P4)

Cierra el módulo de Trend al nivel de Polar/Bode/SCL con paridad
completa: persistencia histórica + polish visual + autodiagnostico
ejecutivo Bently/ISO 20816.

P1 — core/trend_history.py NUEVO. A diferencia de polar/bode/scl
history (snapshot de metricas derivadas), Trend persiste los CSV
CRUDOS bajo {INSTANCES_DIR}/{instance_id}/trend_history/{corrida_id}/
con metadata.json + files/*.csv. save_trend_corrida + list +
load + delete + get_metadata + update_time_range +
list_corridas_summary. Auto-prune a 36 corridas (3 años mensual).

P2 — UI Historico de Tendencias en pages/04_Trends.py.
Sidebar nueva seccion con selector de instancia activa, resumen
de corridas, expander Archivar corrida actual con label/notas y
time_range auto-detectado, multiselect Incluir corridas anteriores
que concatena CSVs historicos al analisis actual, y expander
Administrar corridas con boton de borrado. Helper _NamedBytesIO
wrapping bytes para que los parsers existentes funcionen sin cambio.

P3 — Polish visual del trend figure. Bandas de severidad
(warning->danger ambar 0.08, >danger rojo 0.10) + health chip
top-right (Normal/Vigilancia/Atencion/Accion Requerida) +
linea de pendiente y forecast a Warning/Danger basado en
regresion lineal de los ultimos 60 puntos. Margen top 120px.

P4 — Autodiagnostico ejecutivo (build_trend_autodiagnostic).
Sintesis Bently-style insertada antes de los detectores con
headline, 6 parrafos en prosa (estado vs umbrales + ISO 20816,
pendiente + forecast, anomalias, drift, cambio de regimen,
vinculo operacional) y recomendaciones numeradas por status.
Vocabulario API 670 / ISO 20816 / Bently Technical Training." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.5 (Trend clase mundial) pusheado a dev"
echo "================================================================"
echo ""
echo "Para verlo end-to-end:"
echo "  1. Trends page → seleccionar instancia activa en sidebar."
echo "  2. Cargar CSVs trend + operacional."
echo "  3. Sidebar 'Histórico de Tendencias' → Archivar corrida actual."
echo "  4. Limpiar uploaders, recargar, abrir multiselect de corridas"
echo "     anteriores → ver concatenación cronológica."
echo "  5. El gráfico ahora muestra:"
echo "     - Bandas amber/red en zonas Warning y Danger."
echo "     - Health chip Normal/Vigilancia/Atención/Acción."
echo "     - Pendiente y forecast a umbral abajo del chip."
echo "  6. Bajo el gráfico: 🩺 Autodiagnóstico ejecutivo con prosa"
echo "     Bently/ISO 20816 + recomendaciones numeradas."
echo ""
echo "Estado del histórico multi-modulo:"
echo "  Tabular  ✅ snapshot + comparativo + EVOLUCIÓN PDF"
echo "  Polar    ✅ snapshot + multi-overlay + narrativa Bently"
echo "  Bode     ✅ snapshot + multi-overlay + narrativa Bently"
echo "  SCL      ✅ snapshot + multi-overlay + narrativa Bently/API 670"
echo "  Trend    ✅ corridas CSV completas + autodiag + polish visual"
echo "  Spectrum ⏳ Ciclo 17.6 (próximo)"
echo "================================================================"
