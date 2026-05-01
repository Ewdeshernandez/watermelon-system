#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.5 v2 → DEV: Trend P2-P4
# =============================================================
# Tres frentes pedidos tras revisión del módulo Trend:
#
# (P2) Thresholds Warning/Danger sugeridos desde el Vault con
#      override editable del cliente.
# (P3) Trend completo al PDF (gráfico + autodiagnóstico
#      ejecutivo Bently + recomendaciones + fuente de setpoints).
# (P4) HD export con ambos ejes (eje izquierdo vibración + eje
#      derecho operacional) sin que se corten ni se solapen.
#
# == P2 — THRESHOLDS DESDE EL VAULT ==
#
# Nuevo helper suggest_trend_thresholds(records, sensors,
# metric_key, machine_group) que para los CSVs cargados
# resuelve cada sensor contra el Sensor Map de la instancia y
# devuelve los setpoints más conservadores (mínimo de alarmas
# y mínimo de dangers). Jerarquía de fallback:
#
#   1. Sensor Map per-instance (Vault). Cada sensor tiene sus
#      `alarm` y `danger` en unit_native (mil pp para proximity,
#      mm/s RMS para velocity, g RMS para accelerometer).
#   2. ISO 20816 por machine_group (class I/II/III/IV).
#      Class IV → 2.8/4.5 mm/s, class III → 2.8/7.1 mm/s, etc.
#   3. Defaults Bently 3500 → 3.0/5.0 mil pp para proximity.
#
# UI: chip de fuente (Sensor Map / ISO 20816 / Override) arriba
# de los inputs Warning/Danger; botón "Aplicar setpoints
# sugeridos" que limpia el override; caption "⚙️ Override del
# cliente" cuando el valor manual difiere de la sugerencia.
# Estado persistido en st.session_state["wm_tr_threshold_source"]
# para que el reporte cite la fuente exacta.
#
# Caso real: la norma dice 4 mil pp pero el cliente pide 3 mil
# pp como criterio conservador → escribe manualmente 3.000 y el
# sistema lo respeta + lo documenta en el PDF como Override.
#
# == P3 — TREND COMPLETO AL PDF ==
#
# _send_to_report() ahora prepende al narrative del reporte:
#   - AUTODIAGNÓSTICO EJECUTIVO (headline)
#   - 6 párrafos en prosa Bently/ISO 20816 (estado vs umbrales,
#     pendiente + forecast, anomalías, drift, cambio de
#     régimen, vínculo operacional)
#   - Acciones recomendadas numeradas según status
#   - Linea de fuente de setpoints (Sensor Map / ISO 20816 /
#     Override del cliente con sugerido)
#
# Además el item_payload ahora incluye:
#   - autodiagnostic: { headline, prose[], recommendations[],
#                       status, status_label }
#   - threshold_source: { warning_value, danger_value,
#                         suggested_warning, suggested_danger,
#                         source, detail, *_is_override,
#                         machine_group }
#
# Para que el renderer del PDF (Reports page) pueda destacar
# visualmente el bloque autodiag con su color de status.
#
# == P4 — HD EXPORT CON DOBLE EJE ==
#
# Bug previo: cuando el panel estaba en mixed mode (vibración +
# operacional), el secondary axis quedaba en position=0.80
# fijo, lo que dejaba parte del eje fuera del área visible.
# Y `to_image` ignoraba el width dinámico calculado.
#
# Fix:
#   - Coordenadas dinámicas:
#       has_secondary_y SIN info → xaxis [0,0.935], yaxis2 @ 0.945
#       has_secondary_y CON info → xaxis [0,0.72],  yaxis2 @ 0.735
#       solo info box (sin sec)  → xaxis [0,0.72]
#       caso default              → fullwidth
#   - build_export_png_bytes ahora respeta export_fig.layout.width
#     (4200 / 4700 / 4900 según contenido) en lugar de hardcoded.
#
# Resultado: el PNG HD muestra ambos ejes, etiquetas legibles,
# sin solape entre el eje secundario y el info box derecho.
#
# Ejecutar:
#   bash _publish_ciclo17_5_v2.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/04_Trends.py
git add _publish_ciclo17_5_v2.sh
git status --short | head

git commit -m "feat(trend): Ciclo 17.5 v2 — Vault thresholds + Trend al PDF + HD export doble eje (P2+P3+P4)

P2 — Thresholds Warning/Danger sugeridos desde el Sensor Map de
la instancia activa con override editable del cliente. Nuevo
helper suggest_trend_thresholds() resuelve cada record contra
el Sensor Map y toma los setpoints mas conservadores. Fallback
jerarquico Vault -> ISO 20816 (class I/II/III/IV por
machine_group, 2.8/4.5 mm/s para class IV) -> defaults
Bently 3500 (3.0/5.0 mil pp). UI con chip de fuente, boton
'Aplicar setpoints sugeridos' y caption 'Override del cliente'
cuando difiere. Estado persistido para que el reporte cite la
fuente. Caso de uso real: norma dice 4 mil pp pero cliente
exige 3 mil pp conservador.

P3 — Trend completo al PDF. _send_to_report() prepende al
narrative el autodiagnostico ejecutivo (headline + 6 parrafos
Bently/ISO 20816 + recomendaciones numeradas) y la linea de
fuente de setpoints. item_payload incluye campos estructurados
'autodiagnostic' y 'threshold_source' para que el PDF renderer
pueda destacar visualmente el bloque con color de status.

P4 — HD export con doble eje fixed. Antes el secondary axis
quedaba en position 0.80 fijo y to_image ignoraba el width
dinamico, dejando parte del eje fuera. Ahora coordenadas
dinamicas: has_secondary SIN info xaxis [0,0.935] yaxis2@0.945,
has_secondary CON info xaxis [0,0.72] yaxis2@0.735. PNG HD
ahora muestra ambos ejes legibles sin solape." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.5 v2 (P2+P3+P4) pusheado a dev"
echo "================================================================"
echo ""
echo "Para verlo:"
echo "  1. Trends page → seleccionar instancia con Sensor Map (TES1)."
echo "  2. Cargar CSV → la sidebar Alarms muestra 'Setpoints sugeridos:"
echo "     Sensor Map' con detalle del bearing."
echo "  3. Cambiar Warning a 3.000 → caption muestra Override."
echo "  4. Click Send to Report → el PDF incluye:"
echo "     - Bloque AUTODIAGNOSTICO EJECUTIVO arriba"
echo "     - 6 parrafos prosa Bently"
echo "     - Recomendaciones numeradas"
echo "     - Linea de fuente: 'Setpoints: Sensor Map · Override del"
echo "       cliente activo (sugeridos: W=4 / D=5)'"
echo "  5. Prepare PNG HD en mixed mode → ambos ejes visibles, eje"
echo "     derecho con curva operacional sin solape."
echo "================================================================"
