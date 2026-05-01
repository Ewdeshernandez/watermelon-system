#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.1.2 → DEV: Machine Map en PDF report
# =============================================================
# Inserta una sección "MAPA DE SENSORES" en el PDF report justo
# después del Resumen Ejecutivo y antes de Recomendaciones.
#
# La sección aparece automáticamente cuando hay una Asset
# Instance activa (schematic_instance_id en meta del reporte)
# con Sensor Map configurado. Si no hay, se omite limpio sin
# romper nada.
#
# Composición de la sección:
#
#   1. Título "MAPA DE SENSORES" en estilo WMTOC1 — entra a
#      la Tabla de Contenido automáticamente como una sección
#      más.
#
#   2. Párrafo de síntesis en prosa (no bullets, no tablas
#      markdown) listando totales por zona — coherente con
#      el resto del reporte:
#        "Del Sensor Map configurado para la unidad (N sensores
#        en total), X se mantienen en condición aceptable
#        contra los setpoints individuales del DCS, Y se
#        encuentran en zona de atención y Z requieren acción
#        inmediata. El heatmap a continuación ubica cada sonda
#        en su posición física sobre el tren acoplado y la
#        colorea según el estado actual contra los umbrales
#        de Alarm y Danger del propio sensor (no contra
#        defaults globales)."
#
#   3. Heatmap full (vista lateral + polar por plano)
#      renderizado con render_sensor_map_diagram en modo
#      severity_by_label. Caption corto centrado abajo.
#
#   4. Tabla drill-down de sensores con atención requerida
#      (Alarm + Danger) con: Sensor, Plano, Tipo, Overall,
#      Alarm, Danger, Unidad, Estado coloreado. Si todo
#      aceptable, se omite la tabla y se cierra con una
#      línea positiva en prosa.
#
# Reusa los helpers compartidos del Ciclo 15.1.1:
#   - core.machine_severity.build_severity_table
#   - core.machine_severity.count_status
#   - core.sensor_diagram.render_sensor_map_diagram
#
# Smoke: AST OK; imports lazy en su lugar; sección dentro de
# _build_pdf_bytes; _paragraph_safe / _fit_image_dimensions
# ya existían en el módulo.
#
# Archivos tocados:
#   - pages/16_Reports.py  (nueva sección entre Resumen Ejec y
#                            Recomendaciones)
#
# Ejecutar:
#   bash _publish_ciclo15_1_2_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/16_Reports.py
git status --short | head

git commit -m "feat(reports): seccion MAPA DE SENSORES en el PDF (Ciclo 15.1.2)

Inserta seccion dedicada justo despues del Resumen Ejecutivo y
antes de Recomendaciones. Aparece automaticamente cuando hay
Asset Instance activa con Sensor Map configurado.

Composicion:
* Titulo 'MAPA DE SENSORES' en WMTOC1 (entra al TOC).
* Parrafo de sintesis en prosa con totales por zona.
* Heatmap full (lateral + polar por plano) con severity_by_label.
* Drill-down table de sensores en Alarm/Danger (Sensor, Plano,
  Tipo, Overall, Alarm, Danger, Unidad, Estado coloreado).
  Si todo aceptable, prosa positiva en su lugar.

Reusa helpers compartidos del Ciclo 15.1.1:
  - core.machine_severity.build_severity_table
  - core.machine_severity.count_status
  - core.sensor_diagram.render_sensor_map_diagram

Smoke: AST OK, sección dentro de _build_pdf_bytes, helpers
internos (_paragraph_safe, _fit_image_dimensions) ya existian." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 15.1.2 pusheado a dev"
echo "================================================================"
echo ""
echo "Para verificar:"
echo "  1. Abrir el Reports en Streamlit con una instancia activa"
echo "     que tenga Sensor Map configurado."
echo "  2. Generar el PDF (boton 'Generar PDF')."
echo "  3. Despues del Resumen Ejecutivo y la TOC debe aparecer"
echo "     la seccion 'MAPA DE SENSORES' con heatmap + drill-down."
echo "  4. Si la unidad no tiene sensores con datos cargados, la"
echo "     seccion se renderea pero todos los planos salen grises"
echo "     (No Data) y la tabla drill-down se omite."
echo ""
echo "Cuando lo apruebes, mergeamos a main con un publish v2.5"
echo "(o v2.5 si juntamos con 15.1.1)."
echo "================================================================"
