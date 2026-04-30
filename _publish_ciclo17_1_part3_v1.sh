#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.1 P3+P4+P5 → DEV: trail visual + PDF
# =============================================================
# Cierra el Ciclo 17.1: el snapshot anterior elegido en el
# sidebar ahora se VE sobre el polar mismo (no solo en la tabla)
# y la comparativa se incluye en el PDF Reports cuando enviás
# la figura al reporte.
#
# Cambios en pages/06_Polar_Plot.py:
#
# (P3) build_polar_figure acepta nuevos params opcionales:
#   prev_snapshot_amp, prev_snapshot_phase, prev_snapshot_label,
#   prev_snapshot_op_speed.
#
#   Cuando se pasan + hay operating_rpm valido, el polar dibuja:
#     1. Marker GHOST (estrella hueca gris) en el (amp, phase)
#        de la corrida anterior, con tooltip de Δamp + Δphase.
#     2. Linea conectora dotted roja desde la corrida anterior
#        al punto operativo actual (segmento de 12 puntos para
#        que en el polar quede visualmente recto).
#     3. Marker Op Speed actual (estrella negra) sobre todos
#        los anteriores.
#
#   Si la fase del anterior fue 142° y la actual es 178°, el
#   shift se ve fisicamente como una flecha que cruza ~36°
#   sobre el plano polar.
#
# (P4) render_polar_panel busca el snapshot elegido del session
#   state, identifica el sensor matched para ESTE panel y extrae
#   la lectura previa (amp_at_op, phase_at_op). Pasa los datos
#   a build_polar_figure.
#
# (P5) Inyeccion de narrativa comparativa en text_diag para que
#   el PDF Reports incluya un parrafo ingenieril abajo de la
#   figura cuando esta en el reporte:
#     'Comparativo de balance contra "Coastdown abril 19 2026"
#     del 2026-04-19. A la velocidad operativa (3597 rpm), el
#     sensor pasó de 0.850 mil pp @ 142.5° a 1.200 mil pp @
#     178.0°. La amplitud 1X varió en +0.350 mil pp (+41.2%) y
#     la fase 1X muestra un shift de +35.5° (cambio circular de
#     menor arco). El shift de fase 1X entre 30° y 60° es
#     sintoma diagnostico clasico de cambio de balance del rotor
#     (API 684, ISO 21940-12). Se recomienda programar
#     verificacion de balance en proxima ventana de
#     mantenimiento. El crecimiento de amplitud 1X acompañando
#     al shift de fase refuerza el diagnostico de degradacion
#     activa.'
#
#   _build_polar_report_notes acepta nueva clave
#   'comparison_narrative' en text_diag y la inyecta entre el
#   detalle del diagnostico y las acciones priorizadas.
#
# Como funciona end-to-end:
#   1. Usuario carga corrida polar, va a sidebar Histórico Polar,
#      guarda snapshot 'Coastdown abril 19'.
#   2. Carga corrida nueva, sidebar muestra 'Comparar contra:
#      Coastdown abril 19'.
#   3. Polar plot del panel ahora muestra:
#      * Estrella negra al actual (Op nominal)
#      * Estrella gris hueca al anterior (Snapshot abril 19)
#      * Linea dotted roja conectando ambas
#      * Tooltip al hover del ghost: Δamp y Δphase
#   4. Tabla 'Comparativo Polar' arriba con diagnóstico textual
#      por sensor.
#   5. Usuario clickea 'Send to Report' → la figura va al PDF
#      con el overlay incluido + las notas con la narrativa
#      comparativa ingenieril.
#
# Ejecutar:
#   bash _publish_ciclo17_1_part3_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/06_Polar_Plot.py
git status --short | head

git commit -m "feat(polar): trail visual sobre polar + narrativa PDF (Ciclo 17.1 P3-P5)

Cierra el Ciclo 17.1. El snapshot anterior elegido en el sidebar
ahora se VE sobre el polar mismo (estrella gris hueca + linea
conectora dotted al actual) y la comparativa se incluye en el
PDF Reports cuando se envia la figura al reporte.

(P3) build_polar_figure acepta prev_snapshot_amp/phase/label/
op_speed. Cuando se pasan + operating_rpm valido, dibuja marker
ghost en (amp, phase) anterior + linea dotted al actual + el
ghost tiene tooltip con Δamp y Δphase.

(P4) render_polar_panel busca el snapshot elegido en session
state, resuelve el sensor matched al panel actual, extrae
lectura previa y la pasa a build_polar_figure.

(P5) Inyeccion de comparison_narrative en text_diag con prosa
ingenieril citando shift de fase >30° como sintoma cambio de
balance (API 684 / ISO 21940-12). _build_polar_report_notes
incluye la narrativa entre el detalle y las acciones.

Como funciona: usuario carga corrida + guarda snapshot + carga
nueva corrida + comparar contra snapshot anterior → polar
muestra ghost + flecha + tabla comparativa + al enviar a reporte
la figura va al PDF con overlay y narrativa ingenieril." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.1 completo (P3+P4+P5) pusheado a dev"
echo "================================================================"
echo ""
echo "Para verlo:"
echo "  1. Abrir Polar Plot, cargar corrida + guardar snapshot."
echo "  2. Cargar otra corrida (cambiar archivos)."
echo "  3. En sidebar, elegir el snapshot anterior en 'Comparar"
echo "     contra corrida anterior'."
echo "  4. El polar plot ahora muestra:"
echo "     * Estrella negra (Op actual)"
echo "     * Estrella gris hueca con label 'Anterior · ...'"
echo "     * Linea dotted roja conectando ambas"
echo "     * Hover sobre el ghost = Δamp + Δphase"
echo "  5. Click 'Send to Report' → figura al PDF con narrativa"
echo "     comparativa ingenieril."
echo ""
echo "Ya el flujo Polar history end-to-end esta listo. Próxima"
echo "iteración: aplicar el mismo patron a Bode, SCL, Spectrum."
echo "================================================================"
