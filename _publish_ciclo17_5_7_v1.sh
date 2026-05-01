#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.5.7: Saneamiento del PDF (3 fixes)
# =============================================================
# Tres correcciones que vienen de revisión del PDF generado en
# producción contra una corrida de TES1 LM6000 (data 2 horas,
# transient de arranque):
#
# (1) FORECAST INVÁLIDO "~0 días"
#     PDF mostraba "cruce de umbral proyectado en ~0 días" tres
#     veces porque la pendiente lineal del último tramo era
#     +45.9 mil pp/día (totalmente irreal) y el cálculo
#     `(target - latest) / slope` daba ~0.026 días que con
#     `%.0f` se redondeaba a "0 días".
#
#     Causa raíz: el slope-based forecast no es válido durante
#     transients de arranque o cuando la cola es altamente
#     inestable. La regresión lineal sobre 60 puntos de un
#     transient extrapola la transitoria como si fuera tendencia
#     operacional.
#
#     Fix en _compute_trend_health():
#       - Ventana total < 24h → forecast suprimido
#       - Coef. de variación de la cola > 50% → suprimido
#       - days_to_target < 0.5 días → suprimido (físicamente irreal)
#       - Cuando se emite, format `max(1, round(days))` evita "0 días"
#
#     En la prosa: si hay slope pero forecast suprimido, ahora
#     decimos "ventana actual demasiado corta o cola demasiado
#     inestable como para emitir un forecast lineal confiable;
#     se sugiere repetir la medición con al menos 24h de datos".
#     Honestidad sobre confabulación.
#
#     Headlines y recomendaciones también actualizados para que
#     no muestren "0 días" en ningún path.
#
# (2) CONTRADICCIÓN "comportamiento estable" vs Strong change
#     El PDF tenía dos párrafos pegados en la misma figura:
#       "Se identifican 313 eventos puntuales... Strong change..."
#       "El comportamiento es estable y sin desviaciones..."
#
#     Causa raíz: build_trend_report_narrative_core (legacy)
#     usa _classify_trend_behavior que clasificaba como "stable"
#     basado en slope_ratio/jerk/volatility relativos al span,
#     incluso cuando el cambio absoluto era 7088% (0.025 → 1.8).
#
#     Fix en _classify_trend_behavior:
#       Guardrail anti-contradicción. Si abs(change_pct) >= 100%
#       (la señal cambió más del doble entre primer y último
#       valor), clasificar como "stable" es categoricamente
#       erróneo. Se promueve a "abrupt" (cuando jerk/volatility
#       lo justifica) o "progressive_increase/decrease" según
#       direction.
#
# (3) RESUMEN EJECUTIVO no escalaba con Trend
#     PDF decía "Estado global: CONDICIÓN ACEPTABLE" + "no se
#     identifican acciones de prioridad alta" mientras la única
#     figura reportaba pendiente +45.9, 313 spikes y Strong
#     change. El extractor de findings sólo leía SCL/Polar/Bode/
#     ISO/critical_speeds — los items de Trend pasaban inadvertidos.
#
#     Fix en pages/16_Reports.py:
#       - _extract_findings_from_items lee item.autodiagnostic.status
#         y item.behavior_summary.top_classification cuando type
#         == "trends". Si status in (alarm, action) o behavior
#         == Strong change, agrega a `trend_states` y a
#         `high_priority_actions`.
#       - _global_severity escala el rank global usando
#         _TREND_STATUS_RANK (ok=0, watch=1, alarm=2, action=3)
#         y Strong change → al menos rank 2 (ATENCIÓN).
#       - _compose_executive_summary añade hallazgos Trend a la
#         prosa principal con el mismo nivel de detalle que SCL/
#         Polar/Bode.
#
#     Resultado: si la corrida real tiene Trend en alarm/action
#     o un Strong change, el Resumen Ejecutivo ya NO dirá
#     "CONDICIÓN ACEPTABLE" — escalará a ATENCIÓN o ACCIÓN
#     REQUERIDA y citará el hallazgo concreto.
#
# Ejecutar:
#   bash _publish_ciclo17_5_7_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add pages/04_Trends.py
git add pages/16_Reports.py
git add core/trend_diagnostics.py
git add _publish_ciclo17_5_7_v1.sh
git status --short | head -10

git commit -m "fix(reports): saneamiento PDF — forecast invalido, contradiccion estable, Resumen Ejecutivo escala con Trend (17.5.7)

Tres fixes que vienen de revision del PDF generado contra una
corrida de TES1 LM6000 (transient de arranque, 2h de data):

(1) Forecast '~0 dias' invalidado. _compute_trend_health
suprime forecast cuando ventana < 24h, coef. variacion cola
> 50%, o days_to_target < 0.5 (fisicamente irreal). En la
prosa, headlines y recomendaciones se quita '0 dias';
cuando hay slope pero forecast suprimido, decimos honestamente
'ventana insuficiente para forecast lineal confiable'.

(2) Contradiccion 'comportamiento estable' vs Strong change
arreglada en _classify_trend_behavior. Guardrail: si
abs(change_pct) >= 100% (señal duplicada o mas), clasificar
como stable es erroneo — se promueve a abrupt o
progressive_increase/decrease segun jerk/volatility.

(3) Resumen Ejecutivo no escalaba cuando Trend estaba en
alarm/action o Strong change. Ahora _extract_findings lee
item.autodiagnostic.status y behavior_summary.top_classification
para items type=trends, agrega a trend_states y a
high_priority_actions cuando aplica. _global_severity
escala via _TREND_STATUS_RANK (ok/watch/alarm/action ->
0/1/2/3) + Strong change -> minimo rank 2.
_compose_executive_summary cita el hallazgo concreto en la
prosa principal." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.5.7 (saneamiento PDF) pusheado a dev"
echo "================================================================"
echo ""
echo "Para verificar:"
echo "  1. Trends → cargar corrida corta (<24h) → diagnostico debe"
echo "     decir 'ventana insuficiente para forecast' en lugar de"
echo "     '0 dias'."
echo "  2. Cargar corrida con cambio grande (e.g. arranque 0->1.8) →"
echo "     ya no debe aparecer 'comportamiento estable' en el PDF."
echo "  3. Si Trend tiene alarm/action o Strong change, el Resumen"
echo "     Ejecutivo del PDF debe escalar a ATENCION o ACCION."
echo "================================================================"
