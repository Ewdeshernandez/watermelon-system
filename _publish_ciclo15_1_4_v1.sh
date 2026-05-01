#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.1.4 → DEV: Machine Map alineado al Tabular
# =============================================================
# Cambio arquitectural: el Machine Map ahora es una visualización
# gráfica del Tabular List, no un cálculo paralelo de severidad.
#
# Antes (15.1.0–15.1.3): el Machine Map calculaba RMS por su
# cuenta sobre las signals de la sesión y eso producía un Status
# que potencialmente difería del que mostraba el Tabular en la
# misma corrida (caso real: Tabular veía 2 ATENCIÓN en CRF/TRF
# ACELL, Machine Map seguía diciendo "9 aceptables").
#
# Ahora: el Tabular guarda su DataFrame en
# st.session_state["wm_tabular_df"] al final de su render
# (con columna Sensor=label del Sensor Map, Overall, Status,
# Alarm, Danger, Unit Full). El helper build_severity_table de
# core.machine_severity primero intenta proyectar este df sobre
# la geometría del Sensor Map; si no está, cae al cálculo legacy.
# Esto garantiza que Machine Map (página completa, Mini Map
# arriba del Tabular y sección del PDF) y el Tabular nunca puedan
# contradecirse en la misma sesión.
#
# Bonus 1: la prosa de la sección Machine Map del PDF se
# reescribe en clave ingenieril:
#   - Habla del Sensor Map como conjunto distribuido sobre el
#     tren acoplado.
#   - Diferencia sensores cargados vs sin dato.
#   - Cuando hay Atención/Acción Requerida, lista el sensor con
#     mayor consumo de margen por NOMBRE: "El sensor con mayor
#     margen consumido es 1_RAD_A (TRF) con un Overall de 3.840
#     g peak sobre un Danger de 6.000 g peak, equivalente al
#     64% del umbral. Se recomienda priorizar la verificación
#     de este punto en el siguiente ciclo de inspección."
#   - Cuando todo está aceptable, igual menciona el de mayor
#     margen consumido como evidencia del juicio.
#   - Cierra explicando que los chips circulares bajo cada
#     cojinete indican los tipos de sensor presentes en ese
#     plano (proximidad, velocidad, acelerómetros).
#
# Bonus 2: Machine Map respeta exactamente los overrides de
# criterio del usuario (cuando el ingeniero ajusta alarm/danger
# manualmente en el sidebar avanzado del Tabular, esos valores
# se reflejan automáticamente en el heatmap del PDF).
#
# Archivos:
#   - pages/01__Tabular_List.py   (cachear df en session_state)
#   - core/machine_severity.py    (fast path desde df Tabular,
#                                   legacy fallback intacto)
#   - pages/16_Reports.py         (prosa ingenieril mencionando
#                                   sensores criticos por nombre
#                                   y % de margen consumido)
#
# Ejecutar:
#   bash _publish_ciclo15_1_4_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/machine_severity.py
git add pages/01__Tabular_List.py
git add pages/16_Reports.py
git status --short | head

git commit -m "feat(machine-map): alineado con Tabular como fuente unica de verdad (Ciclo 15.1.4)

Cambio arquitectural: Machine Map ahora es una visualizacion grafica
del Tabular List, no un calculo paralelo. El Tabular guarda su
DataFrame en session_state['wm_tabular_df'] y build_severity_table
lo proyecta sobre la geometria del Sensor Map. Esto garantiza que
nunca puedan contradecirse en la misma sesion.

(1) pages/01__Tabular_List.py — al final del render guarda
df_table.copy() en st.session_state['wm_tabular_df'] junto con
'wm_tabular_active_instance_id'. La columna Sensor (label del
Sensor Map) es la clave de proyeccion.

(2) core/machine_severity.py — build_severity_table usa fast path
con df cacheado primero (proyecta Sensor -> overall/status/
alarm/danger/unit). Si no hay df (Reports / Machine Map abiertos
sin pasar por Tabular), cae al calculo legacy sobre los signals
de session_state['signals']. El legacy ahora es robusto a los 3
formatos: SimpleNamespace.x, .amplitude, dict.

(3) pages/16_Reports.py — prosa de la seccion MAPA DE SENSORES
reescrita en clave ingenieril. Menciona por NOMBRE el sensor con
mayor margen consumido del Danger ('El sensor con mayor margen
consumido es 1_RAD_A (TRF) con un Overall de 3.840 g peak sobre
un Danger de 6.000, equivalente al 64% del umbral'). Distribucion
de severidad orientada a lo critico primero. Cierre que explica
los chips circulares de tipo de sensor.

Smoke: con df Tabular sintetico de 3 Alarms, build_severity_table
proyecta correctamente; sin df cacheado el legacy con SimpleNamespace
sigue funcionando." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 15.1.4 pusheado a dev"
echo "================================================================"
echo ""
echo "Verificacion en Streamlit (1-2 min para redeploy):"
echo "  1. Ir a Tabular List — esto guarda el df en session_state."
echo "  2. Volver a generar el PDF Reports — la seccion MAPA DE"
echo "     SENSORES debe ahora coincidir EXACTAMENTE con el Status"
echo "     y Overall del Tabular (mismas 2 ATENCIÓN sobre CRF/TRF"
echo "     ACELL al 64%/52% del danger respectivamente)."
echo "  3. La prosa debe mencionar al sensor con mayor margen"
echo "     consumido por nombre."
echo ""
echo "Cuando lo veas alineado, lo juntamos con 15.1.1+15.1.2+15.1.3"
echo "en un merge a main como v2.5 definitivo."
echo "================================================================"
