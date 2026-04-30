#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.1.5 → DEV: keyphasor + esquemático vivo
# =============================================================
# Dos refinamientos sobre el feedback del segundo PDF v2.5:
#
# (1) Diferenciar Keyphasor de sensores de vibración.
# Antes la prosa decía "9 sensores configurados". Pero un
# keyphasor es una sonda once-per-rev sobre el coupling/eje
# que provee referencia de fase para orbits, Polar, Bode y
# balanceo — NO mide amplitud de vibración, no se evalúa
# contra setpoints de Alarm/Danger en el Tabular. Reportarlo
# como "un sensor de vibración más" da una imagen incorrecta
# del Sensor Map al cliente.
#
# Ahora count_status separa keyphasor del conteo de vibración
# (devuelve 'vibration_total' y 'keyphasor' además de 'total').
# La prosa del PDF dice ahora algo como:
#   "consta de 8 sensores de vibración de monitoreo continuo
#   distribuidos a lo largo del tren acoplado, complementados
#   por 1 señal de referencia de fase (keyphasor) instalada
#   sobre el eje del rotor para sincronización de medidas
#   vectoriales (orbits, Polar y Bode) y diagnóstico de
#   fenómenos rotodinámicos según API 670."
#
# (2) Esquemático vivo en el Resumen Ejecutivo.
# Antes el Resumen Ejecutivo mostraba el schematic_png estático
# del Vault — solo decoración. El usuario quería que el
# esquemático ENTREGUE información: valores Overall por plano
# coloreados por severidad (verde/ámbar/rojo).
#
# Ahora cuando hay Sensor Map + signals en sesión, el Resumen
# Ejecutivo renderiza un mini-heatmap con la silueta
# turbomachinery (igual a la del Mini Map del Tabular) PERO
# con valores Overall del peor sensor por plano coloreados
# por severidad. El cliente abre la primera página y ve de un
# vistazo "TRF 3.85 g pk" en ámbar, "Driven NDE 1.20 mil pp"
# en verde. El esquemático funciona como dashboard ejecutivo.
#
# Si no hay Sensor Map / sesión / signals, fallback al
# schematic_png estático (Ciclo 14a) — ningún reporte se rompe.
#
# Archivos:
#   - core/machine_severity.py   (count_status separa keyphasor)
#   - core/sensor_diagram.py     (parametros overall_by_label +
#                                  unit_by_label en compact mode)
#   - pages/16_Reports.py        (prosa keyphasor + esquemático
#                                  vivo en Resumen Ejecutivo)
#
# Smoke validado:
#   * count_status sobre 9 sensores (8 vib + 1 kp): vibration_total=8,
#     keyphasor=1, alarm=2, normal=6.
#   * Render compact con overall_by_label muestra valores numéricos
#     bajo cada plano coloreados por severidad.
#
# Ejecutar:
#   bash _publish_ciclo15_1_5_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add core/machine_severity.py
git add core/sensor_diagram.py
git add pages/16_Reports.py
git status --short | head

git commit -m "feat(reports): keyphasor split + esquematico vivo en Resumen Ejecutivo (Ciclo 15.1.5)

(1) count_status ahora separa keyphasor del conteo de vibracion.
Devuelve 'vibration_total', 'keyphasor', y desglose normal/alarm/
danger/no_data SOLO sobre vibracion. Antes la prosa decia '9
sensores configurados' incluyendo el kp; ahora dice '8 sensores
de vibracion + 1 senal de referencia de fase (keyphasor)
instalada sobre el eje del rotor para sincronizacion de medidas
vectoriales y diagnostico rotodinamico segun API 670'.

(2) Esquematico vivo en el Resumen Ejecutivo del PDF. Antes
mostraba el schematic_png estatico del Vault como decoracion.
Ahora renderiza un mini-heatmap con silueta turbomachinery +
valores Overall del peor sensor por plano coloreados por
severidad (verde/ambar/rojo). El cliente abre la primera pagina
y ve 'TRF 3.85 g pk' en ambar, 'CRF 3.10 g pk' en ambar,
'Driven NDE 1.20 mil pp' en verde — dashboard ejecutivo
inmediato.

Render acepta nuevos parametros overall_by_label y unit_by_label
para mostrar valores bajo cada cojinete. Solo activos en
compact mode. Fallback al schematic_png estatico cuando no hay
sensor map o sesion del Tabular." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 15.1.5 pusheado a dev"
echo "================================================================"
echo ""
echo "Verificacion en Streamlit (1-2 min para redeploy):"
echo "  1. Abrir Tabular List (esto guarda df en session_state)."
echo "  2. Ir a Reports y generar PDF."
echo "  3. RESUMEN EJECUTIVO debe mostrar el esquematico VIVO con:"
echo "       - LM6000 silueta turbina + cojinetes 1, 2 ambar"
echo "         (etiquetas TRF / CRF + valor Overall en g pk)"
echo "       - Brush silueta generador + cojinetes 3, 4 verdes"
echo "         (etiquetas Driven NDE / DE + valor Overall en mil pp)"
echo "       - Keyphasor sobre el coupling (estrella ambar 'kp')"
echo "  4. MAPA DE SENSORES debe decir '8 sensores de vibracion +"
echo "     1 senal de referencia de fase (keyphasor)' en lugar de"
echo "     '9 sensores'."
echo ""
echo "Si todo se ve bien, juntamos 15.1.1+...+15.1.5 en merge a"
echo "main como v2.5 definitivo."
echo "================================================================"
