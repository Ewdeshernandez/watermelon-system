#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.1: Machine Map con heatmap de severidad (dev)
# =============================================================
# Primer paso del Ciclo 15: tomar el diagrama del Sensor Map del
# Ciclo 14c.2 y agregarle colores de severidad por sensor según el
# status calculado contra los thresholds individuales.
#
# CAMBIOS:
#
# 1) core/sensor_diagram.py:
#    * Nuevo parámetro severity_by_label: Dict[str, str] que mapea
#      sensor_label → status ("Normal" / "Alarm" / "Danger" / "No Data").
#    * Cuando se provee, los markers se colorean por severidad
#      (verde / amarillo / rojo / gris). Cuando no, mantienen el
#      color por tipo (modo configuración del Library).
#    * Leyenda dinámica: si severity → "CONDICIÓN ACEPTABLE / ATENCIÓN
#      / ACCIÓN REQUERIDA / Sin datos". Si no → "Proximity / Velocity
#      / Accelerometer / Keyphasor".
#    * Paleta _COLOR_SEVERITY coherente con cintas Cat IV.
#
# 2) NUEVA pages/01b_Machine_Map.py:
#    * Lee instancia activa + sensors + signals cargados.
#    * Para cada sensor, busca el signal que matchea via
#      resolve_sensor_for_point. Si encuentra: calcula RMS de la
#      señal, lo convierte a unit_native del sensor (peak/pp/rms),
#      clasifica contra alarm/danger del sensor.
#    * Renderiza el diagrama con colores de severidad.
#    * Resumen con 4 metrics (CONDICIÓN ACEPTABLE / ATENCIÓN /
#      ACCIÓN REQUERIDA / Sin datos).
#    * Tabla drill-down de "Sensores con atención requerida".
#    * Tabla completa colapsada.
#    * Bloqueo si no hay activa O si el activo no tiene sensors.
#
# 3) core/auth.py:
#    * NAV_ITEMS agrega "Machine Map" entre Tabular List y Time
#      Waveforms (orden lógico del flujo: data → tabla → mapa visual
#      → módulos detallados).
#
# RESULTADO:
#
# El ingeniero ve EN UN SOLO VISTAZO el estado de todos los sensores
# del activo. Verde = OK, amarillo = ATENCIÓN, rojo = DANGER. La
# tabla drill-down lista exactamente qué sensores requieren acción.
#
# Smoke validado: TES1 con 9 sensores configurados (2 accel CRF +
# 2 vel CRF/TRF + 4 prox X-Y + 1 keyphasor), severidades simuladas
# mezcladas → PNG de 91 KB con colores correctos.
#
# Compile clean. NO toca matemática de Tabular List ni de Reports —
# solo consume los thresholds y status para visualizar.
#
# Ejecutar:
#   bash _publish_ciclo15_1_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/sensor_diagram.py pages/01b_Machine_Map.py core/auth.py
git status --short | head

git commit -m "feat(machine-map): Ciclo 15.1 — Machine Map con heatmap de severidad (dev)

Primera fase del Ciclo 15: heatmap de severidad por sensor sobre el
diagrama del Sensor Map del Ciclo 14c.2. El ingeniero ve en un solo
vistazo dónde están los problemas antes de entrar al detalle.

core/sensor_diagram.py:
* Nuevo parametro severity_by_label: Dict[str, str].
* Cuando se provee, markers se colorean por severidad (verde Normal /
  amarillo Alarm / rojo Danger / gris Sin datos). Cuando no, mantienen
  color por tipo (modo configuracion).
* Leyenda dinamica segun modo.
* Paleta coherente con cintas Cat IV de los expanders.

pages/01b_Machine_Map.py (NUEVO):
* Lee instancia activa + sensors + signals cargados.
* Para cada sensor: busca signal que matchea via
  resolve_sensor_for_point. Calcula RMS, convierte a unit_native
  (peak/pp/rms), clasifica contra alarm/danger del sensor.
* Renderiza diagrama con colores de severidad.
* Metrics: CONDICION ACEPTABLE / ATENCION / ACCION REQUERIDA / Sin datos.
* Tabla drill-down 'Sensores con atencion requerida' ordenada por
  severidad.
* Tabla completa colapsada con todos los sensores y su matching.
* Bloqueo si no hay activa o sin Sensor Map configurado.

core/auth.py NAV_ITEMS:
* Machine Map agregado entre Tabular List y Time Waveforms (orden
  logico del flujo: data → tabla → mapa visual → modulos detallados).

Smoke validado: TES1 con 9 sensores y severidades mixtas (1 Alarm + 1
Danger + resto Normal) → PNG 91 KB con colores correctos.

Compile clean. NO toca matematica de Tabular List ni Reports."

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 15.1 listo en dev"
echo "================================================================"
echo ""
echo "Plan de pruebas (dev):"
echo ""
echo "  1. Refrescar app (dev branch local o redeploy de la app dev)"
echo "  2. Login → menu lateral muestra 'Machine Map' entre Tabular"
echo "     List y Time Waveforms ✓"
echo "  3. Si TES1 NO esta activa → click 'Machine Map' → bloqueo con"
echo "     mensaje 'No hay máquina activa'"
echo "  4. Activar TES1 en Library, ir a Load Data, subir CSVs"
echo "  5. Click 'Machine Map' →"
echo "     - Banner verde con esquematico + train description"
echo "     - 4 metrics arriba: CONDICION ACEPTABLE / ATENCION /"
echo "       ACCION REQUERIDA / Sin datos"
echo "     - Diagrama con TODOS los sensores coloreados segun"
echo "       severidad (verde/amarillo/rojo/gris)"
echo "     - Si hay sensores en zona Alarm o Danger, aparece tabla"
echo "       'Sensores con atencion requerida' debajo"
echo "     - Tabla completa colapsada al final"
echo "  6. Editar manualmente algun threshold del Sensor Map en"
echo "     Library para forzar uno en zona Alarm → volver a Machine"
echo "     Map → ver el cambio de color"
echo ""
echo "Próximos pasos:"
echo "  - Ciclo 15.2 (futuro, opcional): coordenadas 2D explicitas"
echo "    sobre la imagen real del activo (click-to-place)"
echo "  - Ciclo 15.1.1: integrar mini-mapa en Tabular List arriba"
echo "    de la tabla"
echo "  - Ciclo 15.1.2: integrar Machine Map como seccion del"
echo "    reporte PDF despues del Resumen Ejecutivo"
echo ""
echo "Cuando confirmes que funciona, hacemos publish + merge a main"
echo "(otra vuelta del flujo v2.x)."
echo "================================================================"
