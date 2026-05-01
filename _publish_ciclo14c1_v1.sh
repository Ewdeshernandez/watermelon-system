#!/bin/bash
# =============================================================
# Watermelon — Ciclo 14c.1: Sensor Map per-instancia (dev)
# =============================================================
# Cada Asset Instance tiene ahora un mapa de sensores configurado
# en Machinery Library, donde cada sensor describe su ubicacion
# fisica (API 670 / ISO 20816-1), tipo, unidad nativa y setpoints
# individuales del DCS. Tabular List clasifica cada CSV cargado con
# los thresholds del sensor que matchea, en lugar de un alarm/danger
# global incorrecto para sensores de distintos tipos.
#
# Resuelve el bug del screenshot del usuario: la fila TRF ACELL
# (acelerometro, g) salia con thresholds de proximity (3.000 / 3.500
# mil pp) y se clasificaba como Danger. Ahora cada acelerometro tiene
# sus propios thresholds (e.g. 4.5 / 9.0 g RMS).
#
# CAMBIOS:
#
# 1) NUEVO core/sensor_map.py (~280 lineas):
#    * new_sensor: constructor con defaults
#    * sensor_label: formato industrial 1Y_D, 3X_A, 2_RAD_V
#    * sensor_unit_family: Proximity / Velocity / Acceleration
#    * resolve_sensor_for_point: glob-match (case-insensitive) con
#      fallback heuristico por dirección X/Y + unidad
#    * generate_standard_sensor_map: pre-llena 8-10 sensores tipicos
#      (4 cojinetes × 2 sondas X-Y proximity API 670 a 45° R/L +
#       acelerometros radiales por cada cojinete del driven)
#    * Convencion API 670: planos numerados de driver(1,2) a
#      driven(3,4). Sondas a +45° R (X) y +45° L (Y).
#
# 2) core/instance_state.py:
#    * Instance.sensors: List[Dict] con default_factory=list
#    * from_dict deserializa sensors
#    * update_instance_header acepta sensors via kwargs
#    * Back-compat: instancias previas (sin sensors) siguen siendo
#      validas — sensors arranca como [].
#
# 3) pages/00_Machinery_Library.py:
#    * NUEVA seccion '📍 Mapa de Sensores' DESPUES del header
#      (fuera del form principal porque st.data_editor necesita
#      reactividad).
#    * Botones:
#      - 'Generar mapa estandar': pre-llena 10 sensores tipicos
#        (4 cojinetes proximity + acelerometros driven). Disabled
#        si ya hay sensores configurados.
#      - 'Limpiar mapa': borra todos los sensores. Disabled si vacio.
#      - 'Guardar mapa de sensores': persiste los cambios del
#        st.data_editor.
#    * st.data_editor con num_rows='dynamic' y column_config para
#      cada campo (plane / plane_label / side / angle_deg / direction
#      / sensor_type / unit_native / alarm / danger / csv_match_pattern
#      / notes), con dropdowns donde corresponde.
#    * Preview expander con labels formateados (1Y_D, 3X_A, etc.).
#
# 4) pages/01__Tabular_List.py:
#    * build_table_dataframe acepta sensors_map=List[Dict].
#    * Por cada record llama resolve_sensor_for_point antes que la
#      logica machine_settings/point_settings/global. Si encuentra
#      sensor matching, usa SUS valores granulares:
#      - alarm/danger del sensor
#      - family derivado del sensor_type (proximity → Proximity, etc.)
#      - overall_mode: PP para proximity, RMS para velocity/accel
#      - criterion: API 670 + ISO 7919-3/20816-3 para proximity,
#        ISO 20816-3 para casing
#    * Nueva columna 'Sensor' en el DataFrame con el label industrial
#      (1Y_D, 3X_A, etc.) cuando hay match.
#    * Si NO hay match, fallback al comportamiento legacy
#      (machine_settings → point_settings → global).
#    * Banner verde de Tabular List enriquecido: muestra '📍 Sensor
#      Map: N sensores configurados (X proximity + Y accelerometer)'
#      o warning 'Sensor Map vacio' si no hay nada configurado.
#
# Smoke integral validado:
# * Brush 54 MW + LM6000 con mapa estandar de 10 sensores.
# * Patterns ajustados a Point names reales (5807, 5808, 5809, 5810).
# * Match de 4 CSVs proximity → 1Y_D, 2X_D, 3Y_D, 4X_D con A=4.0 D=6.0
#   mil pp cada uno (thresholds del sensor, no del global).
# * Acelerometro 'TRF ACELL' requiere ajuste manual del pattern
#   (workflow normal post-generacion).
#
# Compatibilidad: NO toca matematica de overall_rms / harmonic_fit /
# render_table — solo cambia COMO se asignan thresholds por fila.
# Compile clean en core/sensor_map.py + core/instance_state.py +
# pages/00_Machinery_Library.py + pages/01__Tabular_List.py.
#
# Ejecutar:
#   bash _publish_ciclo14c1_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/sensor_map.py core/instance_state.py \
        pages/00_Machinery_Library.py pages/01__Tabular_List.py
git status --short | head

git commit -m "feat(library+tabular): Ciclo 14c.1 — Sensor Map per-instancia (dev)

Cada Asset Instance tiene ahora un mapa de sensores configurado en
Machinery Library con la convencion API 670 / ISO 20816-1 (planos
numerados driver→driven, sondas X-Y a 45° R/L). Tabular List
clasifica cada CSV cargado con los thresholds del sensor que matchea,
en lugar de un alarm/danger global incorrecto.

Resuelve el bug donde un acelerometro salia clasificado con
thresholds de proximity y se marcaba Danger por unidades distintas.

NUEVO core/sensor_map.py:
* new_sensor / sensor_label / sensor_unit_family / resolve_sensor_for_point
  / generate_standard_sensor_map.
* Convencion: 1Y_D = plano 1, dirección Y, Desplazamiento. 3X_A =
  plano 3, X, Aceleración.
* Resolución por glob csv_match_pattern (case-insensitive) + fallback
  heuristico por dirección X/Y + unidad (g/mil/mm-s).

core/instance_state.py:
* Instance.sensors: List[Dict] con back-compat (default []).
* from_dict deserializa, update_instance_header acepta sensors.

pages/00_Machinery_Library.py:
* Nueva seccion '📍 Mapa de Sensores' con st.data_editor, botones
  'Generar mapa estandar' / 'Limpiar mapa' / 'Guardar mapa'.
* Preview expander con labels industriales formateados.

pages/01__Tabular_List.py:
* build_table_dataframe acepta sensors_map. Por cada record llama
  resolve_sensor_for_point ANTES que machine/point/global settings.
* Match → usa alarm/danger/family/overall_mode/criterion del sensor.
* Nueva columna 'Sensor' con label industrial.
* Banner verde muestra 'Sensor Map: N sensores configurados (X
  proximity + Y accelerometer)' o warning si vacio.

Smoke integral validado: Brush 54 MW con 10 sensores generados,
patterns ajustados a Point names reales (5807-5810), 4 CSVs match
correctamente con thresholds individuales.

Compile clean. NO toca matematica de overall_rms / harmonic_fit /
render_table — solo cambia COMO se asignan thresholds por fila."

git push origin dev

echo ""
echo "================================================================"
echo " LISTO — Ciclo 14c.1 en dev"
echo "================================================================"
echo ""
echo "Plan de pruebas:"
echo ""
echo "  ===== A. Configurar mapa de sensores en Library ====="
echo "  1. Refrescar app. Login → Machinery Library."
echo "  2. Activar TES1 (badge verde)."
echo "  3. Scroll hasta '📍 Mapa de Sensores' (después del header)."
echo "  4. Click '🪄 Generar mapa estandar'."
echo "     → Aparecen 10 sensores: 8 proximity (planos 1-4 × X-Y a"
echo "       45° R/L) + 2 acelerometros radiales en planos 3-4."
echo "  5. Editar los csv_match_pattern para que matcheen tus Points"
echo "     reales. Para Brush con CSVs VE5807-VE5810:"
echo "     - Sensor 1X_D → pattern '*5807*x*'"
echo "     - Sensor 1Y_D → pattern '*5807*y*'"
echo "     - Sensor 2X_D → pattern '*5808*x*'"
echo "     - Sensor 2Y_D → pattern '*5808*y*'"
echo "     - Sensor 3X_D → pattern '*5809*x*'"
echo "     - Sensor 3Y_D → pattern '*5809*y*'"
echo "     - Sensor 3_RAD_A → pattern '*acell*' o '*trf*' (lo que sea)"
echo "  6. (Opcional) ajustar alarm/danger por sensor segun setpoints"
echo "     reales del DCS de Ecopetrol-Magnex."
echo "  7. Click '💾 Guardar mapa de sensores'."
echo "  8. Verificar el preview expander que muestra los 10 sensores"
echo "     con labels formateados (1X_D, 1Y_D, 2X_D, ..., 3_RAD_A)."
echo ""
echo "  ===== B. Validar match en Tabular List ====="
echo "  9. Click 'Load Data' → subir los 5 CSVs reales del usuario"
echo "     (WF 5807, WF 5808, WF 5809, WF 5810 + uno con TRF ACELL)."
echo " 10. Click 'Generate Time Waveforms'."
echo " 11. Click 'Tabular List'."
echo " 12. Verificar:"
echo "     - Banner verde arriba muestra '📍 Sensor Map: 10 sensores"
echo "       configurados (8 proximity + 2 accelerometer)'"
echo "     - Tabla con columna NUEVA 'Sensor' que muestra el label"
echo "       (1X_D, 1Y_D, etc.) por cada fila"
echo "     - VE5807(Y) → label 1Y_D, alarm/danger 4.0/6.0 mil pp,"
echo "       Status correcto"
echo "     - TRF ACELL → label 3_RAD_A, alarm/danger 4.5/9.0 g RMS"
echo "       (cuando ajustes el pattern), Status correcto"
echo "     - Cada fila clasifica con SUS thresholds reales, no con"
echo "       el global"
echo ""
echo "  ===== C. Test de override y casos edge ====="
echo " 13. (Opcional) Modificar manualmente alarm de algun sensor"
echo "     en Library → guardar → volver a Tabular List → la fila"
echo "     correspondiente reclasifica con el threshold nuevo."
echo " 14. (Opcional) En Library, 'Limpiar mapa' → en Tabular List"
echo "     vuelve al comportamiento legacy con un solo alarm/danger"
echo "     global."
echo ""
echo "Cuando confirmes que el flujo cierra, los proximos pasos:"
echo "  - Ciclo 14c.2: diagrama visual del mapa (polar dividido R/L"
echo "    con sondas marcadas en sus angulos físicos)."
echo "  - Ciclo 14c.3: usar el mapa en Time Waveforms / Spectrum /"
echo "    Polar / Bode / SCL para que también respeten thresholds"
echo "    individuales del sensor."
echo "  - Cuando dev este maduro: tag v2.1 + merge a main."
echo "================================================================"
