#!/bin/bash
# =============================================================
# Watermelon — Ciclo 14b.2: Tabular List auto-derivado (dev)
# =============================================================
# Tabular List es la primera figura del reporte. Antes pedia 5 inputs
# manuales (criterion / family / overall_mode / alarm / danger) que
# eran ruido para el usuario y NO matcheaban automaticamente al perfil
# tecnico real del activo.
#
# Ahora los 5 defaults se derivan automaticamente del Instance.header
# de la maquina activa en Machinery Library, usando un helper nuevo:
#
# CAMBIOS:
#
# 1) NUEVO core/tabular_defaults.py (~280 lineas):
#    Helper centralizado derive_tabular_defaults(instance) → dict.
#    Mapping logic:
#    * criterion: ISO 7919-3/20816-3 si fluid_film, ISO 20816-3 si
#      rolling_element, API 670 si magnetic
#    * machine_class: Class I/II/III/IV segun nominal_power_mw
#      (Brush 54 MW va a Class IV, motor pequeño va a Class II)
#    * family: Proximity si fluid_film/magnetic, Velocity si rolling
#    * overall_mode: PP para Proximity (estandar API 670), RMS para
#      Velocity/Acceleration
#    * alarm/danger: PRIORIDAD 1 = setpoints reales del activo
#      (alert_level/danger_level del DCS); PRIORIDAD 2 = tablas ISO
#      con conversion a unidad apropiada (mil pp para shaft, mm/s
#      RMS para casing). Convencion: Alert = zona B/C, Danger = zona C/D.
#    * Devuelve dict con explanation, sources, unit_hint para UI.
#
# 2) pages/01__Tabular_List.py:
#    * Sidebar: render_instance_selector(module_name='tabular') al
#      inicio + bloqueo con st.error+st.stop si no hay maquina activa.
#    * Banner verde arriba con: esquematico embebido, tag, train
#      description, criterio aplicado + explanation, alarm/danger con
#      sources (de instance setpoints o de tablas ISO).
#    * Eliminados los 5 inputs manuales del sidebar (criterion, family,
#      overall_mode, alarm, danger).
#    * Sigue config_mode (Criterion by Machine / by Point) — eleccion
#      de agrupacion del usuario, no automatizable.
#    * NUEVO expander 'Override criterio para este analisis (avanzado)'
#      colapsado por default. Caso de uso legitimo: comparar el mismo
#      set de datos contra criterios distintos (ej. ISO 20816-2 vs
#      ISO 7919-3). NO modifica la instancia en Library.
#    * Detector de override activo: cuando el usuario modifica algo
#      del expander, aparece warning amarillo arriba "Override criterio
#      activo" con resumen de los 5 valores efectivos. Asi nunca
#      sorprende al ingeniero.
#
# RESULTADO ESPERADO:
#
# Flujo del ingeniero post-ciclo 14b.2:
#   1. Library → activar TES1 (Brush 54 MW)
#   2. Load Data → subir CSVs (banner Ciclo 14b ya muestra que va a TES1)
#   3. Tabular List:
#      - Banner verde: 'Tabular List · TES1' + esquematico + criterio
#        ISO 7919-3/20816-3 Class IV @ 3600 rpm + alarm 5.118 mil pp
#        (zona B/C) + danger 8.268 mil pp (zona C/D)
#      - Sidebar: solo dropdown Configuration mode + expander avanzado
#      - Tabla con todos los signals clasificados con esos thresholds
#   4. Reports → primera figura sale auto-llenada con esa tabla.
#
# Smoke validado:
# * Brush 54 MW @ 3600 rpm fluid_film → ISO 7919-3, Class IV, Proximity,
#   PP, alarm 5.12 mil pp, danger 8.27 mil pp.
# * Con setpoints DCS reales (alert_level/danger_level) → esos overriden
#   las tablas ISO y aparece source 'setpoints reales del activo'.
# * Motor 110 kW rolling → ISO 20816-3, Class II, Velocity, RMS,
#   alarm 4.5 mm/s, danger 11.2 mm/s.
# * Sin instancia → defaults genericos.
#
# Compile clean. NO toca matematica de overall_rms / harmonic_fit /
# build_table_dataframe / render_table — solo cambia como se obtienen
# los defaults de configuracion.
#
# Ejecutar:
#   bash _publish_ciclo14b2_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/tabular_defaults.py pages/01__Tabular_List.py
git status --short | head

git commit -m "feat(tabular): Ciclo 14b.2 — Tabular List auto-derivado de Machinery Library (dev)

Reduce el setup manual de Tabular List de 5 inputs (criterion / family /
overall_mode / alarm / danger) a CERO. Todos los defaults se derivan
del Instance.header de la maquina activa via helper nuevo:

NUEVO core/tabular_defaults.py:
* derive_tabular_defaults(instance) → dict con criterion + family +
  overall_mode + alarm + danger + sources + explanation
* Mapping: support_type → criterion + measurement_type
  (fluid_film → ISO 7919-3/20816-3 shaft, rolling_element → ISO 20816-3
  casing, magnetic → API 670)
* nominal_power_mw → machine_class (Class I/II/III/IV de ISO 20816-3)
* Setpoints PRIORIDAD 1: alert_level/danger_level del activo (DCS).
  PRIORIDAD 2: tablas ISO con conversion de unidad apropiada
  (Alert=zona B/C, Danger=zona C/D)
* Smoke test: Brush 54 MW @ 3600 rpm → 5.12/8.27 mil pp Class IV.
  Motor 110 kW rolling → 4.5/11.2 mm/s RMS Class II.

pages/01__Tabular_List.py:
* Sidebar arranca con render_instance_selector + bloqueo si no hay activa.
* Banner verde arriba: esquematico + tag + train_description +
  criterio aplicado + explanation + alarm/danger con sources.
* 5 inputs default del sidebar ELIMINADOS.
* Expander avanzado 'Override criterio para este analisis' permite
  override puntual sin modificar la instancia (caso de uso: comparar
  ISO 20816-2 vs 7919-3 sobre la misma data). Detector de override
  activo muestra banner amarillo arriba.
* config_mode (Criterion by Machine / by Point) sigue como eleccion
  del usuario — no automatizable, depende de granularidad deseada.

Compile clean. NO toca matematica de overall_rms / harmonic_fit /
build_table_dataframe — solo cambia como se obtienen los defaults."

git push origin dev

echo ""
echo "================================================================"
echo " LISTO — Ciclo 14b.2 en dev"
echo "================================================================"
echo ""
echo "Plan de pruebas:"
echo ""
echo "  1. Refrescar app. Login → Machinery Library."
echo "  2. Activar TES1 (Brush 54 MW + LM6000) si no esta activa."
echo "  3. Click 'Load Data' → subir 3-5 CSVs de waveform o spectrum."
echo "     (Para Tabular List el flow tipico es subir 5-10 CSVs de"
echo "     puntos diferentes de la maquina)."
echo "  4. Click 'Generate Time Waveforms'."
echo "  5. Click 'Tabular List' en menu lateral."
echo "  6. Verificar:"
echo "     - Sidebar: 'Activo monitoreado: TES1' (selector con check verde)"
echo "     - Banner arriba: esquematico de TES1 + 'Tabular List · TES1'"
echo "       + 'Criterio: ISO 7919-3 / ISO 20816-3' + explanation"
echo "       + Alert: 5.118 mil pp (zona B/C) + Danger: 8.268 mil pp (zona C/D)"
echo "     - Sidebar 'Tabular List Setup': SOLO 'Configuration mode'"
echo "       + expander avanzado 'Override criterio...' colapsado"
echo "     - Tabla principal: cada fila clasificada como"
echo "       'CONDICION ACEPTABLE' / 'ATENCION' / 'ACCION REQUERIDA' /"
echo "       'CRITICA' segun los thresholds derivados."
echo ""
echo "  7. (Opcional) Test del override:"
echo "     - Expander 'Override criterio para este analisis'"
echo "     - Cambiar Alarm = 3.0, Danger = 5.0"
echo "     - Aparece banner amarillo 'Override criterio activo' arriba"
echo "     - La tabla se reclasifica con los nuevos thresholds"
echo "     - Cerrar expander y restaurar valores → banner desaparece"
echo ""
echo "  8. (Opcional) Si TES1 tiene setpoints DCS configurados"
echo "     (alert_level / danger_level en metadata Library), esos"
echo "     ganan sobre las tablas ISO y aparecen en banner como"
echo "     'setpoints reales del activo (mil pp)'."
echo ""
echo "  9. Click 'Enviar a Reporte' (si existe el boton) o seguir el"
echo "     flujo normal hasta Reports."
echo ""
echo " 10. En Reports → primera figura del PDF sera la tabular ya"
echo "     clasificada con los thresholds correctos del activo."
echo ""
echo "Cuando confirmes que el flujo se siente automatico, los proximos"
echo "candidatos serian:"
echo "  - Ciclo 14b.3: mismo banner + auto-defaults en Time Waveforms,"
echo "    Spectrum, Orbit, Trends (los modulos que aun no leen instance)."
echo "  - Ciclo 14c: filtro de signals por instance_id (cuando trabajas"
echo "    con multiples maquinas en sesion paralela)."
echo "  - Ciclo 13: Orbit avanzado con clasificador geometrico."
echo "================================================================"
