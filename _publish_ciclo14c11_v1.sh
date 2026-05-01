#!/bin/bash
# =============================================================
# Watermelon — Ciclo 14c.1.1: Generador estándar diferenciado (dev)
# =============================================================
# Hace que generate_standard_sensor_map respete la realidad del campo:
# trenes con instrumentación heterogénea entre driver y driven (típico
# en turbogeneradores aero como TES1: LM6000 + Brush 54 MW).
#
# Antes (Ciclo 14c.1): asumía 8 proxímetros + 2 acelerómetros para
# todos los casos. Eso fallaba cuando el driver es turbina aero
# (rolling element con TRF/CRF) y el driven es generador con cojinetes
# planos (proxímetros X-Y) — situaciones del mundo real.
#
# CAMBIOS:
#
# 1) NUEVO tipo de sensor 'keyphasor' en core/sensor_map.py:
#    * _TYPE_TO_FAMILY: 'Phase Reference'
#    * _TYPE_TO_LETTER: 'K' (label industrial: 0_AX_K)
#    * _DEFAULT_UNIT: 'pulses/rev'
#    * Convención: plane=0 = coupling, axial, sin alarm/danger
#      (es referencia, no medición)
#
# 2) NUEVO modo de instrumentación 'accel_plus_velocity':
#    En cada plano genera 1 acelerómetro + 1 velocímetro radiales
#    juntos. Estándar moderno turbinas aero (LM6000, TM2500) que
#    instrumentan TRF y CRF con par accel+velocity.
#
# 3) Refactor de generate_standard_sensor_map:
#    * driver_instrumentation / driven_instrumentation reemplazan
#      driver_support_type / driven_support_type. Tres opciones:
#      - proximity_xy (default fluid_film, API 670 X-Y)
#      - axial_accel (rolling_element simple, 1 sensor por plano)
#      - accel_plus_velocity (turbinas aero modernas, 2 sensores
#        por plano)
#    * driver_plane_labels / driven_plane_labels: nombres custom por
#      plano (ej. ['CRF (LM6000)', 'TRF (LM6000)']).
#    * include_keyphasor: agrega 1 keyphasor en coupling al final.
#    * Setpoints separados por familia (proximity / accel / velocity).
#    * Back-compat completo con API previo (Ciclo 14c.1).
#
# 4) UI de Machinery Library:
#    * Form 'Generar mapa estándar' rediseñado con:
#      - Selectbox 'Instrumentación driver' con 3 opciones (etiquetas
#        humanas): 'Proxímetros X-Y (API 670)', 'Acelerómetro radial',
#        'Accel + Velocity (turbinas aero)'.
#      - Idem driven.
#      - Prefix Point CSV solo aparece si modo incluye acelerómetros.
#      - Checkbox 'Incluir keyphasor en coupling'.
#      - Default driver = accel_plus_velocity si support_type es
#        rolling_element (auto-derivación inteligente).
#    * Opción 'keyphasor' agregada al dropdown sensor_type del
#      data_editor (para edición manual).
#
# RESULTADO ESPERADO PARA TES1:
#
# Click 'Generar mapa con esta configuración':
#   - Driver = 2 planos, Accel + Velocity, prefix='CRF' (después
#     editar manual el plano 2 a 'TRF')
#   - Driven = 2 planos, Proxímetros X-Y
#   - Keyphasor en coupling = TRUE
#
# Resultado: 9 sensores
#   1_RAD_A (CRF accel) + 1_RAD_V (CRF velocity)
#   2_RAD_A (TRF accel) + 2_RAD_V (TRF velocity)
#   3X_D + 3Y_D (DE Brush proximity X-Y)
#   4X_D + 4Y_D (NDE Brush proximity X-Y)
#   0_AX_K (keyphasor en coupling)
#
# Para casos donde el cliente tenga 10+ sensores (ej. axiales extra,
# velocímetros adicionales), el ingeniero edita manualmente la tabla
# de sensores agregando filas con el botón '+' del data_editor.
#
# Smoke validados:
# * TES1 LM6000+Brush con accel_plus_velocity + proximity_xy +
#   keyphasor → 9 sensores correctos.
# * Compresor centrífugo (todo proximity_xy) → 8 sensores.
# * Motor + bomba (todo axial_accel) → 4 sensores.
# * Back-compat: API viejo sigue funcionando.
#
# Compile clean. NO toca matemática de overall_rms / harmonic_fit /
# resolve_sensor_for_point — solo enriquece el GENERADOR y la UI.
#
# Ejecutar:
#   bash _publish_ciclo14c11_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add core/sensor_map.py pages/00_Machinery_Library.py
git status --short | head

git commit -m "feat(library): Ciclo 14c.1.1 — generador estándar diferenciado driver/driven (dev)

Hace que generate_standard_sensor_map respete trenes con instrumentación
heterogénea entre driver y driven (típico turbogeneradores aero como
TES1: LM6000 + Brush 54 MW).

NUEVO tipo de sensor 'keyphasor':
* core/sensor_map.py: _TYPE_TO_FAMILY/LETTER/UNIT extendidos.
* Convención: plane=0 = coupling, axial, sin alarm/danger.
* Label industrial: 0_AX_K.

NUEVO modo de instrumentación 'accel_plus_velocity':
* Por cada plano genera 1 accel + 1 velocity radiales juntos.
* Estándar turbinas aero modernas (LM6000, TM2500) con TRF/CRF
  instrumentados completos.

generate_standard_sensor_map refactorizado:
* driver_instrumentation / driven_instrumentation reemplazan los
  support_type previos. 3 opciones: proximity_xy / axial_accel /
  accel_plus_velocity.
* driver_plane_labels / driven_plane_labels para nombres custom.
* include_keyphasor para agregar 1 keyphasor en coupling.
* Setpoints separados por familia.
* Back-compat completo con API previo (Ciclo 14c.1).

UI Machinery Library:
* Form 'Generar mapa' rediseñado: selectbox 'Instrumentación' con 3
  opciones (etiquetas humanas), prefix CSV solo si modo incluye
  acelerómetros, checkbox keyphasor.
* Default driver = accel_plus_velocity si support_type es
  rolling_element.
* 'keyphasor' agregado al dropdown sensor_type del data_editor.

Smoke validados:
* TES1 (accel_plus_velocity + proximity_xy + keyphasor) = 9 sensores.
* Compresor (proximity_xy + proximity_xy) = 8 sensores.
* Motor+bomba (axial_accel + axial_accel) = 4 sensores.
* Back-compat: API viejo sigue funcionando.

Compile clean."

git push origin dev

echo ""
echo "================================================================"
echo " LISTO — Ciclo 14c.1.1 en dev"
echo "================================================================"
echo ""
echo "Plan de pruebas para TES1:"
echo ""
echo "  1. Refrescar app. Library → TES1."
echo "  2. Mapa de Sensores → expandir 'Generar mapa estándar'."
echo "  3. Si ya hay sensores, marcar 'Confirmo sobreescribir'."
echo "  4. Configurar:"
echo "     Driver:"
echo "       - Planos: 2"
echo "       - Instrumentación: 'Accel + Velocity (turbinas aero, TRF/CRF)'"
echo "       - Prefijo Point CSV: 'CRF' (o 'acell' genérico)"
echo "     Driven:"
echo "       - Planos: 2"
echo "       - Instrumentación: 'Proxímetros X-Y (API 670, fluid_film)'"
echo "     Keyphasor: ✅ Incluir"
echo "  5. Click '🪄 Generar mapa con esta configuración'"
echo "  6. Resultado: 9 sensores ✓"
echo "     1_RAD_A + 1_RAD_V (CRF) | 2_RAD_A + 2_RAD_V (TRF) | "
echo "     3X_D + 3Y_D + 4X_D + 4Y_D (Brush) | 0_AX_K (coupling)"
echo ""
echo "  7. Editar manualmente:"
echo "     - Plane label del plano 2: 'CRF (LM6000)' → 'TRF (LM6000)'"
echo "     - Texto Point CSV del plano 2 accel: 'CRF' → 'TRF'"
echo "     - Texto Point CSV del plano 2 velocity: idem"
echo "     - Texto Point CSV del keyphasor: 'KPHGEN' o lo que tu CSV use"
echo "  8. '💾 Guardar mapa de sensores'"
echo ""
echo "  9. Load Data → subir todos los CSVs (TRF, CRF, VE5807-VE5810,"
echo "     KPHGEN). Tabular List ahora clasifica cada uno con el sensor"
echo "     correcto."
echo ""
echo "Próximos pasos sugeridos:"
echo "  - Ciclo 14c.2: diagrama visual del mapa (polar dividido R/L)"
echo "  - Ciclo 14c.3: usar el mapa también en Time Waveforms / Spectrum"
echo "    / Polar / Bode / SCL para que TODOS los módulos respeten los"
echo "    thresholds individuales del sensor."
echo "================================================================"
