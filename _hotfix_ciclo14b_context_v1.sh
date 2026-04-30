#!/bin/bash
# =============================================================
# Watermelon — HOTFIX Ciclo 14b: Machine Diagnostic Context auto-derivado
# =============================================================
# El usuario detecto que la seccion 'Machine Diagnostic Context' del
# Load Data quedo redundante despues del Ciclo 14a/14b: pedia que el
# ingeniero tipeara manualmente Asset type, Machine configuration,
# Primary equipment, Secondary equipment y Machine technical description,
# pero TODA esa info ya vive en Instance.header de la maquina activa
# (Machinery Library). Tener que tipearlo dos veces era ruido + bloqueo
# (los 5 campos eran requeridos y deshabilitaban "Generate Time
# Waveforms" si faltaba alguno).
#
# Fix: la seccion ahora se auto-deriva de la instancia activa:
#   asset_type            ← inst.asset_class
#   machine_configuration ← 'Compuesta / tren' si driver+driven, sino 'Simple'
#   primary_equipment     ← driver_manufacturer + driver_model
#   secondary_equipment   ← driven_manufacturer + driven_model
#   machine_description   ← compose_train_description(inst)
#
# Para overrides puntuales (raro) hay un expander '⚙️ Override manual
# del contexto de máquina' colapsado por default, donde puede modificar
# cualquier campo sin tocar la instancia en la Library.
#
# Cero validaciones requeridas, cero errores bloqueantes — el flujo
# fluye porque los datos siempre estan completos. El boton 'Generate
# Time Waveforms' queda habilitado apenas hay CSVs validos cargados.
#
# Ejecutar:
#   bash _hotfix_ciclo14b_context_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add pages/01_Load_Data.py
git status --short | head

git commit -m "fix(load-data): hotfix Ciclo 14b — Machine Diagnostic Context auto-derivado

La seccion Machine Diagnostic Context pedia tipear manualmente 5 campos
(Asset type / Machine configuration / Primary / Secondary / Description)
que ya vivian en Instance.header de la maquina activa post-Ciclo 14a.
Era ruido + bloqueo (campos requeridos deshabilitaban 'Generate Time
Waveforms').

Fix: los 5 campos ahora se auto-derivan de _active_instance:
* asset_type ← asset_class
* machine_configuration ← 'Compuesta / tren' si driver+driven, sino 'Simple'
* primary_equipment ← driver_manufacturer + driver_model
* secondary_equipment ← driven_manufacturer + driven_model
* machine_description ← compose_train_description(inst)

Override manual disponible en expander '⚙️ Override manual del
contexto de máquina' (colapsado por default, no toca la instancia en
Library). Cero validaciones bloqueantes — el boton 'Generate Time
Waveforms' habilita apenas hay CSVs validos."

git push origin dev

echo ""
echo "Refrescar — la seccion Machine Diagnostic Context ya no aparece"
echo "como bloque manual. En su lugar hay un expander discreto"
echo "'Override manual del contexto de máquina' (colapsado por default)."
echo "El boton 'Generate Time Waveforms' habilita inmediatamente despues"
echo "de cargar CSVs validos."
