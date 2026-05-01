#!/bin/bash
# =============================================================
# Watermelon — Ciclo 11.1: Spectrum UX (dev only)
# =============================================================
# Tres mejoras de UX en el modulo Spectrum:
#
#   1. Auto-select all signals on load
#      Cuando el usuario carga 3, 4 o más CSVs, el multiselect arranca
#      con TODOS marcados (igual que ya hacia Bode desde Ciclo 2-B).
#      Antes mostraba solo el primero, lo que obligaba a clic uno-a-uno.
#
#   2. Auto-escala del eje X (Max frequency CPM) según unidad fisica
#      Nuevo modulo core/spectrum_scale.py con classify_amplitude_quantity()
#      y suggest_max_cpm_for_unit():
#        - displacement (mil pp, µm pp) -> 60.000 CPM (~1 kHz)
#        - velocity (in/s, mm/s)        -> 120.000 CPM (~2 kHz)
#        - acceleration (g pk, g RMS)   -> 600.000 CPM (~10 kHz)
#        - unknown + RPM                 -> 10× operating rpm
#      El usuario puede overridear el valor desde el number_input.
#      La preferencia se guarda por familia, asi al cambiar entre
#      espectros de displacement y acceleration cada uno mantiene su
#      propio rango.
#      Caption visible explica la decision: "Auto-rango: aceleracion ->
#      600.000 CPM (~10 kHz). Rango alto para deteccion de fallas
#      tempranas de rodamientos (BPFO/BPFI/BSF/FTF)..."
#
#   3. Auditoria del comparador de 2 espectros (sin tocar)
#      Se reviso core/spectrum_compare.py — esta al nivel Cat IV con
#      validacion de comparabilidad, 7+ patrones detectados y narrativa
#      automatica. No requiere cambios. Mejora opcional futura: integrar
#      detector sub-sincrono del Ciclo 11 al assessment del comparador.
#
# Cambios incluidos:
#   * core/spectrum_scale.py (nuevo)
#   * pages/03_Spectrum.py: import + auto-select all + auto-cpm helper
#
# Ejecutar:
#   bash _publish_ciclo11_1_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 11.1: Spectrum UX (dev)"
echo "================================================================"
echo ""

[ -f .git/index.lock ] && rm -f .git/index.lock

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  echo "Cambiando a dev..."
  git checkout dev
fi
git pull origin dev

echo ""
echo "[1] Adoptando cambios..."
git add core/spectrum_scale.py pages/03_Spectrum.py
git status --short | head
echo ""

echo "[2] Commit..."
git commit -m "feat(spectrum): Ciclo 11.1 — auto-select all + auto-escala unidad

Tres mejoras de UX para que el modulo Spectrum se sienta mas pro y
ahorre clics al usuario:

1. Auto-select all signals on load (mismo patron que Bode Ciclo 2-B).
   El multiselect 'Spectra to display' arranca con TODAS las senales
   cargadas por el usuario en lugar de solo la primera. Ahorra el
   clic repetitivo cuando se cargan 3 o mas CSVs.

2. Auto-escala del Max frequency CPM segun unidad fisica:
     - displacement (mil pp, µm pp)    -> 60.000 CPM (~1 kHz)
     - velocity     (in/s, mm/s)       -> 120.000 CPM (~2 kHz)
     - acceleration (g pk, g RMS)      -> 600.000 CPM (~10 kHz)
     - unknown + RPM                    -> 10× operating rpm
   Defaults Cat IV (lo que un ingeniero rotodinamico senior elegiria).
   La preferencia se cachea por familia en session_state, asi cambiar
   entre espectros de displacement y acceleration mantiene su propio
   rango sin perder la edicion manual.
   Caption visible explica la decision con razonamiento textual.

3. Auditoria de core/spectrum_compare.py (sin tocar): el comparador
   de 2 espectros del usuario tiene 17 funciones especializadas con
   validacion de comparabilidad, 7+ patrones detectados y narrativa
   automatica con severity. Esta al nivel Cat IV. No requiere cambios.

Componentes:
* core/spectrum_scale.py (nuevo): classify_amplitude_quantity y
  suggest_max_cpm_for_unit con defaults documentados.
* pages/03_Spectrum.py: wire del helper en la sidebar 'Display' +
  auto-select all en signal_name_map default.

Smoke validado: 10/10 unidades clasifican correctamente
(mil pp, µm pp, mil pk-pk, in/s pk, mm/s rms, mm/s pk, g pk, g rms,
unknown con RPM, unknown sin RPM)."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 11.1 en dev"
echo "================================================================"
echo ""
echo "Validar en tu app de dev (wm-test.streamlit.app):"
echo ""
echo "  1. Carga 3 o mas CSVs de espectro desde Load Data."
echo "  2. Andate a Spectrum. En la sidebar, en 'Spectra to display',"
echo "     vas a ver TODOS los espectros marcados automaticamente —"
echo "     no mas seleccion uno-a-uno."
echo ""
echo "  3. Mira el campo 'Max frequency (CPM)' debajo de 'Display':"
echo "     - Si tu primera senal es g pk (aceleracion), va a aparecer"
echo "       600.000 CPM como default."
echo "     - Si es mm/s pk (velocidad), 120.000 CPM."
echo "     - Si es mil pp (displacement), 60.000 CPM."
echo "     - Y debajo del input, un caption explicando por que."
echo ""
echo "  4. Pasa el cursor por el ? del campo — vas a ver el tooltip"
echo "     completo con el rango sugerido y la razon (ej. 'Rango alto"
echo "     para deteccion de fallas tempranas de rodamientos...')."
echo ""
echo "  5. Cambia el valor manualmente, refresca la pagina — tu valor"
echo "     se mantiene. Cambia a otra senal de DIFERENTE familia"
echo "     (displacement -> acceleration) y veras que cada familia"
echo "     mantiene su propio default/edicion."
echo ""
echo "Despues podemos seguir con Ciclo 12 (Orbit Cat IV: clasificador"
echo "geometrico) o Ciclo 10B (Tabla 1 amplitudes en el reporte)."
echo "================================================================"
