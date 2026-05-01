#!/bin/bash
# =============================================================
# Watermelon — Ciclo 8: Asset Instances + Auto-cálculos
# =============================================================
# Sube todo el ciclo 8 a dev (NO a main todavía). Main queda en v1.2
# hasta que ciclos 8 + 9 estén validados juntos como v2.0.
#
# Cambios incluidos:
#   - core/instance_state.py: modelo Instance + CRUD per-máquina física
#   - core/instance_selector.py: selector UI con auto-bootstrap de seeds
#   - core/bearing_calculations.py: auto-cálculos Cd, Cr, L/D, unit load
#   - pages/17_Asset_Documents.py: rewrite completo con instance UI
#       + create instance form + edit metadata + danger zone
#   - pages/09_Shaft_Centerline.py: ahora usa render_instance_selector
#   - pages/06_Polar_Plot.py: ahora usa render_instance_selector
#   - pages/07_Bode_Plot.py: ahora usa render_instance_selector
#   - .gitignore: data/instances/ excluido (es data del usuario)
#
# Lo que el usuario verá tras el deploy en dev:
#   - Asset Documents tiene un panel arriba "+ Crear nueva instancia"
#   - Selector lateral pide "Instancia activa" en vez de Profile
#   - Auto-creación de instance "brush_default" desde el seed v1.1
#   - En el formulario aparecen métricas calculadas: Cd, Cr, L/D, unit load
#   - SCL/Polar/Bode toman vault_params per-instancia
#
# Ejecutar desde el root del repo:
#   bash _publish_ciclo8_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

echo ""
echo "================================================================"
echo " Watermelon — Ciclo 8: Asset Instances + Auto-cálculos"
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
echo "[1] Adoptando archivos del Ciclo 8..."
git add .gitignore
git add core/instance_state.py core/instance_selector.py core/bearing_calculations.py
git add pages/06_Polar_Plot.py pages/07_Bode_Plot.py
git add pages/09_Shaft_Centerline.py pages/17_Asset_Documents.py

echo "    Files staged:"
git diff --cached --stat | head -15
echo ""

echo "[2] Commit..."
git commit -m "feat: Ciclo 8 — Asset Instances + Auto-calculos (dev only)

Refactor del modelo de datos: cada maquina fisica registrada en el
sistema tiene un Instance ID unico (slug) con sus propios parametros,
documentos e historico. Antes la asociacion era por profile_key, lo
que mezclaba data entre maquinas distintas del mismo modelo
(bug grave para clientes con flotas).

Ciclo 8 (esto):
* core/instance_state: Instance dataclass + CRUD (create/list/get/
  update_header/delete + parametros + documentos per-instancia).
  Storage en data/instances/{instance_id}/metadata.json + documents/
* core/instance_selector: UI helper con auto-bootstrap. Si existe
  seed para un profile y aun no hay instance, crea {prefix}_default
  automaticamente. Devuelve dict compatible con el formato del
  legacy profile_state para minimizar cambios en modulos.
* core/bearing_calculations: derivaciones live de Cd, Cr, L/D, unit
  load, lift-off speed estimate. compute_all_derived(parameters) -> dict
* pages/17_Asset_Documents: reescritura completa.
    - Boton 'Crear nueva instancia' con formulario inline
    - Editar metadata de instancia (tag, serial, ubicacion, notas)
    - Panel de auto-calculos en vivo (st.metric con tooltips)
    - Documentos per-instancia con upload/download/delete
    - Danger zone para eliminar instancia con confirmacion
* pages/09_Shaft_Centerline, pages/06_Polar_Plot, pages/07_Bode_Plot:
  ahora llaman render_instance_selector en vez de profile_selector.
  Siguen funcionando igual pero leen vault_params per-instancia.
* .gitignore: data/instances/ excluido (data del usuario, igual que
  data/asset_metadata/)

Compatibilidad con Ciclo 7 (vault_seeds):
* Al crear una instancia desde un profile con seed, los parametros
  del seed se inyectan como defaults iniciales en captured_parameters.
* El bootstrap automatico crea {prefix}_default desde cada seed
  disponible para que la app arranque poblada sin requerir crear
  instancias a mano. Brush 54 MW arranca como 'brush_default'.

Smoke tests:
* CRUD de instances OK (create/list/get/update_header)
* Auto-bootstrap crea brush_default con 13 parametros del seed
* compute_all_derived(brush params): Cd=0.382 mm (manual), Cr=0.191 mm
  (calculado), L/D y unit_load aparecen cuando hay datos suficientes
* SCL/Polar/Bode siguen compilando sin errores

Este commit queda SOLO en dev. Main sigue en v1.2 hasta que se sume
Ciclo 9 (Supabase persistence) y se promueva todo junto como v2.0."
echo "    OK"
echo ""

echo "[3] Push a dev..."
git push origin dev
echo "    OK"
echo ""

echo "================================================================"
echo " LISTO — Ciclo 8 publicado en dev"
echo "================================================================"
echo ""
echo "Streamlit Cloud (si tenés la app de dev configurada en"
echo "wm-home-final-2026.streamlit.app o equivalente) va a redesplegar"
echo "en 1-3 minutos. Si la app de dev es watermelonsystem.app, NO va a"
echo "cambiar — esa apunta a main que sigue en v1.2."
echo ""
echo "Para validar el Ciclo 8:"
echo ""
echo "  1. Abrí el despliegue de dev y andá a 'Asset Documents'"
echo "  2. Vas a ver una entrada nueva en la sidebar: 'Instancia activa'"
echo "     con la instancia 'brush_default' auto-creada del seed"
echo "  3. Expandí '+ Crear nueva instancia' y crea una con tag 'TES1'"
echo "     usando el profile Brush. Va a aparecer al lado de la default."
echo "  4. Cambiá entre instancias en la sidebar — los datos NO se"
echo "     mezclan, cada una tiene sus propios parámetros."
echo "  5. En el panel de parámetros, ingresá 'Diámetro del journal'"
echo "     ej. 254.03 mm. Vas a ver aparecer en vivo la métrica"
echo "     'Cd diametral' con valor calculado, 'Cr radial', y si"
echo "     completás 'Longitud axial', 'Peso del rotor', 'Viscosidad'"
echo "     vas a ver L/D, carga unitaria y lift-off estimado."
echo "  6. Andá a Shaft Centerline. La sidebar arriba muestra"
echo "     'Activo monitoreado: brush_default' y los datos del Vault"
echo "     se leen de la instancia, no del profile."
echo ""
echo "Cuando confirmes que todo anda bien en dev, arrancamos el Ciclo 9"
echo "(Supabase para persistencia real) y mergeamos los dos juntos a"
echo "main como v2.0."
echo ""
echo "NOTA: el smoke test inicial dejó una instancia 'test_brush_xyz'"
echo "en tu filesystem local. Borrala desde la UI de Asset Documents"
echo "(seleccionala y usá la 'Zona peligrosa' para eliminar). En"
echo "producción no aparece porque data/instances/ está en .gitignore."
echo "================================================================"
