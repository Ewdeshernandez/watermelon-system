#!/bin/bash
# =============================================================
# Watermelon — Ciclo 14b: Wire selector activo en Load Data (dev)
# =============================================================
# Cierra el cockpit "una sola seleccion de maquina, todo el sistema
# la usa" empezando por Load Data. La maquina seleccionada en
# Machinery Library queda visible en Load Data + cada CSV cargado
# queda etiquetado con su instance_id.
#
# CAMBIOS en pages/01_Load_Data.py:
#
# 1) IMPORTS:
#    + render_instance_selector, get_active_instance_id
#    + get_instance, get_instance_document_bytes, compose_train_description
#
# 2) SIDEBAR — render del instance_selector
#    Igual que Polar/Bode/SCL. Muestra el dropdown 'Instancia activa'
#    y respeta el state cross-page (hotfix 8). El usuario ve siempre
#    cual es su maquina activa.
#
# 3) BANNER ARRIBA DEL UPLOADER
#    st.container con borde:
#    - Esquematico de la maquina activa (PNG/JPG embebido) si tiene
#    - "🟢 Cargando CSVs para: TES1"
#    - Train description (ej. 'Turbogenerador GE LM6000 acoplado a
#      Generador Brush 54 MW')
#    - Meta bits: potencia, RPM, cliente, sitio
#    - Caption explicativo de que los CSVs quedaran etiquetados con
#      instance_id
#
# 4) BLOQUEO SI NO HAY MAQUINA ACTIVA
#    Si get_instance() devuelve None → st.error grande + st.info con
#    instrucciones + st.stop(). El usuario NO puede cargar CSVs sin
#    seleccionar maquina primero. Eso garantiza que cero CSVs queden
#    huerfanos.
#
# 5) ETIQUETADO de cada signal cargado:
#    En build_signal_from_parsed(), despues de canonicalizar metadata,
#    si hay instance_id activo:
#    - metadata['instance_id'] = inst.instance_id
#    - metadata['instance_tag'] = inst.tag
#    - metadata['instance_train'] = compose_train_description(inst)
#    - metadata['instance_client'] = inst.client
#    - metadata['instance_site'] = inst.site
#    - Backfill metadata['Machine'] con instance_tag si llegaba 'Unknown'
#    - Backfill metadata['RPM'] con instance.nominal_rpm si vacio
#    Try/except defensivo: si falla, NO rompe el upload.
#
# RESULTADO:
#
# Flujo del ingeniero post-Ciclo 14b:
#   1. Login → Machinery Library
#   2. Click 'Activar' en TES1 (badge verde)
#   3. Click 'Load Data' en menu lateral
#      → ve banner: "🟢 Cargando CSVs para: TES1" + esquematico
#   4. Sube 5 CSVs del Brush
#      → cada CSV queda con metadata.instance_id='brush_default' y
#        Machine='TES1' (sin tipear nada)
#   5. Click 'Time Waveforms' / 'Polar' / 'Bode' / 'SCL' / etc.
#      → los plots usan los CSVs vinculados a TES1
#   6. Click 'Reports'
#      → auto-fill ya tomaba cliente/sitio/clase/modelo/train_desc/
#        esquematico (Ciclo 14a) y ahora ademas cada figura del
#        reporte sale con la maquina correctamente identificada
#
# Compatibilidad: NO toca Polar/Bode/SCL/Spectrum/Waveform/Trend
# todavia (esos vienen en Ciclo 14b parte 2 si hace falta). Compile
# clean. Si la session no tiene maquina activa al abrir Load Data,
# se bloquea con instrucciones — comportamiento intencional.
#
# Ejecutar:
#   bash _publish_ciclo14b_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev

git add pages/01_Load_Data.py
git status --short | head

git commit -m "feat(load-data): Ciclo 14b — wire selector instancia activa en Load Data (dev)

Cierra el cockpit 'una sola seleccion de maquina, todo el sistema la usa'
empezando por Load Data. La maquina activada en Machinery Library queda
visible en Load Data y cada CSV cargado queda etiquetado con instance_id.

Cambios en pages/01_Load_Data.py:
* Imports: render_instance_selector, get_active_instance_id, get_instance,
  get_instance_document_bytes, compose_train_description.
* Sidebar: render del instance_selector con module_name='load_data'.
  Respeta el state cross-page (hotfix 8).
* Banner arriba del uploader: container con borde mostrando esquematico
  de la maquina activa (si tiene), tag, train_description, potencia, RPM,
  cliente, sitio. Mensaje claro 'Cargando CSVs para: TES1'.
* Bloqueo si no hay maquina activa: st.error grande + st.info con
  instrucciones para ir a Machinery Library + st.stop(). Cero CSVs
  huerfanos.
* Etiquetado por signal en build_signal_from_parsed: metadata.instance_id,
  instance_tag, instance_train, instance_client, instance_site. Backfill
  de Machine y RPM si llegaban vacios desde el CSV. Try/except defensivo.

Compatibilidad: NO toca otros modulos. Compile clean."

git push origin dev

echo ""
echo "================================================================"
echo " LISTO — Ciclo 14b en dev"
echo "================================================================"
echo ""
echo "Plan de pruebas:"
echo ""
echo "  1. Refrescar app. Login → Machinery Library."
echo "  2. TES1 ya esta activa (badge verde) — confirmar."
echo "  3. Click 'Load Data' en menu lateral."
echo "  4. Verificar:"
echo "     - Sidebar muestra dropdown 'Instancia activa = TES1'"
echo "     - Banner verde arriba: '🟢 Cargando CSVs para: TES1' con"
echo "       esquematico embebido + train description"
echo "  5. Subir 3-4 CSVs de Brush 54 MW."
echo "  6. Verificar que cargaron OK (mensaje 'X signals loaded')."
echo "  7. Click 'Time Waveforms' o 'Spectrum' — los plots de los"
echo "     CSVs salen correctamente."
echo "  8. Click 'Reports'. Panel auto-fill verde, esquematico listo,"
echo "     cliente/sitio/train_description prelllenados de TES1."
echo "  9. Generar PDF — sale completo con esquematico en pagina 3."
echo " 10. (Test del bloqueo) Borrar la maquina activa o ir a Library"
echo "     y des-activar TES1 (si se puede). Volver a Load Data."
echo "     → debe aparecer st.error 'No hay maquina activa' + st.stop"
echo "       que bloquea el uploader."
echo ""
echo "Cuando confirmes que funciona, los proximos ciclos serian:"
echo "  - Ciclo 14b parte 2: wire del banner en Tabular List, Time"
echo "    Waveforms, Spectrum, Orbit, Trends (mismo patron)"
echo "  - Ciclo 14c: filtro de signals por instance_id (cuando tenes"
echo "    CSVs de TES1 y TES3 en sesion, cada modulo solo muestra los"
echo "    de la maquina activa)"
echo "  - Ciclo 13: Orbit avanzado"
echo "  - Cuando dev este maduro: tag v2.1 + merge a main"
echo "================================================================"
