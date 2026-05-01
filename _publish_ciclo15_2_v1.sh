#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.2 → DEV: click-to-place sobre el esquemático
# =============================================================
# Permite asignar coordenadas (x_pct, y_pct) a cada plano del Sensor
# Map sobre la imagen schematic_png del activo. Una vez configurado,
# el Resumen Ejecutivo del PDF y la página Machine Map renderizan
# los markers de severidad + valores Overall sobre la foto/dibujo
# REAL del activo en lugar del esquemático genérico.
#
# Si no hay coordenadas configuradas, fallback al render genérico
# turbomachinery silhouette — retro-compatible con instancias antes
# del 15.2.
#
# Adicional (15.1.6): diagnóstico Sin Datos en Machine Map. Cuando
# un sensor sale en "No Data", expander que lista los Point names
# disponibles en sesión, el csv_match_pattern esperado, y un
# indicador de si algun signal cargado matchearia. Self-service para
# diagnosticar mismatches sin pedir ayuda.
#
# Cambios:
#
# (1) requirements.txt — nuevas deps:
#       streamlit-image-coordinates>=0.1.7  (captura clicks sobre imagen)
#       Pillow>=10.0.0                       (overlay markers + texto)
#
# (2) core/sensor_map.py — new_sensor() ahora acepta x_pct y y_pct.
#     Campos opcionales (None por default). Documentado en docstring.
#
# (3) core/sensor_diagram.py — nueva funcion render_on_schematic():
#     toma bytes del schematic_png + lista de sensores con x_pct/y_pct
#     y devuelve PNG con markers circulares (color por worst-of-plane),
#     numero del cojinete, plane label normalizada y valor Overall del
#     peor sensor coloreado por severidad. Keyphasor renderiza estrella
#     ámbar. Sin coords → devuelve None (caller cae al render generico).
#
# (4) pages/00_Machinery_Library.py — nueva seccion "📍 Posicionar
#     sensores sobre el esquemático" debajo del diagrama generico.
#     UI con streamlit_image_coordinates: el usuario selecciona un
#     plano, hace clic en la imagen y se capturan x_pct/y_pct
#     normalizados (0-100%). Tambien fallback de inputs numericos.
#     Boton "Guardar posicion" aplica las coords a TODOS los sensores
#     del plano (varios sensores en un cojinete comparten posicion).
#     Boton "Borrar todas las posiciones" para rehacer desde cero.
#
#     Bonus: el "Guardar mapa de sensores" ahora preserva x_pct/y_pct
#     existentes cuando el usuario edita la tabla del data_editor
#     (que no muestra esos campos pero antes los borraba al guardar).
#
# (5) pages/01b_Machine_Map.py — preferimos render_on_schematic si
#     hay coordenadas; si no, fallback al render generico. Caption
#     se adapta. Tambien: nuevo expander de diagnostico
#     "🔍 por que N sensores aparecen sin datos" que lista los
#     Point names disponibles, los patterns esperados y si alguno
#     matchearia — self-service para diagnosticar mismatches.
#
# (6) pages/16_Reports.py — Resumen Ejecutivo del PDF intenta
#     render_on_schematic primero (foto real con valores y colores),
#     fallback al render generico turbomachinery. Caption del PDF se
#     adapta segun cual se uso.
#
# Smoke validado:
#   * render_on_schematic con synthetic schematic + 6 sensores
#     posicionados muestra markers circulares en sus coords con
#     plane labels (TRF, CRF, GEN DE, GEN NDE), valores Overall
#     coloreados por severidad y keyphasor estrella ambar.
#   * Compile OK en los 6 archivos editados.
#
# Ejecutar:
#   bash _publish_ciclo15_2_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
git pull origin dev || true

git add requirements.txt
git add core/sensor_map.py
git add core/sensor_diagram.py
git add pages/00_Machinery_Library.py
git add pages/01b_Machine_Map.py
git add pages/16_Reports.py
git status --short | head

git commit -m "feat(machine-map): click-to-place sensores sobre esquematico real (Ciclo 15.2)

Permite asignar coordenadas (x_pct, y_pct) a cada plano del Sensor
Map sobre la imagen schematic_png del activo. Una vez configurado,
el Resumen Ejecutivo del PDF y la pagina Machine Map renderizan
los markers de severidad + valores Overall sobre la foto/dibujo
REAL del activo. Sin coords configuradas → fallback al render
generico turbomachinery silhouette.

Cambios:

(1) requirements.txt: streamlit-image-coordinates>=0.1.7 + Pillow>=10.

(2) core/sensor_map.py: new_sensor acepta x_pct, y_pct (Optional[float]).

(3) core/sensor_diagram.py: nueva funcion render_on_schematic() que
overlay markers circulares + plane labels + valores Overall sobre
la imagen base. Worst-of-plane severity. Keyphasor estrella ambar.

(4) pages/00_Machinery_Library.py: seccion 'Posicionar sensores
sobre el esquematico' con streamlit_image_coordinates (clic en
imagen) + fallback inputs numericos. Boton guardar aplica coords a
TODOS los sensores del plano. Bonus: el save del Sensor Map ahora
preserva x_pct/y_pct existentes (el data_editor no los muestra).

(5) pages/01b_Machine_Map.py: preferimos render_on_schematic si hay
coords; fallback al generico. Bonus: expander de diagnostico
'por que N sensores aparecen sin datos' que lista Points en sesion,
patterns esperados y si alguno matchearia — self-service.

(6) pages/16_Reports.py: Resumen Ejecutivo del PDF intenta
render_on_schematic primero, fallback al generico. Caption se
adapta segun cual se uso.

Smoke: synthetic schematic + 6 sensores posicionados → markers en
coords correctas, plane labels, valores Overall coloreados,
keyphasor estrella. Compile OK en los 6 archivos." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 15.2 pusheado a dev"
echo "================================================================"
echo ""
echo "Para usarlo:"
echo "  1. Ir a Machinery Library, activar la instancia."
echo "  2. Sección 'Mapa de Sensores' → debajo del diagrama generico"
echo "     aparecera 'Posicionar sensores sobre el esquematico'."
echo "  3. Seleccionar un plano del dropdown, hacer clic en la imagen"
echo "     donde queda el cojinete, y 'Guardar posicion'."
echo "  4. Repetir para cada plano + el keyphasor."
echo "  5. Generar el PDF Reports — el Resumen Ejecutivo ahora"
echo "     muestra los markers sobre tu foto/dibujo real."
echo ""
echo "Streamlit Cloud va a instalar streamlit-image-coordinates en el"
echo "primer redeploy (1-2 min, una vez). Si tarda, refresca la"
echo "pagina; la dependencia es liviana."
echo "================================================================"
