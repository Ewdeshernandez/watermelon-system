#!/bin/bash
# =============================================================
# Watermelon — Ciclo 15.1.1 → DEV: Mini Machine Map en Tabular
# =============================================================
# Banner colapsable con vista rápida del estado de la máquina
# arriba del Tabular List. Reusa el helper render_sensor_map_diagram
# en modo compact (vista lateral worst-of-plane) y el
# build_severity_table compartido.
#
# Cambios:
#
#   1. core/machine_severity.py (NUEVO): helpers compartidos —
#      classify_severity, build_severity_table, count_status,
#      compute_signal_overall_rms, convert_rms_to_unit.
#      Los usa Machine Map (pagina completa) Y el Mini Machine
#      Map del Tabular. Una sola fuente de verdad para severidad.
#
#   2. core/sensor_diagram.py: nuevo parametro `compact=True` en
#      render_sensor_map_diagram. En compact se renderiza solo la
#      vista lateral del tren (sin polar por plano), con cada
#      cojinete coloreado por la peor severidad de los sensores
#      en ese plano (worst-of). Sin titulo ni leyenda — el banner
#      del Tabular ya tiene 4 KPIs arriba.
#
#   3. pages/01b_Machine_Map.py: refactor para usar los helpers
#      compartidos. Mismo render que antes — solo dejamos de
#      duplicar la logica de severidad.
#
#   4. pages/01__Tabular_List.py: nuevo banner "🗺️ Machine Map
#      (vista rápida)" entre el banner verde del activo y la
#      seccion del criterio. 4 metricas + diagrama compact +
#      page_link al Machine Map completo. Wrapped en st.expander
#      expanded=True para que se pueda colapsar si molesta.
#
# Smoke validado:
#   * compact PNG ~20 KB vs full ~66 KB
#   * worst-of-plane: plano con sensor Danger sale rojo aunque
#     tenga otros normales en ese plano
#   * sin signals → todos los planos neutros (No Data)
#   * compile OK en los 4 archivos editados
#
# Ejecutar:
#   bash _publish_ciclo15_1_1_v1.sh
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
git add pages/01b_Machine_Map.py
git add pages/01__Tabular_List.py

git status --short | head

git commit -m "feat(tabular): Mini Machine Map arriba del Tabular List (Ciclo 15.1.1)

Banner colapsable con vista rapida del estado de la maquina arriba
del Tabular: 4 KPIs (Aceptable/Atencion/Accion/Sin datos) + diagrama
lateral compact con cojinetes coloreados worst-of-plane + link al
Machine Map completo.

Cambios:

(1) core/machine_severity.py NUEVO — helpers compartidos
classify_severity, build_severity_table, count_status,
compute_signal_overall_rms, convert_rms_to_unit. Una sola fuente de
verdad para severidad entre Machine Map y Mini Machine Map.

(2) core/sensor_diagram.py — nuevo parametro compact=True en
render_sensor_map_diagram. Renderiza solo vista lateral, sin polar
por plano. Cojinetes se rellenan con el peor status de los sensores
en cada plano (Danger > Alarm > Normal > No Data). Sin titulo ni
leyenda — el banner del Tabular ya tiene KPIs arriba.

(3) pages/01b_Machine_Map.py — refactor para usar los helpers
compartidos. Mismo render que antes, sin duplicar logica.

(4) pages/01__Tabular_List.py — banner mini Machine Map insertado
entre el banner verde del activo y la seccion del criterio.

Smoke: compact PNG ~20KB vs full ~66KB; worst-of-plane validado;
sin signals -> planos neutros." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 15.1.1 pusheado a dev"
echo "================================================================"
echo ""
echo "Para verificar localmente: reiniciar Streamlit (Ctrl+C + run)"
echo "porque cambios en core/* requieren restart. La pagina"
echo "Tabular List debe mostrar arriba un expander abierto"
echo "'🗺️ Machine Map (vista rápida)' con 4 metricas + diagrama"
echo "lateral chico + link 'Ver Machine Map completo →'."
echo ""
echo "Cuando funcione bien, mergear a main con un publish v2.5."
echo "================================================================"
