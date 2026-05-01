#!/bin/bash
# =============================================================
# Watermelon — v2.5 → MAIN: Machine Map vivo + Click-to-place
# =============================================================
# Release grande: trae a main todo el trabajo del Ciclo 15.1.x +
# Ciclo 15.2.x acumulado en dev desde v2.4. Muchos cambios y
# muchos hotfixes — este es el merge definitivo.
#
# == GRANDES TEMAS DE v2.5 ==
#
# (A) Mini Machine Map arriba del Tabular List (Ciclo 15.1.1)
#     Banner colapsable con vista rápida del estado de la
#     máquina arriba del Tabular: 4 KPIs (Aceptable/Atención/
#     Acción/Sin datos) + diagrama lateral compact con cojinetes
#     coloreados worst-of-plane + link al Machine Map completo.
#     Reusa render_sensor_map_diagram en modo compact y un
#     build_severity_table compartido.
#
# (B) Sección MAPA DE SENSORES en el PDF report (Ciclo 15.1.2)
#     Después del Resumen Ejecutivo, antes de Recomendaciones:
#     título + síntesis en prosa ingenieril que menciona por
#     nombre el sensor con mayor margen consumido del Danger,
#     heatmap full (lateral + polar por plano), tabla
#     drill-down de sensores en Atención / Acción Requerida.
#
# (C) Machine Map alineado con Tabular como fuente única de
#     verdad (Ciclo 15.1.4)
#     Antes el Machine Map calculaba RMS por su cuenta y podía
#     contradecir al Tabular. Ahora el Tabular guarda su DataFrame
#     en st.session_state["wm_tabular_df"] y el Machine Map
#     proyecta esos valores sobre la geometría. Por construcción
#     no pueden contradecirse en la misma sesión.
#
# (D) Esquemático vivo en el Resumen Ejecutivo (Ciclo 15.1.5)
#     Antes el schematic_png estático del Vault era solo
#     decorativo. Ahora si hay Sensor Map + signals en sesión,
#     se renderiza un mini-heatmap del tren con valores Overall
#     del peor sensor por plano coloreados por severidad. El
#     cliente abre la primera página y ve el estado al toque.
#
# (E) Keyphasor diferenciado de sensores de vibración (Ciclo
#     15.1.5). count_status separa keyphasor del conteo de
#     vibración. La prosa dice "8 sensores de vibración +
#     1 señal de referencia de fase (keyphasor) instalada
#     sobre el eje del rotor para sincronización de medidas
#     vectoriales según API 670" en lugar de "9 sensores".
#
# (F) CLICK-TO-PLACE sobre el esquemático real (Ciclo 15.2)
#     Permite asignar coordenadas (x_pct, y_pct) a cada plano
#     del Sensor Map sobre la imagen schematic_png del activo
#     usando streamlit-image-coordinates (clic en imagen).
#     Una vez configurado, el Resumen Ejecutivo y la página
#     Machine Map muestran los markers de severidad +
#     valores Overall sobre la foto/dibujo REAL del activo
#     en lugar del esquemático genérico turbomachinery.
#     Sin coords configuradas → fallback al render genérico.
#
#     UI en Machinery Library → Mapa de Sensores →
#     "📍 Posicionar sensores sobre el esquemático":
#     dropdown del plano + clic en la imagen + Guardar.
#     Coords se aplican a TODOS los sensores del plano.
#
# (G) Multi-sensor display por plano (Ciclo 15.2.1)
#     Antes el render mostraba solo el peor sensor por plano.
#     Ahora muestra TODOS los sensores con sus valores
#     Overall individuales coloreados por SU propia severidad.
#     En TRF/CRF se ven velocity Y accelerometer; en planos
#     del generador se ven X y Y.
#
# (H) Aire en el render del esquemático (Ciclo 15.2.2)
#     Más espacio entre valores stackeados, fonts más grandes,
#     padding y opacidad del fondo blanco para legibilidad
#     sobre fotos reales del activo.
#
# (I) Caption ejecutiva corta (Ciclo 15.2.3)
#     Reemplazada la caption verbose del Resumen Ejecutivo
#     por una sola línea: "Estado actual del tren · {train}".
#     La imagen + colores + valores hablan solos; detalle va
#     en MAPA DE SENSORES.
#
# == HOTFIXES INCLUIDOS ==
#
# - Bug de severidad en Machine Map: signals son
#   SimpleNamespace(.x), no .amplitude. compute_signal_overall_rms
#   ahora prueba .amplitude → .x → dict en orden.
#
# - Resumen Ejecutivo perdido cuando el campo está vacío: ahora
#   se auto-redacta con _autodraft_executive_summary.
#
# - Diagrama generico mejorado: silueta turbomachinery (turbina
#   con inlet vanes + stage rings + exhaust cone, generador con
#   end shields + cooling vanes radiales, coupling con
#   tornillería) en lugar de cajas redondas. Bearings coloreados
#   por worst-of-plane en compact y full mode.
#
# - Plane labels normalizadas: si el usuario nombra los planos
#   "TRF Vel" / "TRF Accel", el diagrama muestra solo "TRF" y
#   chips coloreados debajo del cojinete indicando los tipos
#   de sensor presentes (proximity violeta, velocity cian,
#   accelerometer rojo). Antes la velocity se "perdía" cuando
#   compartía plano con la accel.
#
# - Driver/driven detection con fallback robusto cuando solo un
#   lado tiene tokens "driver"/"driven" en plane_label.
#
# - Hotfix import compose_train_description en Machinery Library
#   (la función se usaba en el render del diagrama sin estar
#   importada).
#
# - Diagnóstico self-service en Machine Map: cuando hay sensores
#   en "No Data", expander que lista los Point names cargados,
#   los patterns esperados y si alguno matcheria — para que el
#   ingeniero ajuste el csv_match_pattern sin pedir ayuda.
#
# - Hotfix bytes → PIL.Image (streamlit-image-coordinates exige
#   PIL.Image, no bytes).
#
# - Hotfix dropdown click-to-place vuelve al keyphasor: sort y
#   persistencia de la KEY del plano en session_state para
#   sobrevivir reruns que cambian la label del selectbox.
#
# == NUEVAS DEPENDENCIAS ==
#
#   streamlit-image-coordinates>=0.1.7
#   Pillow>=10.0.0
#
# Streamlit Cloud va a instalarlas automáticamente al primer
# redeploy desde main. Para correr local:
#   pip install streamlit-image-coordinates Pillow
#
# == ARCHIVOS PRINCIPALES TOCADOS ==
#
#   core/sensor_diagram.py         (silueta, render_on_schematic,
#                                    multi-sensor stack, normalize
#                                    plane labels, sensor type chips)
#   core/sensor_map.py             (x_pct, y_pct en new_sensor)
#   core/machine_severity.py       (NUEVO — fuente de verdad
#                                    compartida entre Tabular y
#                                    Machine Map; keyphasor split
#                                    en count_status)
#   pages/00_Machinery_Library.py  (sección click-to-place,
#                                    import compose_train_description,
#                                    preservar x_pct/y_pct en save)
#   pages/01__Tabular_List.py      (mini Machine Map, expone
#                                    df_table en session_state)
#   pages/01b_Machine_Map.py       (render_on_schematic con
#                                    fallback, diagnóstico Sin Datos)
#   pages/16_Reports.py            (esquemático vivo en Resumen
#                                    Ejecutivo, sección MAPA DE
#                                    SENSORES con prosa ingenieril,
#                                    auto-draft Resumen Ejecutivo)
#   requirements.txt               (streamlit-image-coordinates,
#                                    Pillow)
#
# Ejecutar:
#   bash _publish_v2_5_to_main.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " STEP 0: Verificar branch dev limpio"
echo "================================================================"
CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  git checkout dev
fi
git pull origin dev || true

echo ""
echo "================================================================"
echo " STEP 1: Commit de la caption corta + cambios pendientes"
echo "================================================================"
git add pages/16_Reports.py 2>/dev/null || true

git commit -m "feat(reports): caption ejecutiva corta del esquematico (Ciclo 15.2.3)

Reemplaza la caption verbose ('Estado actual del tren sobre el
esquematico del activo · cojinetes coloreados segun severidad por
plano (verde = aceptable, ambar = atencion, rojo = accion
requerida); valores Overall del peor sensor por plano sobre la
etiqueta. Detalle por sonda en la seccion Mapa de Sensores.') por
una sola linea: 'Estado actual del tren · {train_description}'.

La imagen + colores + valores hablan solos; el detalle ingenieril
ya esta en la seccion MAPA DE SENSORES." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " STEP 2: Tag pre-merge de main para rollback"
echo "================================================================"
git fetch origin
PRE_MERGE_TAG="v2.5-pre-main-$(date +%Y%m%d-%H%M%S)"
git tag -a "$PRE_MERGE_TAG" origin/main -m "Snapshot de main antes del merge v2.5"
git push origin "$PRE_MERGE_TAG"
echo "  Tag de rollback creado: $PRE_MERGE_TAG"

echo ""
echo "================================================================"
echo " STEP 3: Merge dev → main"
echo "================================================================"
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev → main — Watermelon v2.5

Release grande: Machine Map vivo + click-to-place sobre el
esquemático real del activo + alineamiento de Machine Map con
Tabular como fuente única de verdad.

Highlights:

(A) Mini Machine Map arriba del Tabular List — vista rápida
    del estado del tren coloreada por severidad worst-of-plane.

(B) Seccion MAPA DE SENSORES en el PDF report con prosa
    ingenieril que menciona el sensor de mayor margen consumido
    por nombre, heatmap full + drill-down de criticos.

(C) Machine Map alineado con Tabular como fuente unica de
    verdad — el Tabular guarda su DataFrame en session_state y
    el Machine Map proyecta esos valores. Antes podian
    contradecirse; ahora no.

(D) Esquematico vivo en el Resumen Ejecutivo — antes era una
    foto decorativa, ahora muestra el estado actual del tren
    con valores Overall coloreados por severidad.

(E) Keyphasor diferenciado en la prosa: 'sensores de vibracion +
    senales de referencia de fase' segun API 670, no 'N sensores'
    todos juntos.

(F) CLICK-TO-PLACE: nueva UI en Machinery Library que permite
    asignar coordenadas a cada plano sobre la imagen real del
    activo (foto/dibujo). El reporte ejecutivo entonces overlaya
    los markers de severidad sobre TU foto, no sobre un
    esquematico generico. Usa streamlit-image-coordinates.

(G) Multi-sensor display por plano: ahora muestra TODOS los
    sensores con valores propios. En TRF/CRF se ven vel + accel,
    en planos del generador se ven X y Y.

(H) Diagrama generico mejorado: silueta turbomachinery (turbina
    + generador + coupling + cojinetes coloreados) en lugar de
    cajas redondas. Sensor type chips bajo cada cojinete.

(I) Aire visual en el render: fonts mas grandes, mas espacio
    entre valores stackeados, fondos blancos opacos para
    legibilidad sobre cualquier foto.

Hotfixes incluidos:

  * BUG critico de severidad: signals son SimpleNamespace(.x).
    compute_signal_overall_rms ahora prueba .amplitude → .x →
    dict en orden. Antes daba 0 silencioso para todo.
  * Resumen Ejecutivo perdido cuando vacio → ahora auto-draft.
  * Diagnostico self-service en Machine Map para sensores en
    No Data (lista Point names cargados + patterns esperados).
  * Plane labels normalizadas (TRF Vel + TRF Accel → 'TRF' con
    chips de tipo de sensor debajo).
  * Driver/driven detection con fallback robusto.
  * Import compose_train_description en Machinery Library.
  * bytes → PIL.Image para streamlit-image-coordinates.
  * Dropdown click-to-place vuelve al keyphasor → sort fix +
    persistencia de la key seleccionada.

Nuevas dependencias en requirements.txt:
  streamlit-image-coordinates>=0.1.7
  Pillow>=10.0.0"

echo ""
echo "================================================================"
echo " STEP 4: Tag v2.5 y push"
echo "================================================================"
git tag -a "v2.5" -m "Watermelon v2.5 — Machine Map vivo + Click-to-place sobre esquematico real"
git push origin main
git push origin v2.5

echo ""
echo "================================================================"
echo " STEP 5: Volver a dev"
echo "================================================================"
git checkout dev

echo ""
echo "================================================================"
echo " ✓ MERGE A MAIN COMPLETADO — v2.5 LIVE"
echo "================================================================"
echo ""
echo "Tags:"
echo "  - $PRE_MERGE_TAG (rollback)"
echo "  - v2.5 (release)"
echo ""
echo "ROLLBACK:"
echo "  git checkout main && git reset --hard $PRE_MERGE_TAG && \\"
echo "  git push --force-with-lease origin main"
echo ""
echo "Streamlit Cloud va a redeployar en 1-2 min desde main."
echo "Atentos al primer redeploy: instala las nuevas deps"
echo "(streamlit-image-coordinates + Pillow). Si tarda, refrescar."
echo ""
echo "Próximas mejoras pendientes en dev:"
echo "  - Persistir la sesion del Tabular en disco para que el"
echo "    Resumen Ejecutivo del PDF use el df aun si el usuario"
echo "    no visito Tabular antes de generar el reporte."
echo "  - Patterns mas robustos para Sensor Map standard que"
echo "    cubran nomenclaturas Bently (VE####) ademas de la"
echo "    convencion API 670 (3X / 3Y / 4X / 4Y)."
echo "================================================================"
