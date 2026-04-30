#!/bin/bash
# =============================================================
# Watermelon — v2.6 → MAIN: Histórico multi-módulo + Bently/API 684
# =============================================================
# Release grande consolidando todo lo acumulado en dev desde v2.5.1.
# El producto entra en una nueva categoría: monitoreo continuo
# con histórico persistente y diagnóstico modal completo.
#
# == HIGHLIGHTS DE v2.6 ==
#
# 🪄 (16.1) WIZARD AUTO-PATTERN para Sensor Map
#   Mira los CSVs en sesión, detecta sensores sin match definitivo y
#   propone csv_match_pattern concreto basado en el Point name del
#   signal más compatible (tipo + dirección). Tabla con checkbox por
#   fila + bulk apply. Reduce el setup manual de Sensor Map a 1 click.
#
# 📚 (16.2) HISTÓRICO TABULAR + COMPARATIVO MULTI-FECHA
#   Snapshots por instancia con Overall + Status + Alarm + Danger
#   por sensor. JSON liviano (~5KB por corrida). Sidebar con
#   guardar/comparar/borrar. Tabla comparativa con tendencias
#   (▲ up_critical / ↑ up / → stable / ↓ down / ▼ down_good).
#   Sección EVOLUCIÓN en el PDF con prosa ingenieril citando el
#   sensor de mayor incremento por nombre.
#
# 📈 (16.3) TRENDS MULTI-SNAPSHOT EN PDF
#   Mini line charts por sensor crítico mostrando últimos N snapshots
#   con threshold lines (Alarm/Danger) y markers coloreados por
#   status. Permite ver la trayectoria temporal del sensor además del
#   simple delta vs corrida anterior.
#
# 📐 (17.1) HISTÓRICO POLAR — vector polar history
#   Snapshot del 1X amp + fase + trayectoria completa downsampleada
#   por sensor. Multi-overlay sobre el polar con gradiente cronológico
#   (azul claro → ámbar → rojo). Diamond markers en cada peak histórico.
#   Trail visual + tabla comparativa + narrativa modal completa estilo
#   Bently Nevada Technical Training / API 684:
#     1) Encabezado factual con vector change
#     2) Caracterización del modo (translacional/cónico/flexural)
#     3) Diagnóstico diferencial del shift
#     4) Análisis de sensitividad / damping
#     5) Distinción modal rotor vs estructural
#
# 📈 (17.2) HISTÓRICO BODE — mismo patrón aplicado al Bode
#   Snapshot de criticas + Q + amp/fase vs RPM. Multi-overlay con
#   gradiente cronológico sobre amp vs RPM y phase vs RPM. Diamond
#   en cada peak histórico = critical speed por corrida. Narrativa
#   modal completa en PDF con migración del modo + Q evolution.
#
# == HOTFIXES INCLUIDOS ==
#
# 🐛 BUG INSTANCE SWITCHING en Machinery Library: el callback del
#   "Activar" en cada card no actualizaba el selectbox del sidebar
#   porque las keys estaban desincronizadas (widget=
#   wm_instance_select_documents, callback seteaba
#   wm_instance_select_library). Fix: render_instance_selector
#   sincroniza SIEMPRE el widget desde la key persistente antes de
#   instanciar; callback itera todas las keys
#   wm_instance_select_*. Resuelve el "no me deja cambiar de
#   máquina" reportado en main.
#
# 🐛 Bode "ningún CSV matchea" mensaje vago — ahora distingue 3
#   casos (no Sensor Map / no CSVs / matcher falla) y abre un
#   expander de diagnóstico con tabla de CSVs cargados + tabla de
#   sensores con patterns + tip al wizard auto-pattern.
#
# 🐛 hotfix_skip_identical_to en comparativo (16.2.1) — el comparativo
#   del PDF saltaba snapshots cuyas lecturas son idénticas a la
#   corrida actual (caso: usuario guarda manualmente y genera PDF
#   inmediato, comparaba contra sí mismo y decía "todo estable").
#
# == NUEVAS DEPENDENCIAS ==
#
# Ninguna nueva. matplotlib, streamlit-image-coordinates y Pillow
# ya estaban en v2.5.1.
#
# == ARCHIVOS PRINCIPALES TOCADOS ==
#
#   core/instance_history.py        (NUEVO: snapshots Tabular)
#   core/polar_history.py           (NUEVO: snapshots Polar)
#   core/bode_history.py            (NUEVO: snapshots Bode)
#   core/trend_charts.py            (NUEVO: mini line charts PDF)
#   core/sensor_map.py              (suggest_pattern_for_sensor +
#                                     detect_definitive_matches)
#   core/instance_selector.py       (FIX bug instance switching)
#   pages/00_Machinery_Library.py   (Wizard UI + callback fix)
#   pages/01__Tabular_List.py       (sidebar histórico + comparativo)
#   pages/06_Polar_Plot.py          (sidebar + multi-overlay + narrativa)
#   pages/07_Bode_Plot.py           (sidebar + multi-overlay + narrativa)
#   pages/16_Reports.py             (sección EVOLUCIÓN + trends grid)
#
# Ejecutar:
#   bash _publish_v2_6_to_main.sh
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
echo " STEP 1: Commit de cambios pendientes en dev"
echo "================================================================"
git add core/instance_selector.py 2>/dev/null || true
git add pages/00_Machinery_Library.py 2>/dev/null || true
git add pages/07_Bode_Plot.py 2>/dev/null || true
git status --short | head

git commit -m "fix(instance-selector + bode): switching bug + Bode diagnostic message

(1) instance_selector ahora SIEMPRE sincroniza el widget desde la
key persistente antes de instanciar el selectbox. Resuelve el bug
en Machinery Library donde clickear 'Activar' en una card no
actualizaba el selectbox del sidebar (las keys estaban
desincronizadas: widget=wm_instance_select_documents, callback
seteaba wm_instance_select_library).

(2) Callback _set_active_instance ahora itera TODAS las keys
wm_instance_select_* en session_state y las sincroniza al
target_instance_id, asi cualquier widget de instance_selector
ya instanciado se actualiza correctamente.

(3) Bode page: mensaje 'ningun CSV matchea' ahora distingue 3
casos (no Sensor Map / no CSVs / matcher falla) y abre un
expander de diagnostico con tabla de CSVs cargados + tabla de
sensores con patterns + tip al wizard auto-pattern de Machinery
Library." || echo "Nothing to commit"

git push origin dev

echo ""
echo "================================================================"
echo " STEP 2: Tag pre-merge de main para rollback"
echo "================================================================"
git fetch origin
PRE_MERGE_TAG="v2.6-pre-main-$(date +%Y%m%d-%H%M%S)"
git tag -a "$PRE_MERGE_TAG" origin/main -m "Snapshot main antes del merge v2.6"
git push origin "$PRE_MERGE_TAG"

echo ""
echo "================================================================"
echo " STEP 3: Merge dev → main"
echo "================================================================"
git checkout main
git pull origin main
git merge --no-ff dev -m "merge: dev → main — Watermelon v2.6

Release grande: histórico multi-módulo + diagnóstico modal
completo nivel Bently Nevada / API 684.

HIGHLIGHTS:

(16.1) Wizard auto-pattern Sensor Map — propone csv_match_pattern
concreto basado en CSVs en sesion. Reduce setup manual a 1 click.

(16.2) Histórico Tabular — snapshots persistentes con comparativo
multi-fecha, tabla con tendencias, sección EVOLUCIÓN en PDF
mencionando sensor de mayor incremento por nombre.

(16.3) Trends multi-snapshot en PDF — mini line charts por sensor
con threshold lines y markers coloreados por status.

(17.1) Histórico Polar — vector polar history con multi-overlay
de trayectorias completas, gradiente cronológico, diamond en
peaks, narrativa modal completa estilo Bently/API 684:
caracterización del modo (translacional/cónico/flexural),
diagnóstico diferencial, sensitividad, distinción modal rotor
vs estructural.

(17.2) Histórico Bode — mismo patrón aplicado al Bode con
overlays de amp vs RPM y phase vs RPM, migración del modo, Q
evolution.

HOTFIXES:
- BUG instance switching: el callback de cards 'Activar' no
  actualizaba el selectbox del sidebar por keys desincronizadas.
  Fix en instance_selector + callback. Resuelve 'no me deja
  cambiar de máquina' reportado en main.
- Bode 'ningún CSV matchea' mensaje vago → ahora distingue casos
  + diagnóstico con tabla CSVs vs patterns.
- skip_identical_to en comparativo evita comparar contra
  snapshot recén guardado de la corrida actual."

echo ""
echo "================================================================"
echo " STEP 4: Tag v2.6 y push"
echo "================================================================"
git tag -a "v2.6" -m "Watermelon v2.6 — Histórico multi-módulo + Bently/API 684 narrative"
git push origin main
git push origin v2.6

echo ""
echo "================================================================"
echo " STEP 5: Volver a dev"
echo "================================================================"
git checkout dev

echo ""
echo "================================================================"
echo " ✓ MERGE A MAIN COMPLETADO — v2.6 LIVE"
echo "================================================================"
echo ""
echo "Tags:"
echo "  - $PRE_MERGE_TAG (rollback)"
echo "  - v2.6 (release)"
echo ""
echo "ROLLBACK:"
echo "  git checkout main && git reset --hard $PRE_MERGE_TAG && \\"
echo "  git push --force-with-lease origin main"
echo ""
echo "Streamlit Cloud va a redeployar en 1-2 min desde main."
echo ""
echo "Pendientes en dev para próxima iteración:"
echo "  - 17.3 Histórico SCL (X/Y migration + eccentricity)"
echo "  - 17.4 Histórico Spectrum (peaks por orden)"
echo "  - 16.2 Parte 3 Trend chips visuales en esquemático"
echo "================================================================"
