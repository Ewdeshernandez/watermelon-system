#!/bin/bash
# =============================================================
# Watermelon — v3.0 publish dev → main (con hotfix 17.5.9)
# =============================================================
# Hace TODO en una sola corrida:
#   1. Reconcilia divergencia local↔origin/dev (rebase)
#   2. Commit del hotfix 17.5.9 (chip health + planos hasta 50)
#   3. Push dev a origin
#   4. Switch a main + pull main
#   5. Merge dev → main (--no-ff) con changelog completo
#   6. Tag v3.0 anotado
#   7. Push main + tag
#   8. Vuelve a dev
#
# Si algo falla, el script aborta con instrucciones claras.
#
# Cierra el lote más grande desde v2.6:
#
#   • Ciclo 17.3 — SCL history (snapshot + multi-overlay +
#     narrativa Bently/API 670)
#   • Ciclo 17.5 — Trend module clase mundial (corridas CSV
#     completas + Vault thresholds editables + autodiagnóstico
#     ejecutivo Bently/ISO 20816 + bandas de severidad +
#     health chip + PNG HD doble eje)
#   • Hotfix 17.5.5 — PNG export con marker.size por punto
#   • Hotfix 17.5.6 — Sobriedad autodiag + bug global send-to-
#     report (helper compartido + 7 módulos migrados)
#   • Ciclo 17.5.7 — Saneamiento PDF: forecast inválido, anti-
#     contradicción "estable", Resumen Ejecutivo escala con Trend
#   • Ciclo 17.5.8 — Resumen Ejecutivo live (regenera draft
#     stale automáticamente + cosmética)
#   • Hotfix 17.5.9 — Chip health por max de ventana reciente
#     + Machinery Library hasta 50 apoyos por sección
#
# Ejecutar:
#   bash _publish_v3_0_main_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0"
RELEASE_TITLE="Trend module clase mundial + SCL history + Reports saneados"

# ---------- Limpieza de locks (defensivo) ----------
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) Reconciliar divergencia con rebase ----------
echo ""
echo "▶ Reconciliando dev local ↔ origin/dev (rebase)..."
git fetch origin dev
git pull --rebase origin dev || {
    echo ""
    echo "✗ Rebase falló. Resolvé los conflictos a mano:"
    echo "    git status"
    echo "    # editá los archivos en conflicto"
    echo "    git add <archivos>"
    echo "    git rebase --continue"
    echo "    bash _publish_v3_0_main_v1.sh"
    exit 1
}

# ---------- 2) Commit del hotfix 17.5.9 (si hay cambios) ----------
echo ""
echo "▶ Commiteando hotfix 17.5.9 (chip health + planos 50)..."
git add pages/04_Trends.py pages/00_Machinery_Library.py _publish_v3_0_main_v1.sh
if ! git diff --staged --quiet; then
    git commit -m "fix(trend+library): chip health por max reciente + Machinery hasta 50 apoyos (17.5.9)

(1) BUG REPORTADO: Chip 'Normal' aunque hay datos sobre Alarm.
Antes _compute_trend_health clasificaba SOLO por latest_value
(ultimo sample). Si el ultimo punto estaba debajo de Warning
aunque la ventana tuviera multiples picos sobre Danger, decia
'Normal'. Caso reportado: pico 2.319 in/s pk con Danger 1.230,
chip decia Normal porque latest era 0.057.

Fix: clasificar por el WORST de la ventana reciente (ultimos
7 dias o ultimos 100 samples). Si cualquier punto reciente
cruza Danger -> action; cualquiera cruza Warning -> alarm.
Latest sigue siendo el numero reportado, pero el status
refleja el peor reciente. Nuevo campo recent_max_value en
health dict. Prosa explicita el pico cuando latest << recent_max.

(2) BUG REPORTADO: Machinery Library no permitia mas de 16
apoyos (8 driver + 8 driven). Maquinas grandes (compresores
multi-etapa, trenes turbina-gearbox-generador con multiples
soportes) requerian mas. Subido a 50 por seccion (100 max
total) + columna plane del data editor a 100." || echo "  (no hay cambios para commitear)"
else
    echo "  (no hay cambios staged)"
fi

# ---------- 3) Push dev a origin ----------
echo ""
echo "▶ Pusheando dev a origin..."
git push origin dev

# ---------- 4) Cambiar a main + pull ----------
echo ""
echo "▶ Cambiando a main..."
git checkout main
git pull origin main

# ---------- 5) Merge dev → main ----------
echo ""
echo "▶ Mergeando dev → main..."
git merge dev --no-ff -m "release(${VERSION}): ${RELEASE_TITLE}

Lote acumulado desde v2.6:

== TREND MODULE CLASE MUNDIAL (Ciclo 17.5) ==

P1 - core/trend_history.py NUEVO. Persistencia de CSVs CRUDOS
por instancia bajo {INSTANCES_DIR}/{instance_id}/trend_history/
{corrida_id}/. A diferencia de polar/bode/scl que snapshotean
metricas derivadas, Trend persiste los CSV completos porque el
valor del modulo es la serie temporal de meses/años.
Auto-prune a 36 corridas (3 años mensual).

P2 - UI Historico de Tendencias en sidebar con instance
selector + multiselect de corridas anteriores que CONCATENAN
con la corrida actual + administrador con borrado.

P3 - Polish visual: bandas de severidad ambar y roja, health
chip top-right (Normal/Vigilancia/Atencion/Accion Requerida),
linea de pendiente y forecast a umbral.

P4 - Autodiagnostico ejecutivo Bently-style: headline + 6
parrafos en prosa + recomendaciones numeradas.

== VAULT THRESHOLDS EDITABLES (17.5.2) ==

suggest_trend_thresholds() resuelve cada record contra el
Sensor Map y toma los setpoints mas conservadores. Fallback
Vault -> ISO 20816 -> defaults Bently 3500. UI con chip de
fuente + boton 'Aplicar setpoints sugeridos' + caption
'Override del cliente'. Estado persistido para reporte.

== TREND COMPLETO AL PDF (17.5.3) ==

_send_to_report prepende headline + 6 parrafos + acciones.
item_payload incluye 'autodiagnostic' y 'threshold_source'.

== HD EXPORT CON DOBLE EJE (17.5.4) ==

Coordenadas dinamicas yaxis2 + width respetado.

== SCL HISTORY (Ciclo 17.3) ==

P1 core/scl_history.py + P2 UI multi-overlay del centerline
con gradiente cronologico + diamond markers en operating
points + P3 Narrativa migracion 5 bloques Bently/API 670 §6.7
(encabezado, evolucion clearance, clasificacion migration +
shift attitude, lift-off speed, distincion ESTRUCTURAL vs
OPERACIONAL).

== HOTFIXES Y SANEAMIENTO ==

17.5.5 - PNG export reventaba con float(list) cuando
marcadores de anomalia tenian size por punto.

17.5.6 - Bug GLOBAL del send-to-report: report_state.json
sobreescribia items en memoria la primera vez que se entraba
a Reports. Helper compartido core.report_state.
append_report_item_and_persist() y ensure_report_state_loaded()
con merge memoria<->disco. 7 modulos migrados. Autodiag mas
sobrio estilo Polar/Bode.

17.5.7 - Saneamiento PDF: forecast '~0 dias' invalidado
(ventana<24h, CV>50%, dias<0.5). Anti-contradiccion
'comportamiento estable' vs Strong change (guardrail
change_pct>=100% en _classify_trend_behavior). Resumen
Ejecutivo escala con Trend autodiag.

17.5.8 - Resumen Ejecutivo siempre live: badge se recomputa
desde findings en cada PDF + draft stale se regenera
automaticamente. Cosmetica: '1 figura' singular, 'variacion
absoluta' cuando inicial es ~0.

17.5.9 - Chip health por max de ventana reciente (no solo
latest_value). Machinery Library hasta 50 apoyos por seccion."

# ---------- 6) Tag de release ----------
echo ""
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Cambios mayores desde v2.6:

  Trend module clase mundial: corridas CSV completas + Vault
  thresholds editables + autodiagnostico Bently/ISO 20816 +
  bandas de severidad + health chip por max reciente + PNG HD
  doble eje.

  SCL history: snapshot + multi-overlay + narrativa Bently/
  API 670 §6.7 (5 bloques).

  Reports saneados: bug global de send-to-report resuelto via
  helper compartido. Resumen Ejecutivo escala severidad con
  Trend findings y se regenera automaticamente cuando esta
  stale. Forecast inválido invalidado. Anti-contradiccion
  estable vs Strong change. Pluralizacion correcta.

  Machinery Library hasta 50 apoyos por seccion (driver+driven).

Estado del historico multi-modulo:
  Tabular  ✅
  Polar    ✅ snapshot + multi-overlay + narrativa Bently
  Bode     ✅ snapshot + multi-overlay + narrativa Bently
  SCL      ✅ snapshot + multi-overlay + narrativa Bently/API 670
  Trend    ✅ corridas CSV completas + autodiag + polish visual
  Spectrum ⏳ proximo ciclo"

# ---------- 7) Push main + tags ----------
echo ""
echo "▶ Pusheando main + tag ${VERSION} a origin..."
git push origin main
git push origin "${VERSION}"

# ---------- 8) Volver a dev ----------
echo ""
echo "▶ Volviendo a dev..."
git checkout dev

echo ""
echo "================================================================"
echo " ✓ ${VERSION} publicado a main (con hotfix 17.5.9)"
echo "================================================================"
echo ""
echo " Branch actual: dev (lista para próximos ciclos)"
echo " Tag creado:    ${VERSION}"
echo " Main contiene: todo dev al 100%"
echo ""
echo " Verificación rápida:"
echo "   git log main --oneline | head -5"
echo "   git tag --list | tail -5"
echo "   git diff main dev --stat   # debe estar vacío"
echo "================================================================"
