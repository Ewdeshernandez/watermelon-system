#!/bin/bash
# =============================================================
# Watermelon — v3.0.2 hotfix: Reports auto-fill por instancia
# =============================================================
# BUG REPORTADO EN MAIN:
#   Activé TES1 en Machinery Library → en Reports el "Activo:"
#   dice TES1 pero Cliente, Sitio, Clase, Modelo, Train
#   description y esquemático son de C200C (la instancia previa).
#   No me deja generar reportes correctamente.
#
# CAUSA:
# _autofill_report_meta_from_active_instance() hacía back-fill
# "no destructivo" — solo escribía meta[key] cuando estaba vacío.
# Eso preserva ediciones manuales del usuario, pero también deja
# pegados los datos de la instancia previa al cambiar de
# máquina activa.
#
# FIX:
# Agregamos meta["_autofilled_from_instance_id"] como huella del
# último auto-fill. Si la instancia activa cambió desde entonces,
# los campos heredados de la instancia previa están stale y se
# sobrescriben (incluido el cleanup del schematic_doc_id viejo y
# la invalidación del executive_summary cached).
#
# Se sincronizan también los widget keys (report_meta_<key>)
# para que los textbox de la UI reflejen el cambio sin
# necesidad de refresh manual.
#
# Persistimos a disco cuando hay cambio efectivo, así el meta
# en report_state.json no queda con datos cruzados.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0.2"
RELEASE_TITLE="Hotfix: Reports refresca meta al cambiar de instancia activa"

# ---------- Locks ----------
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) STAGE primero (antes del rebase) ----------
echo ""
echo "▶ Stageando cambios 17.5.10..."
git add pages/16_Reports.py
git add _publish_v3_0_2_hotfix.sh 2>/dev/null || true

# ---------- 2) Commit (si hay cambios staged) ----------
if ! git diff --staged --quiet; then
    echo "▶ Commiteando hotfix v3.0.2..."
    git commit -m "fix(reports): auto-fill se pegaba a la instancia previa al cambiar de activo (17.5.10)

Bug reportado: 'active TES1 en Machinery Library pero en Reports
me aparecen Cliente PAREX, Sitio La Belleza, Clase COMPRESOR
RECIPROCANTE KBK-4, Modelo HNP2 503-68E, Train description y
esquematico de la instancia previa C200C — no puedo generar
reportes correctos'.

Causa: _autofill_report_meta_from_active_instance() hacia
back-fill 'no destructivo', solo escribia meta[key] cuando
estaba vacio. Eso preserva ediciones manuales pero deja pegados
los datos de la instancia previa al cambiar la activa.

Fix: meta['_autofilled_from_instance_id'] guarda la huella del
ultimo auto-fill. Si la instancia activa cambio desde entonces,
los campos heredados (client, asset_class, asset_model,
location, asset, unit, train_description, schematic_doc_id,
schematic_instance_id) se sobrescriben. El executive_summary
cached tambien se invalida para que el Resumen Ejecutivo se
regenere con findings de la nueva maquina. Widget keys
report_meta_<key> sincronizadas para refresh inmediato de la
UI. Persistencia a disco al detectar cambio efectivo."
else
    echo "  (no hay cambios staged — saltando commit)"
fi

# ---------- 3) Rebase contra origin ----------
echo ""
echo "▶ Reconciliando contra origin/dev..."
git fetch origin dev
git pull --rebase origin dev || {
    echo ""
    echo "✗ Rebase falló por conflictos. Resolvelos:"
    echo "    git status"
    echo "    git add <archivos>"
    echo "    git rebase --continue"
    echo "    bash _publish_v3_0_2_hotfix.sh"
    exit 1
}

# ---------- 4) Push dev ----------
echo ""
echo "▶ Pusheando dev..."
git push origin dev

# ---------- 5) Switch a main + pull ----------
echo ""
echo "▶ Cambiando a main..."
git checkout main
git pull origin main

# ---------- 6) Merge dev → main ----------
echo ""
echo "▶ Mergeando dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Bug critico en main: cambiar la instancia activa en Machinery
Library no refrescaba los campos del reporte. El usuario
activaba TES1 pero el reporte mostraba Cliente, Sitio, Clase,
Modelo, train_description y esquematico de la instancia previa
(C200C). Imposibilidad de generar reportes correctos para mas
de una maquina por sesion.

Fix en _autofill_report_meta_from_active_instance():
meta['_autofilled_from_instance_id'] track del origen del
auto-fill. Cambio de instancia detectado -> sobrescribe los
campos heredados + limpia schematic + invalida executive_summary
cached + sincroniza widget keys + persiste a disco."

# ---------- 7) Tag v3.0.2 ----------
echo ""
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix critico sobre v3.0.1:

Reports refresca correctamente cliente/sitio/clase/modelo/
train_description/esquematico al cambiar la instancia activa.
Antes los datos quedaban pegados a la instancia previa, lo
que generaba reportes con metadata cruzada entre maquinas."

# ---------- 8) Push main + tag ----------
echo ""
echo "▶ Pusheando main + ${VERSION}..."
git push origin main
git push origin "${VERSION}"

# ---------- 9) Volver a dev ----------
echo ""
echo "▶ Volviendo a dev..."
git checkout dev

echo ""
echo "================================================================"
echo " ✓ ${VERSION} publicado a main"
echo "================================================================"
echo ""
echo " Branch actual: dev"
echo " Tag creado:    ${VERSION}"
echo ""
echo " Verificación:"
echo "   git log main --oneline | head -5"
echo "   git tag --list | grep v3.0"
echo ""
echo " Streamlit Cloud auto-redeploya en ~2 min."
echo " Para verificar: cambiá entre TES1 y C200C en Machinery"
echo " Library y entrá a Reports — los campos deben mostrar la"
echo " maquina correcta cada vez."
echo "================================================================"
