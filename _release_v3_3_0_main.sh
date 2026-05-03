#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.3.0 → MAIN
# =============================================================
# Promueve a producción el Ciclo 17.15: aislamiento de reportes
# por owner_email + archivo histórico inmutable de PDFs.
#
# Lo que MAIN va a recibir:
#
# Ciclo 17.15 — Aislamiento de Reportes + Archivo histórico
#   ► Cada usuario tiene su propio espacio de reportes:
#       data/users/{email_slug}/report_state.json
#       data/users/{email_slug}/report_drafts/
#     (antes era global → bug de pisarse el trabajo)
#
#   ► Migración automática del estado legacy al admin (1 vez)
#
#   ► UI "Inspeccionar reporte de otro especialista" para
#     admin/specialist en modo solo-lectura
#
#   ► Botón "Duplicar a mi reporte" para copiar a espacio propio
#
#   ► Archivo histórico inmutable de PDFs aprobados:
#       data/reports_archive/{email_slug}/{YYYY}/{MM}/...
#     con sidecar JSON metadata + visibilidad por role:
#       admin       → todo
#       specialist  → suyos + otros @sigasas
#       client      → solo shared_with_client=True
#
#   ► Tab "Archivo histórico" en Reports con filtros + búsqueda +
#     descarga + acciones (compartir con cliente / eliminar)
#
#   ► Activity feed del Home con toggle "Solo mía / Toda" para
#     admin/specialist + avatar de iniciales coloreado por owner
#
# =============================================================
# IMPORTANTE — ANTES DE CORRER:
#
#   1. Verificar que dev está limpio:
#        cd ~/Documents/WatermelonSystem
#        git status   (debería decir 'nothing to commit')
#
#   2. Pushear los commits pendientes a dev (si no se hizo):
#        git push origin dev
#
#   3. Después corré ESTE script:
#        bash _release_v3_3_0_main.sh
#
# El script hace:
#   - sync dev con remoto
#   - checkout main + pull
#   - merge dev (no fast-forward, commit explícito de release)
#   - tag v3.3.0
#   - push origin main + tag
#   - vuelve a dev
#
# Tarda ~30 segundos + ~30-60 seg de redeploy de Streamlit Cloud.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚀 RELEASE v3.3.0 → MAIN"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.15  Aislamiento de Reportes por owner + Archivo histórico"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/6  Sincronizando dev con origin..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
if ! git diff --quiet || ! git diff --staged --quiet; then
    echo "✗ Hay cambios sin commitear en dev. Commiteá primero."
    exit 1
fi
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Pull dev falló."; exit 1; }
git push origin dev || { echo "✗ Push dev falló."; exit 1; }
echo "  ✓ dev sincronizado"
echo ""

echo "▶ 2/6  Cambiando a main..."
git checkout main || { echo "✗ No se pudo cambiar a main."; exit 1; }
git fetch origin main
git pull --rebase origin main || { echo "✗ Pull main falló."; exit 1; }
echo "  ✓ main actualizado"
echo ""

echo "▶ 3/6  Mergeando dev → main..."
MERGE_MSG="release(v3.3.0): merge dev -> main

Ciclo 17.15 — Aislamiento de Reportes por owner + Archivo
historico inmutable de PDFs.

Cada usuario tiene su propio espacio de reportes
(data/users/{email_slug}/...) — fin del bug de pisarse el
trabajo entre Ewdes y J Suarez.

Permisos por role:
- admin       ve y edita todo
- specialist  ve los suyos + lectura de otros @sigasas con
              opcion 'duplicar a mi reporte'
- client      solo ve archivados marcados shared_with_client

Archivo historico inmutable: cada PDF generado puede archivarse
permanentemente, con sidecar JSON metadata, ACL por role y
filtros multi-criterio.

Activity feed del Home con toggle por usuario y avatares de
iniciales coloreados por owner.

Tests smoke OK. Commit detallado: b2f2330."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    echo "  Resolvé con 'git mergetool' o abortá con 'git merge --abort'."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 4/6  Creando tag v3.3.0..."
TAG_EXISTS=$(git tag -l "v3.3.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.3.0 ya existe. Saltando creación."
else
    git tag -a v3.3.0 -m "Release v3.3.0 — Aislamiento Reports + Archivo historico inmutable"
    echo "  ✓ Tag v3.3.0 creado"
fi
echo ""

echo "▶ 5/6  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.3.0 || { echo "  ⚠ Push del tag falló (ya existía remoto?)"; }
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 6/6  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.3.0 COMPLETADO"
echo "================================================================"
echo ""
echo " La app productiva (https://wm-home-final-2026.streamlit.app)"
echo " va a redeployar en ~30-60 segundos."
echo ""
echo " QUEDA POR HACER después del redeploy:"
echo ""
echo "  1. Recargá el app productivo y verificá:"
echo "     - El footer SCADA dice v3.3.0"
echo "     - Reports tiene la nueva sección 'Archivo histórico' al pie"
echo "     - El Home tiene el toggle 'Solo mía / Toda la actividad'"
echo ""
echo "  2. Probá el aislamiento con vos + J Suarez:"
echo "     Vos:    crear reporte con 5 items, archivar PDF marcando"
echo "             'compartir con cliente'"
echo "     Jsuar:  abrir Reports → debe estar VACÍO (no los tuyos)"
echo "             usar 'Inspeccionar reporte de otro especialista'"
echo "             para ver el tuyo en read-only"
echo "     Cliente: cuando entre, en Archivo histórico solo va a ver"
echo "             los PDFs marcados 'compartir con cliente'"
echo ""
echo "  3. Si en producción aparece bug, podés rebobinar a v3.2.0:"
echo "     git checkout main && git reset --hard v3.2.0 && \\"
echo "       git push --force origin main"
echo ""
echo "================================================================"
