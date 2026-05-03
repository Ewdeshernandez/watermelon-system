#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.7 → DEV: Versión visible cross-app
# =============================================================
# Pedido del usuario: tener visible la versión del software en
# lugares estratégicos, "tipo software internacional pro".
#
# IMPLEMENTACIÓN:
#
# (1) core/version.py NUEVO — single source of truth
#     Auto-deriva la versión vía resolución por prioridad:
#       1. Variable de entorno WM_VERSION (override producción)
#       2. _git_latest_semver_tag() = `git tag --list 'v*'
#          --sort=-v:refname | head -1` — devuelve el tag semver
#          más alto del repo, sin importar branch. ESTO es clave
#          porque nuestros tags v3.0.x viven en main (sobre merge
#          commits) y desde dev `git describe` no los encuentra.
#       3. `git describe --tags --dirty` (ancestor only)
#       4. Archivo VERSION en la raíz
#       5. Constante hardcodeada _FALLBACK_VERSION
#
#     Environment se infiere de la branch:
#       main  → production
#       dev   → development
#       <otra> → branch name
#       (override con WM_ENVIRONMENT)
#
#     API: get_version_info() devuelve dict con version, commit,
#     branch, date, environment, commits_ahead, is_dirty,
#     full_label, release_name. lru_cache(1) para no fork-ear
#     git en cada rerun de Streamlit.
#
# (2) Login footer (pages/00_Login.py)
#     Reemplaza el hardcoded "build dev" por:
#         v3.0.8 [PRODUCTION]·22b268f3·2026-05-03
#         © 2026 SIGASAS · All rights reserved
#     El environment va en chip coloreado (verde production,
#     azul development, ámbar staging).
#
# (3) Sidebar de toda página autenticada (core/auth.py)
#     Caption pequeño monoespaciado al pie del sidebar, debajo
#     del botón Cerrar sesión:
#         v3.0.8 ● development · 22b268f3
#     Color tenue (#94a3b8) para no competir con la navegación,
#     bullet coloreado por environment.
#
# (4) Footer del PDF Reports
#     Tanto en la PORTADA como en páginas internas, esquina
#     inferior derecha en gris muy tenue (5.6pt):
#         Generado con Watermelon System v3.0.8 · build 22b268f3
#         · 2026-05-03
#     Trazabilidad: cualquier reporte impreso en el futuro
#     dice exactamente con qué build se generó.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.7..."
git add core/version.py
git add core/auth.py
git add pages/00_Login.py
git add pages/16_Reports.py
git add _publish_ciclo17_7_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.7..."
    git commit -m "feat(version): version visible cross-app (login + sidebar + PDF) (17.7)

Pedido del usuario: tener la version del software visible 'tipo
software internacional pro'.

(1) core/version.py NUEVO — single source of truth.
Auto-deriva via prioridad:
  1. env var WM_VERSION
  2. _git_latest_semver_tag() — tag semver mas alto del repo
     (clave: nuestros tags v3.0.x viven en main sobre merge
     commits; desde dev git describe no los encuentra)
  3. git describe --tags --dirty
  4. archivo VERSION
  5. fallback hardcoded
Environment inferido de branch (main=production, dev=development).
API: get_version_info() con lru_cache(1) para no fork-ear git
en cada rerun.

(2) Login footer: reemplaza 'build dev' por version + chip de
environment coloreado + commit SHA + fecha + copyright.

(3) Sidebar: caption monoespaciado pequeño debajo del logout,
con bullet coloreado por environment. No compite con la nav.

(4) PDF Reports: footer en portada Y paginas internas con
'Generado con Watermelon System v3.0.8 · build SHA · fecha'
en gris tenue 5.6pt esquina inferior derecha. Trazabilidad
de version en cualquier reporte impreso." || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo ""
echo "▶ Reconciliando..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Rebase falló."; exit 1; }

echo "▶ Push dev..."
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.7 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Cuando reinicies Streamlit local vas a ver:"
echo "   - Login footer: v3.0.8 [DEVELOPMENT] · 22b268f3 · 2026-05-03"
echo "   - Sidebar inferior: v3.0.8 ● development · 22b268f3"
echo "   - PDF generado: 'Watermelon System v3.0.8' en footer"
echo ""
echo " En producción (Streamlit Cloud / main) automaticamente"
echo " mostrará 'production' en verde porque la branch es main."
echo "================================================================"
