#!/bin/bash
# =============================================================
# Watermelon — HOTFIX 2 Ciclo 14a: warning module_name='library' (dev)
# =============================================================
# El render_instance_selector valida que el module_name pasado esté
# declarado en los modulos aplicables del profile activo. Pasamos
# 'library' al promover la pagina, pero ningun profile en
# core/machine_profiles.py declara 'library' como modulo aplicable
# (es universal por diseño), asi que mostraba un warning naranja
# "El profile X no incluye 'library' en sus modulos aplicables".
#
# Fix: revertir module_name a 'documents' (alias historico ya
# aceptado por todos los profiles desde Ciclo 8). La Library hereda
# de Asset Documents, asi que el alias es correcto semanticamente.
#
# Ejecutar:
#   bash _hotfix_ciclo14a_warning_v1.sh
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock

CURRENT_BRANCH=$(git branch --show-current)
if [ "$CURRENT_BRANCH" != "dev" ]; then
  git checkout dev
fi
git pull origin dev

git add pages/00_Machinery_Library.py
git status --short | head

git commit -m "fix(library): hotfix 2 Ciclo 14a — module_name 'library' no esta declarado en profiles

render_instance_selector validaba 'library' contra los modulos aplicables
del profile, pero ningun profile lo declara. Mostrando warning naranja:
'El profile X no incluye library en sus modulos aplicables'.

Fix: revertir a module_name='documents' (alias historico aceptado por
todos los profiles desde Ciclo 8). La Library hereda de Asset Documents,
el alias es correcto semanticamente."

git push origin dev

echo ""
echo "================================================================"
echo " HOTFIX 2 aplicado — refrescar app y el warning debe desaparecer"
echo "================================================================"
