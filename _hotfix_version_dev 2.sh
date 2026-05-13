#!/bin/bash
# Hotfix: archivo VERSION + fallback en code para que si Streamlit
# Cloud no consigue git tags, lea v3.15.0 desde VERSION en vez de
# v3.0.8 hardcodeado.
# IMPORTANTE: si la app en Streamlit Cloud sigue mostrando v2.1
# después de este hotfix, hay que ir a Settings → Secrets y BORRAR
# la línea WM_VERSION="v2.1".
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B hotfix/version-fallback

git add VERSION core/version.py
git commit -m "fix: archivo VERSION + bump _FALLBACK_VERSION a v3.15.0

Si Streamlit Cloud no consigue git tags al deployar (caso conocido
en su build), version.py caía a _FALLBACK_VERSION=v3.0.8. Combinado
con un override viejo WM_VERSION=v2.1 en Streamlit secrets, mostraba
versiones obsoletas.

Cambios:
- Nuevo archivo VERSION (sourced as 4to fallback en version.py).
- _FALLBACK_VERSION subido de v3.0.8 a v3.15.0.

NOTA: si después del deploy sigue mostrando v2.1, hay que ir a
Streamlit Cloud Settings → Secrets de cada app (wm-test y
wm-home-final-2026) y BORRAR la línea WM_VERSION."

git push -u origin hotfix/version-fallback
git checkout dev
git merge --no-ff hotfix/version-fallback -m "Merge hotfix/version-fallback into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Hotfix VERSION en DEV"
echo "================================================================"
echo " Después del redeploy debe verse v3.14.0 (tag actual de dev)"
echo " — A MENOS que tengas WM_VERSION='v2.1' en Streamlit secrets."
echo " "
echo " Si sigue saliendo v2.1: borrá la env var en Streamlit Cloud."
echo "================================================================"
