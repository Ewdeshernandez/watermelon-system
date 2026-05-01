#!/bin/bash
# Hotfix Ciclo 8: profile.profile_key -> profile.key
set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock

git checkout dev
git pull origin dev

git add core/instance_selector.py
git commit -m "fix(ciclo8): use profile.key instead of profile.profile_key

MachineProfile dataclass usa el atributo 'key', no 'profile_key'.
Estaba accediendo al nombre equivocado y producia AttributeError
en render_instance_selector cuando module_name no estaba en
applicable_modules del profile."

git push origin dev
echo ""
echo "OK — hotfix publicado en dev. Refrescá el navegador (Cmd+Shift+R)"
echo "y la página Asset Documents debería cargar sin error."
