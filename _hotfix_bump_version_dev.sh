#!/bin/bash
# Bump VERSION file en dev a la próxima release (v3.17.0).
# Convención: dev siempre lleva la versión que va a salir, así
# wm-test muestra 'vX.Y.Z DEVELOPMENT' coherente con el roadmap.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only

git add VERSION
git commit -m "chore: bump VERSION → v3.17.0 (próxima release con Ciclo 18.3)"
git push origin dev

echo "================================================================"
echo " ✓ VERSION bumpeada a v3.17.0 en dev"
echo " Después del redeploy wm-test debe mostrar v3.17.0 DEVELOPMENT."
echo " Si OK → bash _publish_v3_17_0_to_main.sh"
echo "================================================================"
