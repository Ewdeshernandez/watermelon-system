#!/bin/bash
# Ciclo 22.2a → DEV: refactor visual del grid de Machinery Library.
# Cards modernos con severidad coloreada, schematic embebido, chips
# de metadata, layout consistente.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo22-2a-machinery-grid-modern

echo "v3.28.0" > VERSION

git add pages/00_Machinery_Library.py VERSION
git commit -m "feat(22.2a): grid moderno de Machinery Library con cards visuales

  ► render_machinery_grid_v2() — cards con:
    - Header: tag + badge de severidad coloreada (CRÍTICA roja,
      ATENCIÓN ámbar, VIGILANCIA azul, CONDICIÓN ACEPTABLE verde,
      SIN ANÁLISIS gris)
    - Schematic PNG embebido (si existe el schematic_png del activo;
      placeholder con ícono ⚙️ si no)
    - Título: driver → driven (formato visual)
    - Chips de metadata: cliente 👤, sitio 📍, RPM ⚡, potencia 🔋
    - Footer: # sensores 📡, # documentos 📄
    - Border azul + 'ACTIVA' badge cuando es la activa
    - Botón 'Activar →' cuando no

  ► render_machinery_grid() viejo NO se elimina — queda como
    fallback legacy. Si v2 da problemas, basta con cambiar la
    llamada en main() a la versión vieja.

  ► CSS embebido en HTML inline (textwrap.dedent para evitar el bug
    de CommonMark que ya conocimos en Reports).

VERSION → v3.28.0"

git push -u origin feat/ciclo22-2a-machinery-grid-modern
git checkout dev
git merge --no-ff feat/ciclo22-2a-machinery-grid-modern -m "Merge feat/ciclo22-2a-machinery-grid-modern into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 22.2a en DEV — v3.28.0"
echo " Probar wm-test → Machinery Library: cards modernos"
echo " Si OK → bash _publish_v3_28_0_to_main.sh"
echo "================================================================"
