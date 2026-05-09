#!/bin/bash
# Ciclo 22.2c+22.2d → DEV: refactor visual Captured Parameters + Documents Vault.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo22-2cd-params-docs-visual

echo "v3.30.0" > VERSION

git add pages/00_Machinery_Library.py VERSION
git commit -m "feat(22.2c+d): Captured Parameters con progreso + Documents grid

  ► 22.2c — Captured Parameters
    - Barra de progreso global (filled/total · %) con color dinámico
      (verde >=80%, ámbar >=40%, rojo <40%)
    - Chips por categoría con % completado y emoji icónico
      (🆔 Identificación, 🔩 Geometría cojinete, ⚙️ Rodamiento,
       ⚖️ Cargas, 📐 Tolerancias, ⚡ Operación, 🌀 Rotor, 🔗 Acople,
       💧 Lubricación, 📋 Otros)
    - Cada expander muestra '🔩 Cojinete - geometría — 5/8 (62%)'

  ► 22.2d — Documents Vault
    - Grid de cards (3 col) en lugar de tabla plana
    - Iconos por tipo: 📕 PDF/manual, 📊 datasheet, 📐 drawing,
      🏆 certificate, 📄 report, 📷 photo, 🗺️ schematic, 🛠️ maintenance
    - Filtros: dropdown tipo + buscador (título/descripción/tags)
    - Counter '15 de 42 documentos'
    - Cada card: icono grande + título (clamp 2 líneas) + filename +
      descripción (clamp) + tags chips + tamaño + fecha + botones
      Descargar/Eliminar
    - Empty states amigables

VERSION → v3.30.0"

git push -u origin feat/ciclo22-2cd-params-docs-visual
git checkout dev
git merge --no-ff feat/ciclo22-2cd-params-docs-visual -m "Merge feat/ciclo22-2cd-params-docs-visual into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 22.2c+d en DEV — v3.30.0"
echo " Probar wm-test → ML → seleccionar activo:"
echo "  • Sección 'Parámetros técnicos': barra progreso + chips %"
echo "  • Sección 'Documentos': grid de cards con filtros"
echo " Si OK → bash _publish_v3_30_0_to_main.sh"
echo "================================================================"
