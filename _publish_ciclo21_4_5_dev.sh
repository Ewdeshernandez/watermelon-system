#!/bin/bash
# Ciclo 21.4 (schematic recip correcto) + 21.5 (click-to-place visual editor) → DEV.
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo21-4-5-recip-schematic-editor

echo "v3.25.0" > VERSION

git add core/recip_schematic.py pages/_machinery_wizard.py VERSION
git commit -m "feat(21.4+21.5): schematic reciprocante + editor visual click-to-place

  ► 21.4 — core/recip_schematic.py (NUEVO)
    generate_recip_png(n_cylinders, n_motor_planes) → PNG bytes con
    motor + pieza distancia + acople + cigüeñal + compresor con N
    cilindros físicos correctos. Reemplaza el render genérico que
    confundía cojinetes con sensores.

    sensor_default_position(sensor) → (x_pct, y_pct) basado en rol
    (motor DE/NDE, frame top/side, cilindro N, rod drop, keyphasor).

  ► 21.5 — pages/_machinery_wizard.py
    Nuevo tab 'Editor visual' en Paso 4 para reciprocantes:
      - Genera PNG del activo con sensores como markers numerados
      - Lista lateral: botón por sensor para 'seleccionar'
      - Click sobre la imagen reposiciona el sensor seleccionado
      - Coordenadas se persisten en x_pct/y_pct del sensor
    Usa streamlit_image_coordinates (ya en requirements.txt).

  ► _build_reciprocating_sensor_map: asigna x_pct/y_pct sensatos
    por defecto antes de ir al editor.

  ► _execute_creation: persiste el PNG generado como schematic_png
    en el Document Vault del activo.

VERSION → v3.25.0 en dev."

git push -u origin feat/ciclo21-4-5-recip-schematic-editor
git checkout dev
git merge --no-ff feat/ciclo21-4-5-recip-schematic-editor -m "Merge feat/ciclo21-4-5 into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 21.4+21.5 en DEV — v3.25.0"
echo " Probar: wizard reciprocante con 4 cilindros → en Paso 4 ver"
echo "         tab 'Editor visual' con motor + 4 cilindros dibujados"
echo "         click sensor en lista → click sobre imagen → reposiciona"
echo " Si OK → bash _publish_v3_25_0_to_main.sh"
echo "================================================================"
