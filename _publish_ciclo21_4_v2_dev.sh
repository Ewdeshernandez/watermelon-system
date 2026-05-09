#!/bin/bash
# Ciclo 21.4 v2: rediseño schematic recip + 4 plantillas LATAM nuevas.
# - core/recip_schematic.py: sin pieza distancia, acople 3 líneas,
#   cilindros TODOS arriba alineados, motor con tapas + aletas estilizadas
# - data/machine_templates.json: + GE LM6000, LM5000, TM2500, SGT300+gearbox
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo21-4-v2-recip-design

echo "v3.26.0" > VERSION

git add core/recip_schematic.py data/machine_templates.json VERSION
git commit -m "feat(21.4 v2): rediseño schematic recip + 4 plantillas turbogen LATAM

  ► core/recip_schematic.py
    - Eliminada 'pieza distancia' (era error técnico — va el acople)
    - Acople = 3 líneas verticales (estilo industrial real)
    - Cilindros TODOS arriba alineados (estilo Ariel KBK / Burckhardt)
    - Motor: tapas DE/NDE diferenciadas + aletas decorativas
    - Cigüeñal interno al frame del compresor, más gruesa
    - Conexión cilindro→cuerpo con cuello visible

  ► data/machine_templates.json: 4 nuevas plantillas turbogeneradores
    aeroderivate LATAM:
      - GE LM6000 + Brush (~45 MW, 3600 RPM)        [TES1/TES3]
      - GE LM5000 + Brush (~33 MW)
      - TM2500 + generador (~25 MW)
      - Siemens SGT-300 + gearbox planetario + generador (~8 MW)

VERSION → v3.26.0"

git push -u origin feat/ciclo21-4-v2-recip-design
git checkout dev
git merge --no-ff feat/ciclo21-4-v2-recip-design -m "Merge feat/ciclo21-4-v2-recip-design into dev"
git push origin dev

# Direct to main
git checkout main
git pull origin main --ff-only
echo "v3.26.0" > VERSION
git add VERSION
git commit -m "chore: bump VERSION to v3.26.0" --allow-empty
git merge --no-ff dev -m "release(v3.26.0): schematic recip rediseñado + 4 plantillas turbogen aeroderivate"
git tag -a "v3.26.0" -m "Release v3.26.0"
git push origin main
git push origin "v3.26.0"

echo ""
echo "================================================================"
echo " ✅ v3.26.0 en MAIN"
echo " Probar: wizard recip → Paso 4 → Editor visual con dibujo limpio"
echo "         wizard turbomáquina → plantilla LM6000/LM5000/TM2500/SGT300"
echo "================================================================"
