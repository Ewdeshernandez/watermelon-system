#!/bin/bash
# =============================================================
# Watermelon — v3.0.6 hotfix: motor sin grilla + 4 cilindros
# =============================================================
# QUEJA DEL USUARIO:
#   "terrible es terrible, te hablé un motor eléctrico para que
#   le colocas esa rueda de carreta horrible al lado libre, y el
#   compresor te dije 4 cilindros dos por cada lado de ese cajón"
#
# DOS FIXES:
#
# (1) MOTOR — quitar la "rueda de carreta" (fan grille circular
#     con cruz que metí en v3.0.5). Ahora el motor es SOBRIO:
#       - Frame con 5 cooling fins horizontales sutiles
#       - End shields planos en ambos extremos (sin bolt circles
#         ni grilla de ventilador)
#       - Caja de bornes pequeña arriba con línea de partición
#       - Eje sobresale del lado DE
#       - Patas de montaje abajo
#     Diseño minimal, parece un motor industrial sin recargar
#     el dibujo.
#
# (2) COMPRESOR — los 4 cilindros se veían como 2 porque los
#     DOWN parecían patas del crankcase. Ahora:
#       - Crankcase MÁS BAJO (crank_h 0.50 vs 0.85)
#       - Cilindros MÁS ALTOS (cyl_h 1.20 vs 0.95)
#       - Cilindros MÁS ANCHOS (cyl_w hasta 0.55)
#       - Cabezales prominentes con válvulas visibles (2 stubs
#         por cabezal representando válvula succión/descarga)
#       - Cada cilindro NUMERADO en el centro del cuerpo (1, 2,
#         3, 4) para que sea inconfundible
#       - Línea del cigüeñal visible dentro del crankcase
#
#     Resultado para ARIEL KBK/4:
#       throw 1: cilindro 1 ARRIBA + cilindro 2 ABAJO
#       throw 2: cilindro 3 ARRIBA + cilindro 4 ABAJO
#     Los 4 son inmediatamente visibles, con número, válvulas y
#     cabezales claros. No se confunden con patas.
#
# (3) ylim expandido a (-0.4, 4.7) cuando hay recip_compressor
#     para que ni los cilindros DOWN ni la etiqueta UP queden
#     cortadas.
# =============================================================

set -e
cd "$(dirname "$0")"

VERSION="v3.0.6"
RELEASE_TITLE="Hotfix: motor sin grilla + 4 cilindros del recip claramente visibles"

# ---------- Locks ----------
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

# ---------- 0) Asegurar branch dev ----------
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# ---------- 1) Stage ----------
echo ""
echo "▶ Stageando cambios v3.0.6..."
git add core/sensor_diagram.py
git add _publish_v3_0_6_hotfix.sh 2>/dev/null || true

# ---------- 2) Commit ----------
if ! git diff --staged --quiet; then
    echo "▶ Commiteando hotfix v3.0.6..."
    git commit -m "fix(sensor_diagram): motor sin grilla + 4 cilindros recip prominentes (17.5.14)

Quejas del usuario:
- 'rueda de carreta horrible' en el motor (la grilla circular
  del ventilador que meti en v3.0.5)
- 'solo 2 cilindros' aunque eran 4 (los DOWN parecian patas
  del crankcase, no cilindros)

Fix motor: remueve completamente la fan grille + bolt circles
+ cable glands. Diseño minimal: frame con 5 cooling fins
sutiles, end shields planos, caja de bornes pequeña, eje, patas.
No recarga el dibujo, parece un motor industrial sobrio.

Fix compresor:
- Crankcase mas bajo (crank_h 0.50 vs 0.85)
- Cilindros mas altos (cyl_h 1.20 vs 0.95)
- Cilindros mas anchos (cyl_w hasta 0.55)
- Cabezales prominentes con 2 valvulas visibles cada uno
- Cada cilindro NUMERADO (1, 2, 3, 4) en el centro del body
- Linea del cigueñal visible dentro del crankcase

Para ARIEL KBK/4: throw 1 = cilindros 1 (UP) + 2 (DOWN);
throw 2 = cilindros 3 (UP) + 4 (DOWN). Los 4 inmediatamente
visibles con numero, valvulas y cabezales claros. No se
confunden con patas del frame.

ylim expandido a (-0.4, 4.7) cuando hay recip para que ni
DOWN ni etiqueta UP queden cortadas." || echo "  (sin cambios)"
else
    echo "  (no hay cambios staged)"
fi

# ---------- 3) Rebase ----------
echo ""
echo "▶ Reconciliando contra origin/dev..."
git fetch origin dev
git pull --rebase origin dev || {
    echo "✗ Rebase falló."
    exit 1
}

# ---------- 4) Push dev ----------
echo "▶ Pusheando dev..."
git push origin dev

# ---------- 5) Switch a main ----------
echo "▶ Cambiando a main..."
git checkout main
git pull origin main

# ---------- 6) Merge ----------
echo "▶ Mergeando dev → main..."
git merge dev --no-ff -m "hotfix(${VERSION}): ${RELEASE_TITLE}

Sobre v3.0.5:

Motor electrico: quita la fan grille circular ('rueda de
carreta') + bolt circles + cable glands. Diseño sobrio: frame
con cooling fins sutiles, end shields planos, caja de bornes,
eje, patas. Sin recargar.

Compresor reciprocante: cilindros AHORA son grandes (1.20
alto, hasta 0.55 ancho), con cabezales prominentes + 2
valvulas visibles por cabezal + numero del cilindro en el
centro. Para ARIEL KBK/4: 4 cilindros inmediatamente
visibles con etiquetas 1, 2, 3, 4 distribuidos en pares
opuestos por throw."

# ---------- 7) Tag ----------
echo "▶ Creando tag ${VERSION}..."
git tag -a "${VERSION}" -m "${RELEASE_TITLE}

Hotfix sobre v3.0.5: motor sin grilla circular ('rueda de
carreta') + compresor reciprocante con los 4 cilindros
claramente visibles, numerados, con cabezales prominentes y
valvulas. ARIEL KBK/4 ahora es inconfundible."

# ---------- 8) Push main + tag ----------
echo "▶ Pusheando main + ${VERSION}..."
git push origin main
git push origin "${VERSION}"

# ---------- 9) Volver a dev ----------
echo "▶ Volviendo a dev..."
git checkout dev

echo ""
echo "================================================================"
echo " ✓ ${VERSION} publicado a main"
echo "================================================================"
echo ""
echo " Streamlit Cloud auto-redeploya en ~2 min."
echo " Después refrescá Cmd+Shift+R y vas a ver:"
echo "   - Motor sin la 'rueda de carreta' (sobrio)"
echo "   - 4 cilindros del compresor con números 1, 2, 3, 4"
echo "     en cada uno, claramente distintos del crankcase"
echo "================================================================"
