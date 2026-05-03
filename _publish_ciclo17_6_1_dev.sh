#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.6.1 → DEV (no toca main)
# =============================================================
# Quejas del usuario tras revisar 17.6:
#
# (1) "sigue apareciendo el recuadro rojo sobre usuario y clave"
# (2) "porque esa línea blanca o bloque blanco arriba de Ingresar
#      sin nada igual que encima del logo de watermelon al lado
#      izquierdo"
#
# CAUSA Y FIX:
#
# (1) BORDE ROJO PERSISTENTE
#     Mi CSS solo overrideaba [data-testid="stTextInput"] > div > div,
#     pero Streamlit usa BaseWeb que tiene OTROS selectores:
#       [data-baseweb="input"]
#       [data-baseweb="base-input"]
#     y además Chrome agrega outline rojo en :focus-visible y
#     fondo rojizo/amarillo en :-webkit-autofill.
#
#     Override total ahora cubre:
#       - data-baseweb=input + base-input (BaseWeb)
#       - :focus + :focus-visible (Chrome outline)
#       - :-webkit-autofill (Chrome password manager)
#       - aria-invalid + :invalid + style*="rgb(255..." (heurísticas)
#       - :has(input:-webkit-autofill) (autofill via parent)
#     Todos forzando border-color #d3dde9 y box-shadow:none.
#
#     También agregué autocomplete="username" / "current-password"
#     a los st.text_input para que Chrome los reconozca como
#     campos válidos de login y no aplique validation styling.
#
# (2) CAJAS BLANCAS FANTASMA
#     Antes envolvía con <div class="wm-login-card">...</div> y
#     <div class="wm-logo-box">...</div> en st.markdown SEPARADOS,
#     pero Streamlit renderiza CADA st.markdown como bloque
#     independiente — el div abierto y el div cerrado no envuelven
#     los componentes intermedios (form, image), quedan como
#     bloques vacíos con background blanco.
#
#     Fix con técnica :has():
#       - <span class="wm-login-marker"></span> (invisible) dentro
#         del column derecho
#       - CSS: [data-testid="column"]:has(.wm-login-marker) {
#           background: rgba(255,255,255,0.88);
#           border-radius: 22px; padding: 2rem ...; ... }
#       - El estilo se aplica a TODA la columna (que sí envuelve
#         el form y todo el contenido). Sin div fantasma.
#
#     Mismo trick para el logo: [data-testid="column"]:has(
#     .wm-logo-marker) [data-testid="stImage"] img { width: 56px;
#     height: 56px; border-radius: 14px; box-shadow: ... }
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.6.1..."
git add pages/00_Login.py
git add _publish_ciclo17_6_1_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.6.1..."
    git commit -m "fix(login): borde rojo persistente + cajas blancas fantasma (17.6.1)

Quejas tras 17.6:
- 'Sigue apareciendo el recuadro rojo sobre usuario y clave'
- 'Porque esa linea blanca o bloque blanco arriba de Ingresar
   sin nada igual que encima del logo de watermelon'

(1) BORDE ROJO: Override solo cubria
    [data-testid=stTextInput] > div > div, faltaban los
    selectores BaseWeb [data-baseweb=input] y [data-baseweb=
    base-input], plus :focus-visible (Chrome outline) y
    :-webkit-autofill (password manager). Override total
    ahora cubre TODOS los selectores BaseWeb + autofill +
    focus-visible, todos forzando border #d3dde9 + box-shadow
    none. Agregue autocomplete=username/current-password a
    st.text_input para que Chrome los trate como login fields.

(2) CAJAS BLANCAS: Antes envolvia con <div class=wm-login-card>
    y <div class=wm-logo-box> en st.markdown separados, pero
    Streamlit renderiza cada st.markdown como bloque
    independiente — los div abierto+cerrado no envuelven los
    componentes intermedios (form, image), quedan vacios con
    background blanco visible arriba.

    Fix con :has(): inyecto <span class=wm-login-marker>
    invisible y CSS [data-testid=column]:has(.wm-login-marker)
    aplica el card style a TODA la columna. Mismo trick para
    el logo. Sin div fantasma." || echo "  (sin cambios)"
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
echo " ✓ Ciclo 17.6.1 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " Refrescá Cmd+Shift+R en el login y verificá:"
echo "   - Sin recuadro rojo en username/password (ni en focus,"
echo "     ni cuando Chrome autofill ofrece sugerencias)"
echo "   - Sin caja blanca fantasma encima del logo ni encima"
echo "     de 'Ingresar'"
echo "   - Card del login limpia, alineada"
echo "================================================================"
