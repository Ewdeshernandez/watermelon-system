#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.4.6 UX → MAIN
# =============================================================
# Release de mejoras visuales del sidebar. NO urgente, NO toca
# lógica funcional. Solo CSS/HTML del sidebar en core/auth.py.
#
# Cambios incluidos:
#
#   CICLO 17.22 — Avatar mini al fondo del sidebar (Opción A UX)
#   ─────────────────────────────────────────────────────────────
#   Antes: card grande "Sesión" arriba con Usuario/Nombre/Correo/
#   Rol (4 líneas, el correo aparecía 2 veces, ocupaba ~25% del
#   scroll inicial).
#
#   Ahora: avatar circular gradient con la inicial del nombre,
#   nombre + rol en pequeño, al fondo del sidebar antes de
#   "Cambiar mi password" / "Cerrar sesión". Tooltip con el email
#   completo al hover.
#
#   CICLO 17.23 — Sacar header "NAVEGACIÓN" del sidebar
#   ─────────────────────────────────────────────────────────────
#   Linear/Notion/Stripe no ponen "NAVIGATION" arriba de su menú
#   principal — los botones SON la navegación, no necesitan título
#   que lo diga. El divider de arriba ya da separación visual.
#
#   Mantenemos los headers de "Administración" (separa items
#   técnicos de admin) y "Sesión" (separa acciones de navegación).
#
# Resultado:
#   Sidebar más limpio, profesional y escaneable. Mismo contenido
#   funcional, mejor jerarquía visual.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " ✨ RELEASE v3.4.6 UX → MAIN  (sidebar polish, no urgente)"
echo "================================================================"
echo ""
echo "Esto va a promover a producción:"
echo "  17.22  Avatar mini al fondo del sidebar (Opción A UX)"
echo "  17.23  Sacar header 'NAVEGACIÓN' (los botones son navegación)"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

read -p "¿Confirmás el release a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

echo "▶ 1/7  Commit del 17.22 + 17.23 en dev..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

# Descartar mods locales a scripts ya merged
git checkout HEAD -- _release_v3_4_2_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_3_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_4_hotfix.sh 2>/dev/null || true
git checkout HEAD -- _release_v3_4_5_hotfix.sh 2>/dev/null || true

if ! git diff --quiet || ! git diff --staged --quiet; then
    git add core/auth.py _release_v3_4_6_ux.sh
    git commit -m "ux(17.22+17.23): avatar mini sidebar + sacar header 'Navegación'

17.22 — Avatar mini al fondo del sidebar (Opción A UX)
======================================================
Antes: card grande 'Sesión' arriba con Usuario/Nombre/Correo/Rol
(4 líneas, el correo aparecía 2 veces, ocupaba ~25% del scroll
inicial del sidebar).

Ahora: avatar circular gradient con la inicial del nombre, nombre
+ rol en pequeño, al fondo del sidebar antes de 'Cambiar mi
password' / 'Cerrar sesión'. Tooltip con el email completo al
hover sobre el bloque.

CSS nuevo .wm-user-mini con gradient celeste, fondo translúcido y
borde sutil. Inicial del nombre como avatar (Ewdes → 'E') con
fallback al primer carácter del email si no hay full_name.

17.23 — Sacar header 'NAVEGACIÓN' del sidebar
=============================================
Linear/Notion/Stripe no ponen 'NAVIGATION' arriba de su menú
principal — los botones SON la navegación, no necesitan título
que lo diga. El divider de arriba ya da separación visual.

Mantenemos los headers de 'Administración' (separa items
técnicos de admin) y 'Sesión' (separa acciones de navegación
de los botones de página).

Resultado:
  Sidebar más limpio y profesional. Mismo contenido funcional,
  mejor jerarquía visual. Sin cambios en lógica ni en otras
  páginas de la app." || echo "  (nada nuevo para commitear)"
fi

echo "  ✓ dev tiene los cambios commiteados"
echo ""

echo "▶ 2/7  Push de dev a origin..."
git fetch origin dev
git pull --rebase origin dev || { echo "✗ Pull dev falló."; exit 1; }
git push origin dev || { echo "✗ Push dev falló."; exit 1; }
echo "  ✓ dev en origin actualizado"
echo ""

echo "▶ 3/7  Cambiando a main..."
git checkout main || { echo "✗ No se pudo cambiar a main."; exit 1; }
git fetch origin main
git pull --rebase origin main || { echo "✗ Pull main falló."; exit 1; }
echo "  ✓ main actualizado"
echo ""

echo "▶ 4/7  Mergeando dev → main..."
MERGE_MSG="ux(v3.4.6): merge dev -> main · sidebar polish (17.22 + 17.23)

Release de mejoras visuales del sidebar. NO toca lógica
funcional. Solo CSS/HTML en core/auth.py.

17.22 Avatar mini:
  Reemplaza el card grande Sesión/Usuario/Nombre/Correo/Rol de
  arriba (info redundante, ocupaba 25% del scroll) por un
  avatar circular con inicial + nombre + rol al fondo del
  sidebar. Tooltip con email completo al hover.

17.23 Sacar header 'Navegación':
  Linear/Notion/Stripe tampoco lo ponen — los botones son la
  navegación. Mantenemos 'Administración' y 'Sesión' que sí
  cumplen función de agrupación.

Resultado: sidebar más limpio y profesional, mismo contenido
funcional, mejor jerarquía visual."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (conflictos). NO se subió nada."
    exit 1
}
echo "  ✓ Merge OK"
echo ""

echo "▶ 5/7  Creando tag v3.4.6..."
TAG_EXISTS=$(git tag -l "v3.4.6")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.4.6 ya existe. Saltando creación."
else
    git tag -a v3.4.6 -m "Release v3.4.6 — Sidebar polish (17.22 avatar mini + 17.23 sacar Navegación)"
    echo "  ✓ Tag v3.4.6 creado"
fi
echo ""

echo "▶ 6/7  Pusheando main + tag a origin..."
git push origin main || { echo "✗ Push main falló."; exit 1; }
git push origin v3.4.6 || echo "  ⚠ Push del tag falló (ya existía remoto?)"
echo "  ✓ main y tag pusheados"
echo ""

echo "▶ 7/7  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.4.6 COMPLETADO"
echo "================================================================"
echo ""
echo " ⏱  Streamlit Cloud va a redeployar wm-home-final-2026 en 1-2 min."
echo ""
echo " 👁  Cambios visibles en producción:"
echo ""
echo "    Sidebar:"
echo "    - Arriba arranca DIRECTO en la lista de páginas"
echo "      (sin el header 'NAVEGACIÓN')"
echo "    - El card grande de usuario YA NO ESTÁ arriba"
echo "    - Al fondo: avatar circular con inicial + nombre + rol,"
echo "      antes de 'Cambiar mi password' / 'Cerrar sesión'"
echo "    - Hover sobre el avatar → tooltip con el email completo"
echo ""
echo " 📊 Sin cambios funcionales:"
echo ""
echo "    Toda la navegación, autenticación, generación de PDF,"
echo "    cargas, etc. sigue funcionando exactamente igual que en"
echo "    v3.4.5. Solo cambia cómo se ve el sidebar."
echo ""
echo " 💡 Pendientes en backlog (no urgentes):"
echo ""
echo "    - Rotar Azure client_secret (expuesto en chat)"
echo "    - 5ta Redirect URL de wm-test en Supabase"
echo ""
echo "================================================================"
