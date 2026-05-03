#!/bin/bash
# =============================================================
# Watermelon — RELEASE v3.2.0 → MAIN
# =============================================================
# Promueve a producción (main) el bloque acumulado de los ciclos
# 17.9 → 17.14.1 que vivieron en dev las últimas semanas.
#
# Lo que MAIN va a recibir tras este script:
#
# Ciclo 17.9 — Catálogo ISO/API thresholds + UI selección + override
#   - core/iso_thresholds.py NUEVO (7 normas iniciales)
#   - Instance dataclass +5 campos (iso_norm_code/class + override)
#   - Tab "Norma ISO" en Machinery Library
#   - Cableado en Trends + PDF cita la norma + override justificado
#
# Ciclo 17.10 — Catálogo expandido (26 normas + API 684 + balanceo)
#   - +19 normas: API 684, ISO 21940-11/12 (balanceo rígido/flex),
#     API 671 (couplings), API 619/617/612/611/610/541/546,
#     IEC 60034-14, NEMA MG-1 P7, ANSI S2.41, VDI 2056/2059,
#     ISO 20816-5 (hidro), ISO 10816-7 (bombas), ISO 14694 (fans)
#   - Agrupación por dominio en UI (📳 VIB / 🛡️ EJE / ⚖️ BAL / 🔬 ROT)
#   - Heurísticas extendidas (Brush → API 546, hidro → 20816-5, etc.)
#
# Ciclo 17.11 — Home rediseño Nivel 1+2 (HMI premium)
#   - core/home_metrics.py NUEVO (fleet status, activity, sparklines)
#   - pages/_landing.py REWRITE: hero compacto con saludo + reloj +
#     turno + status, KPI band con sparklines, quick actions strip,
#     grid de active assets, activity feed, SCADA status footer
#
# Ciclo 17.12 — Nivel 3 (Health Score + Omnibox + Modo turno)
#   - core/health_score.py NUEVO (algoritmo 0-100 + gauge SVG)
#   - core/omnibox_search.py NUEVO (búsqueda fuzzy global)
#   - Modo turno auto: hero vira a rojo entre 22:00 y 06:00
#
# Ciclo 17.12.1 — Hotfix gauge SVG inline (1 línea sin comments)
#
# Ciclo 17.13 — Briefing diario + Severidad ejecutiva real + Cmd+K
#   - core/briefing.py NUEVO (PDF de 1 página)
#   - scripts/generate_daily_briefing.py NUEVO (cron-ready + SMTP)
#   - Instance +3 campos: last_executive_severity/summary/date
#   - Reports persiste severidad real al generar PDF
#   - Home muestra severity REAL (no heurística) cuando hay PDF previo
#   - Cmd+K (Mac) / Ctrl+K (Win/Linux) enfoca el omnibox
#
# Ciclo 17.14 — Sistema de usuarios real con Supabase Auth
#   - core/supabase_auth.py NUEVO (wrapper Auth admin API)
#   - core/auth.py prioriza Supabase, fallback al sistema legacy
#   - Login por email corporativo
#   - pages/_admin_users.py NUEVO (Admin Panel completo)
#   - scripts/bootstrap_admin.py NUEVO (crear admin única vez)
#   - scripts/test_supabase_auth.py NUEVO (smoke test)
#   - Roles automáticos por dominio (admin/specialist/client)
#
# Ciclo 17.14.1 HOTFIX CRÍTICO — Anti-pérdida de Reports
#   - save_report_state ATÓMICO (tmp + os.replace)
#   - Backups rotativos automáticos (.bak.1 → .bak.5)
#   - Recovery automático desde backup si JSON principal está corrupto
#   - Banner visible al usuario si hubo recovery o pérdida real
#   - Resuelve el bug que perdía un día de trabajo con 50-100 imágenes
#
# =============================================================
# IMPORTANTE — ANTES DE CORRER:
#
#   1. Verificar que estás en la carpeta correcta:
#        cd ~/Documents/WatermelonSystem
#
#   2. Verificar que dev está limpio y al día:
#        git status   (debería decir 'nothing to commit')
#        git log dev --oneline -10  (deberías ver el commit
#                                    de0a009 del 17.14.1 arriba)
#
#   3. Si ya pusheaste antes a dev y querés sincronizar:
#        git push origin dev
#
#   4. Después corré ESTE script:
#        bash _release_v3_2_0_main.sh
#
# El script hace:
#   - checkout main
#   - pull origin main (sincroniza)
#   - merge dev (no fast-forward para crear commit explícito de release)
#   - tag v3.2.0
#   - push origin main
#   - push origin v3.2.0
#   - vuelve a dev
#
# Si el merge tiene conflictos (no debería, dev está adelante puro),
# el script aborta y te indica cómo resolver. NO se borra nada.
# =============================================================

set -e
cd "$(dirname "$0")"

# Limpieza de locks
[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

echo ""
echo "================================================================"
echo " 🚀 RELEASE v3.2.0 → MAIN"
echo "================================================================"
echo ""
echo "Esto va a promover los siguientes ciclos a producción:"
echo "  17.9    Catálogo ISO/API + override"
echo "  17.10   Catálogo expandido (26 normas + API 684 + balanceo)"
echo "  17.11   Home Nivel 1+2 (HMI premium)"
echo "  17.12   Home Nivel 3 (Health Score + Omnibox + Modo turno)"
echo "  17.12.1 Hotfix gauge SVG"
echo "  17.13   Briefing diario + severidad ejecutiva real + Cmd+K"
echo "  17.14   Sistema de usuarios Supabase Auth + Admin Panel"
echo "  17.14.1 HOTFIX anti-pérdida de Reports (write atómico + backups)"
echo ""
echo "Estado actual:"
git log dev --oneline -1 | sed 's/^/  dev:  /'
git log main --oneline -1 | sed 's/^/  main: /'
echo ""

# ─── Confirmación interactiva
read -p "¿Confirmás el release a MAIN? (escribí 'si' para continuar): " CONFIRM
if [ "$CONFIRM" != "si" ]; then
    echo "✗ Release cancelado."
    exit 0
fi
echo ""

# ─── 1. Sync dev con remoto
echo "▶ 1/6  Sincronizando dev con origin..."
CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev
if ! git diff --quiet || ! git diff --staged --quiet; then
    echo "✗ Hay cambios sin commitear en dev. Commiteá primero."
    exit 1
fi
git fetch origin dev
git pull --rebase origin dev || {
    echo "✗ Pull de dev falló (probablemente conflicto). Resolvé y reintentá."
    exit 1
}
git push origin dev || {
    echo "✗ Push de dev falló. Verificá conexión / credenciales."
    exit 1
}
echo "  ✓ dev sincronizado"
echo ""

# ─── 2. Checkout main
echo "▶ 2/6  Cambiando a main..."
git checkout main || {
    echo "✗ No se pudo cambiar a main. ¿Existe la rama?"
    exit 1
}
git fetch origin main
git pull --rebase origin main || {
    echo "✗ Pull de main falló."
    exit 1
}
echo "  ✓ main actualizado"
echo ""

# ─── 3. Merge dev → main
echo "▶ 3/6  Mergeando dev → main (no fast-forward, commit explícito)..."
MERGE_MSG="release(v3.2.0): merge dev -> main

Acumulado de los ciclos 17.9 - 17.14.1:

- 17.9    Catalogo ISO/API thresholds + UI seleccion + override
- 17.10   Catalogo expandido (26 normas + API 684 + balanceo
          ISO 21940-11/12 rigidos y flexibles)
- 17.11   Home rediseno Nivel 1+2 (HMI premium con fleet status)
- 17.12   Nivel 3 - Health Score + Omnibox + Modo turno
- 17.12.1 Hotfix gauge SVG inline
- 17.13   Briefing diario PDF + severidad ejecutiva real + Cmd+K
- 17.14   Sistema de usuarios Supabase Auth + Admin Panel
- 17.14.1 HOTFIX anti-perdida de Reports (write atomico + backups)

Cambio de modelo de auth: usuarios reales en Supabase Auth con
roles automaticos por dominio. Ehernandez@sigasas.com es admin
unico. Admin Panel para crear/eliminar/bloquear usuarios.

Bug critico resuelto: perdida de trabajo en Reports con muchas
imagenes (write no atomico + silent fail). Ahora con backups
rotativos automaticos y notificacion visible al usuario."

git merge --no-ff dev -m "$MERGE_MSG" || {
    echo "✗ Merge falló (probablemente conflictos). NO se subió nada."
    echo "  Para resolver: 'git mergetool', luego 'git commit', luego correr otra vez."
    echo "  Para abortar: 'git merge --abort'"
    exit 1
}
echo "  ✓ Merge OK"
echo ""

# ─── 4. Tag v3.2.0
echo "▶ 4/6  Creando tag v3.2.0..."
TAG_EXISTS=$(git tag -l "v3.2.0")
if [ -n "$TAG_EXISTS" ]; then
    echo "  ⚠  Tag v3.2.0 ya existe. Saltando creación."
else
    git tag -a v3.2.0 -m "Release v3.2.0 — Auth real + Home premium + Briefing diario + Anti-perdida Reports"
    echo "  ✓ Tag v3.2.0 creado"
fi
echo ""

# ─── 5. Push main + tags
echo "▶ 5/6  Pusheando main + tag a origin..."
git push origin main || {
    echo "✗ Push de main falló."
    exit 1
}
git push origin v3.2.0 || {
    echo "  ⚠ Push del tag falló (ya existía remoto?)"
}
echo "  ✓ main y tag pusheados"
echo ""

# ─── 6. Volver a dev
echo "▶ 6/6  Volviendo a dev..."
git checkout dev
echo "  ✓ Estás de vuelta en dev"
echo ""

echo "================================================================"
echo " ✅ RELEASE v3.2.0 COMPLETADO"
echo "================================================================"
echo ""
echo " La app productiva (https://wm-home-final-2026.streamlit.app)"
echo " va a redeployar en ~30-60 segundos."
echo ""
echo " QUEDA POR HACER después del redeploy:"
echo ""
echo "  1. Abrir el app productivo y verificar:"
echo "     - El Home se ve con el rediseño nuevo"
echo "     - Login con ehernandez@sigasas.com (con la password del bootstrap)"
echo "     - Aparece botón 'Admin · Usuarios' en sidebar"
echo ""
echo "  2. Crear el primer specialist desde el Admin Panel:"
echo "     - Click en 'Admin · Usuarios'"
echo "     - 'Crear nuevo usuario'"
echo "     - Email: jsuarez@sigasas.com"
echo "     - Nombre: J Suarez"
echo "     - Role: specialist (auto-sugerido por dominio @sigasas.com)"
echo "     - Copiar password temporal y entregársela"
echo ""
echo "  3. Verificar que jsuarez puede:"
echo "     - Hacer login normal"
echo "     - Ver TODAS las páginas del menú"
echo "     - Editar instancias, cargar CSVs, generar reportes"
echo "     - NO ve el botón 'Admin · Usuarios' (correcto, no es admin)"
echo ""
echo "  4. Si algo no anda bien, podés rebobinar a v3.1.5 con:"
echo "     git checkout main && git reset --hard 9ca245e && git push --force origin main"
echo "     (NO recomendado salvo emergencia, pierde el release)"
echo ""
echo "================================================================"
