#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.14 → DEV: Sistema de usuarios real
#                              (Supabase Auth + Admin Panel)
# =============================================================
# Reemplaza el sistema viejo de admin/demo hardcoded por usuarios
# REALES persistidos en Supabase Auth, con roles automáticos por
# dominio y un Admin Panel completo para gestionarlos.
#
# Lo nuevo que ve el usuario:
#
#   ► LOGIN POR EMAIL CORPORATIVO
#     - Antes: usuario "admin" con password fija para todos
#     - Ahora: cada persona usa su correo @sigasas.com con password
#       propia. Los clientes externos también pueden tener usuario
#       (con role limitado), pero con su propio email
#     - Mensajes de error claros: incorrectos / bloqueado / no existe
#
#   ► ROLES AUTOMÁTICOS POR DOMINIO
#     - ehernandez@sigasas.com         → admin (único)
#     - cualquier *@sigasas.com         → specialist
#     - cualquier otro dominio          → client
#     - El admin puede sobreescribir manualmente desde el panel
#
#   ► ADMIN PANEL (botón nuevo en sidebar, SOLO visible para admin)
#     - Tabla con todos los usuarios: email, nombre, role, status,
#       fecha de creación, último login
#     - Crear usuario: form con email, nombre, role, password
#       temporal auto-generada (12 chars sin caracteres ambiguos)
#     - Por usuario: cambiar role, resetear password, bloquear/
#       desbloquear, editar nombre, eliminar (con doble confirm)
#     - El admin único está PROTEGIDO contra modificación desde la UI
#       (anti-lockout)
#     - Búsqueda por email/nombre, KPI band con conteos por role
#
#   ► FALLBACK BACKWARDS-COMPAT
#     - Si Supabase Auth falla por config/red, cae al sistema viejo
#       de admin/cliente1 hardcoded en .streamlit/secrets.toml
#     - Los usuarios viejos siguen funcionando hasta que se migren
#       manualmente desde el Admin Panel
#
# Cambios técnicos:
#
# (NUEVO) core/supabase_auth.py
#   - Wrapper completo del Auth admin API (supabase-py 2.29)
#   - infer_role_from_email: regla de dominio determinística
#   - is_admin_email: helper para guard del único admin
#   - is_supabase_auth_enabled: detecta si secrets están configurados
#   - get_admin_client: cliente Supabase con service_key (cached
#     por sesión Streamlit)
#   - create_user: con email_confirm=True (SMTP no listo aún)
#   - signin_user: con manejo legible de errores (incorrectos /
#     bloqueado / no confirmado)
#   - list_all_users / get_user_by_email
#   - update_user_role, update_user_full_name, reset_user_password
#   - block_user (ban_duration='876000h' = 100 años)
#   - unblock_user (ban_duration='none')
#   - delete_user (irreversible)
#   - generate_temp_password: 12 chars alfanuméricos sin ambiguos
#     (sin 0, O, l, 1)
#   - Helpers _user_to_dict / _session_to_dict
#
# (MODIFICADO) core/auth.py
#   - login() prioriza Supabase Auth, fallback al sistema viejo
#   - get_current_user() devuelve nuevos campos: user_id, is_admin, source
#   - logout() limpia también las claves nuevas
#   - render_user_menu() agrega botón "👥 Admin · Usuarios" SOLO
#     si user.is_admin es True (visible solo para ehernandez@sigasas.com)
#   - El render_user_menu sigue mostrando el menú normal para todos
#
# (MODIFICADO) pages/00_Login.py
#   - Label "Usuario o correo corporativo" → "Correo corporativo"
#   - Placeholder "nombre.apellido@empresa.com" → "@sigasas.com"
#   - Help text explicando el origen del email para SIGASAS vs cliente
#   - autocomplete="email" en lugar de "username"
#
# (NUEVO) pages/_admin_users.py
#   - Página completa del Admin Panel
#   - Guard: solo accesible si is_admin_email(current_user.email)
#   - 5 KPI cards: total, admin, specialist, client, bloqueados
#   - Form de crear usuario con password auto-generada
#   - Búsqueda + lista de cards con acciones por usuario via popovers
#   - Estilos con prefix .wmu- para no chocar con el resto del app
#
# (NUEVO) scripts/bootstrap_admin.py
#   - Script standalone para crear el admin único la PRIMERA vez
#   - Verifica conexión, busca si ya existe, lo crea o resetea password
#   - Imprime credenciales en terminal (UNA sola vez)
#   - Necesario porque sin un admin creado, no se puede acceder al
#     Admin Panel para crear más usuarios (huevo y gallina)
#
# CÓMO USARLO TRAS EL MERGE:
#
#   1. Pushear estos cambios a dev (este script lo hace)
#   2. Esperar al deploy de Streamlit Cloud
#   3. EN TU MAC LOCAL, correr UNA vez:
#        cd ~/Documents/WatermelonSystem
#        python scripts/bootstrap_admin.py
#      → te imprime tu password temporal en pantalla
#   4. Ir al Login del app, entrar con ehernandez@sigasas.com + esa pwd
#   5. Vas a ver el botón "👥 Admin · Usuarios" en el sidebar
#   6. Desde ahí crear los demás usuarios @sigasas.com de tu equipo
#   7. Cambiar tu password después del primer login (próximo ciclo
#      agregamos UI de "cambiar mi password" propia para usuarios
#      no-admin)
#
# IMPORTANTE — SMTP no listo todavía:
#   email_confirm=True en create_user porque el SMTP de Office 365
#   no está conectado aún (eso es Ciclo 17.16). Por eso los usuarios
#   no reciben mail de confirmación, se crean directamente activos.
#   Cuando hagamos 17.16, podemos prender la confirmación si querés.
#
# IMPORTANTE — fallback legacy SIGUE FUNCIONANDO:
#   Los usuarios admin@watermelon.com y cliente@empresa.com (los
#   hardcoded del .streamlit/secrets.toml) SIGUEN PUDIENDO LOGUEARSE
#   normalmente. Esto es a propósito para no romper nada durante la
#   migración. Cuando termines de migrar, podés borrar la sección
#   [auth.users.*] del secrets.toml.
#
# Solo push a DEV. Pausamos antes de main para que pruebes con tu
# equipo (3-5 personas) y veas si la UX está bien.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.14..."
git add core/supabase_auth.py
git add core/auth.py
git add pages/00_Login.py
git add pages/_admin_users.py
git add scripts/bootstrap_admin.py
git add scripts/test_supabase_auth.py 2>/dev/null || true
git add _publish_ciclo17_14_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.14..."
    git commit -m "feat(auth): sistema de usuarios real con Supabase Auth + Admin Panel (17.14)

Reemplaza admin/demo hardcoded por usuarios reales en Supabase Auth.

NUEVO core/supabase_auth.py — wrapper Auth admin API:
- infer_role_from_email: regla dominio (admin solo ehernandez@sigasas,
  specialist todos @sigasas, client otros)
- is_admin_email: helper para guards
- is_supabase_auth_enabled: detecta config en st.secrets
- get_admin_client cached por sesión
- create_user con email_confirm=True (SMTP pendiente para 17.16)
- signin_user con errores legibles
- list_all_users, get_user_by_email
- update_user_role, update_user_full_name, reset_user_password
- block_user / unblock_user (ban_duration)
- delete_user (irreversible)
- generate_temp_password 12 chars sin chars ambiguos

MODIFICADO core/auth.py:
- login() prioriza Supabase, fallback al sistema legacy hardcoded
- get_current_user devuelve user_id, is_admin, source nuevos
- logout limpia tambien las nuevas keys
- render_user_menu agrega boton 'Admin · Usuarios' SOLO si is_admin

MODIFICADO pages/00_Login.py:
- Label 'Correo corporativo' (era 'Usuario o correo')
- Placeholder con @sigasas.com
- Help text para SIGASAS vs clientes
- autocomplete=email

NUEVO pages/_admin_users.py — Admin Panel:
- Guard: solo ehernandez@sigasas.com puede entrar
- 5 KPI cards (total, admin, specialist, client, bloqueados)
- Form de crear usuario con password auto-generada
- Busqueda por email/nombre
- Lista de cards con popovers para cambiar role, reset password,
  bloquear/desbloquear, editar nombre, eliminar (con confirm)
- Admin unico PROTEGIDO contra modificacion (anti-lockout)

NUEVO scripts/bootstrap_admin.py — setup inicial:
- Crea ehernandez@sigasas.com como admin la primera vez
- --force-reset para resetear password despues
- Imprime credenciales en terminal una sola vez
- Necesario porque sin admin no se puede entrar al Admin Panel

Fallback legacy mantenido: usuarios viejos del secrets.toml siguen
funcionando hasta migracion manual desde Admin Panel.

Solo push a DEV. Pausar antes de main para probar con equipo." || echo "  (sin cambios)"
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
echo " ✓ Ciclo 17.14 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " ► PASO INMEDIATO POST-PUSH (1 vez):"
echo "  Crear el admin único en Supabase Auth con:"
echo ""
echo "    cd ~/Documents/WatermelonSystem"
echo "    python scripts/bootstrap_admin.py"
echo ""
echo "  El script te imprime una password temporal en pantalla."
echo "  Guardala — solo se muestra UNA vez."
echo ""
echo " ► PROBAR EL LOGIN NUEVO:"
echo "  1. Andá al app (cloud o localhost)"
echo "  2. Login con ehernandez@sigasas.com + la password temporal"
echo "  3. Vas a ver un botón nuevo en sidebar: '👥 Admin · Usuarios'"
echo "  4. Click → entrás al Admin Panel"
echo "  5. Crear más usuarios @sigasas.com para tu equipo"
echo ""
echo " ► QUEDA PENDIENTE PARA SIGUIENTE CICLO (17.15):"
echo "  - Aislamiento de reportes y CSVs por owner_email"
echo "  - Repositorio de PDFs aprobados (no editables)"
echo ""
echo " ► RECOMENDACIÓN:"
echo "  Probá el flow con 2-3 personas reales de tu equipo ANTES de"
echo "  arrancar el 17.15. Si hay UX que ajustar en el Admin Panel,"
echo "  mejor descubrirlo ahora."
echo "================================================================"
