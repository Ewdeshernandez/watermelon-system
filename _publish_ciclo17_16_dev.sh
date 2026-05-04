#!/bin/bash
# =============================================================
# Watermelon — Ciclo 17.16 → DEV: Recovery password +
#                                  Restricciones cliente +
#                                  SMTP Microsoft 365
# =============================================================
# Cierra la primera lista de necesidades del usuario:
# items 2 (recovery email) + 6 (vista cliente bloqueada).
#
# Lo nuevo:
#
#   ► RECOVERY DE PASSWORD POR EMAIL
#     - Login: link "¿Olvidaste tu contraseña?" → form que pide email
#     - request_reset(): genera token UUID + TTL 1 hora + envía email
#       vía SMTP Office 365 o Microsoft Graph (configurable via secrets)
#     - Por seguridad NO revela si el email existe (anti-enumeration)
#     - El usuario recibe email con link a /reset_password?token=xxx
#     - pages/_reset_password.py: form para nueva password
#     - Token one-shot use, se invalida al consumir
#
#   ► RESTRICCIONES DEL ROLE CLIENT
#     - NAV_ITEMS filtrado en sidebar: client NO ve Machinery Library,
#       Load Data, Diagnostics, Machine Map
#     - Cada página restringida tiene guard require_role() al inicio
#       (defensa en profundidad si entran por URL directa)
#     - pages/16_Reports.py para client: redirect a vista limitada
#       que solo muestra el archivo histórico filtrado (ACL del 17.15
#       garantiza que solo ve los marcados shared_with_client)
#     - Hero negro "🔐 Acceso de cliente — Solo lectura"
#
#   ► CAMBIAR MI PASSWORD
#     - Botón en sidebar "🔑 Cambiar mi password" para usuarios
#       Supabase Auth (popover con form: nueva + confirmar)
#     - No requiere admin, usa supabase_auth.reset_user_password con
#       el user_id de la sesión activa
#
#   ► EMAIL BACKEND DUAL
#     - core/email_sender.py soporta dos backends:
#         a) SMTP con AUTH (Office 365, Gmail App Password)
#            Config: [email.smtp] host/port/starttls/username/password/from_email
#         b) Microsoft Graph API (OAuth client_credentials)
#            Config: [email.graph] tenant_id/client_id/client_secret/from_email
#     - get_email_backend_status() para diagnóstico
#     - Templates HTML branded para reset password
#
# Cambios técnicos:
#
# (NUEVO) core/email_sender.py
#   - send_email(to, subject, text, html, attachments)
#   - send_password_reset_email(email, token_url, full_name, ttl_minutes)
#   - send_briefing_email(to, pdf_bytes, date_str, full_name)
#   - get_email_backend_status() para admin diagnostic
#   - Detección automática SMTP vs Graph según secrets
#
# (NUEVO) core/password_reset.py
#   - request_reset(email, base_url, ttl_minutes=60)
#   - validate_token(token) → {valid, email, expired, consumed}
#   - consume_token(token, new_password) → cambia pwd + invalida
#   - cleanup_expired_tokens(max_age_days=7)
#   - Storage: data/password_reset_tokens/{token}.json
#
# (NUEVO) pages/_reset_password.py
#   - Landing del email link
#   - Lee ?token=xxx de la URL
#   - Valida + form de nueva password + consume
#   - UI consistente con el login
#
# MODIFICADO core/auth.py:
#   - CLIENT_BLOCKED_PAGES + is_page_allowed_for_role()
#   - require_role(allowed_roles) helper para guards
#   - render_user_menu filtra NAV_ITEMS según role
#   - Botón "🔑 Cambiar mi password" en sidebar (si auth_source=supabase)
#
# MODIFICADO pages/00_Login.py:
#   - Expander "¿Olvidaste tu contraseña?" debajo del form
#   - Form de email + llamada a request_reset
#
# MODIFICADO pages/16_Reports.py:
#   - Si role=client: render vista limitada con archivo + st.stop()
#   - Solo ve PDFs marcados shared_with_client
#
# MODIFICADO pages/00_Machinery_Library.py, 01_Load_Data.py,
#            15_Diagnostics.py, 01b_Machine_Map.py:
#   - require_role(("admin", "specialist")) al inicio
#
# IMPORTANTE — NO funciona end-to-end sin SMTP configurado.
# Después de mergear, hay que agregar a Streamlit Cloud Secrets:
#
#   [email.smtp]
#   host = "smtp.office365.com"
#   port = 587
#   starttls = true
#   username = "noreply@sigasas.com"   # mailbox dedicado
#   password = "<app password>"
#   from_email = "noreply@sigasas.com"
#   from_name = "Watermelon System"
#
# Si el tenant Office 365 no permite SMTP AUTH (común en Microsoft 365
# moderno), usar el backend Graph API:
#
#   [email.graph]
#   tenant_id = "xxxxx"
#   client_id = "xxxxx"
#   client_secret = "xxxxx"
#   from_email = "noreply@sigasas.com"
#   from_name = "Watermelon System"
#
# Eso requiere registrar una App en Azure AD con permiso Mail.Send
# (Application permission, con admin consent).
#
# Solo push a DEV. Pausar antes de main hasta validar el flow de
# email end-to-end.
# =============================================================

set -e
cd "$(dirname "$0")"

[ -f .git/index.lock ] && rm -f .git/index.lock
[ -f .git/HEAD.lock ] && rm -f .git/HEAD.lock

CURRENT=$(git branch --show-current)
[ "$CURRENT" != "dev" ] && git checkout dev

echo ""
echo "▶ Stageando 17.16..."
git add core/email_sender.py
git add core/password_reset.py
git add core/auth.py
git add pages/00_Login.py
git add pages/_reset_password.py
git add pages/16_Reports.py
git add pages/00_Machinery_Library.py
git add pages/01_Load_Data.py
git add pages/15_Diagnostics.py
git add pages/01b_Machine_Map.py
git add _publish_ciclo17_16_dev.sh 2>/dev/null || true

if ! git diff --staged --quiet; then
    echo "▶ Commiteando 17.16..."
    git commit -m "feat(auth): recovery password + restricciones cliente + SMTP/Graph (17.16)

NUEVO core/email_sender.py — backend dual:
- SMTP (Office 365, Gmail App Password) si secrets [email.smtp]
- Microsoft Graph API si secrets [email.graph]
- send_email/send_password_reset_email/send_briefing_email
- get_email_backend_status para diagnostico

NUEVO core/password_reset.py — tokens TTL 1h:
- request_reset(email): genera UUID + envia email + anti-enumeration
- validate_token / consume_token (one-shot)
- Storage data/password_reset_tokens/

NUEVO pages/_reset_password.py:
- Landing del link de email (?token=xxx)
- Form nueva password + consume

MODIFICADO core/auth.py:
- CLIENT_BLOCKED_PAGES + is_page_allowed_for_role + require_role
- NAV_ITEMS filtrado en render_user_menu segun role
- Boton 'Cambiar mi password' en sidebar (popover)

MODIFICADO pages/00_Login.py:
- Expander 'Olvidaste tu contrasena' con form de email

MODIFICADO pages/16_Reports.py:
- Si role=client: vista solo-lectura del archivo + st.stop()

MODIFICADO pages/00_Machinery_Library/01_Load_Data/15_Diagnostics/
            01b_Machine_Map.py:
- require_role(admin, specialist) guards al inicio

NECESITA configurar SMTP de Office 365 en Streamlit Cloud Secrets
para que el flujo end-to-end funcione. Doc en el publish script.

Solo push DEV. Pausar antes main hasta validar email end-to-end." || echo "  (sin cambios)"
else
    echo "  (no hay cambios)"
fi

echo ""
echo "▶ Push dev..."
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 17.16 pusheado a DEV (main intacto)"
echo "================================================================"
echo ""
echo " ► CONFIGURACIÓN PENDIENTE — SMTP de Microsoft 365:"
echo ""
echo "   Para que el reset password funcione, agregar en Streamlit Cloud"
echo "   Secrets (las dos apps: wm-home-final-2026 + wm-test):"
echo ""
echo "   [email.smtp]"
echo "   host = \"smtp.office365.com\""
echo "   port = 587"
echo "   starttls = true"
echo "   username = \"noreply@sigasas.com\""
echo "   password = \"<app password del mailbox>\""
echo "   from_email = \"noreply@sigasas.com\""
echo "   from_name = \"Watermelon System\""
echo ""
echo "   Si el tenant Office 365 tiene SMTP AUTH desactivado (común en"
echo "   tenants modernos), usar Microsoft Graph API en su lugar:"
echo ""
echo "   [email.graph]"
echo "   tenant_id = \"<tenant id de Azure>\""
echo "   client_id = \"<app id de Azure>\""
echo "   client_secret = \"<secret de la app>\""
echo "   from_email = \"noreply@sigasas.com\""
echo "   from_name = \"Watermelon System\""
echo ""
echo " ► PASOS DE VERIFICACIÓN tras configurar SMTP:"
echo "  1. En wm-test.streamlit.app, ir a Login"
echo "  2. Click '¿Olvidaste tu contraseña?' → ingresar tu email"
echo "  3. Mirá tu inbox: debería llegar un mail con link de reset"
echo "  4. Click en el link → te lleva a /reset_password con token"
echo "  5. Elegí nueva password → confirma"
echo "  6. Probá login con la nueva password"
echo ""
echo "  Si todo OK → mergeamos a main como v3.4.0"
echo "================================================================"
