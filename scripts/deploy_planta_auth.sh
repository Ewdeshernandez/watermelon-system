#!/usr/bin/env bash
# =============================================================
# deploy_planta_auth.sh — Despliega la Edge Function planta-auth
# =============================================================
# Setea los secrets de la función (GRAPH_* + OTP_SIGNING_SECRET) leyéndolos
# de .streamlit/secrets.toml (así el client_secret NUNCA se copia a mano) y
# despliega la función con --no-verify-jwt.
#
# SUPABASE_URL / SUPABASE_ANON_KEY / SUPABASE_SERVICE_ROLE_KEY los inyecta
# Supabase automáticamente en la función — NO hay que setearlos.
#
# Requisitos:
#   - Supabase CLI instalado y logueado (supabase login)
#   - .streamlit/secrets.toml con la sección [email.graph]
#
# Uso:
#   bash scripts/deploy_planta_auth.sh
# =============================================================
set -euo pipefail
PROJECT_REF="yxeqwkhybueelmkrdkgq"
cd "$(dirname "$0")/.."

if ! command -v supabase >/dev/null 2>&1; then
  echo "ERROR: falta el Supabase CLI. Instalá con: brew install supabase/tap/supabase"
  exit 1
fi

TMP_ENV="$(mktemp)"
trap 'rm -f "$TMP_ENV"' EXIT

# En el Mac de Ewdes el intérprete correcto es `python` (conda 3.13, trae tomllib).
python - "$TMP_ENV" <<'PY'
import sys, secrets, pathlib
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # fallback py<3.11
out = sys.argv[1]
p = pathlib.Path(".streamlit/secrets.toml")
if not p.exists():
    sys.exit("No encontré .streamlit/secrets.toml")
data = tomllib.loads(p.read_text())
g = (data.get("email", {}) or {}).get("graph", {}) or {}
required = ["tenant_id", "client_id", "client_secret", "from_email"]
missing = [k for k in required if not g.get(k)]
if missing:
    sys.exit(f"Faltan claves en [email.graph]: {missing}")
lines = [
    f"GRAPH_TENANT_ID={g['tenant_id']}",
    f"GRAPH_CLIENT_ID={g['client_id']}",
    f"GRAPH_CLIENT_SECRET={g['client_secret']}",
    f"GRAPH_FROM_EMAIL={g['from_email']}",
    f"GRAPH_FROM_NAME={g.get('from_name', 'Watermelon System')}",
    f"OTP_SIGNING_SECRET={secrets.token_hex(32)}",
]
pathlib.Path(out).write_text("\n".join(lines) + "\n")
print("→ Secrets preparados (GRAPH_* + OTP_SIGNING_SECRET aleatorio).")
PY

echo "→ Seteando secrets en la Edge Function..."
supabase secrets set --env-file "$TMP_ENV" --project-ref "$PROJECT_REF"

echo "→ Desplegando planta-auth (--no-verify-jwt)..."
supabase functions deploy planta-auth --no-verify-jwt --project-ref "$PROJECT_REF"

echo ""
echo "✅ Listo: planta-auth desplegada y secrets cargados."
echo "   Probá:  curl -s -X POST \\"
echo "     \"https://${PROJECT_REF}.supabase.co/functions/v1/planta-auth\" \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"action\":\"request\",\"email\":\"macevedo@sigasas.com\"}'"
