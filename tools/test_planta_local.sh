#!/bin/bash
# ============================================================================
# tools/test_planta_local.sh — Test Watermelon Planta en el Mac
# ============================================================================
#
# Levanta Streamlit local con planta/app_planta.py para probar la app
# completa ANTES de pushear a GitHub Actions.
#
# Hace:
#   1. Verifica que existe tools/.keys/private_key.pem (sino sale)
#   2. Verifica que existe una licencia test válida o la emite
#   3. Copia la licencia a planta/data/license.token
#   4. Arranca Streamlit apuntando a planta/app_planta.py
#   5. Abre el browser default en http://localhost:8501
#   6. Al cerrar la app (Ctrl+C), restaura el estado original
#
# Uso:
#   bash tools/test_planta_local.sh
#
# Para emitir una licencia test custom:
#   bash tools/test_planta_local.sh "Mi Empresa Test"
# ============================================================================

set -e

# Detectar el directorio del repo (parent de tools/)
REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$REPO_ROOT"

# Colores para el output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  WATERMELON PLANTA — Test Local en Mac${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

# ----------------------------------------------------------------------------
# 1. Verificar dependencias críticas
# ----------------------------------------------------------------------------
echo -e "${YELLOW}[1/5]${NC} Verificando dependencias..."

if [ ! -f "tools/.keys/private_key.pem" ]; then
    echo -e "${RED}ERROR:${NC} No existe tools/.keys/private_key.pem"
    echo ""
    echo "Genera las keys RSA primero con:"
    echo "  python tools/license_keygen.py"
    exit 1
fi

if ! command -v streamlit &> /dev/null; then
    echo -e "${RED}ERROR:${NC} streamlit no está instalado en este Python"
    echo ""
    echo "Instala con: pip install streamlit"
    exit 1
fi

if ! python -c "import jwt" 2>/dev/null; then
    echo -e "${RED}ERROR:${NC} PyJWT no está instalado"
    echo ""
    echo "Instala con: pip install pyjwt cryptography"
    exit 1
fi

echo -e "${GREEN}  ✓${NC} private_key.pem OK"
echo -e "${GREEN}  ✓${NC} streamlit OK"
echo -e "${GREEN}  ✓${NC} pyjwt OK"
echo ""

# ----------------------------------------------------------------------------
# 2. Emitir o reutilizar licencia test
# ----------------------------------------------------------------------------
CUSTOMER="${1:-SIGA Mac Test Lab}"
SLUG=$(echo "$CUSTOMER" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9' '-' | sed 's/^-*//;s/-*$//')
LICENSE_DIR="tools/licenses_issued/$SLUG"
LICENSE_FILE="$LICENSE_DIR/license.token"

echo -e "${YELLOW}[2/5]${NC} Preparando licencia test..."
echo "      Cliente: $CUSTOMER"

if [ -f "$LICENSE_FILE" ]; then
    echo -e "${GREEN}  ✓${NC} Reutilizando licencia existente: $LICENSE_FILE"
else
    echo "      Emitiendo licencia nueva (válida 1 año)..."
    EXPIRES=$(date -v+1y +"%Y-%m-%d" 2>/dev/null || date -d "+1 year" +"%Y-%m-%d")
    python tools/license_issue.py \
        --customer "$CUSTOMER" \
        --email "test@sigasas.com" \
        --plan enterprise \
        --expires "$EXPIRES" \
        > /tmp/license_issue_output.log 2>&1
    if [ ! -f "$LICENSE_FILE" ]; then
        echo -e "${RED}ERROR:${NC} No se pudo emitir la licencia"
        cat /tmp/license_issue_output.log
        exit 1
    fi
    echo -e "${GREEN}  ✓${NC} Licencia emitida: $LICENSE_FILE"
fi
echo ""

# ----------------------------------------------------------------------------
# 3. Copiar licencia a planta/data/
# ----------------------------------------------------------------------------
echo -e "${YELLOW}[3/5]${NC} Instalando licencia en planta/data/..."
mkdir -p planta/data
cp "$LICENSE_FILE" planta/data/license.token
echo -e "${GREEN}  ✓${NC} Copiada a planta/data/license.token"
echo ""

# ----------------------------------------------------------------------------
# 4. Limpiar cache de revocación + update (para test limpio)
# ----------------------------------------------------------------------------
echo -e "${YELLOW}[4/5]${NC} Limpiando cache local (para test fresco)..."
rm -f planta/data/.revocation_cache.json
rm -f planta/data/.update_check_cache.json
echo -e "${GREEN}  ✓${NC} Cache borrado"
echo ""

# ----------------------------------------------------------------------------
# 5. Arrancar Streamlit
# ----------------------------------------------------------------------------
echo -e "${YELLOW}[5/5]${NC} Arrancando Streamlit..."
echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  Watermelon Planta corriendo${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""
echo "  URL:           http://localhost:8501"
echo "  Cliente test:  $CUSTOMER"
echo "  Para cerrar:   Ctrl+C en esta terminal"
echo ""
echo -e "${YELLOW}Lo que tenés que ver al abrir el browser:${NC}"
echo "  ✓ Chip verde arriba con el nombre del cliente"
echo "  ✓ Sidebar industrial con logo, estado del sistema"
echo "  ✓ Botones de Acceso rápido EMA + OMA"
echo "  ✓ Header con gradient navy → teal"
echo ""
echo -e "${YELLOW}Streamlit logs abajo:${NC}"
echo ""

cd planta

# Abrir el browser automáticamente en 3 segundos (en background)
(sleep 3 && open "http://localhost:8501") &

exec streamlit run app_planta.py \
    --server.port 8501 \
    --server.headless true \
    --browser.gatherUsageStats false \
    --server.runOnSave false
