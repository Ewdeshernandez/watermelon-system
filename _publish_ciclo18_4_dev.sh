#!/bin/bash
# Ciclo 18.4 → DEV: deploy config para API REST en Render.
# Sin esto, el bot WhatsApp v1.0 no puede consumir la API.
set -e
cd "$(dirname "$0")"

PRE_TAG="pre-ciclo18-4-$(date +%Y%m%d)"
git tag -f "${PRE_TAG}"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo18-4-api-render-deploy

git add requirements-api-minimal.txt render-api.yaml
git commit -m "feat(18.4): config Render para deploy del API REST público

- requirements-api-minimal.txt: deps livianas (fastapi+uvicorn+numpy+pandas)
  sin streamlit/plotly/kaleido/reportlab. Reduce cold start del free
  tier de Render de ~3min a <30s.
- render-api.yaml: config declarativa para el web service.
  Convive sin conflicto con la app Streamlit (wm-home-final-2026).

Sin cambios en código de producción. Solo archivos nuevos.

Próximo paso (manual): user crea el servicio en Render dashboard
y agrega WATERMELON_API_URL al bot."

git push -u origin feat/ciclo18-4-api-render-deploy
git checkout dev
git merge --no-ff feat/ciclo18-4-api-render-deploy -m "Merge feat/ciclo18-4-api-render-deploy into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 18.4 en DEV"
echo "================================================================"
echo " Pasos manuales en Render dashboard:"
echo " 1. https://dashboard.render.com/ → '+ New' → Web Service"
echo " 2. Connect repo 'watermelon-system'"
echo " 3. Branch: main (después de merge a main)"
echo " 4. Build cmd: pip install -r requirements-api-minimal.txt"
echo " 5. Start cmd: uvicorn api.app:app --host 0.0.0.0 --port \$PORT"
echo " 6. Env var: WATERMELON_API_KEYS = watermelon-bot-key-2026"
echo " 7. Deploy → te da URL https://watermelon-api-XYZ.onrender.com"
echo ""
echo " Después en el bot (Render → watermelon-whatsapp-bot → Env):"
echo " - WATERMELON_API_URL = https://watermelon-api-XYZ.onrender.com"
echo " - WATERMELON_API_KEY = watermelon-bot-key-2026"
echo " - Save → bot redeploya solo"
echo ""
echo " Después /rodamiento, /plantillas y /estado funcionan completos."
echo " Si OK → bash _publish_v3_18_0_to_main.sh"
echo "================================================================"
