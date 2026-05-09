#!/bin/bash
# Ciclo 19 → DEV: WhatsApp Asset Query.
# - api/services.py: list_archived_assets, list_reports_for_asset,
#   get_latest_report_pdf, get_report_pdf_by_id
# - api/app.py: 4 endpoints nuevos /v1/assets, /v1/assets/{a}/reports,
#   /v1/assets/{a}/reports/latest/pdf, /v1/reports/{id}/pdf
set -e
cd "$(dirname "$0")"

git checkout dev
git pull origin dev --ff-only
git checkout -B feat/ciclo19-asset-query

git add api/services.py api/app.py
git commit -m "feat(19): API endpoints para WhatsApp Asset Query

Nuevos endpoints (read-only autenticados):
  GET /v1/assets — lista activos del archivo (deduplicados).
  GET /v1/assets/{asset}/reports — reportes archivados del activo.
  GET /v1/assets/{asset}/reports/latest/pdf — PDF binario del último.
  GET /v1/reports/{archive_id}/pdf — PDF por archive_id directo.

Capa servicios pura en api/services.py (testeable sin FastAPI).
Usa core.reports_archive con viewer=admin global.

Diseñado para que el bot WhatsApp v1.1 pueda servir PDFs por chat:
  '/reporte TES1' → bot llama API → recibe PDF → lo manda a WA.

Sin cambios en core/ ni pages/."

git push -u origin feat/ciclo19-asset-query
git checkout dev
git merge --no-ff feat/ciclo19-asset-query -m "Merge feat/ciclo19-asset-query into dev"
git push origin dev

echo ""
echo "================================================================"
echo " ✓ Ciclo 19 (Watermelon API) en DEV"
echo " Después: bash _publish_v3_19_0_to_main.sh"
echo "================================================================"
