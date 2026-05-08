"""
api.app
=======

Factory FastAPI para el servicio público read-only de Watermelon System.

Para correr localmente:

    pip install -r requirements-api.txt
    export WATERMELON_API_KEYS="dev-secret-key-1,dev-secret-key-2"
    uvicorn api.app:app --host 0.0.0.0 --port 8000

Una vez levantado:
    GET  http://localhost:8000/v1/health
    GET  http://localhost:8000/docs            (Swagger UI auto)
    GET  http://localhost:8000/redoc           (Redoc UI auto)

Todos los endpoints (excepto health y docs) requieren cabecera:
    Authorization: Bearer <API_KEY>
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

# Imports diferidos: si FastAPI no está instalado, este módulo lanza
# ImportError sólo cuando alguien lo IMPORTA (no por la mera presencia
# de api/app.py). Esto permite que api/services.py y api/auth.py
# permanezcan testeables sin instalar FastAPI.
try:
    from fastapi import Body, Depends, FastAPI, HTTPException, Header, Query, Response
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    Body = None  # type: ignore
    Depends = None  # type: ignore
    HTTPException = None  # type: ignore
    Header = None  # type: ignore
    Query = None  # type: ignore
    Response = None  # type: ignore
    JSONResponse = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore


from api.auth import is_valid_api_key, hash_for_log
from api import services


log = logging.getLogger("watermelon.api")


# =============================================================================
# Pydantic models — definidos a nivel módulo para que FastAPI pueda resolver
# los type hints en `payload: LiveIngestPayload = Body(...)`.
# =============================================================================

if _FASTAPI_AVAILABLE:

    class LiveReadingItem(BaseModel):
        """Una lectura individual enviada por el collector."""
        variable: str = Field(..., description="Nombre de la variable, ej. '1YV VEL CRF'")
        metric: str = Field(..., description="Direct | Gap | BiasVoltage | 1X_Ampl | 1X_Phase | 2X_Ampl | 2X_Phase")
        value: Optional[float] = Field(None, description="Valor numérico; null si fallo de comm")
        unit: Optional[str] = Field(None, description="Unidad nativa, ej. 'in/s pk', 'mil pp', 'V DC', 'deg', 'rpm'")
        sensor_label: Optional[str] = Field(None, description="Label industrial del sensor del WM Sensor Map, ej. '1Y_V'")
        register: Optional[int] = Field(None, description="Modbus register address (debug)")
        quality: Optional[str] = Field("good", description="good | stale | overrange | comm_fail")
        captured_at: Optional[str] = Field(None, description="ISO8601; si null usa el del payload base")
        metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class LiveIngestPayload(BaseModel):
        """Payload completo: 1 batch del collector."""
        instance_id: str = Field(..., description="ID del activo en WM, ej. 'tes1'")
        captured_at: str = Field(..., description="Timestamp ISO8601 base del batch")
        readings: List[LiveReadingItem] = Field(..., min_length=1, max_length=200)
        metadata: Optional[Dict[str, Any]] = Field(
            default_factory=dict,
            description="Info del collector: version, host, modbus byte order, etc.",
        )

else:
    # Stubs para que el módulo importe sin FastAPI (tests)
    LiveReadingItem = object  # type: ignore
    LiveIngestPayload = object  # type: ignore


def _require_fastapi():
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI no está instalado. Ejecutá: pip install -r requirements-api.txt"
        )


def _api_key_dependency(authorization: Optional[str] = (Header(default=None, alias="Authorization") if _FASTAPI_AVAILABLE else None)):
    """
    Validador de API key. Acepta:
      - Authorization: Bearer <key>
      - Authorization: <key>           (sin prefijo, también válido)
    """
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not is_valid_api_key(token):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return hash_for_log(token)


def _scope_dependency(
    authorization: Optional[str] = (Header(default=None, alias="Authorization") if _FASTAPI_AVAILABLE else None),
    x_caller_phone: Optional[str] = (Header(default=None, alias="X-Caller-Phone") if _FASTAPI_AVAILABLE else None),
):
    """
    Resuelve el scope del caller para multi-tenant ACL (Ciclo 20A).

    Reglas:
      1. La key Authorization debe ser válida (Bearer admin key o
         api_key de un cliente registrado).
      2. Si el caller resuelve a admin (típicamente el bot WhatsApp)
         Y vino X-Caller-Phone, se hace una segunda resolución:
            - Si phone matchea registry → ese scope manda
            - Si phone NO matchea → 403 (rechaza phone desconocido)
      3. Si el caller resuelve a client por api_key, X-Caller-Phone
         se ignora (no hay escalación de privilegios cross-cliente).
      4. Caller unknown → 401.
    """
    from core.clients import resolve_by_api_key, resolve_by_phone

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization.strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not is_valid_api_key(token):
        # Antes de rechazar, ¿coincide con api_key de algún cliente?
        scope_by_key = resolve_by_api_key(token, admin_keys=[])
        if scope_by_key.is_authorized:
            return scope_by_key
        raise HTTPException(status_code=401, detail="Invalid API key")

    scope_by_key = resolve_by_api_key(token)
    # Si la key es admin y vino phone → resolver scope real por phone
    if scope_by_key.role == "admin" and x_caller_phone:
        scope_by_phone = resolve_by_phone(x_caller_phone)
        if not scope_by_phone.is_authorized:
            raise HTTPException(
                status_code=403,
                detail=f"Phone {x_caller_phone} not registered in clients.json",
            )
        return scope_by_phone

    return scope_by_key


def create_app() -> "FastAPI":
    """
    Factory de la app FastAPI. Los routers se registran aquí, lo cual
    facilita que tests instancien fresh apps sin globals colgados.
    """
    _require_fastapi()

    app = FastAPI(
        title="Watermelon System API",
        version="1.0.0",
        description=(
            "API REST pública read-only de Watermelon System. "
            "Hardware-agnostic: lista activos, plantillas de máquinas, "
            "normas ISO/API y catálogo de rodamientos para integradores "
            "y clientes que vienen de System1, AMS, @ptitude y otros."
        ),
        contact={"name": "SIGA Group SAS"},
        license_info={"name": "Proprietary"},
    )

    # =========================================================
    # Health (público)
    # =========================================================
    @app.get("/v1/health", tags=["health"], summary="Estado del servicio")
    def health():
        return services.get_health()

    # =========================================================
    # Templates (autenticado)
    # =========================================================
    @app.get("/v1/templates", tags=["templates"], summary="Lista plantillas de máquinas")
    def list_templates_endpoint(
        category: Optional[str] = Query(default=None),
        manufacturer: Optional[str] = Query(default=None),
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        return {
            "items": services.list_machine_templates_summary(
                category=category, manufacturer=manufacturer
            )
        }

    @app.get("/v1/templates/{template_id}", tags=["templates"],
             summary="Detalle de una plantilla")
    def template_detail(
        template_id: str,
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        out = services.get_machine_template_detail(template_id)
        if out is None:
            raise HTTPException(status_code=404, detail="Template not found")
        return out

    @app.get("/v1/templates/{template_id}/norm-recommendation", tags=["templates"],
             summary="Norma + clase ISO recomendada para una plantilla")
    def template_norm_reco(
        template_id: str,
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        return services.get_norm_recommendation_for_template(template_id)

    @app.get("/v1/templates/categories", tags=["templates"],
             summary="Categorías disponibles")
    def template_categories(api_key_hash: str = Depends(_api_key_dependency)):
        return {"items": services.list_template_categories()}

    # =========================================================
    # Norms (autenticado)
    # =========================================================
    @app.get("/v1/norms", tags=["norms"], summary="Lista normas registradas")
    def list_norms_endpoint(api_key_hash: str = Depends(_api_key_dependency)):
        return {"items": services.list_norms_summary()}

    @app.get("/v1/norms/groups", tags=["norms"], summary="Normas agrupadas por familia")
    def list_norm_groups_endpoint(api_key_hash: str = Depends(_api_key_dependency)):
        return services.list_norm_groups_summary()

    @app.get("/v1/norms/{norm_code}", tags=["norms"], summary="Detalle de norma")
    def norm_detail(
        norm_code: str,
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        out = services.get_norm_detail(norm_code)
        if out is None:
            raise HTTPException(status_code=404, detail="Norm not found")
        return out

    @app.get("/v1/norms/{norm_code}/classes/{class_code}/thresholds", tags=["norms"],
             summary="Thresholds de una clase específica")
    def norm_class_thresholds(
        norm_code: str,
        class_code: str,
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        out = services.get_norm_class_thresholds(norm_code, class_code)
        if out is None:
            raise HTTPException(status_code=404, detail="Norm/class not found")
        return out

    # =========================================================
    # Loaders (autenticado, advertising)
    # =========================================================
    @app.get("/v1/loaders", tags=["capabilities"],
             summary="Formatos de archivo soportados por el sistema")
    def supported_loaders(api_key_hash: str = Depends(_api_key_dependency)):
        return {"items": services.list_supported_loaders()}

    # =========================================================
    # Bearings (autenticado)
    # =========================================================
    @app.get("/v1/bearings", tags=["bearings"], summary="Catálogo público de rodamientos")
    def bearings_list(
        limit: int = Query(default=200, ge=1, le=2000),
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        return {"items": services.list_bearings_summary(limit=limit)}

    @app.get("/v1/bearings/overlay", tags=["bearings"],
             summary="Cálculo de frecuencias de falla para un rodamiento")
    def bearings_overlay(
        model: str = Query(..., description="Modelo de rodamiento (p.ej. 'SKF 6319')"),
        rpm: float = Query(..., gt=0),
        harmonics: int = Query(default=3, ge=1, le=10),
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        return services.get_bearing_overlay(model=model, rpm=rpm, harmonics=harmonics)

    # =========================================================
    # Assets & Reports (Ciclo 19 + 20A multi-tenant ACL)
    # =========================================================
    @app.get("/v1/assets", tags=["assets"],
             summary="Lista activos con reportes archivados (filtra por scope)")
    def assets_list(
        limit: int = Query(default=200, ge=1, le=1000),
        scope=Depends(_scope_dependency),
    ):
        return {
            "scope": scope.as_dict(),
            "items": services.list_archived_assets(limit=limit, scope=scope),
        }

    @app.get("/v1/assets/{asset}/reports", tags=["assets"],
             summary="Reportes archivados de un activo (con ACL)")
    def asset_reports(
        asset: str,
        limit: int = Query(default=50, ge=1, le=500),
        scope=Depends(_scope_dependency),
    ):
        return {
            "scope": scope.as_dict(),
            "items": services.list_reports_for_asset(asset=asset, limit=limit, scope=scope),
        }

    @app.get("/v1/assets/{asset}/reports/latest/pdf", tags=["assets"],
             summary="PDF binario del último reporte (ACL aplicado)",
             response_class=Response if _FASTAPI_AVAILABLE else None)
    def asset_latest_report_pdf(
        asset: str,
        scope=Depends(_scope_dependency),
    ):
        out = services.get_latest_report_pdf(asset, scope=scope)
        if out is None:
            raise HTTPException(status_code=404, detail="No reports found for asset")
        filename = out["filename"]
        return Response(
            content=out["pdf_bytes"],
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-Archive-Id": out["archive_id"],
            },
        )

    @app.get("/v1/reports/{archive_id:path}/pdf", tags=["assets"],
             summary="PDF binario por archive_id directo",
             response_class=Response if _FASTAPI_AVAILABLE else None)
    def report_by_id_pdf(
        archive_id: str,
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        out = services.get_report_pdf_by_id(archive_id)
        if out is None:
            raise HTTPException(status_code=404, detail="Report not found")
        filename = out["filename"]
        return Response(
            content=out["pdf_bytes"],
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    # =========================================================
    # Live Ingestion (Tier 0 A — Ciclo 23.1)
    # ---------------------------------------------------------
    # Endpoint POST /v1/ingest/live: recibe lecturas en tiempo real
    # desde wm-collector (Bently 3500/92 Modbus) e inserta en
    # Supabase tabla live_readings. Auth: Bearer API key.
    # =========================================================
    @app.post("/v1/ingest/live", tags=["ingest"],
              summary="Ingesta de lecturas en tiempo real (Tier 0 A)")
    def ingest_live(
        payload: LiveIngestPayload = Body(...),
        api_key_hash: str = Depends(_api_key_dependency),
    ):
        from core.live_readings import LiveReading, ingest_batch
        from datetime import datetime as _dt
        from datetime import timezone as _tz

        if not payload.readings:
            return {"ingested": 0, "instance_id": payload.instance_id}

        # captured_at base — si una reading no trae el suyo, usamos el del payload
        base_time_str = payload.captured_at
        try:
            base_time = _dt.fromisoformat(base_time_str.replace("Z", "+00:00"))
        except Exception:
            base_time = _dt.now(tz=_tz.utc)

        rows: List[LiveReading] = []
        for r in payload.readings:
            try:
                rt = (
                    _dt.fromisoformat(r.captured_at.replace("Z", "+00:00"))
                    if r.captured_at else base_time
                )
            except Exception:
                rt = base_time
            rows.append(LiveReading(
                instance_id=payload.instance_id,
                sensor_label=r.sensor_label,
                variable=r.variable,
                metric=r.metric,
                value=r.value,
                unit=r.unit,
                captured_at=rt,
                register=r.register,
                quality=r.quality or "good",
                metadata=r.metadata or {},
            ))

        n = ingest_batch(rows)
        log.info(
            "ingest_live: instance=%s sent=%d inserted=%d caller_hash=%s",
            payload.instance_id, len(rows), n, api_key_hash,
        )
        return {
            "ingested": n,
            "instance_id": payload.instance_id,
            "received": len(rows),
        }

    # =========================================================
    # Manejo global de errores
    # =========================================================
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request, exc):
        log.exception("Unhandled exception in %s %s", request.method, request.url)
        return JSONResponse(
            status_code=500,
            content={"detail": "internal_server_error"},
        )

    return app


# Instancia top-level para `uvicorn api.app:app`. Sólo se construye si
# FastAPI está instalado.
if _FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None  # type: ignore


__all__ = ["create_app", "app"]
