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
    from fastapi import Depends, FastAPI, HTTPException, Header, Query
    from fastapi.responses import JSONResponse
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore
    Depends = None  # type: ignore
    HTTPException = None  # type: ignore
    Header = None  # type: ignore
    Query = None  # type: ignore
    JSONResponse = None  # type: ignore


from api.auth import is_valid_api_key, hash_for_log
from api import services


log = logging.getLogger("watermelon.api")


def _require_fastapi():
    if not _FASTAPI_AVAILABLE:
        raise RuntimeError(
            "FastAPI no está instalado. Ejecutá: pip install -r requirements-api.txt"
        )


def _api_key_dependency(authorization: Optional[str] = None):
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
