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
    from fastapi import Body, Depends, FastAPI, HTTPException, Header, Query, Request, Response
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
    Request = None  # type: ignore
    JSONResponse = None  # type: ignore
    BaseModel = object  # type: ignore
    Field = lambda *a, **k: None  # type: ignore


from api.auth import is_valid_api_key, hash_for_log
from api import services


log = logging.getLogger("watermelon.api")


class PublicQuote(BaseModel):
    """Formulario 'Cotice en 1 minuto' de la web de ventas (sin llave)."""
    company: str = Field("", max_length=160)
    name: str = Field("", max_length=120)
    email: str = Field(..., max_length=160)
    phone: str = Field("", max_length=40)
    machine_types: str = Field("", max_length=200)
    machines: int = Field(1, ge=1, le=60)
    plan: str = Field("profesional", max_length=20)
    entity: str = Field("siga_co", max_length=10)
    pilot: bool = False                           # quiere piloto de 30 días en 1 máquina
    lang: str = Field("es", max_length=2)
    website: str = Field("", max_length=200)      # trampa para bots: debe venir vacío


_QUOTE_HITS: dict = {}


def _quote_rate_ok(ip: str, limit: int = 6, window_s: int = 3600) -> bool:
    import time as _t
    now = _t.time()
    hits = [h for h in _QUOTE_HITS.get(ip, []) if now - h < window_s]
    if len(hits) >= limit:
        _QUOTE_HITS[ip] = hits
        return False
    hits.append(now)
    _QUOTE_HITS[ip] = hits
    return True


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

    class CollectorAck(BaseModel):
        """Confirmación del nodo: versión de config de máquina aplicada/rechazada."""
        instance_id: str
        version: int
        status: str = Field("applied", description="applied | rejected")
        detail: Optional[str] = Field(None, description="motivo / resumen (máx 300 chars)")

else:
    # Stubs para que el módulo importe sin FastAPI (tests)
    LiveReadingItem = object  # type: ignore
    LiveIngestPayload = object  # type: ignore
    CollectorAck = object  # type: ignore


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

    # Web de ventas (watermelonsystem.com) llama a /v1/public/* desde el navegador.
    from fastapi.middleware.cors import CORSMiddleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://watermelonsystem.com", "https://www.watermelonsystem.com",
                       "http://localhost:3000", "http://localhost:3300"],
        allow_origin_regex=r"https://watermelon-web[a-z0-9-]*\.vercel\.app",
        allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
    )

    # =========================================================
    # Web de ventas — planes y cotización (PÚBLICO, sin llave)
    # =========================================================
    @app.get("/v1/public/plans", tags=["public"], summary="Planes y descuentos vigentes")
    def public_plans():
        from core.billing import pricing
        return {
            "currency": "COP",
            "plans": [{"key": p.key, "name": p.name, "price_cop": p.price_cop,
                       "includes": list(p.includes), "excludes": list(p.excludes)}
                      for p in pricing.PLANS.values()],
            "volume_tiers": [{"from": n, "discount_pct": round(d * 100)}
                             for n, d in sorted(pricing.VOLUME_TIERS)],
            "usd_discount_pct": round(pricing.USD_DISCOUNT * 100),
            "annual_months_charged": pricing.ANNUAL_MONTHS_CHARGED,
        }

    _SHOWCASE: dict = {}

    @app.get("/v1/public/showcase/train", tags=["public"],
             summary="Tren en vivo de la máquina de muestra (web de ventas)")
    def public_showcase_train():
        """Tren de Machines EN VIVO de la máquina elegida en Administración
        (wm_billing_settings.showcase_instance). Sin tag ni cliente en el dibujo:
        la web lo rotula 'Turbogenerador'. Caché 60 s."""
        import time as _t
        hit = _SHOWCASE.get("v")
        if hit and _t.time() - hit[0] < 60:
            return hit[1]
        from core.billing.store import get_setting
        iid = str(get_setting("showcase_instance", "") or "").strip()
        if not iid:
            raise HTTPException(status_code=404, detail="sin máquina de muestra")
        from core.instance_state import get_instance
        from core.live_readings import _get_supabase_client, latest_for_instance
        from core.live_report_builder import _build_sensor_lookup, pick_speeds
        from core.live_train import live_payload, machines_train_for
        from core.live_train_svg import apply_live_severity, build_live_train_svg
        inst = get_instance(iid)
        mt = machines_train_for(iid, client=_get_supabase_client())
        if inst is None or not mt:
            raise HTTPException(status_code=404, detail="sin tren")
        latest = latest_for_instance(iid) or []
        setup, layout = mt
        payload = live_payload(setup, latest)
        apply_live_severity(payload, latest, _build_sensor_lookup(inst), inst, setup)
        _, speeds, _ = pick_speeds(latest)
        newest = max((str(r.get("captured_at") or "") for r in latest), default="")
        pts = []
        for pr in payload.get("probes") or []:
            try:
                v = float(pr.get("v"))
            except (TypeError, ValueError):
                continue
            al = pr.get("alarm") or 0
            pts.append({"n": pr.get("n"), "v": round(v, 2), "u": pr.get("u") or "",
                        "sev": pr.get("sev") or "ok",
                        "pct": round(v / float(al) * 100) if al else None})
        out = {"label": str(get_setting("showcase_label", "Turbogenerador") or "Turbogenerador"),
               "svg": build_live_train_svg(layout, payload), "speeds": speeds, "updated": newest,
               "points": pts}
        _SHOWCASE["v"] = (_t.time(), out)
        return out

    _SHOW_TREND: dict = {}

    @app.get("/v1/public/showcase/trend", tags=["public"],
             summary="Tendencia de un sensor de la máquina de muestra (página /demo)")
    def public_showcase_trend(sensor: str = Query("", max_length=12),
                              days: int = Query(30, ge=1, le=30)):
        """Máximo y promedio cada 2 h de un sensor de la máquina de muestra, con
        sus niveles de alerta y peligro. Sin tag ni cliente. Caché 10 min."""
        import time as _t
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz
        key = f"{sensor.upper()}:{days}"
        hit = _SHOW_TREND.get(key)
        if hit and _t.time() - hit[0] < 600:
            return hit[1]
        from core.billing.store import get_setting
        iid = str(get_setting("showcase_instance", "") or "").strip()
        if not iid:
            raise HTTPException(status_code=404, detail="sin máquina de muestra")
        from core.instance_state import get_instance
        from core.live_readings import history_rollup, latest_for_instance
        from core.live_report_builder import _build_sensor_lookup, _compute_rendered_rows
        inst = get_instance(iid)
        latest = latest_for_instance(iid) or []
        rows, _ = _compute_rendered_rows(latest, _build_sensor_lookup(inst), inst)
        direct = {r["sensor_label"]: r for r in rows if r.get("variable")}
        var_by = {}
        for r in latest:
            if (r.get("metric") or "") == "Direct" and r.get("sensor_label") and r.get("variable"):
                var_by.setdefault(r["sensor_label"], r["variable"])
        from core.report_extras import _axis_rank, _bearing
        labels = sorted(direct, key=lambda x: (_bearing(x), _axis_rank(x), x))
        want = sensor.strip().upper().replace("_", "")
        pick = next((l for l in labels if l.replace("_", "").upper() == want), None)
        if pick is None:
            pick = labels[0] if labels else None
        if pick is None or pick not in var_by:
            raise HTTPException(status_code=404, detail="sin datos")
        frm = (_dt.now(_tz.utc) - _td(days=days)).isoformat()
        pts = history_rollup(iid, var_by[pick], "Direct", frm, "2 hours") or []
        r = direct[pick]
        out = {
            "sensor": pick.replace("_", ""), "unit": r.get("unit") or "",
            "alarm": r.get("alarm_used") or None, "danger": r.get("danger_used") or None,
            "sensors": [l.replace("_", "") for l in labels],
            "points": [{"t": p.get("bucket"), "max": p.get("max_val"), "avg": p.get("avg_val")}
                       for p in pts if p.get("max_val") is not None],
        }
        _SHOW_TREND[key] = (_t.time(), out)
        return out

    @app.post("/v1/public/quote", tags=["public"], summary="Cotización en PDF (web de ventas)",
              response_class=Response if _FASTAPI_AVAILABLE else None)
    def public_quote(request: "Request", payload: PublicQuote = Body(...)):
        import re as _re
        import secrets as _secrets
        from datetime import date as _date, datetime as _dt, timezone as _tz
        from core.billing import pricing
        from core.billing.quote_pdf import build_quote_pdf, quote_number
        if payload.website:                       # bot
            raise HTTPException(status_code=400, detail="invalid")
        ip = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "")
              ).split(",")[0].strip()
        if not _quote_rate_ok(ip or "?"):
            raise HTTPException(status_code=429, detail="Demasiadas solicitudes, intente más tarde.")
        email = payload.email.strip().lower()
        if not _re.match(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", email):
            raise HTTPException(status_code=422, detail="Correo inválido")
        if payload.plan not in pricing.PLANS or payload.entity not in pricing.ENTITIES:
            raise HTTPException(status_code=422, detail="Plan o facturación inválidos")
        trm = None
        if payload.entity == "wm_us":
            from core.billing.cycle import get_trm
            trm = get_trm()
            if not trm:
                raise HTTPException(status_code=503, detail="TRM no disponible, intente en un momento")
        lead = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        lead.update(email=email, trm=trm)
        number = quote_number(_date.today(), _secrets.token_hex(3))
        pdf = build_quote_pdf(lead, number)
        try:                                      # prospecto → Administración (aviso + cron correo)
            from core.billing import store
            reqs = list(store.get_setting("sales_requests", []) or [])
            m = pricing.quote(lead["plan"], lead["machines"], lead["entity"], "mensual", trm=trm)
            reqs.insert(0, {
                "at": _dt.now(_tz.utc).isoformat(timespec="seconds"), "kind": "cotizacion",
                "client_id": lead.get("company") or email, "by": email, "status": "nueva",
                "notified": False, "number": number,
                "lead": {k: lead.get(k) for k in ("company", "name", "email", "phone",
                                                  "machine_types", "machines", "plan",
                                                  "entity", "trm", "pilot", "lang")},
                "detail": (("PILOTO 30 DÍAS · " if lead.get("pilot") else "")
                           + f"{pricing.PLANS[lead['plan']].name} · {lead['machines']} máquina(s) · "
                           f"{pricing.fmt_money(m['amount'], m['currency'])}/mes · "
                           f"{lead.get('name') or ''} {lead.get('phone') or ''} · "
                           f"{lead.get('machine_types') or ''}").strip(),
            })
            store.set_setting("sales_requests", reqs[:300])
        except Exception as e:  # noqa: BLE001
            log.warning("cotización: no se pudo guardar el prospecto: %s", e)
        log.info("public_quote: %s plan=%s n=%s", number, lead["plan"], lead["machines"])
        # Correo al prospecto + aviso a SIGA AL INSTANTE (hilo aparte para no
        # demorar la descarga). Si la API no tiene backend de correo, lo envía
        # el cron horario (notify_new_leads marca notified y no duplica).
        try:
            from core.email_sender import get_email_backend_status
            if get_email_backend_status().get("configured"):
                import threading as _th
                from core.billing.upsell import notify_new_leads
                _th.Thread(target=notify_new_leads, daemon=True).start()
        except Exception as e:  # noqa: BLE001
            log.warning("cotización: aviso inmediato no disparado: %s", e)
        return Response(content=pdf, media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{number}.pdf"'})

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

        # Origen de la señal (modbus | ni) en CADA lectura → permite el periodo
        # en paralelo Modbus vs NI y marcar el corte en el histórico. Los
        # collectors Modbus viejos (≤1.1.0) no lo mandan → "modbus".
        batch_meta = payload.metadata or {}
        batch_source = str(batch_meta.get("source") or "modbus")

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
                metadata={"source": batch_source, **(r.metadata or {})},
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
    # Mapa Modbus GENERADO desde la config de la máquina (Machines /
    # rm_setups). El collector lo descarga al arrancar y cada N min; si
    # falla, sigue con su copia local. Fase 1 migración Modbus → NI.
    # =========================================================
    @app.get("/v1/collector/map/{instance_id}", tags=["ingest"],
             summary="Mapa Modbus del collector generado desde la config de la máquina")
    def collector_map(instance_id: str, api_key_hash: str = Depends(_api_key_dependency)):
        from core.live_readings import _get_supabase_client
        from core.remote_monitoring.config import find_setup_cloud_by_instance
        from core.remote_monitoring.modbus_binding import modbus_map_from_setup

        found = find_setup_cloud_by_instance(instance_id, client=_get_supabase_client())
        if not found:
            raise HTTPException(status_code=404, detail="No machine config for this instance")
        setup = found["setup"]
        if not any(c.modbus for c in setup.channels):
            raise HTTPException(status_code=404, detail="Machine has no Modbus source configured")
        out = modbus_map_from_setup(setup)
        out["_config"] = {"setup_id": found.get("id"), "updated_at": found.get("updated_at"),
                          "version": found.get("version"), "source": setup.source}
        log.info("collector_map: instance=%s regs=%d caller_hash=%s",
                 instance_id, len(out.get("registers") or {}), api_key_hash)
        return out

    # =========================================================
    # Confirmación del nodo (Fase 4): el collector / nodo en línea avisa qué
    # versión de la config de máquina está aplicando ("applied") o por qué la
    # rechazó ("rejected"). Machines lo muestra en el historial de versiones.
    # =========================================================
    @app.post("/v1/collector/ack", tags=["ingest"],
              summary="El nodo confirma la versión de config aplicada (o la rechaza)")
    def collector_ack(payload: CollectorAck = Body(...),
                      api_key_hash: str = Depends(_api_key_dependency)):
        from core.live_readings import _get_supabase_client
        from core.remote_monitoring.config import find_setup_cloud_by_instance
        from core.remote_monitoring.setup_versions import node_ack

        status = payload.status if payload.status in ("applied", "rejected") else "applied"
        client = _get_supabase_client()
        found = find_setup_cloud_by_instance(payload.instance_id, client=client)
        if not found:
            raise HTTPException(status_code=404, detail="No machine config for this instance")
        res = node_ack(found["id"], int(payload.version), status,
                       (payload.detail or "")[:300], client=client)
        if not res.get("ok"):
            raise HTTPException(status_code=502, detail=res.get("reason") or "ack failed")
        log.info("collector_ack: instance=%s v%s %s caller_hash=%s",
                 payload.instance_id, payload.version, status, api_key_hash)
        return {"ok": True, "setup_id": found["id"], "version": int(payload.version), "status": status}

    # =========================================================
    # Figuras — FUENTE ÚNICA (core.figure_series). Web, reportes y app
    # consumen las MISMAS series (espectro / onda / órbita). v1.
    # =========================================================
    @app.get("/v1/figure-style", tags=["figures"],
             summary="Estilo ÚNICO de los gráficos (colores/rangos/marcadores)")
    def figure_style(api_key_hash: str = Depends(_api_key_dependency)):
        from core.figure_style import as_dict
        return as_dict()

    @app.get("/v1/assets/{asset}/points", tags=["figures"],
             summary="Puntos de medición disponibles (dynamic_raw)")
    def figure_points(asset: str,
                      api_key_hash: str = Depends(_api_key_dependency)):
        from core import figure_series as _fs
        return {"asset": asset, "items": _fs.list_points(asset)}

    @app.get("/v1/assets/{asset}/spectrum", tags=["figures"],
             summary="Espectro (FFT Hann) de un sensor")
    def figure_spectrum(asset: str, token: str = Query(..., description="ej. 1XD, 4XA"),
                        api_key_hash: str = Depends(_api_key_dependency)):
        from core import figure_series as _fs
        out = _fs.spectrum_series(asset, token)
        if not out:
            raise HTTPException(status_code=404, detail="Sin datos para ese sensor")
        return out

    @app.get("/v1/assets/{asset}/waveform", tags=["figures"],
             summary="Forma de onda de un sensor")
    def figure_waveform(asset: str, token: str = Query(..., description="ej. 1XD, 4XA"),
                        api_key_hash: str = Depends(_api_key_dependency)):
        from core import figure_series as _fs
        out = _fs.waveform_series(asset, token)
        if not out:
            raise HTTPException(status_code=404, detail="Sin datos para ese sensor")
        return out

    @app.get("/v1/assets/{asset}/orbit", tags=["figures"],
             summary="Órbita X vs Y de un cojinete")
    def figure_orbit(asset: str, bearing: str = Query(..., description="ej. 1, BRG1"),
                     api_key_hash: str = Depends(_api_key_dependency)):
        from core import figure_series as _fs
        out = _fs.orbit_series_app(asset, bearing)
        if not out:
            raise HTTPException(status_code=404, detail="Sin datos para ese cojinete")
        return out

    @app.get("/v1/assets/{asset}/train-svg", tags=["figures"],
             summary="Diagrama del tren (SVG) con sensores por severidad — mismo de la web")
    def figure_train_svg(asset: str, api_key_hash: str = Depends(_api_key_dependency)):
        from core.instance_state import get_instance
        from core.live_readings import latest_for_instance
        from core.train_svg import build_train_svg
        inst = get_instance(asset)
        if inst is None:
            raise HTTPException(status_code=404, detail="asset no encontrado")
        latest = latest_for_instance(asset) or []
        lookup = {}
        try:
            from core.sensor_map import sensor_label as _slbl
            for s in (getattr(inst, "sensors", None) or []):
                try:
                    lookup[_slbl(s)] = s
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        # 2026-11-23 — Si la máquina tiene TREN dibujado en Machines, se entrega ese
        # tren EN VIVO (mismo de Live Monitoring y del reporte PDF web): la app
        # móvil y su reporte lo muestran sin cambios de código propios.
        try:
            from core.live_readings import _get_supabase_client
            from core.live_train import live_payload, machines_train_for
            from core.live_train_svg import apply_live_severity, build_live_train_svg
            mt = machines_train_for(asset, client=_get_supabase_client())
            if mt:
                setup, layout = mt
                payload = live_payload(setup, latest)
                apply_live_severity(payload, latest, lookup, inst, setup)
                return {"asset": asset, "svg": build_live_train_svg(layout, payload),
                        "kind": "machines"}
        except Exception as e:  # noqa: BLE001
            log.warning("train-svg machines falló (%s) → dibujo anterior", e)
        svg = build_train_svg(inst, latest, lookup)
        if not svg:
            raise HTTPException(status_code=404,
                                detail="sin diagrama (faltan driver/driven_icon_key o sensores mapeables)")
        return {"asset": asset, "svg": svg, "kind": "legacy"}

    # =========================================================
    # Reporte de condición en PDF = EL MISMO del cron diario y de la descarga
    # desde Live (core.live_report_builder). La app móvil lo pide aquí para que
    # su reporte sea copia exacta del de la web. Detecta solo fuera de línea.
    # =========================================================
    @app.get("/v1/assets/{asset}/report-pdf", tags=["figures"],
             summary="Reporte de condición (PDF) — mismo de la web",
             response_class=Response if _FASTAPI_AVAILABLE else None)
    def asset_report_pdf(asset: str, api_key_hash: str = Depends(_api_key_dependency)):
        from core.instance_state import get_instance
        inst = get_instance(asset)
        if inst is None:
            raise HTTPException(status_code=404, detail="asset no encontrado")
        pdf, meta = _report_pdf_isolated(asset)   # se arma en un proceso aparte
        if not pdf:
            raise HTTPException(status_code=404, detail="sin lecturas para armar el reporte")
        log.info("report_pdf: asset=%s status=%s caller_hash=%s",
                 asset, (meta or {}).get("status"), api_key_hash)
        return Response(content=pdf, media_type="application/pdf")

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


# =============================================================
# Reporte PDF aislado (memoria)
# =============================================================
# Armar el reporte sube el proceso de ~45 MB a ~380-420 MB (plotly, reportlab,
# svglib, numpy) y esa memoria NO se devuelve: dos reportes seguidos tumbaban
# el servicio (límite 512 MB, aviso de Render 2026-11-23). Ahora cada reporte
# corre en un proceso hijo que termina al entregar el PDF (la memoria vuelve
# al sistema), UNO a la vez, y el mismo reporte se reusa 120 s.
import threading as _threading
import time as _time

_REPORT_LOCK = _threading.Lock()
_REPORT_CACHE: dict = {}
_REPORT_TTL_S = 120.0


def _report_pdf_isolated(asset: str):
    hit = _REPORT_CACHE.get(asset)
    if hit and _time.time() - hit[0] < _REPORT_TTL_S:
        return hit[1], hit[2]
    with _REPORT_LOCK:
        hit = _REPORT_CACHE.get(asset)               # otro request lo pudo armar
        if hit and _time.time() - hit[0] < _REPORT_TTL_S:
            return hit[1], hit[2]
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=1, mp_context=_mp.get_context("spawn")) as ex:
            from core.report_worker import build_report
            pdf, meta = ex.submit(build_report, asset).result(timeout=170)
        if pdf:
            _REPORT_CACHE.clear()                    # guarda solo el último (memoria)
            _REPORT_CACHE[asset] = (_time.time(), pdf, meta)
        return pdf, meta


# Instancia top-level para `uvicorn api.app:app`. Sólo se construye si
# FastAPI está instalado.
if _FASTAPI_AVAILABLE:
    app = create_app()
else:
    app = None  # type: ignore


__all__ = ["create_app", "app"]
