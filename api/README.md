# Watermelon System API (v1.0)

Servicio HTTP REST público read-only.

## Run local

```bash
pip install -r requirements-api.txt
export WATERMELON_API_KEYS="dev-key-1,dev-key-2"
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI: http://localhost:8000/docs
Redoc:     http://localhost:8000/redoc

## Auth

Todos los endpoints (excepto `/v1/health`) requieren cabecera:

```
Authorization: Bearer <API_KEY>
```

## Endpoints v1.0

| Método | Path                                                       | Descripción |
|--------|------------------------------------------------------------|-------------|
| GET    | `/v1/health`                                               | Estado del servicio |
| GET    | `/v1/templates`                                            | Lista plantillas (filtros: category, manufacturer) |
| GET    | `/v1/templates/{id}`                                       | Detalle de plantilla |
| GET    | `/v1/templates/{id}/norm-recommendation`                   | Norma ISO recomendada |
| GET    | `/v1/templates/categories`                                 | Categorías |
| GET    | `/v1/norms`                                                | Lista normas (ISO/API) |
| GET    | `/v1/norms/groups`                                         | Normas agrupadas |
| GET    | `/v1/norms/{code}`                                         | Detalle norma |
| GET    | `/v1/norms/{code}/classes/{class}/thresholds`              | Thresholds de clase |
| GET    | `/v1/loaders`                                              | Formatos soportados |
| GET    | `/v1/bearings`                                             | Catálogo de rodamientos |
| GET    | `/v1/bearings/overlay?model=...&rpm=...`                   | BPFO/BPFI/BSF/FTF |

## Próximos (v1.1)

- POST `/v1/diagnostics/spectrum`  — diagnóstico AI Cat IV+ vía API
- POST `/v1/loaders/parse`         — parser universal CSI/ADRE/UFF
- WebSocket `/v1/stream/diagnostic`  — diagnóstico en tiempo real

## Tests

```bash
pytest tests/test_api_services.py -v
```

La capa de servicios es testeable sin FastAPI/uvicorn instalados.
