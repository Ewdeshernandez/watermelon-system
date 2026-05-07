"""
api
===

Servicio HTTP REST público de Watermelon System.

Diseño en dos capas para que la lógica sea testeable sin tener FastAPI
instalado:

  - api.services  : capa de servicios pura (funciones Python). No
                    depende de FastAPI/uvicorn. Toda la inteligencia de
                    qué se expone, qué se filtra y qué shape se devuelve
                    vive aquí. Es 100% testeable con pytest sin servidor.

  - api.app       : factory FastAPI. Importa servicios y expone routers.
                    Esta capa es delgada: validación de auth, parámetros,
                    serialización JSON, manejo de errores HTTP.

  - api.auth      : validación de API key (header Authorization o
                    query param) contra una whitelist en variables de
                    entorno o Supabase.

Filosofía:
  - **Read-only en v1.0**. Esta API sólo expone GET endpoints. Mutaciones
    (POST/PUT/DELETE) llegan en v1.1 con pruebas de penetración.
  - Versionado semántico: prefijo `/v1/`. Breaking changes implican v2.
  - Hardware-agnostic: la API expone el mismo shape para activos
    cargados desde Bently CSV, CSI 2140, ADRE 408, UFF — esto es
    PRECISAMENTE el cuchillo contra el lock-in de System1.
"""
