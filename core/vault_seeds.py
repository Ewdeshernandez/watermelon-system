"""
core.vault_seeds
================

Datos por defecto del Document Vault que viven en el código (no en
filesystem efímero). Sobreviven cualquier redeploy de Streamlit Cloud y
permiten que la app salga al aire con los activos conocidos ya
"poblados", sin requerir que el usuario re-ingrese parámetros físicos
después de cada reinicio del container.

**Por qué existen:**

Streamlit Cloud tiene filesystem efímero. Los datos persistidos en
``data/asset_metadata/`` y ``data/asset_documents/`` se borran cuando
el container se reinicia (cosa que ocurre en cada redeploy y cuando la
app está inactiva). Para activos críticos (turbogeneradores grandes,
máquinas con presupuesto OEM bien documentado), no podemos depender de
que el usuario los rellene cada vez. Las semillas resuelven eso:

- ``get_captured_parameters(profile_key)`` primero intenta leer del
  filesystem (datos que el usuario ingresó manualmente y aún no se han
  borrado), y si no hay nada allí, cae a las semillas de este módulo.
- Si el usuario ingresa nuevos valores, esos se persisten al
  filesystem y ganan sobre la semilla. La semilla solo actúa como
  fallback de arranque en frío.

**Cómo agregar un activo nuevo:**

Añadir una entrada al diccionario ``VAULT_SEEDS`` con el ``profile_key``
exacto del perfil (ver ``core/machine_profiles.py``). Los campos
disponibles son los 26 ``CAPTURED_PARAMETER_FIELDS`` definidos en
``core/document_vault.py``. Los valores deben respetar las unidades
declaradas allí (mm para diámetros, mm para clearances, °C para
temperaturas, fechas en ISO yyyy-mm-dd, etc.).

Solo poblar los campos para los que tengamos dato OEM o de
mantenimiento confiable. Los demás se dejan vacíos y la app los
preguntará al usuario.
"""

from __future__ import annotations

from typing import Any, Dict


# =============================================================
# SEMILLAS DE VAULT POR PROFILE_KEY
# =============================================================
#
# IMPORTANTE: el profile_key debe coincidir EXACTAMENTE con el de
# core/machine_profiles.py (ver MACHINE_PROFILES dict).

VAULT_SEEDS: Dict[str, Dict[str, Any]] = {

    # =========================================================
    # Brush turbogenerator 54 MW @ 3600 rpm
    # =========================================================
    # Activo de referencia del usuario. Datos extraídos del
    # reporte de rebabbiting Wersin (23 oct 2018) y del manual
    # de cojinetes radiales Brush HA.
    "brush_turbogenerator_54mw_3600": {
        "bearing_inner_diameter_mm": 254.41,
        "diametral_clearance_mm": 0.382,
        "babbitt_material": "ASTM B-23 Grade 2 / BERA 90",
        "last_rebabbiting_date": "2018-10-23",
        "oil_grade": "ISO VG 32",
        "design_oil_temperature_c": 49.0,
        "alarm_oil_temperature_c": 75.0,
        "trip_oil_temperature_c": 85.0,
        "design_bearing_temperature_c": 90.0,
        "alarm_bearing_temperature_c": 105.0,
        "trip_bearing_temperature_c": 115.0,
        "rated_power_mw": 54.0,
        "rated_speed_rpm": 3600.0,
    },

    # Más activos pueden agregarse aquí siguiendo el mismo formato.
    # Cuando un cliente nuevo aporte datos OEM confiables de su
    # máquina, se incorporan acá para que la app los traiga "de
    # fábrica" sin requerir ingreso manual repetido.
}


def get_seed_parameters(profile_key: str) -> Dict[str, Any]:
    """
    Devuelve los parámetros semilla para un profile, o dict vacío si
    no hay semilla definida. Nunca lanza — falla a {} si la entrada
    no existe.
    """
    return dict(VAULT_SEEDS.get(profile_key, {}))


def has_seed(profile_key: str) -> bool:
    """True si existe una semilla para ese profile_key."""
    return profile_key in VAULT_SEEDS


__all__ = [
    "VAULT_SEEDS",
    "get_seed_parameters",
    "has_seed",
]
