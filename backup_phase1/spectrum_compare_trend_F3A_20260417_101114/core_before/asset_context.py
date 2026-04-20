
from typing import Dict


def get_asset_type(metadata: Dict) -> str:
    return str(metadata.get("Asset Type", "")).strip().lower()


def is_hydraulic_machine(asset_type: str) -> bool:
    return asset_type in ["bomba", "compresor"]


def is_electrical_machine(asset_type: str) -> bool:
    return asset_type in ["generador eléctrico", "motor eléctrico"]


def is_turbomachinery(asset_type: str) -> bool:
    return asset_type in ["turbogenerador", "turbina de gas"]


def allow_hydraulic_faults(asset_type: str) -> bool:
    if is_electrical_machine(asset_type):
        return False
    if is_turbomachinery(asset_type):
        return False
    return True


def adjust_trend_diagnostic_text(text: str, asset_type: str) -> str:
    """
    Limpia o ajusta narrativa según tipo de activo
    """
    if not allow_hydraulic_faults(asset_type):
        # elimina palabras peligrosas
        forbidden = [
            "cavitación",
            "turbulencia",
            "flujo",
            "recirculación",
        ]
        for word in forbidden:
            text = text.replace(word, "")
    return text.strip()
