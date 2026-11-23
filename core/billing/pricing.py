"""Planes por máquina y cálculo del cobro (puro, sin base de datos).

Precio de LISTA por máquina al mes en COP. A un cliente se le puede pactar
otro valor (wm_subscriptions.price_per_machine); el descuento por volumen y el
de USD se aplican igual.

  · Volumen: 1–3 máquinas precio base · 4–5 −10 % · 6–9 −15 % · 10+ −20 %
  · Facturar con Watermelon System (EE. UU., USD, Mercury): −5 % y precio en
    USD congelado 12 meses
  · Anual adelantado: se cobran 11 meses de 12
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Plan:
    key: str
    name: str
    price_cop: int
    includes: Tuple[str, ...]
    excludes: Tuple[str, ...] = ()


PLANS: Dict[str, Plan] = {
    "esencial": Plan(
        "esencial", "Esencial", 11_500_000,
        ("Monitoreo en línea 24/7 (Live Monitoring)",
         "Alarmas por correo",
         "Reporte mensual firmado por analista"),
        ("Reporte semanal", "Balanceo, análisis modal y confiabilidad API 670")),
    "profesional": Plan(
        "profesional", "Profesional", 15_397_259,
        ("Todo lo del plan Esencial",
         "Reporte semanal firmado por analista Cat. IV",
         "App móvil Watermelon Live",
         "Análisis de arranques: Bode, polar y full spectrum"),
        ("Balanceo, análisis modal y confiabilidad API 670 (se cotizan aparte)",)),
    "integral": Plan(
        "integral", "Integral", 18_045_692,
        ("Todo lo del plan Profesional",
         "Balanceo incluido",
         "Análisis modal incluido",
         "Confiabilidad del sistema de monitoreo y protección (API 670)")),
}

VOLUME_TIERS: List[Tuple[int, float]] = [(10, 0.20), (6, 0.15), (4, 0.10), (1, 0.0)]
USD_DISCOUNT = 0.05
ANNUAL_MONTHS_CHARGED = 11

ENTITIES = {
    "siga_co": {"name": "SIGA GROUP SAS", "country": "Colombia", "currency": "COP",
                "methods": ("PSE", "Bre-B (llave)", "Tarjeta de crédito",
                            "Tarjeta débito", "Nequi", "Daviplata",
                            "Transferencia bancaria")},
    "wm_us": {"name": "Watermelon System", "country": "Estados Unidos", "currency": "USD",
              "methods": ("Transferencia internacional (wire)", "ACH",
                          "Tarjeta de crédito", "Tarjeta débito",
                          "Apple Pay", "Google Pay")},
}


def volume_discount(machines: int) -> float:
    for n, pct in VOLUME_TIERS:
        if machines >= n:
            return pct
    return 0.0


def next_tier(machines: int) -> Optional[Tuple[int, float]]:
    """(máquinas, descuento) del siguiente escalón, o None si ya es el máximo."""
    for n, pct in sorted(VOLUME_TIERS):
        if n > machines and pct > volume_discount(machines):
            return n, pct
    return None


def unit_price_cop(plan: str, override: Optional[float] = None) -> float:
    if override:
        return float(override)
    return float(PLANS[plan].price_cop)


def usd_unit_from_cop(unit_cop: float, trm: float) -> float:
    """Precio por máquina en USD (sin descuentos), redondeado a dólar."""
    if not trm or trm <= 0:
        raise ValueError("TRM inválida")
    return float(round(unit_cop / trm))


def quote(plan: str, machines: int, entity: str = "siga_co", cycle: str = "mensual",
          *, price_override: Optional[float] = None, usd_unit: Optional[float] = None,
          trm: Optional[float] = None) -> Dict[str, object]:
    """Cobro de UN periodo (mes o año).

    COP: unit = precio pactado o de lista. USD: unit = `usd_unit` (congelado)
    o, si no hay, el COP convertido con `trm`. El descuento USD solo aplica a
    la entidad de EE. UU.
    """
    if plan not in PLANS:
        raise ValueError(f"plan desconocido: {plan}")
    if entity not in ENTITIES:
        raise ValueError(f"entidad desconocida: {entity}")
    machines = max(0, int(machines))
    currency = ENTITIES[entity]["currency"]
    unit_cop = unit_price_cop(plan, price_override)
    if currency == "USD":
        unit = float(usd_unit) if usd_unit else usd_unit_from_cop(unit_cop, trm or 0)
    else:
        unit = unit_cop
    vol = volume_discount(machines)
    ent = USD_DISCOUNT if entity == "wm_us" else 0.0
    months = 12 if cycle == "anual" else 1
    charged = ANNUAL_MONTHS_CHARGED if cycle == "anual" else 1
    subtotal = unit * machines * months
    amount = unit * machines * (1 - vol) * (1 - ent) * charged
    ndp = 2 if currency == "USD" else 0
    return {
        "plan": plan, "machines": machines, "currency": currency,
        "unit_price": round(unit, ndp), "volume_discount_pct": vol * 100,
        "entity_discount_pct": ent * 100, "months": months,
        "months_charged": charged, "subtotal": round(subtotal, ndp),
        "amount": round(amount, ndp), "savings": round(subtotal - amount, ndp),
        "trm": trm if currency == "USD" else None,
    }


def fmt_money(value: float, currency: str) -> str:
    if currency == "USD":
        return "USD " + f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return "COP " + f"{round(value):,}".replace(",", ".")
