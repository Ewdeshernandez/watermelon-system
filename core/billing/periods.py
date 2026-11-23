"""Periodos de facturación, numeración y vencimientos (puro)."""

from __future__ import annotations

import calendar
import re
from datetime import date, timedelta
from typing import Tuple


def add_months(d: date, n: int) -> date:
    """Suma n meses conservando el día (si no existe, último día del mes)."""
    m = d.month - 1 + n
    y = d.year + m // 12
    m = m % 12 + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def period_bounds(start: date, cycle: str = "mensual") -> Tuple[date, date]:
    """(inicio, fin inclusive) del periodo que arranca en `start`."""
    nxt = add_months(start, 12 if cycle == "anual" else 1)
    return start, nxt - timedelta(days=1)


def next_renewal_after(start: date, cycle: str = "mensual") -> date:
    return add_months(start, 12 if cycle == "anual" else 1)


def invoice_number(client_id: str, period_start: date, cycle: str = "mensual") -> str:
    code = re.sub(r"[^A-Z0-9]+", "-", (client_id or "CLIENTE").upper()).strip("-")
    tag = f"{period_start.year}A" if cycle == "anual" else f"{period_start:%Y-%m}"
    return f"WM-{tag}-{code}"


def due_date(issued: date, due_days: int = 10) -> date:
    return issued + timedelta(days=max(0, int(due_days)))


def effective_status(status: str, due: date, today: date) -> str:
    """'pendiente' pasada la fecha de vencimiento se muestra como 'vencida'."""
    if status == "pendiente" and today > due:
        return "vencida"
    return status


MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
         "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def period_label(start: date, end: date, cycle: str = "mensual") -> str:
    if cycle == "anual":
        return f"{MESES[start.month - 1]} {start.year} – {MESES[end.month - 1]} {end.year}"
    return f"{MESES[start.month - 1]} {start.year}"


def fmt_date(v) -> str:
    """'2026-12-01' → '1 dic 2026'."""
    try:
        d = v if isinstance(v, date) else date.fromisoformat(str(v)[:10])
    except Exception:  # noqa: BLE001
        return str(v or "—")
    return f"{d.day} {MESES[d.month - 1][:3]} {d.year}"
