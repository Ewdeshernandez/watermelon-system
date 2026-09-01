"""
core/modal/campbell.py — Diagrama de Campbell con detección AUTOMÁTICA de cruces
===============================================================================

Núcleo de cómputo (numpy puro) para el diagrama de Campbell según API 684:
cruza las frecuencias naturales identificadas (EMA/OMA) contra las líneas de
orden (1×, 2×, ...) y detecta automáticamente las velocidades de coincidencia
(potenciales resonancias), su ubicación respecto a las bandas de operación y su
margen de separación.

Reemplaza el trabajo manual "a ojo" sobre el gráfico de ARTeMIS: entrega la
TABLA de cruces lista para el reporte + una figura (Plotly) con las bandas de
operación sombreadas y los cruces marcados.

Referencia: API 684 §1.2.2.3 / §1.6 — velocidades críticas y márgenes de
separación. La coincidencia fn ↔ N× NO implica resonancia por sí sola; marca una
zona de interés que debe correlacionarse con amplitud/fase en operación.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

# Órdenes de excitación por defecto (incluye ½× para holgura/aceite y sub-armónicos)
DEFAULT_ORDERS: Tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 4.0)


@dataclass
class SpeedBand:
    """Banda de velocidad de interés (p.ej. velocidad máx de operación ± tol.)."""
    center_rpm: float
    tol_rpm: float
    label: str = ""

    @property
    def low(self) -> float:
        return self.center_rpm - self.tol_rpm

    @property
    def high(self) -> float:
        return self.center_rpm + self.tol_rpm

    def contains(self, rpm: float) -> bool:
        return self.low <= rpm <= self.high


@dataclass
class Crossing:
    """Un cruce modo ↔ orden en el diagrama de Campbell."""
    mode_hz: float
    mode_label: str
    order: float
    crossing_rpm: float          # N = 60·fn/orden
    in_band: bool                # cae dentro de alguna banda de operación
    band_label: str              # banda más cercana / que lo contiene
    sep_margin_pct: float        # margen de separación al centro de esa banda (API 684)
    severity: str                # "coincidence" | "near" | "clear"

    def describe(self) -> str:
        z = f" (banda {self.band_label})" if self.in_band else ""
        return (f"{self.mode_label} {self.mode_hz:.3f} Hz cruza {self.order:g}× a "
                f"{self.crossing_rpm:.0f} RPM{z} · margen {self.sep_margin_pct:.1f}%")


def separation_margin_pct(crossing_rpm: float, reference_rpm: float) -> float:
    """Margen de separación API 684: |Nref - Ncruce| / Nref · 100.
    Pequeño = cruce cerca de la velocidad de referencia (peor)."""
    if reference_rpm <= 0:
        return float("inf")
    return abs(reference_rpm - crossing_rpm) / reference_rpm * 100.0


def compute_crossings(
    modes_hz: Sequence[float],
    rpm_min: float,
    rpm_max: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
    bands: Optional[Sequence[SpeedBand]] = None,
    mode_labels: Optional[Sequence[str]] = None,
    classification: Optional[Sequence[str]] = None,
    near_margin_pct: float = 10.0,
) -> List[Crossing]:
    """Detecta todos los cruces fn ↔ orden dentro de [rpm_min, rpm_max].

    - in_band: el cruce cae dentro de alguna SpeedBand.
    - severity: 'coincidence' si in_band; 'near' si el margen a una banda < near_margin_pct
      (o, sin bandas, si el margen a rpm_max < near_margin_pct); 'clear' en otro caso.
    """
    bands = list(bands or [])
    n = len(modes_hz)
    labels = list(mode_labels) if mode_labels else [f"Modo {i+1}" for i in range(n)]
    cls = list(classification) if classification else ["natural"] * n
    out: List[Crossing] = []
    for fn, lab, c in zip(modes_hz, labels, cls):
        if c not in ("natural", "") and c is not None:   # armónicos/espurios no generan crítica
            continue
        for order in orders:
            if order <= 0:
                continue
            rpm = 60.0 * float(fn) / float(order)
            if not (rpm_min <= rpm <= rpm_max):
                continue
            # banda más cercana (por distancia al centro)
            in_band = False; band_label = ""; ref = rpm_max
            if bands:
                nearest = min(bands, key=lambda b: abs(b.center_rpm - rpm))
                ref = nearest.center_rpm
                band_label = nearest.label or f"{nearest.center_rpm:.0f}±{nearest.tol_rpm:.0f}"
                in_band = any(b.contains(rpm) for b in bands)
            sm = separation_margin_pct(rpm, ref)
            if in_band:
                sev = "coincidence"
            elif sm < near_margin_pct:
                sev = "near"
            else:
                sev = "clear"
            out.append(Crossing(mode_hz=float(fn), mode_label=lab, order=float(order),
                                crossing_rpm=rpm, in_band=in_band, band_label=band_label,
                                sep_margin_pct=sm, severity=sev))
    out.sort(key=lambda x: (0 if x.severity == "coincidence" else 1 if x.severity == "near" else 2,
                            x.sep_margin_pct))
    return out


def crossings_table(crossings: Sequence[Crossing]) -> List[dict]:
    """Filas listas para el reporteador (dict por cruce)."""
    return [{
        "Modo": c.mode_label,
        "fn [Hz]": round(c.mode_hz, 3),
        "Orden": f"{c.order:g}×",
        "RPM cruce": round(c.crossing_rpm, 0),
        "Banda": c.band_label if c.in_band else "—",
        "Margen [%]": round(c.sep_margin_pct, 1),
        "Estado": {"coincidence": "Coincidencia", "near": "Cercano", "clear": "Libre"}[c.severity],
    } for c in crossings]


def summarize(crossings: Sequence[Crossing]) -> str:
    """Texto de hallazgos automático (para el reporte)."""
    coin = [c for c in crossings if c.severity == "coincidence"]
    near = [c for c in crossings if c.severity == "near"]
    if not coin and not near:
        return ("No se identifican coincidencias relevantes entre las frecuencias naturales "
                "y las órdenes de excitación dentro del rango de velocidad evaluado.")
    parts = []
    if coin:
        parts.append("Coincidencias dentro de las bandas de operación: "
                     + "; ".join(c.describe() for c in coin[:6]) + ".")
    if near:
        parts.append("Cruces próximos (margen reducido): "
                     + "; ".join(c.describe() for c in near[:4]) + ".")
    parts.append("La coincidencia fn↔orden no confirma resonancia por sí sola (API 684): "
                 "debe correlacionarse con amplitud y fase durante la operación.")
    return " ".join(parts)


# =====================================================================
# Figura (Plotly) — para web y reporteador (render PNG vía kaleido)
# =====================================================================
def build_campbell_figure(
    modes_hz: Sequence[float],
    rpm_min: float,
    rpm_max: float,
    orders: Sequence[float] = DEFAULT_ORDERS,
    bands: Optional[Sequence[SpeedBand]] = None,
    mode_labels: Optional[Sequence[str]] = None,
    classification: Optional[Sequence[str]] = None,
    operating_rpm: Optional[float] = None,
    title: str = "Diagrama de Campbell",
    crossings: Optional[Sequence[Crossing]] = None,
):
    """Figura de Campbell con bandas de operación sombreadas + cruces marcados."""
    import plotly.graph_objects as go
    if crossings is None:
        crossings = compute_crossings(modes_hz, rpm_min, rpm_max, orders, bands,
                                      mode_labels, classification)
    n = len(modes_hz)
    labels = list(mode_labels) if mode_labels else [f"Modo {i+1}" for i in range(n)]
    cls = list(classification) if classification else ["natural"] * n
    fig = go.Figure()
    rpm = np.linspace(rpm_min, rpm_max, 60)

    # bandas de operación (sombreadas)
    for b in (bands or []):
        fig.add_vrect(x0=max(rpm_min, b.low), x1=min(rpm_max, b.high),
                      fillcolor="#f59e0b", opacity=0.12, line_width=0,
                      annotation_text=b.label or f"{b.center_rpm:.0f}±{b.tol_rpm:.0f}",
                      annotation_position="top left",
                      annotation_font_size=10, annotation_font_color="#b45309")

    # líneas de orden
    for order in orders:
        fig.add_trace(go.Scatter(x=rpm, y=order * rpm / 60.0, mode="lines",
                                 name=f"{order:g}× rpm",
                                 line=dict(color="#6B7280", width=1, dash="dot"),
                                 hovertemplate=f"{order:g}× rpm<extra></extra>"))
        fig.add_annotation(x=rpm_max * 0.98, y=order * rpm_max / 60.0,
                           text=f"{order:g}×", showarrow=False,
                           font=dict(size=10, color="#6B7280"))

    # modos (líneas horizontales)
    cc = {"natural": "#16a34a", "harmonic": "#dc2626", "spurious": "#9ca3af"}
    for fn, lab, c in zip(modes_hz, labels, cls):
        fig.add_trace(go.Scatter(x=[rpm_min, rpm_max], y=[fn, fn], mode="lines",
                                 name=f"{lab} · {fn:.2f} Hz",
                                 line=dict(color=cc.get(c, "#0F7FB0"), width=2),
                                 hovertemplate=f"{lab}<br>fn={fn:.3f} Hz<extra></extra>"))

    # operating speed
    if operating_rpm and rpm_min <= operating_rpm <= rpm_max:
        fig.add_vline(x=operating_rpm, line=dict(color="#0F1E3D", width=2, dash="dash"),
                      annotation_text=f"Operación {operating_rpm:.0f} RPM",
                      annotation_position="bottom right")

    # cruces
    sev_color = {"coincidence": "#dc2626", "near": "#f59e0b", "clear": "#94a3b8"}
    for c in crossings:
        fig.add_trace(go.Scatter(
            x=[c.crossing_rpm], y=[c.mode_hz], mode="markers", showlegend=False,
            marker=dict(color=sev_color[c.severity], size=11, symbol="x",
                        line=dict(width=2, color="#7f1d1d")),
            hovertemplate=(f"<b>{c.crossing_rpm:.0f} RPM</b><br>{c.mode_label} × {c.order:g}×"
                           f"<br>fn={c.mode_hz:.3f} Hz · margen {c.sep_margin_pct:.1f}%<extra></extra>")))

    ymax = max(modes_hz) * 1.25 if len(modes_hz) else 200
    fig.update_layout(title=title, xaxis_title="Velocidad (RPM)", yaxis_title="Frecuencia (Hz)",
                      height=520, template="plotly_white", hovermode="closest",
                      margin=dict(l=60, r=40, t=70, b=50),
                      legend=dict(orientation="h", y=-0.2, x=0))
    fig.update_xaxes(range=[rpm_min, rpm_max]); fig.update_yaxes(range=[0, ymax])
    return fig
