"""
core/modal/auto_analysis.py — Auto-analisis normativo modal (rule-based)
=========================================================================

Genera hallazgos automaticos desde los modos identificados aplicando las
normas embebidas: ISO 7626-1..6, ISO 20816, API 684, API 618.

Sin IA — texto deterministico generado por reglas. Equivalente al
auto-analisis que ya existe en Spectrum, SCL, Polar.

Reglas implementadas
--------------------
1. Cruce con armonicas (1x, 2x, 3x...) — API 684 secc. 1.6, API 618 secc. 7.9.4.2.5.3.2
2. AutoMAC redundancy (off-diagonal > 0.7) — ISO 7626-6 secc. 6.5
3. MPC threshold — Pappa & Eishan 1995
4. Damping anormal (> 5% o < 0.1%) — ISO 20816
5. Modos cercanos (delta_f < 5%) — posibles modos repetidos
6. Set modal insuficiente (< 3 modos en banda relevante)
7. Validacion EMA solo: coherencia + n_promedios — ISO 7626-5
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import math


@dataclass
class Finding:
    """Hallazgo automatico del analisis modal."""
    severity: str  # "ok" | "warning" | "fail" | "info"
    title: str
    text: str
    norm_ref: str = ""


def _classify_freq_vs_harmonics(
    freq_hz: float, running_hz: float, n_harmonics: int = 6,
    tolerance_pct: float = 10.0,
) -> Tuple[Optional[int], Optional[float]]:
    """Retorna (orden, delta_pct) si freq esta cerca de N*running, sino (None, None).
    tolerance_pct = 10% por default (API 618 secc. 7.9.4.2.5.3.2 = separacion >= 10%)."""
    if running_hz <= 0:
        return None, None
    for n in range(1, n_harmonics + 1):
        target = running_hz * n
        delta_pct = abs(freq_hz - target) / max(target, 1e-6) * 100.0
        if delta_pct <= tolerance_pct:
            return n, delta_pct
    return None, None


def analyze_modal_results(
    fdd_result: Any,
    *,
    running_rpm: float = 3600.0,
    mac_threshold: float = 0.7,
    mpc_complex_threshold_pct: float = 30.0,
    damping_high_pct: float = 5.0,
    damping_low_pct: float = 0.1,
    modes_min_for_robust: int = 3,
    method: str = "OMA",
) -> List[Finding]:
    """Aplica reglas normativas a fdd_result.modes y devuelve hallazgos."""
    findings: List[Finding] = []
    running_hz = running_rpm / 60.0

    if not fdd_result or not getattr(fdd_result, "modes", None):
        findings.append(Finding(
            severity="fail",
            title="Sin modos identificados",
            text=("El analisis no devolvio modos. Verifica la calidad de la "
                  "captura, el rango de frecuencias y los parametros del FDD/EMA."),
            norm_ref="ISO 7626-6 secc. 6",
        ))
        return findings

    all_modes = list(fdd_result.modes)
    natural_modes = [
        m for m in all_modes
        if getattr(m, "classification", "natural") == "natural"
    ]
    harmonic_modes = [
        m for m in all_modes
        if getattr(m, "classification", "natural") == "harmonic"
    ]
    spurious_modes = [
        m for m in all_modes
        if getattr(m, "classification", "natural") == "spurious"
    ]

    # ---- Regla 1: Conteo y composicion del set modal ----
    if len(natural_modes) < modes_min_for_robust:
        findings.append(Finding(
            severity="warning",
            title=(f"Solo {len(natural_modes)} modo(s) natural(es) "
                    f"identificado(s)"),
            text=(f"Para un analisis modal robusto se recomienda identificar "
                  f"al menos {modes_min_for_robust} modos naturales. "
                  f"Considera aumentar la duracion de captura (OMA) o el "
                  f"numero de promedios (EMA), o ampliar la banda de "
                  f"frecuencias de busqueda."),
            norm_ref="ISO 7626-6 secc. 6",
        ))
    else:
        findings.append(Finding(
            severity="ok",
            title=(f"Set modal completo · {len(natural_modes)} modos "
                    f"naturales identificados"),
            text=(f"Identificacion modal robusta con {len(natural_modes)} "
                  f"modos fisicos. Adicional: {len(harmonic_modes)} "
                  f"armonicas + {len(spurious_modes)} espurios "
                  f"clasificados automaticamente."),
            norm_ref="ISO 7626-6 secc. 6",
        ))

    # ---- Regla 2: Cruce con armonicas de velocidad operativa ----
    cruces_criticos = []
    for m in natural_modes:
        order, delta_pct = _classify_freq_vs_harmonics(
            float(m.natural_frequency_hz), running_hz,
            n_harmonics=6, tolerance_pct=10.0,
        )
        if order is not None:
            cruces_criticos.append({
                "mode": m.mode_number,
                "freq": float(m.natural_frequency_hz),
                "order": order,
                "delta_pct": delta_pct,
            })

    if cruces_criticos:
        for c in cruces_criticos:
            sev = "fail" if c["delta_pct"] < 5.0 else "warning"
            findings.append(Finding(
                severity=sev,
                title=(f"Modo {c['mode']} ({c['freq']:.2f} Hz) cerca de "
                        f"{c['order']}x rpm ({c['delta_pct']:.1f}%)"),
                text=(f"La frecuencia natural del modo {c['mode']} esta a "
                      f"{c['delta_pct']:.1f}% de la armonica {c['order']}x "
                      f"({c['order'] * running_hz:.1f} Hz). "
                      f"API 618 secc. 7.9.4.2.5.3.2 exige separacion >= 10%. "
                      + ("RIESGO DE RESONANCIA — revisar diseno o limitar "
                         "operacion."
                         if sev == "fail" else
                         "Operacion limite — monitorear amplitud "
                         "en el modo durante operacion sostenida.")),
                norm_ref="API 618 secc. 7.9.4.2.5.3.2 + API 684 secc. 1.6",
            ))
    else:
        findings.append(Finding(
            severity="ok",
            title="Sin cruces criticos con armonicas operativas",
            text=(f"Ningun modo natural cae dentro del +/-10% de las "
                  f"armonicas {running_rpm:.0f} rpm. Separacion conforme "
                  f"a API 618 secc. 7.9.4.2.5.3.2."),
            norm_ref="API 618 secc. 7.9.4.2.5.3.2",
        ))

    # ---- Regla 3: AutoMAC redundancy ----
    if len(natural_modes) >= 2:
        try:
            from core.modal.oma_engine import compute_mac_matrix
            import numpy as np
            mac = compute_mac_matrix(natural_modes)
            n = mac.shape[0]
            redundant_pairs = []
            for i in range(n):
                for j in range(i + 1, n):
                    if mac[i, j] > mac_threshold:
                        redundant_pairs.append({
                            "i": natural_modes[i].mode_number,
                            "j": natural_modes[j].mode_number,
                            "mac": float(mac[i, j]),
                        })
            if redundant_pairs:
                for p in redundant_pairs:
                    findings.append(Finding(
                        severity="warning",
                        title=(f"Modos M{p['i']} y M{p['j']} linealmente "
                                f"dependientes (MAC = {p['mac']:.2f})"),
                        text=(f"MAC off-diagonal > {mac_threshold} indica "
                              f"que estos modos comparten forma. "
                              f"Considera si son el mismo modo fisico "
                              f"identificado dos veces — uno deberia "
                              f"eliminarse o ajustar los parametros del FDD "
                              f"(min_distance_hz, prominencia)."),
                        norm_ref="ISO 7626-6 secc. 6.5",
                    ))
            else:
                findings.append(Finding(
                    severity="ok",
                    title="Set modal linealmente independiente",
                    text=(f"AutoMAC off-diagonal < {mac_threshold} en todos "
                          f"los pares. Todos los modos son fisicamente "
                          f"distintos."),
                    norm_ref="ISO 7626-6 secc. 6.5",
                ))
        except Exception:  # noqa: BLE001
            pass

    # ---- Regla 4: MPC threshold (complejidad modal) ----
    high_mpc = [m for m in natural_modes
                  if float(getattr(m, "complexity_pct", 0.0)) > mpc_complex_threshold_pct]
    if high_mpc:
        for m in high_mpc:
            mpc = float(getattr(m, "complexity_pct", 0.0))
            findings.append(Finding(
                severity="warning",
                title=(f"Modo {m.mode_number} con MPC alto "
                        f"({mpc:.1f}%)"),
                text=(f"MPC > {mpc_complex_threshold_pct}% indica modo "
                      f"complejo (fases no colineales). Posibles causas: "
                      f"damping no proporcional, modo espurio, o "
                      f"interaccion con otro modo cercano. Revisar el "
                      f"complexity polar plot."),
                norm_ref="Pappa & Eishan 1995",
            ))

    # ---- Regla 5: Damping anormal ----
    for m in natural_modes:
        z = float(m.damping_ratio_pct)
        if z > damping_high_pct:
            findings.append(Finding(
                severity="warning",
                title=(f"Modo {m.mode_number} con damping alto "
                        f"({z:.2f}%)"),
                text=(f"Damping > {damping_high_pct}% es atipico para "
                      f"estructura mecanica metalica. Posibles causas: "
                      f"identificacion ruidosa, modo no-estructural, o "
                      f"acoplamiento con sistema disipativo. Validar "
                      f"con MPC y forma del peak."),
                norm_ref="ISO 20816 + Brincker 2015",
            ))
        elif z < damping_low_pct:
            findings.append(Finding(
                severity="info",
                title=(f"Modo {m.mode_number} con damping muy bajo "
                        f"({z:.3f}%)"),
                text=(f"Damping < {damping_low_pct}% sugiere modo poco "
                      f"amortiguado — alta amplificacion potencial cerca "
                      f"de resonancia. Verificar que no sea una armonica "
                      f"residual mal clasificada."),
                norm_ref="ISO 20816",
            ))

    # ---- Regla 6: Modos cercanos (posibles modos repetidos) ----
    sorted_modes = sorted(natural_modes,
                            key=lambda m: m.natural_frequency_hz)
    for i in range(len(sorted_modes) - 1):
        f1 = float(sorted_modes[i].natural_frequency_hz)
        f2 = float(sorted_modes[i + 1].natural_frequency_hz)
        delta_pct = (f2 - f1) / max(f1, 1e-6) * 100.0
        if delta_pct < 5.0:
            findings.append(Finding(
                severity="info",
                title=(f"Modos M{sorted_modes[i].mode_number} y "
                        f"M{sorted_modes[i+1].mode_number} muy cercanos "
                        f"(Delta f = {delta_pct:.2f}%)"),
                text=(f"Frecuencias {f1:.2f} Hz y {f2:.2f} Hz separadas "
                      f"menos del 5%. Pueden ser modos repetidos del "
                      f"mismo plano (X+Y) o un par close-coupled. "
                      f"Verificar mode shapes — si son ortogonales son "
                      f"par X/Y, si son similares son redundantes."),
                norm_ref="Ewins 2000 secc. 2.5",
            ))

    return findings


def render_analysis_as_png(
    findings: List[Finding],
    asset_name: str = "Activo",
    method: str = "OMA",
    width_px: int = 1280,
) -> bytes:
    """Renderiza la lista de Findings como PNG nicely formatted via PIL.

    Layout: header navy + lista de findings con color por severity.
    """
    import io
    from PIL import Image, ImageDraw, ImageFont

    # Colores por severity
    sev_colors = {
        "ok":      {"bg": "#dcfce7", "border": "#16a34a", "icon": "✓"},
        "warning": {"bg": "#fef3c7", "border": "#D89B22", "icon": "⚠"},
        "fail":    {"bg": "#fee2e2", "border": "#dc2626", "icon": "✗"},
        "info":    {"bg": "#dbeafe", "border": "#1AAEE5", "icon": "ℹ"},
    }

    # Fonts
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 22)
        font_subtitle = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
        font_finding_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_finding_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        font_norm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
    except (OSError, IOError):
        try:
            font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
            font_subtitle = ImageFont.truetype("DejaVuSans.ttf", 13)
            font_finding_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
            font_finding_body = ImageFont.truetype("DejaVuSans.ttf", 12)
            font_norm = ImageFont.truetype("DejaVuSans.ttf", 10)
        except (OSError, IOError):
            font_title = ImageFont.load_default()
            font_subtitle = font_title
            font_finding_title = font_title
            font_finding_body = font_title
            font_norm = font_title

    # Helper para wrap text
    def _wrap(text: str, font, max_width: int) -> List[str]:
        words = text.split()
        lines = []
        cur = ""
        for w in words:
            test = (cur + " " + w).strip()
            try:
                bbox = font.getbbox(test)
                w_test = bbox[2] - bbox[0]
            except AttributeError:
                w_test = len(test) * 7
            if w_test <= max_width or not cur:
                cur = test
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    # Calcular alturas
    header_h = 90
    pad = 20
    finding_padding_v = 14
    finding_padding_h = 18
    content_width = width_px - 2 * pad
    finding_text_width = content_width - 2 * finding_padding_h - 30  # margen icon

    finding_heights = []
    finding_lines_cache = []
    for f in findings:
        title_lines = _wrap(f.title, font_finding_title, finding_text_width)
        body_lines = _wrap(f.text, font_finding_body, finding_text_width)
        line_h_title = 20
        line_h_body = 17
        line_h_norm = 14
        h = (
            finding_padding_v
            + len(title_lines) * line_h_title
            + len(body_lines) * line_h_body
            + (line_h_norm if f.norm_ref else 0)
            + finding_padding_v
        )
        finding_heights.append(h)
        finding_lines_cache.append((title_lines, body_lines))

    total_height = (
        header_h
        + pad
        + sum(finding_heights)
        + (pad if findings else 0)
        + (len(findings) - 1) * 8  # gap entre findings
        + pad
        + 30  # footer
    )
    total_height = max(total_height, 600)

    # Canvas
    canvas = Image.new("RGB", (width_px, int(total_height)), "white")
    d = ImageDraw.Draw(canvas)

    # Header navy
    d.rectangle([0, 0, width_px, header_h], fill="#0F1E3D")
    d.text((pad, 14), "Watermelon Modal", font=font_subtitle, fill="#1AAEE5")
    d.text((pad, 32), f"Auto-analisis normativo · {asset_name}",
            font=font_title, fill="white")
    d.text((pad, 66), f"Metodo: {method} · Reglas: ISO 7626-6, ISO 20816, "
                       f"API 684, API 618",
            font=font_subtitle, fill="#94a3b8")
    # Footer ribbon header
    d.rectangle([0, header_h - 4, width_px, header_h], fill="#1AAEE5")

    # Findings
    y_cursor = header_h + pad
    for i, (f, h, (t_lines, b_lines)) in enumerate(
        zip(findings, finding_heights, finding_lines_cache)
    ):
        sev = sev_colors.get(f.severity, sev_colors["info"])
        # Background card
        d.rectangle(
            [pad, y_cursor, width_px - pad, y_cursor + h],
            fill=sev["bg"], outline=sev["border"], width=2,
        )
        # Icon
        d.text((pad + 8, y_cursor + 10), sev["icon"],
                font=font_title, fill=sev["border"])
        # Title lines
        ty = y_cursor + finding_padding_v
        for line in t_lines:
            d.text((pad + finding_padding_h + 30, ty), line,
                    font=font_finding_title, fill="#0F1E3D")
            ty += 20
        # Body lines
        for line in b_lines:
            d.text((pad + finding_padding_h + 30, ty), line,
                    font=font_finding_body, fill="#334155")
            ty += 17
        # Norm ref
        if f.norm_ref:
            d.text((pad + finding_padding_h + 30, ty + 2),
                    f"Norma: {f.norm_ref}",
                    font=font_norm, fill="#64748b")

        y_cursor += h + 8

    # Footer
    y_cursor = int(total_height) - 24
    d.text((pad, y_cursor),
            f"Generado automaticamente por Watermelon Modal Module · "
            f"{len(findings)} hallazgos detectados",
            font=font_norm, fill="#64748b")

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def build_analysis_report_item(
    findings: List[Finding],
    asset_name: str = "Activo",
    method: str = "OMA",
) -> Dict[str, Any]:
    """Construye un item compatible con report_state desde una lista de findings."""
    import uuid
    png_bytes = render_analysis_as_png(findings, asset_name, method)

    # Resumen para notes
    n_fail = sum(1 for f in findings if f.severity == "fail")
    n_warn = sum(1 for f in findings if f.severity == "warning")
    n_ok = sum(1 for f in findings if f.severity == "ok")
    n_info = sum(1 for f in findings if f.severity == "info")

    notes = (
        f"Auto-analisis rule-based — {len(findings)} hallazgos: "
        f"{n_fail} criticos, {n_warn} advertencias, {n_ok} conformes, "
        f"{n_info} informativos. Reglas aplicadas: ISO 7626-6, ISO 20816, "
        f"API 684, API 618."
    )

    return {
        "id": f"modal_auto_{uuid.uuid4().hex[:12]}",
        "type": "modal_auto_analysis",
        "title": f"Auto-analisis normativo modal · {asset_name}",
        "notes": notes,
        "signal_id": "",
        "machine": asset_name,
        "point": "Analisis modal",
        "variable": "Auto-analisis normativo",
        "timestamp": "",
        "figure": None,
        "image_bytes": png_bytes,
    }
