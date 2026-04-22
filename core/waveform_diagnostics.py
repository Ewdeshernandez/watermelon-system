from __future__ import annotations

from typing import Dict

SAFE_COLOR = "#16a34a"
WARNING_COLOR = "#f59e0b"
DANGER_COLOR = "#dc2626"


def build_waveform_report_notes(text_diag: Dict[str, str]) -> str:
    headline = str(text_diag.get("headline", "") or "").strip()
    narrative = str(text_diag.get("narrative", "") or "").strip()

    blocks = []
    if headline:
        blocks.append(headline)
    if narrative:
        blocks.append(narrative)
    return "\n\n".join(blocks).strip()



def _waveform_machine_context_adjustment(machine: str, narrative: str) -> str:
    machine_upper = (machine or "").upper()

    is_recip = "RECIP" in machine_upper or "RECIPROC" in machine_upper
    is_compressor = "COMPRESOR" in machine_upper or "COMPRESSOR" in machine_upper
    is_plain_bearing_context = is_recip or is_compressor

    updated = narrative

    if is_plain_bearing_context:
        updated = updated.replace("rodamientos", "cojinetes planos")
        updated = updated.replace("rodamiento", "cojinete plano")
        updated = updated.replace(
            "presencia moderada de transitorios de alta energía",
            "presencia moderada de transitorios de alta energía y excitaciones no lineales"
        )
        updated = updated.replace(
            "Se recomienda correlacionar este comportamiento con Spectrum, condiciones operativas y estado de lubricación antes de definir intervención.",
            "Se recomienda correlacionar este comportamiento con Spectrum, condiciones operativas, holgura dinámica y estado de lubricación del cojinete plano antes de definir intervención."
        )

    return updated

def generate_waveform_diagnostic(record, summary: Dict[str, float]) -> Dict[str, str]:
    machine = str(getattr(record, "machine", "") or "máquina no identificada")
    point = str(getattr(record, "point", "") or "punto no identificado")

    cf = summary.get("Crest Factor")
    rms = summary.get("RMS")
    pp = summary.get("Peak-Peak")

    if cf is None:
        return {
            "status": "SAFE",
            "color": SAFE_COLOR,
            "headline": "Sin diagnóstico válido de forma de onda",
            "narrative": (
                f"No fue posible generar un diagnóstico confiable para la forma de onda correspondiente al punto {point} "
                f"en la máquina {machine} por falta de datos válidos."
            ),
        }

    try:
        cf = float(cf)
    except Exception:
        cf = float("nan")

    try:
        rms_txt = f"{float(rms):.3f}"
    except Exception:
        rms_txt = "—"

    try:
        pp_txt = f"{float(pp):.3f}"
    except Exception:
        pp_txt = "—"

    if cf != cf:
        return {
            "status": "SAFE",
            "color": SAFE_COLOR,
            "headline": "Sin diagnóstico válido de forma de onda",
            "narrative": (
                f"No fue posible generar un diagnóstico confiable para la forma de onda correspondiente al punto {point} "
                f"en la máquina {machine}."
            ),
        }

    if cf < 3.0:
        return {
            "status": "SAFE",
            "color": SAFE_COLOR,
            "headline": "Forma de onda con comportamiento estable",
            "narrative": (
                f"Se analizó la forma de onda de vibración correspondiente al punto {point} en la máquina {machine}. "
                f"La señal presenta un comportamiento estable, con un factor de cresta de {cf:.2f}, un valor RMS de {rms_txt} "
                f"y una amplitud pico a pico de {pp_txt}, sin evidencia significativa de transitorios de alta energía o comportamiento no lineal. "
                f"Se recomienda continuar el monitoreo periódico y correlacionar este resultado con el análisis espectral para seguimiento de condición."
            ),
        }

    if cf <= 5.0:
        return {
            "status": "WARNING",
            "color": WARNING_COLOR,
            "headline": "Forma de onda con indicios moderados de transitorios de alta energía",
            "narrative": (
                f"Se analizó la forma de onda de vibración correspondiente al punto {point} en la máquina {machine}. "
                f"La señal presenta un comportamiento ligeramente no lineal, con un factor de cresta de {cf:.2f}, un valor RMS de {rms_txt} "
                f"y una amplitud pico a pico de {pp_txt}, lo cual sugiere presencia moderada de transitorios de alta energía. "
                f"Se recomienda correlacionar este comportamiento con Spectrum, condiciones operativas y estado de lubricación antes de definir intervención."
            ),
        }

    return {
        "status": "DANGER",
        "color": DANGER_COLOR,
        "headline": "Forma de onda con comportamiento altamente impulsivo",
        "narrative": (
            f"Se analizó la forma de onda de vibración correspondiente al punto {point} en la máquina {machine}. "
            f"La señal presenta un comportamiento altamente impulsivo, con un factor de cresta de {cf:.2f}, un valor RMS de {rms_txt} "
            f"y una amplitud pico a pico de {pp_txt}, condición compatible con transitorios mecánicos, contacto intermitente o daño incipiente en elementos rodantes. "
            f"Se recomienda priorizar la inspección mecánica, validar la condición del rodamiento y correlacionar con Envelope Spectrum antes de operación repetida."
        ),
    }
