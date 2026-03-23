from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
PAGES_DIR = ROOT_DIR / "pages"


PAGE_FILE_MAP = {
    "tabular": "01__Tabular_List.py",
    "waveform": "02_Time_Waveforms.py",
    "spectrum": "03_Spectrum.py",
    "trends": "04_Trends.py",
    "orbit": "05_Orbit_Analysis.py",
    "reports": "16_Reports.py",
}


def _module_name_for_key(page_key: str) -> str:
    return f"wm_runtime_bridge_{page_key}"


@lru_cache(maxsize=None)
def load_page_module(page_key: str) -> ModuleType:
    if page_key not in PAGE_FILE_MAP:
        raise KeyError(f"Unknown page key: {page_key}")

    file_name = PAGE_FILE_MAP[page_key]
    file_path = PAGES_DIR / file_name

    if not file_path.exists():
        raise FileNotFoundError(f"Page file not found: {file_path}")

    module_name = _module_name_for_key(page_key)

    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {file_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_page_attr(page_key: str, attr_name: str, default=None):
    try:
        module = load_page_module(page_key)
        return getattr(module, attr_name, default)
    except Exception:
        return default


def has_page_attr(page_key: str, attr_name: str) -> bool:
    try:
        module = load_page_module(page_key)
        return hasattr(module, attr_name)
    except Exception:
        return False


def get_tabular_module() -> Optional[ModuleType]:
    try:
        return load_page_module("tabular")
    except Exception:
        return None


def get_waveform_module() -> Optional[ModuleType]:
    try:
        return load_page_module("waveform")
    except Exception:
        return None


def get_spectrum_module() -> Optional[ModuleType]:
    try:
        return load_page_module("spectrum")
    except Exception:
        return None


def get_orbit_module() -> Optional[ModuleType]:
    try:
        return load_page_module("orbit")
    except Exception:
        return None


def get_trends_module() -> Optional[ModuleType]:
    try:
        return load_page_module("trends")
    except Exception:
        return None


def get_reports_module() -> Optional[ModuleType]:
    try:
        return load_page_module("reports")
    except Exception:
        return None


def get_waveform_builder():
    return get_page_attr("waveform", "build_waveform_figure")


def get_spectrum_builder():
    return get_page_attr("spectrum", "build_spectrum_figure")


def get_orbit_builder():
    return get_page_attr("orbit", "build_orbit_figure")


def get_trend_builder():
    return get_page_attr("trends", "build_trend_figure")


def get_waveform_signal_record():
    return get_page_attr("waveform", "SignalRecord")


def get_spectrum_signal_record():
    return get_page_attr("spectrum", "SignalRecord")


def get_orbit_signal_record():
    return get_page_attr("orbit", "SignalRecord")


def get_trend_signal_record():
    return get_page_attr("trends", "SignalRecord")
