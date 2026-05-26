"""
core/modal/ni_daq.py — Shim back-compat (DEPRECATED)
=====================================================

v3.31.245 — Este módulo fue renombrado a `core/modal/acq_backend.py`.

Para nuevos imports usar:

    from core.modal.acq_backend import (
        ChannelConfig, AcquisitionConfig,
        discover_acq_modules, capture_oma, capture_ema,
        ...
    )

Este shim re-exporta todo desde el nuevo módulo para mantener
compatibilidad con código viejo que aún haga `from core.modal.ni_daq import ...`.

A eliminar cuando todos los consumers hayan migrado a `acq_backend`.
"""
from core.modal.acq_backend import *  # noqa: F401,F403
from core.modal.acq_backend import (  # noqa: F401 — explicit re-exports
    ChannelConfig,
    AcquisitionConfig,
    discover_acq_modules,
)

# Aliases legacy (back-compat con nombres antiguos)
try:
    from core.modal.acq_backend import discover_acq_modules as discover_ni9234_modules  # noqa: F401
    from core.modal.acq_backend import _ACQ_VALID_RATES as _NI9234_VALID_RATES  # noqa: F401
except ImportError:
    pass
