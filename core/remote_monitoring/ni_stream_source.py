"""
core/remote_monitoring/ni_stream_source.py — Fuente real NI (streaming)
=======================================================================

StreamSource sobre nidaqmx en modo CONTINUO. Corre SOLO en Windows/Linux
con NI-DAQmx instalado (el 9178 es USB). Import lazy → no rompe en Mac ni
en Cloud; solo falla al llamar start() sin driver.

Reusa las convenciones de core/modal/acq_backend:
  · _build_phys_channel: ChannelConfig → "cDAQ1Mod{slot}/ai{idx}"
  · IEPE  → add_ai_accel_chan (con sensitivity)
  · AC/DC → add_ai_voltage_chan (+ ai_coupling)

Hardware objetivo: cDAQ-9178 (8 slots) + N×NI-9234 (4ch IEPE 24-bit 51.2kHz).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from core.modal.acq_backend import _build_phys_channel, _nearest_valid_rate
from core.remote_monitoring.stream_source import StreamConfig, StreamSource


class NIStreamSource(StreamSource):
    """Streaming continuo desde el chasis NI vía nidaqmx.

    A diferencia de acq_backend._capture_oma (que graba a TDMS y termina),
    esta fuente NO tiene fin: read_block() se llama en loop desde el Agent.
    El buffer interno del driver (samps_per_chan grande) absorbe jitter
    entre lecturas para evitar overruns.
    """

    def __init__(self, config: StreamConfig, read_timeout_s: float = 10.0) -> None:
        super().__init__(config)
        self._task = None
        self._read_timeout_s = read_timeout_s

    def start(self) -> None:
        try:
            import nidaqmx
            from nidaqmx.constants import AcquisitionType, Coupling
        except ImportError as exc:  # noqa: F841
            raise ImportError(
                "nidaqmx no está instalado. NIStreamSource solo corre en el PC "
                "de sitio (Windows/Linux con NI-DAQmx). En Mac/dev usá "
                "SimulatedStreamSource."
            ) from exc

        cfg = self.config
        fs = _nearest_valid_rate(cfg.sample_rate_hz)
        chunk = cfg.block_samples

        task = nidaqmx.Task()
        try:
            for ch in cfg.channels:
                phys = _build_phys_channel(cfg.chassis_name, ch)
                coup = (ch.coupling or "").upper()
                if coup == "IEPE":
                    task.ai_channels.add_ai_accel_chan(
                        phys, sensitivity=ch.sensitivity_mv_per_eu,
                        max_val=ch.voltage_range, min_val=-ch.voltage_range,
                    )
                else:
                    vch = task.ai_channels.add_ai_voltage_chan(
                        phys, max_val=ch.voltage_range, min_val=-ch.voltage_range,
                    )
                    # Coupling AC/DC best-effort (el 9234 lo soporta;
                    # otros módulos pueden tenerlo fijo → ignora el error)
                    try:
                        vch.ai_coupling = Coupling.AC if coup == "AC" else Coupling.DC
                    except Exception:  # noqa: BLE001
                        pass

            task.timing.cfg_samp_clk_timing(
                rate=fs,
                sample_mode=AcquisitionType.CONTINUOUS,
                samps_per_chan=chunk * 4,  # buffer NI 4× el bloque → anti-overrun
            )
            task.start()
        except Exception:
            try:
                task.close()
            except Exception:  # noqa: BLE001
                pass
            raise

        self._task = task
        # fs real puede diferir del pedido (rate válido más cercano)
        self.config.sample_rate_hz = float(fs)
        self._running = True

    def read_block(self) -> np.ndarray:
        if not self._running or self._task is None:
            raise RuntimeError("NIStreamSource no está corriendo (llama start())")
        n = self.config.block_samples
        data = self._task.read(number_of_samples_per_channel=n,
                               timeout=self._read_timeout_s)
        # nidaqmx devuelve lista-de-listas (multi-canal) o lista plana (1 canal)
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 1:
            arr = arr[None, :]
        return arr

    def stop(self) -> None:
        self._running = False
        if self._task is not None:
            try:
                self._task.stop()
            except Exception:  # noqa: BLE001
                pass
            try:
                self._task.close()
            except Exception:  # noqa: BLE001
                pass
            self._task = None
