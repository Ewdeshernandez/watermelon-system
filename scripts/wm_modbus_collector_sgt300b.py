#!/usr/bin/env python3
"""
OBSOLETO — no usar.
===================
SGT300 B/A ahora usan el colector OFICIAL (igual que TES1/TES3):

    collector/wm_collector.py  --config wm_collector.config.json

Mapa Modbus:  data/modbus_maps/sgt300b.json
Config:       collector/wm_collector.config.sgt300b.example.json
Despliegue:   collector/DEPLOY_SGT300B.md

Este archivo se puede borrar con `git rm`.
"""
import sys

if __name__ == "__main__":
    sys.exit(
        "OBSOLETO. Usá: collector/wm_collector.py --config wm_collector.config.json\n"
        "Guía: collector/DEPLOY_SGT300B.md"
    )
