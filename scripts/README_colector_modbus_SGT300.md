# ⚠️ OBSOLETO — usar el colector oficial

Este archivo (y `scripts/wm_modbus_collector_sgt300b.py`) quedaron **deprecados**.

SGT300 B/A se conectan en vivo con el **colector oficial** que ya usan TES1/TES3:

- Colector: `collector/wm_collector.py` (genérico, lee mapa JSON → API → Supabase)
- Mapa SGT300B: `data/modbus_maps/sgt300b.json`
- Config: `collector/wm_collector.config.sgt300b.example.json`
- **Guía de despliegue: `collector/DEPLOY_SGT300B.md`**

Podés borrar este archivo y `scripts/wm_modbus_collector_sgt300b.py` con `git rm`.
