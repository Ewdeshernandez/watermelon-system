"""core.billing — suscripción del cliente: planes, precios, facturas, pagos.

pricing  → planes por máquina, descuento por volumen, USD, anual (puro)
periods  → periodos, numeración y vencimientos (puro)
store    → tablas Supabase wm_subscriptions / wm_invoices / wm_billing_settings
cycle    → facturación automática del cron (run_due) + avisos por correo
"""
