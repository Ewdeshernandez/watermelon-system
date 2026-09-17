-- Watermelon Modal — IP + ubicación aproximada de la corrida subida desde el campo.
-- Trazabilidad INFORMATIVA: desde qué IP pública y ubicación (aprox. por IP, nivel ISP)
-- se subió cada corrida. Lo captura el PC de campo al subir (v0.9.87+) y lo muestra el
-- aviso por correo (notify-run) y la web. Aplica en el SQL editor de Supabase. Idempotente.

alter table public.modal_runs add column if not exists ip  text;   -- IP pública del PC de campo al subir
alter table public.modal_runs add column if not exists geo text;   -- 'Ciudad, Región, PAÍS' aprox. por IP
