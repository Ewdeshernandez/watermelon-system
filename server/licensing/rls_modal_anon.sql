-- Watermelon Modal — Fase 2c: el .exe usa ANON key (NO service key) para subir.
-- ================================================================================
-- El .exe corre headless SIN login → rol `anon`. Estas policies le dan EXACTO lo que
-- necesita (subir setups/runs + data cruda) y NADA más. La WEB (Streamlit) lee/analiza
-- con su service key del lado servidor → hace BYPASS de RLS, así que no se afecta.
--
-- Antes: el .exe embebía el service key = god-mode (podía borrar/leer TODA la base si se
-- extraía del binario). Ahora: anon + RLS = superficie mínima.
--
-- Aplica en el SQL editor de Supabase. Idempotente (drop/create).

-- ---- modal_setups: config compartida campo<->web (el .exe lee y escribe) ----
alter table if exists public.modal_setups enable row level security;
drop policy if exists "anon read setups"   on public.modal_setups;
drop policy if exists "anon write setups"  on public.modal_setups;
drop policy if exists "anon update setups" on public.modal_setups;
create policy "anon read setups"   on public.modal_setups for select to anon using (true);
create policy "anon write setups"  on public.modal_setups for insert to anon with check (true);
create policy "anon update setups" on public.modal_setups for update to anon using (true) with check (true);

-- ---- modal_runs: el .exe SUBE corridas (insert/update/upsert). ----
-- (select permitido a anon para que el upsert devuelva la fila; la data sensible cruda va
--  en Storage, no acá. Tightening futuro: por-licencia con JWT firmado.)
alter table if exists public.modal_runs enable row level security;
drop policy if exists "anon read runs"   on public.modal_runs;
drop policy if exists "anon write runs"  on public.modal_runs;
drop policy if exists "anon update runs" on public.modal_runs;
create policy "anon read runs"   on public.modal_runs for select to anon using (true);
create policy "anon write runs"  on public.modal_runs for insert to anon with check (true);
create policy "anon update runs" on public.modal_runs for update to anon using (true) with check (true);

-- ---- Storage: bucket `modal-raw` (data cruda gzip). anon SOLO sube (insert/update). ----
-- Sin policy de select → nadie descarga con la anon key; la web descarga con service key.
drop policy if exists "anon upload modal-raw" on storage.objects;
drop policy if exists "anon update modal-raw" on storage.objects;
create policy "anon upload modal-raw" on storage.objects for insert to anon
  with check (bucket_id = 'modal-raw');
create policy "anon update modal-raw" on storage.objects for update to anon
  using (bucket_id = 'modal-raw') with check (bucket_id = 'modal-raw');

-- NOTA: el bucket `modal-raw` debe existir y ser PRIVADO (Storage → New bucket, sin marcar
-- "Public"). El .exe ya no puede crear buckets (no es admin) → créalo una vez a mano si falta.
