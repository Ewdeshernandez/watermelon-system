// Supabase Edge Function: deactivate — libera ESTA máquina de la licencia.
// El cliente envía {license_key, machine_fp}. Se valida la clave y se BORRA la activación
// de esa máquina (libera el cupo para moverla a otra PC). No firma nada.
// Requiere "Verify JWT" = OFF en la config. NO expone datos de otras máquinas.
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

Deno.serve(async (req) => {
  try {
    const { license_key, machine_fp } = await req.json();
    if (!license_key || !machine_fp)
      return new Response(JSON.stringify({ error: "missing_fields" }), { status: 400 });
    const admin = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
    const { data: lic } = await admin.from("licenses").select("id")
      .eq("key", license_key).maybeSingle();
    if (!lic) return new Response(JSON.stringify({ error: "invalid_key" }), { status: 403 });
    // Borra SOLO la activación de esta máquina bajo esta licencia → libera el cupo.
    const { error } = await admin.from("activations").delete()
      .eq("license_id", lic.id).eq("machine_fp", machine_fp);
    if (error) return new Response(JSON.stringify({ error: String(error.message) }), { status: 500 });
    return new Response(JSON.stringify({ ok: true, deactivated: true }),
      { headers: { "Content-Type": "application/json" } });
  } catch (e) { return new Response(JSON.stringify({ error: String(e) }), { status: 500 }); }
});
