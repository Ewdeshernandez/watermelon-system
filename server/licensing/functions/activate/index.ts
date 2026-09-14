// Supabase Edge Function: activate (modelo por CLAVE DE LICENCIA — sin login/OTP)
// El cliente envía {license_key, machine_fp}; se valida la clave, cupo y binding, y se
// devuelve un TOKEN firmado (Ed25519). Requiere "Verify JWT" = OFF en la config.
// Secret necesario: WM_LICENSE_PRIVKEY.
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const b64u = (b: Uint8Array) =>
  btoa(String.fromCharCode(...b)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const LICENSE_PUBKEY = "6yd-Rfp0GEdlFo_hLZ3O0oQD890vc_ylecJi4TyYWzA";

async function signToken(payload: Record<string, unknown>): Promise<string> {
  const jwk = { kty: "OKP", crv: "Ed25519",
    d: Deno.env.get("WM_LICENSE_PRIVKEY")!, x: LICENSE_PUBKEY, key_ops: ["sign"], ext: true };
  const key = await crypto.subtle.importKey("jwk", jwk, { name: "Ed25519" }, false, ["sign"]);
  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, key, bytes));
  return `${b64u(bytes)}.${b64u(sig)}`;
}

Deno.serve(async (req) => {
  try {
    const { license_key, machine_fp, is_vm } = await req.json();
    if (!license_key || !machine_fp)
      return new Response(JSON.stringify({ error: "missing_fields" }), { status: 400 });
    const admin = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
    const { data: lic } = await admin.from("licenses").select("*")
      .eq("key", license_key).eq("status", "active").maybeSingle();
    if (!lic) return new Response(JSON.stringify({ error: "invalid_key" }), { status: 403 });
    if (new Date(lic.expires_at).getTime() < Date.now())
      return new Response(JSON.stringify({ error: "license_expired" }), { status: 403 });
    const { data: acts } = await admin.from("activations").select("*").eq("license_id", lic.id);
    const existing = acts?.find((a: any) => a.machine_fp === machine_fp);
    if (existing?.revoked) return new Response(JSON.stringify({ error: "machine_revoked" }), { status: 403 });
    if (!existing && (acts?.filter((a: any) => !a.revoked).length ?? 0) >= lic.seats)
      return new Response(JSON.stringify({ error: "no_seats" }), { status: 403 });
    await admin.from("activations").upsert({ license_id: lic.id, account: lic.account, machine_fp,
      is_vm: !!is_vm, last_seen: new Date().toISOString() }, { onConflict: "license_id,machine_fp" });
    const exp = Math.min(Date.now() / 1000 + 14 * 86400, new Date(lic.expires_at).getTime() / 1000);
    const token = await signToken({ account: lic.account, machine_fp, exp, seat: lic.seats,
      features: lic.features, iat: Date.now() / 1000 });
    return new Response(JSON.stringify({ token, exp, account: lic.account }),
      { headers: { "Content-Type": "application/json" } });
  } catch (e) { return new Response(JSON.stringify({ error: String(e) }), { status: 500 }); }
});
