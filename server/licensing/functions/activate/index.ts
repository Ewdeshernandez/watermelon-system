// Supabase Edge Function: activate
// El cliente (login Supabase) envía su huella de máquina; se valida cupo, se registra la
// activación y se devuelve un TOKEN firmado (Ed25519) con la llave PRIVADA del servidor.
// Deploy: supabase functions deploy activate
//   secrets: WM_LICENSE_PRIVKEY (base64url raw 32B), SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const b64u = (b: Uint8Array) =>
  btoa(String.fromCharCode(...b)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const b64uDec = (s: string) =>
  Uint8Array.from(atob(s.replace(/-/g, "+").replace(/_/g, "/")), (c) => c.charCodeAt(0));

// Llave pública (x) — pública por diseño; debe coincidir con la del .exe.
const LICENSE_PUBKEY = "6yd-Rfp0GEdlFo_hLZ3O0oQD890vc_ylecJi4TyYWzA";

async function signToken(payload: Record<string, unknown>): Promise<string> {
  // Ed25519 en WebCrypto: la privada se importa como JWK (d = seed raw, x = pública).
  const jwk = {
    kty: "OKP", crv: "Ed25519",
    d: Deno.env.get("WM_LICENSE_PRIVKEY")!, x: LICENSE_PUBKEY,
    key_ops: ["sign"], ext: true,
  };
  const key = await crypto.subtle.importKey("jwk", jwk, { name: "Ed25519" }, false, ["sign"]);
  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, key, bytes));
  return `${b64u(bytes)}.${b64u(sig)}`;
}

Deno.serve(async (req) => {
  try {
    const { machine_fp, is_vm } = await req.json();
    if (!machine_fp) return new Response("machine_fp required", { status: 400 });

    // Usuario autenticado (JWT del header Authorization)
    const authed = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_ANON_KEY")!, {
      global: { headers: { Authorization: req.headers.get("Authorization")! } },
    });
    const { data: u } = await authed.auth.getUser();
    const email = u?.user?.email;
    if (!email) return new Response("unauthorized", { status: 401 });

    // service role para leer licencias / escribir activaciones (bypassa RLS en el server)
    const admin = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
    const { data: lic } = await admin.from("licenses").select("*")
      .eq("account", email).eq("status", "active").order("expires_at", { ascending: false }).limit(1).single();
    if (!lic) return new Response(JSON.stringify({ error: "no_active_license" }), { status: 403 });
    if (new Date(lic.expires_at).getTime() < Date.now())
      return new Response(JSON.stringify({ error: "license_expired" }), { status: 403 });

    // ¿ya activada esta máquina? si no, revisa cupo
    const { data: acts } = await admin.from("activations").select("*").eq("license_id", lic.id);
    const existing = acts?.find((a: any) => a.machine_fp === machine_fp);
    if (existing?.revoked) return new Response(JSON.stringify({ error: "machine_revoked" }), { status: 403 });
    if (!existing && (acts?.filter((a: any) => !a.revoked).length ?? 0) >= lic.seats)
      return new Response(JSON.stringify({ error: "no_seats" }), { status: 403 });

    await admin.from("activations").upsert({
      license_id: lic.id, account: email, machine_fp, is_vm: !!is_vm, last_seen: new Date().toISOString(),
    }, { onConflict: "license_id,machine_fp" });

    // Token de corta vida (re-check periódico); exp <= expiración de la licencia
    const exp = Math.min(Date.now() / 1000 + 14 * 86400, new Date(lic.expires_at).getTime() / 1000);
    const token = await signToken({
      account: email, machine_fp, exp, seat: lic.seats, features: lic.features, iat: Date.now() / 1000,
    });
    return new Response(JSON.stringify({ token, exp }), { headers: { "Content-Type": "application/json" } });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), { status: 500 });
  }
});
