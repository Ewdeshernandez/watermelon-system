// Supabase Edge Function: activate (modelo por CLAVE DE LICENCIA — sin login/OTP)
// El cliente envía {license_key, machine_fp, is_vm, hostname, app}; se valida la clave,
// el estado comercial, el cupo y el binding, se registra la activación (PC + IP + geo +
// última conexión) y se devuelve un TOKEN firmado (Ed25519).
// Requiere "Verify JWT" = OFF en la config. Secret necesario: WM_LICENSE_PRIVKEY.
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const b64u = (b: Uint8Array) =>
  btoa(String.fromCharCode(...b)).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
const LICENSE_PUBKEY = "ozjVKHml8OE4E-1h07evyw5IOcGnO-IbdB7lV_OUaeM";

async function signToken(payload: Record<string, unknown>): Promise<string> {
  const jwk = { kty: "OKP", crv: "Ed25519",
    d: Deno.env.get("WM_LICENSE_PRIVKEY")!, x: LICENSE_PUBKEY, key_ops: ["sign"], ext: true };
  const key = await crypto.subtle.importKey("jwk", jwk, { name: "Ed25519" }, false, ["sign"]);
  const bytes = new TextEncoder().encode(JSON.stringify(payload));
  const sig = new Uint8Array(await crypto.subtle.sign({ name: "Ed25519" }, key, bytes));
  return `${b64u(bytes)}.${b64u(sig)}`;
}

// IP pública del cliente (Supabase/Cloudflare pone x-forwarded-for).
function clientIp(req: Request): string {
  const xff = req.headers.get("x-forwarded-for") || "";
  const first = xff.split(",")[0].trim();
  return first || req.headers.get("x-real-ip") || "";
}

// Geolocalización best-effort de la IP → "Ciudad, PAÍS". Nunca rompe la activación.
async function geoLookup(ip: string): Promise<string> {
  if (!ip) return "";
  try {
    const ctrl = new AbortController();
    const to = setTimeout(() => ctrl.abort(), 2500);
    const r = await fetch(`https://ipapi.co/${ip}/json/`, { signal: ctrl.signal });
    clearTimeout(to);
    if (!r.ok) return "";
    const j = await r.json();
    const city = j.city || j.region || "";
    const cc = j.country_code || j.country || "";
    return [city, cc].filter(Boolean).join(", ");
  } catch (_e) { return ""; }
}

Deno.serve(async (req) => {
  try {
    const { license_key, machine_fp, is_vm, hostname, app } = await req.json();
    if (!license_key || !machine_fp)
      return new Response(JSON.stringify({ error: "missing_fields" }), { status: 400 });
    const admin = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);

    // Buscar por clave SIN filtrar status → así podemos distinguir suspendida/revocada
    // y devolver el motivo correcto (el cliente muestra el letrero adecuado).
    const { data: lic } = await admin.from("licenses").select("*")
      .eq("key", license_key).maybeSingle();
    if (!lic) return new Response(JSON.stringify({ error: "invalid_key" }), { status: 403 });

    // Estado comercial de la licencia completa.
    if (lic.status === "suspended")
      return new Response(JSON.stringify({ error: "payment_due",
        reason: lic.suspended_reason || "Licencia no renovada por falta de pago" }), { status: 402 });
    if (lic.status === "revoked")
      return new Response(JSON.stringify({ error: "license_revoked" }), { status: 403 });
    if (lic.status !== "active")
      return new Response(JSON.stringify({ error: "invalid_key" }), { status: 403 });
    if (new Date(lic.expires_at).getTime() < Date.now())
      return new Response(JSON.stringify({ error: "license_expired" }), { status: 403 });

    const { data: acts } = await admin.from("activations").select("*").eq("license_id", lic.id);
    const existing = acts?.find((a: any) => a.machine_fp === machine_fp);
    if (existing?.revoked) return new Response(JSON.stringify({ error: "machine_revoked" }), { status: 403 });
    if (!existing && (acts?.filter((a: any) => !a.revoked).length ?? 0) >= lic.seats)
      return new Response(JSON.stringify({ error: "no_seats" }), { status: 403 });

    // Registrar/actualizar la activación con identidad de equipo + red.
    const ip = clientIp(req);
    const now = new Date().toISOString();
    const row: Record<string, unknown> = {
      license_id: lic.id, account: lic.account, machine_fp,
      is_vm: !!is_vm, hostname: hostname || null, app: app || null,
      last_seen: now, updated_at: now,
    };
    if (ip) {
      row.ip = ip;
      // Geolocaliza sólo si es una IP nueva/cambió (ahorra llamadas). Best-effort.
      if (!existing || existing.ip !== ip) row.ip_geo = await geoLookup(ip);
    }
    await admin.from("activations").upsert(row, { onConflict: "license_id,machine_fp" });

    // Historial (Nivel B): una fila por arranque → traza de auditoría (módulo/IP/hora).
    // Best-effort: nunca hace fallar la activación.
    try {
      await admin.from("license_events").insert({
        license_id: lic.id, account: lic.account, machine_fp,
        hostname: hostname || null, app: app || null,
        ip: ip || null, ip_geo: (row.ip_geo as string) || existing?.ip_geo || null,
      });
    } catch (_e) { /* ignore */ }

    const exp = Math.min(Date.now() / 1000 + 30 * 86400, new Date(lic.expires_at).getTime() / 1000);
    const token = await signToken({ account: lic.account, machine_fp, exp, seat: lic.seats,
      features: lic.features, iat: Date.now() / 1000 });
    return new Response(JSON.stringify({ token, exp, account: lic.account }),
      { headers: { "Content-Type": "application/json" } });
  } catch (e) { return new Response(JSON.stringify({ error: String(e) }), { status: 500 }); }
});
