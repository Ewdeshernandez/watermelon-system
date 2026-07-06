// =============================================================
// supabase/functions/planta-auth/index.ts
// Plan B — OTP unificado para Watermelon Planta
// =============================================================
//
// Reemplaza el OTP NATIVO de Supabase (que dependía de la plantilla de
// email "Magic link or OTP") por un OTP propio server-side, IGUAL en
// espíritu al de la app principal (core/auth_otp.py):
//   - El código de 6 dígitos se envía por TU correo (Microsoft Graph,
//     desde ehernandez@sigasas.com), no por el email default de Supabase.
//   - Al verificar, acuñamos una SESIÓN de Supabase (JWT + refresh) y se la
//     devolvemos a la Planta, para que sync_uploader siga subiendo con RLS
//     exactamente como hoy.
//
// El client_secret de Graph y el service_role viven SOLO acá (server),
// nunca en el .exe del cliente.
//
// Endpoint (POST JSON):
//   { "action": "request", "email": "..." }
//     → genera + envía el código. Responde { ok: true } (genérico).
//   { "action": "verify", "email": "...", "code": "123456" }
//     → valida el código y responde
//       { ok: true, access_token, refresh_token, expires_at, user_id, email }
//
// Despliegue:
//   supabase functions deploy planta-auth --no-verify-jwt
//   (--no-verify-jwt: la Planta lo llama con la anon key; la auth real la
//    hace este código validando el código OTP.)
//
// Secrets requeridos (supabase secrets set ...):
//   SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY   → inyectados por Supabase
//   SUPABASE_ANON_KEY                         → anon pública (para verifyOtp)
//   GRAPH_TENANT_ID, GRAPH_CLIENT_ID, GRAPH_CLIENT_SECRET, GRAPH_FROM_EMAIL
//   GRAPH_FROM_NAME (opcional)
//   OTP_SIGNING_SECRET                        → secreto para el HMAC del código
//
// Tabla requerida: planta_otp_challenges (ver migración SQL adjunta).

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SERVICE_ROLE = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";
const ANON_KEY = Deno.env.get("SUPABASE_ANON_KEY") ?? "";

const GRAPH_TENANT = Deno.env.get("GRAPH_TENANT_ID") ?? "";
const GRAPH_CLIENT_ID = Deno.env.get("GRAPH_CLIENT_ID") ?? "";
const GRAPH_CLIENT_SECRET = Deno.env.get("GRAPH_CLIENT_SECRET") ?? "";
const GRAPH_FROM_EMAIL = Deno.env.get("GRAPH_FROM_EMAIL") ?? "";
const GRAPH_FROM_NAME = Deno.env.get("GRAPH_FROM_NAME") ?? "Watermelon System";
const OTP_SIGNING_SECRET = Deno.env.get("OTP_SIGNING_SECRET") ?? "";

const OTP_TTL_SECONDS = 10 * 60; // el código vale 10 min (igual que core/auth_otp)
const MAX_ATTEMPTS = 5;

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization, apikey",
  "Content-Type": "application/json",
  "Cache-Control": "no-store",
};

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: CORS });
}

function normEmail(e: unknown): string {
  return String(e ?? "").trim().toLowerCase();
}

// HMAC-SHA256(secret, message) → hex. Mismo espíritu que core/auth_otp.hash_code.
async function hmacHex(secret: string, message: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const sig = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(message),
  );
  return Array.from(new Uint8Array(sig))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
}

// Comparación en tiempo (casi) constante de dos hex del mismo largo.
function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

function sixDigitCode(): string {
  const n = crypto.getRandomValues(new Uint32Array(1))[0] % 1_000_000;
  return n.toString().padStart(6, "0");
}

// ¿El email está registrado como usuario de Watermelon Cloud?
async function userExists(admin: any, email: string): Promise<string | null> {
  // Paginamos users (base pequeña). Devuelve user_id si existe, si no null.
  for (let page = 1; page <= 20; page++) {
    const { data, error } = await admin.auth.admin.listUsers({ page, perPage: 200 });
    if (error) throw new Error("listUsers: " + error.message);
    const users = data?.users ?? [];
    const hit = users.find((u: any) => (u.email ?? "").toLowerCase() === email);
    if (hit) return hit.id;
    if (users.length < 200) break; // última página
  }
  return null;
}

// Envía el código por Microsoft Graph (OAuth client_credentials + sendMail).
async function sendCodeEmail(email: string, code: string): Promise<void> {
  const tokenRes = await fetch(
    `https://login.microsoftonline.com/${GRAPH_TENANT}/oauth2/v2.0/token`,
    {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: new URLSearchParams({
        client_id: GRAPH_CLIENT_ID,
        client_secret: GRAPH_CLIENT_SECRET,
        scope: "https://graph.microsoft.com/.default",
        grant_type: "client_credentials",
      }),
    },
  );
  if (!tokenRes.ok) {
    throw new Error("graph token: " + (await tokenRes.text()).slice(0, 300));
  }
  const { access_token } = await tokenRes.json();

  const html = `
    <div style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:440px;margin:0 auto">
      <h2 style="color:#0f172a;margin:0 0 8px">Watermelon System</h2>
      <p style="color:#475569;margin:0 0 16px">Tu código de acceso para vincular la Planta al Cloud:</p>
      <div style="font-size:34px;font-weight:800;letter-spacing:8px;color:#0f172a;background:#f1f5f9;border-radius:10px;padding:16px 0;text-align:center">${code}</div>
      <p style="color:#94a3b8;font-size:13px;margin:16px 0 0">Vence en 10 minutos. Si no lo solicitaste, ignora este correo.</p>
    </div>`;

  const sendRes = await fetch(
    `https://graph.microsoft.com/v1.0/users/${GRAPH_FROM_EMAIL}/sendMail`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${access_token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: {
          subject: `Tu código Watermelon: ${code}`,
          body: { contentType: "HTML", content: html },
          from: { emailAddress: { address: GRAPH_FROM_EMAIL, name: GRAPH_FROM_NAME } },
          toRecipients: [{ emailAddress: { address: email } }],
        },
        saveToSentItems: false,
      }),
    },
  );
  if (!sendRes.ok) {
    throw new Error("graph sendMail: " + (await sendRes.text()).slice(0, 300));
  }
}

serve(async (req: Request) => {
  if (req.method === "OPTIONS") return new Response(null, { status: 204, headers: CORS });
  if (req.method !== "POST") return json({ error: "method_not_allowed" }, 405);

  if (!SUPABASE_URL || !SERVICE_ROLE || !ANON_KEY || !OTP_SIGNING_SECRET) {
    return json({ error: "server_misconfigured" }, 500);
  }

  let payload: any;
  try {
    payload = await req.json();
  } catch {
    return json({ error: "invalid_json" }, 400);
  }

  const action = String(payload?.action ?? "");
  const email = normEmail(payload?.email);
  if (!email || !email.includes("@")) return json({ error: "invalid_email" }, 400);

  const admin = createClient(SUPABASE_URL, SERVICE_ROLE, {
    auth: { autoRefreshToken: false, persistSession: false },
  });

  try {
    // -------------------------------------------------------------------
    // REQUEST — genera y envía el código
    // -------------------------------------------------------------------
    if (action === "request") {
      const uid = await userExists(admin, email);
      // Respuesta GENÉRICA para no filtrar qué correos existen. Solo enviamos
      // el código si el usuario está registrado.
      if (uid) {
        if (!GRAPH_TENANT || !GRAPH_CLIENT_ID || !GRAPH_CLIENT_SECRET || !GRAPH_FROM_EMAIL) {
          return json({ error: "email_backend_misconfigured" }, 500);
        }
        const code = sixDigitCode();
        const codeHash = await hmacHex(OTP_SIGNING_SECRET, `${email}::${code}`);
        const expiresAt = new Date(Date.now() + OTP_TTL_SECONDS * 1000).toISOString();
        const { error: upErr } = await admin
          .from("planta_otp_challenges")
          .upsert(
            { email, code_hash: codeHash, expires_at: expiresAt, attempts: 0, created_at: new Date().toISOString() },
            { onConflict: "email" },
          );
        if (upErr) return json({ error: "db_error", detail: upErr.message }, 500);
        await sendCodeEmail(email, code);
      }
      return json({ ok: true });
    }

    // -------------------------------------------------------------------
    // VERIFY — valida el código y acuña una sesión Supabase
    // -------------------------------------------------------------------
    if (action === "verify") {
      const code = String(payload?.code ?? "").trim().replace(/\s/g, "");
      if (!/^\d{6}$/.test(code)) return json({ error: "invalid_code_format" }, 400);

      const { data: ch, error: chErr } = await admin
        .from("planta_otp_challenges")
        .select("email, code_hash, expires_at, attempts")
        .eq("email", email)
        .maybeSingle();
      if (chErr) return json({ error: "db_error", detail: chErr.message }, 500);
      if (!ch) return json({ error: "no_challenge" }, 400);

      if (new Date(ch.expires_at).getTime() < Date.now()) {
        await admin.from("planta_otp_challenges").delete().eq("email", email);
        return json({ error: "code_expired" }, 400);
      }
      if ((ch.attempts ?? 0) >= MAX_ATTEMPTS) {
        await admin.from("planta_otp_challenges").delete().eq("email", email);
        return json({ error: "too_many_attempts" }, 429);
      }

      const providedHash = await hmacHex(OTP_SIGNING_SECRET, `${email}::${code}`);
      if (!safeEqual(providedHash, ch.code_hash)) {
        await admin
          .from("planta_otp_challenges")
          .update({ attempts: (ch.attempts ?? 0) + 1 })
          .eq("email", email);
        return json({ error: "invalid_code" }, 401);
      }

      // Código correcto → acuñar sesión Supabase para el usuario.
      const { data: linkData, error: linkErr } = await admin.auth.admin.generateLink({
        type: "magiclink",
        email,
      });
      const tokenHash = linkData?.properties?.hashed_token;
      const vType = linkData?.properties?.verification_type ?? "magiclink";
      if (linkErr || !tokenHash) {
        return json({ error: "session_mint_failed", detail: linkErr?.message }, 500);
      }

      const anon = createClient(SUPABASE_URL, ANON_KEY, {
        auth: { autoRefreshToken: false, persistSession: false },
      });
      const { data: vData, error: vErr } = await anon.auth.verifyOtp({
        type: vType as any,
        token_hash: tokenHash,
      });
      if (vErr || !vData?.session) {
        return json({ error: "session_mint_failed", detail: vErr?.message }, 500);
      }

      // Consumir el challenge (un solo uso).
      await admin.from("planta_otp_challenges").delete().eq("email", email);

      const s = vData.session;
      return json({
        ok: true,
        access_token: s.access_token,
        refresh_token: s.refresh_token,
        expires_at: s.expires_at,
        user_id: vData.user?.id ?? null,
        email: vData.user?.email ?? email,
      });
    }

    return json({ error: "unknown_action" }, 400);
  } catch (e) {
    return json({ error: "internal_error", detail: String(e).slice(0, 300) }, 500);
  }
});
