// =============================================================
// supabase/functions/license-check/index.ts
// FASE J2 — Heartbeat endpoint para Watermelon Planta
// =============================================================
//
// Endpoint PÚBLICO (sin auth requerida — el license_id ya es público
// porque vive en el JWT del cliente).
//
// Recibe:
//   GET /functions/v1/license-check?jti=<license_id>
//
// Devuelve JSON:
//   { "status": "active" }
//   o
//   { "status": "revoked", "reason": "...", "revoked_at": "..." }
//
// Errores:
//   400 si falta el parámetro jti o no es UUID válido
//   500 si la BD está caída
//
// Despliegue:
//   supabase functions deploy license-check --no-verify-jwt
//   (--no-verify-jwt porque queremos endpoint público sin auth)
//
// URL pública resultante:
//   https://<proyecto>.supabase.co/functions/v1/license-check
//
// CORS: habilitado para que se pueda llamar desde cualquier cliente.

import { serve } from "https://deno.land/std@0.177.0/http/server.ts";
import { createClient } from "https://esm.sh/@supabase/supabase-js@2";

const SUPABASE_URL = Deno.env.get("SUPABASE_URL") ?? "";
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") ?? "";

// Helper: validar formato UUID v4 (relax — solo chequear longitud y hex)
function isValidUuid(s: string): boolean {
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(s);
}

// Headers CORS para que el endpoint sea callable desde cualquier origen
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, Authorization",
  "Content-Type": "application/json",
  "Cache-Control": "no-store, no-cache, must-revalidate",
};

serve(async (req: Request) => {
  // Preflight CORS
  if (req.method === "OPTIONS") {
    return new Response(null, { status: 204, headers: CORS_HEADERS });
  }

  if (req.method !== "GET") {
    return new Response(
      JSON.stringify({ error: "method_not_allowed" }),
      { status: 405, headers: CORS_HEADERS },
    );
  }

  // ----------------------------------------------------------------------
  // 1. Parsear y validar parámetro jti
  // ----------------------------------------------------------------------
  const url = new URL(req.url);
  const jti = url.searchParams.get("jti")?.trim() ?? "";

  if (!jti) {
    return new Response(
      JSON.stringify({ error: "missing_jti_parameter" }),
      { status: 400, headers: CORS_HEADERS },
    );
  }

  if (!isValidUuid(jti)) {
    return new Response(
      JSON.stringify({ error: "invalid_jti_format" }),
      { status: 400, headers: CORS_HEADERS },
    );
  }

  // ----------------------------------------------------------------------
  // 2. Consultar tabla revoked_licenses (usa service role para bypass RLS)
  // ----------------------------------------------------------------------
  if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    return new Response(
      JSON.stringify({ error: "server_misconfigured" }),
      { status: 500, headers: CORS_HEADERS },
    );
  }

  const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
    auth: { autoRefreshToken: false, persistSession: false },
  });

  try {
    const { data, error } = await supabase
      .from("revoked_licenses")
      .select("license_id, revoked_at, reason")
      .eq("license_id", jti)
      .maybeSingle();

    if (error) {
      // Error de BD — devolvemos 500 para que el cliente entre en modo
      // "no pude validar" y use su cache
      return new Response(
        JSON.stringify({ error: "database_error", detail: error.message }),
        { status: 500, headers: CORS_HEADERS },
      );
    }

    if (data) {
      // La licencia está en la blacklist → REVOCADA
      return new Response(
        JSON.stringify({
          status: "revoked",
          reason: data.reason,
          revoked_at: data.revoked_at,
        }),
        { status: 200, headers: CORS_HEADERS },
      );
    }

    // No está en blacklist → ACTIVA
    return new Response(
      JSON.stringify({ status: "active" }),
      { status: 200, headers: CORS_HEADERS },
    );
  } catch (e) {
    return new Response(
      JSON.stringify({ error: "internal_error", detail: String(e) }),
      { status: 500, headers: CORS_HEADERS },
    );
  }
});
