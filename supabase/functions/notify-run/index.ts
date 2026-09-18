// Supabase Edge Function: notify-run — envía un EMAIL cuando llega una corrida nueva.
// Se dispara con un Database Webhook (INSERT en public.modal_runs) que hace POST aquí.
// Envía el correo vía Resend (https://resend.com). Requiere secrets:
//   RESEND_API_KEY  — API key de Resend
//   NOTIFY_EMAILS   — destinatarios separados por coma (ej. ehernandez@sigasas.com,otro@...)
//   RESEND_FROM     — remitente (opcional; por defecto avisos@watermelonsystem.app, dominio verificado)
// "Verify JWT" puede quedar OFF (lo protege que solo el webhook conoce la URL) o ON con header.

Deno.serve(async (req) => {
  try {
    const body = await req.json().catch(() => ({}));
    const rec = (body.record || body.new || body) as Record<string, unknown>;
    const key = Deno.env.get("RESEND_API_KEY");
    const to = (Deno.env.get("NOTIFY_EMAILS") || "").split(",").map((s) => s.trim()).filter(Boolean);
    if (!key || to.length === 0) {
      return new Response(JSON.stringify({ skipped: "falta RESEND_API_KEY o NOTIFY_EMAILS" }), { status: 200 });
    }
    const from = Deno.env.get("RESEND_FROM") || "Watermelon System <avisos@watermelonsystem.app>";
    const name = String(rec.name ?? "corrida");
    const client = String(rec.client ?? "—");
    const host = String(rec.hostname ?? "—");
    const acc = String(rec.account ?? "—");
    const ip = String(rec.ip ?? "").trim();
    const geo = String(rec.geo ?? "").trim();
    const when = String(rec.created_at ?? rec.updated_at ?? "");
    // Módulo (Modal / Torsional) — el mismo notify sirve a ambos.
    const moduleName = String(rec.module ?? "Modal");
    const isTors = moduleName.toLowerCase().startsWith("tors");
    const kind = isTors ? "torsional" : "OMA";
    const webMod = isTors ? "Torsional" : "Modal";
    const rowIp = ip ? `<tr><td style="padding:4px 10px;color:#64748b">IP de conexión</td><td style="padding:4px 10px">${ip}</td></tr>` : "";
    const rowGeo = geo ? `<tr><td style="padding:4px 10px;color:#64748b">Ubicación aprox.</td><td style="padding:4px 10px">${geo}</td></tr>` : "";
    const html = `
      <div style="font-family:Segoe UI,Arial,sans-serif;color:#1f2937">
        <h2 style="color:#0f2a4a">🍉 Nueva corrida de campo en Watermelon System</h2>
        <p>Llegó una corrida ${kind} nueva desde el campo. Ábrela en la web → módulo <b>${webMod}</b> → <b>Data source</b>.</p>
        <table style="border-collapse:collapse;font-size:14px">
          <tr><td style="padding:4px 10px;color:#64748b">Corrida</td><td style="padding:4px 10px"><b>${name}</b></td></tr>
          <tr><td style="padding:4px 10px;color:#64748b">Cliente</td><td style="padding:4px 10px">${client}</td></tr>
          <tr><td style="padding:4px 10px;color:#64748b">PC de campo</td><td style="padding:4px 10px">${host}</td></tr>
          <tr><td style="padding:4px 10px;color:#64748b">Cuenta</td><td style="padding:4px 10px">${acc}</td></tr>
          ${rowIp}
          ${rowGeo}
          <tr><td style="padding:4px 10px;color:#64748b">Fecha</td><td style="padding:4px 10px">${when}</td></tr>
        </table>
        <p style="color:#94a3b8;font-size:12px;margin-top:16px">Aviso automático de Watermelon System.</p>
      </div>`;
    const r = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: { "Authorization": `Bearer ${key}`, "Content-Type": "application/json" },
      body: JSON.stringify({ from, to, subject: `🔔 Nueva corrida ${webMod}: ${name} (${client})`, html }),
    });
    const data = await r.json().catch(() => ({}));
    return new Response(JSON.stringify({ ok: r.ok, data }), { status: r.ok ? 200 : 500 });
  } catch (e) {
    return new Response(JSON.stringify({ error: String(e) }), { status: 500 });
  }
});
