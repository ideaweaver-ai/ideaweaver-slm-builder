const BACKEND = process.env.TRAIN_BACKEND_URL ?? "http://127.0.0.1:8001";

export async function GET() {
  try {
    const upstream = await fetch(`${BACKEND}/train/checkpoint`, { cache: "no-store" });
    if (!upstream.ok || !upstream.body) {
      const text = await upstream.text();
      return new Response(text, { status: upstream.status, headers: { "Content-Type": "application/json" } });
    }
    return new Response(upstream.body, {
      headers: {
        "Content-Type": "application/octet-stream",
        "Content-Disposition": upstream.headers.get("content-disposition") ?? "attachment; filename=checkpoint.pt",
      },
    });
  } catch {
    return Response.json({ error: "Training backend isn't reachable." }, { status: 502 });
  }
}
