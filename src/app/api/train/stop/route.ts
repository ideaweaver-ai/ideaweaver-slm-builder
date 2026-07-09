const BACKEND = process.env.TRAIN_BACKEND_URL ?? "http://127.0.0.1:8001";

export async function POST() {
  try {
    const upstream = await fetch(`${BACKEND}/train/stop`, { method: "POST" });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json({ ok: false, error: "Training backend isn't reachable." }, { status: 502 });
  }
}
