const BACKEND = process.env.TRAIN_BACKEND_URL ?? "http://127.0.0.1:8001";

export async function POST(req: Request) {
  const body = await req.text();
  try {
    const upstream = await fetch(`${BACKEND}/train/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json(
      { ok: false, error: "Training backend isn't reachable. Is train_service.py running?" },
      { status: 502 }
    );
  }
}
