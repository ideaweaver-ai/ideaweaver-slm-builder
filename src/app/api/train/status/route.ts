const BACKEND = process.env.TRAIN_BACKEND_URL ?? "http://127.0.0.1:8001";

export async function GET() {
  try {
    const upstream = await fetch(`${BACKEND}/train/status`, { cache: "no-store" });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch {
    return Response.json({ status: "unreachable" }, { status: 502 });
  }
}
