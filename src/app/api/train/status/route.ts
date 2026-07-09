const BACKEND = process.env.TRAIN_BACKEND_URL ?? "http://127.0.0.1:8001";

export async function GET() {
  try {
    // A busy backend (e.g. mid-tokenization on a CPU-starved Colab instance)
    // can be slow to schedule this request rather than truly down — without
    // a timeout this fetch (and the client's "checking backend..." state)
    // can hang indefinitely instead of resolving either way.
    const upstream = await fetch(`${BACKEND}/train/status`, {
      cache: "no-store",
      signal: AbortSignal.timeout(4000),
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    const timedOut = err instanceof Error && err.name === "TimeoutError";
    return Response.json(
      { status: "unreachable", reason: timedOut ? "timeout" : "connection_failed" },
      { status: 502 }
    );
  }
}
