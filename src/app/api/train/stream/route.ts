export const dynamic = "force-dynamic";

const BACKEND = process.env.TRAIN_BACKEND_URL ?? "http://127.0.0.1:8001";

export async function GET() {
  try {
    const upstream = await fetch(`${BACKEND}/train/stream`, { cache: "no-store" });
    if (!upstream.body) {
      return new Response("data: {\"type\":\"status\",\"status\":\"error\",\"message\":\"Empty stream from backend\"}\n\n", {
        headers: { "Content-Type": "text/event-stream" },
      });
    }
    return new Response(upstream.body, {
      headers: {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        Connection: "keep-alive",
      },
    });
  } catch {
    const message = JSON.stringify({
      type: "status",
      status: "error",
      message: "Training backend isn't reachable. Is train_service.py running?",
    });
    return new Response(`data: ${message}\n\n`, {
      headers: { "Content-Type": "text/event-stream" },
    });
  }
}
