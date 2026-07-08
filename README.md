# IdeaWeaver SLM Builder

An interactive configurator for a from-scratch, Gemma-4-Nano-style small language model —
interleaved local/global attention, grouped-query attention, QK-RMSNorm, partial RoPE, and
cross-layer KV-cache sharing. Tune every architecture and training variable and see a live
parameter-count and VRAM estimate update as you go.

Built by [IdeaWeaver AI Labs](https://www.ideaweaver.ai) to go with the
[Building Small Language Models from Scratch](https://www.ideaweaver.ai/courses) course.

## Run it on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ideaweaver-ai/ideaweaver-slm-builder/blob/main/IdeaWeaver_SLM_Builder.ipynb)

1. **Open the notebook.** Click the badge above, or go to
   [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub**,
   paste `ideaweaver-ai/ideaweaver-slm-builder`, and pick `IdeaWeaver_SLM_Builder.ipynb`.
2. **(Optional) pick a runtime.** *Runtime → Change runtime type → T4 GPU* — the app itself is a
   static UI and doesn't use the GPU, but a T4 runtime keeps this notebook consistent with a real
   training session later.
3. **Run everything.** *Runtime → Run all* (or `Cmd/Ctrl + F9`). Approve the "run anyway" warning
   if Colab shows one — that's normal for any notebook not authored by Google.
4. **Watch it work through 3 cells:**
   - **Clone** — pulls this repo into the Colab VM (`/content/ideaweaver-slm-builder`).
   - **Install Node.js 20 + dependencies** — Colab's default Node is too old for Next.js 16, so
     this installs Node 20 from NodeSource first, then runs `npm install`. Takes ~30–60s.
   - **Start the app** — launches the dev server in the background, waits for the "Ready in…" log
     line, then embeds it as an iframe right in that cell's output.
5. **Use the app.** Scroll to the last cell — the IdeaWeaver SLM Builder UI renders directly in the
   notebook, full width. Configure the architecture, watch the live parameter/VRAM estimates, and
   try "Start Training" for the simulated loss curve.
6. **To stop or restart**, use *Runtime → Restart session* and *Run all* again — the server doesn't
   need any manual cleanup.

**If the iframe shows blank or an error:** the server usually just needs a moment — re-run the last
cell. If it still fails, check the log printed above the iframe for the actual Node/npm error.

## Run it locally

```bash
npm install
npm run dev
```

Then open http://localhost:3000.

## What's real vs. simulated

- Parameter count, VRAM estimate, and hardware warnings are computed live from the same
  architecture math as the actual PyTorch model — not placeholders.
- "Start Training" renders a simulated loss curve so the panel isn't static; it does not train a
  real model in the browser or in this Colab session.
