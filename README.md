# IdeaWeaver SLM Builder

An interactive configurator for a from-scratch, Gemma-4-Nano-style small language model —
interleaved local/global attention, grouped-query attention, QK-RMSNorm, partial RoPE, and
cross-layer KV-cache sharing. Tune every architecture and training variable, see a live
parameter-count and VRAM estimate update as you go, then **actually train it** on TinyStories
and watch a real loss curve.

Built by [IdeaWeaver AI Labs](https://www.ideaweaver.ai) to go with the
[Building Small Language Models from Scratch](https://www.ideaweaver.ai/courses) course.

## How it fits together

- **`src/`** — the Next.js frontend: the configurator UI, live parameter/VRAM estimator, and the
  loss chart. Talks to the backend only through its own `/api/train/*` routes (never directly),
  so it works the same locally and inside the Colab iframe.
- **`backend/`** — a small FastAPI service (`train_service.py`) that builds the *exact* model
  you configured (`model.py`, ported from the reference training script) and actually trains it
  on TinyStories (`build_tokenizer.py`), streaming real loss back over Server-Sent Events.

## Run it on Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ideaweaver-ai/ideaweaver-slm-builder/blob/main/IdeaWeaver_SLM_Builder.ipynb)

1. **Open the notebook.** Click the badge above, or go to
   [colab.research.google.com](https://colab.research.google.com) → **File → Open notebook → GitHub**,
   paste `ideaweaver-ai/ideaweaver-slm-builder`, and pick `IdeaWeaver_SLM_Builder.ipynb`.
2. **Pick a GPU runtime.** *Runtime → Change runtime type → T4 GPU.* Training works on CPU too,
   but it's slow enough to not be worth it — use a GPU.
3. **Run everything.** *Runtime → Run all* (or `Cmd/Ctrl + F9`). Approve the "run anyway" warning
   if Colab shows one — that's normal for any notebook not authored by Google.
4. **Watch it work through the setup cells:** clone the repo → install Node 20 + frontend deps →
   install the backend's Python deps (skipping `torch`/`numpy`, which Colab already has with CUDA)
   → start the training backend on port 8001 → start the frontend on port 3000 and embed it.
5. **Use the app.** Scroll to the last cell — the IdeaWeaver SLM Builder UI renders directly in the
   notebook. Configure the architecture, then click **Start Training**. The first run spends
   several minutes downloading and tokenizing TinyStories (watch the status line under the chart);
   every run after that reuses the cached data. **Stop** anytime and download the checkpoint.
6. **To stop or restart**, use *Runtime → Restart session* and *Run all* again.

**If the iframe shows blank or an error:** the frontend usually just needs a moment — re-run its
cell. If "Start Training" says the backend isn't reachable, check the backend cell's log output
for the actual error.

## Run it locally

Needs two processes: the backend (Python/PyTorch) and the frontend (Next.js).

```bash
# Backend — trains the real model
cd backend
pip install -r requirements.txt   # or your own torch install if you already have one
uvicorn train_service:app --port 8001

# Frontend — in a second terminal, from the repo root
npm install
npm run dev
```

Then open http://localhost:3000. Training runs on GPU automatically if `torch.cuda.is_available()`;
otherwise it falls back to CPU (correct, but slow for anything beyond a tiny config).

## What's real vs. estimated

- Parameter count, VRAM estimate, and hardware warnings are computed client-side from the same
  architecture math as the actual PyTorch model — accurate, but still *estimates* (e.g. peak VRAM
  is a rough heuristic, not a measurement).
- **Start Training is real.** It builds the model you configured, trains it on TinyStories with a
  real optimizer and LR schedule, and streams real loss back into the chart. Nothing is faked.
- The training backend only supports TinyStories right now — the "Dataset" card is informational,
  not configurable.
