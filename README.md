# IdeaWeaver SLM Builder

An interactive configurator for a from-scratch, Gemma-4-Nano-style small language model —
interleaved local/global attention, grouped-query attention, QK-RMSNorm, partial RoPE, and
cross-layer KV-cache sharing. Tune every architecture and training variable, see a live
parameter-count and VRAM estimate, and export a ready-to-train Python config.

Built by [IdeaWeaver AI Labs](https://www.ideaweaver.ai) to go with the
[Building Small Language Models from Scratch](https://www.ideaweaver.ai/courses) course.

## Run it on Google Colab

Open **`IdeaWeaver_SLM_Builder.ipynb`** in Colab and press **Runtime → Run all**. It clones this
repo, installs Node.js + dependencies, and displays the app in an embedded iframe.

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
- **Export** produces a real `slm_config.py` / `slm_config.json` you can drop into an actual
  training script.
