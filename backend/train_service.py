"""
Real training backend for IdeaWeaver SLM Builder.

Single-job FastAPI service: the frontend POSTs a config, this builds the
actual Gemma4Model, trains it on TinyStories, and streams real loss values
back over Server-Sent Events. Next.js API routes proxy to this service so
the browser never talks to it directly (needed for the Colab iframe, where
only one port is exposed to the browser).

Run directly: uvicorn train_service:app --port 8001 --app-dir backend
"""

import asyncio
import json
import os
import threading
import uuid
from contextlib import nullcontext
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from build_tokenizer import ensure_tinystories_ready
from model import Gemma4Model

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

app = FastAPI()


class TrainConfig(BaseModel):
    vocabSize: int
    contextLength: int
    embDim: int
    nHeads: int
    nLayers: int
    hiddenDim: int
    headDim: int
    nKvHeads: int
    slidingWindow: int
    globalHeadDim: int
    nGlobalKvHeads: int
    fullAttnEvery: int
    qkNorm: bool
    attentionKEqV: bool
    ropeLocalBase: float
    ropeBase: float
    partialRotaryFactor: float
    numKvSharedLayers: int
    pleDim: int
    finalLogitSoftcapping: float
    learningRate: float
    maxIters: int
    warmupSteps: int
    minLr: float
    batchSize: int
    blockSize: int
    gradAccumSteps: int
    weightDecay: float
    gradClip: float
    adamEps: float
    beta1: float
    beta2: float
    precision: str  # "bfloat16" | "float16"


class Job:
    def __init__(self):
        self.status = "idle"  # idle | preparing_data | training | done | error | stopped
        self.message = ""
        self.step = 0
        self.max_iters = 0
        self.device = None
        self.queue: Optional[asyncio.Queue] = None
        self.stop_flag = threading.Event()
        self.checkpoint_path: Optional[str] = None
        self.error: Optional[str] = None


job = Job()
job_lock = threading.Lock()


def cfg_to_gemma_config(cfg: TrainConfig) -> dict:
    layer_types = [
        "full_attention" if (i + 1) % cfg.fullAttnEvery == 0 else "sliding_attention"
        for i in range(cfg.nLayers)
    ]
    # Parameters always live in fp32 ("master weights") — cfg.precision only
    # controls the autocast dtype used for the matmul-heavy forward pass on
    # CUDA. Building the parameters themselves in raw float16 (no autocast,
    # no GradScaler) overflows to NaN within a handful of steps; this is the
    # standard mixed-precision recipe instead.
    return {
        "vocab_size": cfg.vocabSize,
        "context_length": cfg.contextLength,
        "emb_dim": cfg.embDim,
        "n_heads": cfg.nHeads,
        "n_layers": cfg.nLayers,
        "hidden_dim": cfg.hiddenDim,
        "head_dim": cfg.headDim,
        "global_head_dim": cfg.globalHeadDim,
        "n_kv_heads": cfg.nKvHeads,
        "n_global_kv_heads": cfg.nGlobalKvHeads,
        "qk_norm": cfg.qkNorm,
        "attention_k_eq_v": cfg.attentionKEqV,
        "rope_local_base": cfg.ropeLocalBase,
        "rope_base": cfg.ropeBase,
        "partial_rotary_factor": cfg.partialRotaryFactor,
        "sliding_window": cfg.slidingWindow,
        "num_kv_shared_layers": cfg.numKvSharedLayers,
        "ple_dim": cfg.pleDim,
        "final_logit_softcapping": cfg.finalLogitSoftcapping or None,
        "dtype": torch.float32,
        "layer_types": layer_types,
    }


def emit(loop: asyncio.AbstractEventLoop, event: dict):
    if job.queue is not None:
        loop.call_soon_threadsafe(job.queue.put_nowait, event)


def set_status(loop, status: str, message: str = ""):
    job.status = status
    job.message = message
    emit(loop, {"type": "status", "status": status, "message": message})


def training_worker(cfg: TrainConfig, loop: asyncio.AbstractEventLoop):
    prev_cwd = os.getcwd()
    try:
        os.chdir(DATA_DIR)

        set_status(loop, "preparing_data", "Checking TinyStories tokenizer + data...")
        ensure_tinystories_ready(progress=lambda msg: set_status(loop, "preparing_data", msg))

        from build_tokenizer import TinyStoriesTokenizer, MODEL_FILE
        enc = TinyStoriesTokenizer(MODEL_FILE)

        gemma_cfg = cfg_to_gemma_config(cfg)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        job.device = device
        set_status(loop, "training", f"Building model on {device}...")

        model = Gemma4Model(gemma_cfg).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.learningRate,
            betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weightDecay, eps=cfg.adamEps,
        )
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_steps = max(1, min(cfg.warmupSteps, cfg.maxIters - 1)) if cfg.maxIters > 1 else 1
        warmup = LinearLR(optimizer, total_iters=warmup_steps)
        decay = CosineAnnealingLR(optimizer, T_max=max(1, cfg.maxIters - warmup_steps), eta_min=cfg.minLr)
        scheduler = SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_steps])

        train_data = np.memmap("train.bin", dtype=np.uint16, mode="r")

        def get_batch():
            ix = torch.randint(len(train_data) - cfg.blockSize, (cfg.batchSize,))
            x = torch.stack([torch.from_numpy(train_data[i:i + cfg.blockSize].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(train_data[i + 1:i + 1 + cfg.blockSize].astype(np.int64)) for i in ix])
            return x.to(device), y.to(device)

        ptdtype = torch.bfloat16 if cfg.precision == "bfloat16" else torch.float16
        ctx = torch.amp.autocast(device_type="cuda", dtype=ptdtype) if device == "cuda" else nullcontext()
        # float16 (unlike bf16) needs loss scaling to avoid gradient underflow;
        # GradScaler(enabled=False) is a no-op, so this is safe for bf16/CPU too.
        scaler = torch.amp.GradScaler(enabled=(device == "cuda" and cfg.precision == "float16"))

        job.max_iters = cfg.maxIters
        job.step = 0
        set_status(loop, "training", f"Training on {device}...")

        model.train()
        smoothed = None
        for step in range(cfg.maxIters):
            if job.stop_flag.is_set():
                break

            X, Y = get_batch()
            with ctx:
                _, loss = model(X, Y)
                loss_scaled = loss / cfg.gradAccumSteps
            scaler.scale(loss_scaled).backward()

            if (step + 1) % cfg.gradAccumSteps == 0 or (step + 1) == cfg.maxIters:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.gradClip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()

            loss_val = float(loss.item())
            smoothed = loss_val if smoothed is None else smoothed * 0.9 + loss_val * 0.1
            job.step = step + 1
            emit(loop, {
                "type": "point",
                "step": step + 1,
                "loss": loss_val,
                "smoothed": smoothed,
                "lr": optimizer.param_groups[0]["lr"],
            })

        stopped_early = job.stop_flag.is_set()

        ckpt_path = os.path.join(CKPT_DIR, f"slm_{uuid.uuid4().hex[:8]}.pt")
        torch.save(model.state_dict(), ckpt_path)
        job.checkpoint_path = ckpt_path

        # Emit checkpoint availability *before* the terminal status — the SSE
        # generator closes the stream as soon as it sees a terminal status
        # event, so anything queued after it would never reach the client.
        emit(loop, {"type": "checkpoint", "available": True})
        final_status = "stopped" if stopped_early else "done"
        set_status(loop, final_status, f"{'Stopped' if stopped_early else 'Finished'} at step {job.step}/{cfg.maxIters}.")

    except Exception as e:  # noqa: BLE001 - surface any failure to the frontend
        job.error = str(e)
        set_status(loop, "error", str(e))
    finally:
        os.chdir(prev_cwd)


@app.post("/train/start")
async def start_train(cfg: TrainConfig):
    with job_lock:
        if job.status in ("preparing_data", "training"):
            return JSONResponse({"ok": False, "error": "A training run is already in progress."}, status_code=409)
        job.status = "starting"
        job.step = 0
        job.max_iters = cfg.maxIters
        job.error = None
        job.checkpoint_path = None
        job.stop_flag = threading.Event()
        job.queue = asyncio.Queue()
        loop = asyncio.get_event_loop()
        thread = threading.Thread(target=training_worker, args=(cfg, loop), daemon=True)
        thread.start()
    return {"ok": True}


@app.post("/train/stop")
async def stop_train():
    job.stop_flag.set()
    return {"ok": True}


@app.get("/train/status")
async def status():
    return {
        "status": job.status,
        "message": job.message,
        "step": job.step,
        "maxIters": job.max_iters,
        "device": job.device,
        "error": job.error,
        "hasCheckpoint": job.checkpoint_path is not None,
    }


@app.get("/train/stream")
async def stream():
    async def event_gen():
        if job.queue is None:
            yield f"data: {json.dumps({'type': 'status', 'status': job.status, 'message': job.message})}\n\n"
            return
        q = job.queue
        while True:
            event = await q.get()
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("type") == "status" and event.get("status") in ("done", "error", "stopped"):
                break

    return StreamingResponse(event_gen(), media_type="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.get("/train/checkpoint")
async def checkpoint():
    if not job.checkpoint_path or not os.path.exists(job.checkpoint_path):
        return JSONResponse({"error": "No checkpoint available yet."}, status_code=404)
    return FileResponse(
        job.checkpoint_path,
        filename=os.path.basename(job.checkpoint_path),
        media_type="application/octet-stream",
    )


@app.get("/health")
async def health():
    return {"ok": True, "cuda": torch.cuda.is_available()}
