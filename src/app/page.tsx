"use client";

import { useEffect, useMemo, useRef, useState } from "react";

/* ────────────────────────────────────────────────────────────────────────
   IdeaWeaver SLM Builder
   Interactive configurator for a from-scratch, Gemma-4-Nano-style small
   language model (interleaved local/global attention, GQA, QK-norm,
   partial RoPE, cross-layer KV-cache sharing). The parameter count, VRAM
   estimate, and warnings are computed client-side from the same
   architecture this page lets you configure. "Start Training" calls the
   real Python backend (backend/train_service.py) — it builds this exact
   model, trains it on TinyStories, and streams real loss back over SSE.
   Requires the backend running locally or in Colab (see README).
   ──────────────────────────────────────────────────────────────────────── */

const IDEAWEAVER_HOME = "https://www.ideaweaver.ai";
const IDEAWEAVER_COURSES = "https://www.ideaweaver.ai/courses";
const GITHUB_REPO = "https://github.com/ideaweaver-ai/ideaweaver-slm-builder";

type Precision = "bfloat16" | "float16";
type GPU = "T4" | "L4" | "A100" | "H100";

type Config = {
  // architecture — core
  vocabSize: number;
  contextLength: number;
  embDim: number;
  nHeads: number;
  nLayers: number;
  hiddenDim: number;
  // local / sliding attention
  headDim: number;
  nKvHeads: number;
  slidingWindow: number;
  // global / full attention
  globalHeadDim: number;
  nGlobalKvHeads: number;
  fullAttnEvery: number;
  // toggles
  qkNorm: boolean;
  attentionKEqV: boolean;
  // rope
  ropeLocalBase: number;
  ropeBase: number;
  partialRotaryFactor: number;
  // memory optimizations
  numKvSharedLayers: number;
  pleDim: number;
  // output
  finalLogitSoftcapping: number;
  // training
  learningRate: number;
  maxIters: number;
  warmupSteps: number;
  minLr: number;
  evalInterval: number;
  evalIters: number;
  batchSize: number;
  blockSize: number;
  gradAccumSteps: number;
  weightDecay: number;
  gradClip: number;
  adamEps: number;
  beta1: number;
  beta2: number;
  // hardware
  gpu: GPU;
  precision: Precision;
};

const PRESET_T4_20L: Config = {
  vocabSize: 8000,
  contextLength: 2048,
  embDim: 384,
  nHeads: 8,
  nLayers: 20,
  hiddenDim: 1536,
  headDim: 48,
  nKvHeads: 2,
  slidingWindow: 512,
  globalHeadDim: 96,
  nGlobalKvHeads: 1,
  fullAttnEvery: 6,
  qkNorm: true,
  attentionKEqV: true,
  ropeLocalBase: 10000,
  ropeBase: 1000000,
  partialRotaryFactor: 0.25,
  numKvSharedLayers: 6,
  pleDim: 0,
  finalLogitSoftcapping: 30,
  learningRate: 0.0005,
  maxIters: 10000,
  warmupSteps: 300,
  minLr: 0.00001,
  evalInterval: 500,
  evalIters: 200,
  batchSize: 16,
  blockSize: 512,
  gradAccumSteps: 4,
  weightDecay: 0.1,
  gradClip: 0.5,
  adamEps: 1e-8,
  beta1: 0.9,
  beta2: 0.95,
  gpu: "T4",
  precision: "float16",
};

const PRESET_COLAB_15L: Config = {
  ...PRESET_T4_20L,
  nLayers: 15,
  hiddenDim: 656,
  numKvSharedLayers: 6,
  batchSize: 24,
  gpu: "T4",
  precision: "float16",
};

const PRESET_A100_24L: Config = {
  ...PRESET_T4_20L,
  nLayers: 24,
  hiddenDim: 1536,
  embDim: 512,
  headDim: 64,
  globalHeadDim: 128,
  batchSize: 48,
  gradAccumSteps: 2,
  gpu: "A100",
  precision: "bfloat16",
};

const PRESETS: { id: string; label: string; blurb: string; cfg: Config }[] = [
  { id: "t4-20l", label: "T4 · 20 layers", blurb: "Free-tier friendly, ~2–3 GB VRAM", cfg: PRESET_T4_20L },
  { id: "colab-15l", label: "Colab Free · 15 layers", blurb: "Smaller + faster iteration loop", cfg: PRESET_COLAB_15L },
  { id: "a100-24l", label: "A100 · 24 layers", blurb: "Wider model, bf16 tensor cores", cfg: PRESET_A100_24L },
];

// This build only trains on TinyStories — its size is fixed here rather
// than exposed as an editable field.
const TINYSTORIES_TOTAL_TOKENS = 450_000_000;

const GPU_INFO: Record<GPU, { vramGB: number; bf16TensorCores: boolean; arch: string }> = {
  T4: { vramGB: 16, bf16TensorCores: false, arch: "Turing (SM 7.5)" },
  L4: { vramGB: 24, bf16TensorCores: true, arch: "Ada Lovelace (SM 8.9)" },
  A100: { vramGB: 40, bf16TensorCores: true, arch: "Ampere (SM 8.0)" },
  H100: { vramGB: 80, bf16TensorCores: true, arch: "Hopper (SM 9.0)" },
};

/* ── Derived architecture math (mirrors the actual PyTorch module tree) ── */

function layerTypes(cfg: Config): ("sliding" | "full")[] {
  return Array.from({ length: cfg.nLayers }, (_, i) =>
    (i + 1) % cfg.fullAttnEvery === 0 ? "full" : "sliding"
  );
}

function estimateParams(cfg: Config) {
  const types = layerTypes(cfg);
  const firstShared = cfg.nLayers - cfg.numKvSharedLayers;
  let blockParams = 0;

  for (let i = 0; i < cfg.nLayers; i++) {
    const isFull = types[i] === "full";
    const headDim = isFull ? cfg.globalHeadDim : cfg.headDim;
    const numKvHeads = isFull ? cfg.nGlobalKvHeads : cfg.nKvHeads;
    const useKEqV = isFull && cfg.attentionKEqV;
    const dOut = cfg.nHeads * headDim;
    const isKvShared = cfg.numKvSharedLayers > 0 && i >= firstShared;

    let p = cfg.embDim * dOut; // W_query
    p += headDim; // q_norm scale
    if (!isKvShared) {
      p += cfg.embDim * (numKvHeads * headDim); // W_key
      p += headDim; // k_norm scale
      if (!useKEqV) p += cfg.embDim * (numKvHeads * headDim); // W_value
    }
    p += dOut * cfg.embDim; // out_proj
    p += cfg.embDim * 4; // 4x sandwich RMSNorm scales
    p += 3 * cfg.embDim * cfg.hiddenDim; // gate + up + down

    if (cfg.pleDim > 0) {
      p += cfg.embDim * cfg.pleDim + cfg.pleDim * cfg.embDim + cfg.embDim;
    }
    blockParams += p;
  }

  const embedding = cfg.vocabSize * cfg.embDim; // tied with out_head
  const finalNorm = cfg.embDim;
  let pleExtra = 0;
  if (cfg.pleDim > 0) {
    pleExtra += cfg.vocabSize * cfg.nLayers * cfg.pleDim;
    pleExtra += cfg.embDim * cfg.nLayers * cfg.pleDim;
    pleExtra += cfg.pleDim;
  }

  const total = embedding + finalNorm + blockParams + pleExtra;
  return { total, embedding, blockParams, finalNorm, pleExtra, types, firstShared };
}

function fmtParams(n: number) {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

function fmtTokens(n: number) {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return `${n}`;
}

/* ── UI primitives ─────────────────────────────────────────────────────── */

function Card({
  title,
  subtitle,
  icon,
  children,
}: {
  title: string;
  subtitle?: string;
  icon: string;
  children: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-white/[0.06] bg-[#111116] p-6">
      <div className="mb-5 flex items-start gap-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-violet-500/10 text-base">
          {icon}
        </div>
        <div>
          <h3 className="text-sm font-bold text-white">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs text-zinc-500">{subtitle}</p>}
        </div>
      </div>
      {children}
    </section>
  );
}

function Group({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-5 last:mb-0">
      <div className="mb-2.5 text-[11px] font-semibold uppercase tracking-wide text-zinc-500">
        {label}
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">{children}</div>
    </div>
  );
}

function NumField({
  label,
  value,
  onChange,
  step = 1,
  min,
  max,
  hint,
  mono = true,
}: {
  label: string;
  value: number;
  onChange: (v: number) => void;
  step?: number;
  min?: number;
  max?: number;
  hint?: string;
  mono?: boolean;
}) {
  return (
    <label className="block">
      <span className="mb-1 flex items-center gap-1 text-[11px] font-medium text-zinc-400">
        {label}
        {hint && (
          <span className="group relative inline-flex cursor-help text-zinc-600">
            <span className="text-[10px]">ⓘ</span>
            <span className="pointer-events-none absolute bottom-full left-1/2 z-10 mb-1.5 w-48 -translate-x-1/2 rounded-lg border border-white/10 bg-[#1a1a20] px-2.5 py-1.5 text-[11px] font-normal leading-snug text-zinc-300 opacity-0 shadow-xl transition group-hover:opacity-100">
              {hint}
            </span>
          </span>
        )}
      </span>
      <input
        type="number"
        value={Number.isFinite(value) ? value : 0}
        step={step}
        min={min}
        max={max}
        onChange={(e) => onChange(e.target.value === "" ? 0 : parseFloat(e.target.value))}
        className={`w-full rounded-lg border border-white/[0.08] bg-black/30 px-2.5 py-1.5 text-sm text-white outline-none transition focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/30 ${
          mono ? "font-mono" : ""
        }`}
      />
    </label>
  );
}

function ToggleField({
  label,
  checked,
  onChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onChange: (v: boolean) => void;
  hint?: string;
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className="flex w-full items-center justify-between rounded-lg border border-white/[0.08] bg-black/30 px-3 py-2 text-left transition hover:border-white/[0.15]"
    >
      <span className="pr-2 text-[11px] font-medium text-zinc-300">
        {label}
        {hint && <span className="mt-0.5 block text-[10px] font-normal text-zinc-500">{hint}</span>}
      </span>
      <span
        className={`relative h-5 w-9 shrink-0 rounded-full transition ${
          checked ? "bg-violet-600" : "bg-zinc-700"
        }`}
      >
        <span
          className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition ${
            checked ? "left-[18px]" : "left-0.5"
          }`}
        />
      </span>
    </button>
  );
}

function SelectField<T extends string>({
  label,
  value,
  options,
  onChange,
  hint,
}: {
  label: string;
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
  hint?: string;
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium text-zinc-400">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="w-full appearance-none rounded-lg border border-white/[0.08] bg-black/30 px-2.5 py-1.5 text-sm text-white outline-none transition focus:border-violet-500/50 focus:ring-1 focus:ring-violet-500/30"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value} className="bg-[#111116]">
            {o.label}
          </option>
        ))}
      </select>
      {hint && <span className="mt-1 block text-[10px] text-zinc-500">{hint}</span>}
    </label>
  );
}

function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-black/20 px-3 py-2.5">
      <div className="text-[10px] font-semibold uppercase tracking-wide text-zinc-500">{label}</div>
      <div className="mt-0.5 font-mono text-lg font-bold text-white">{value}</div>
      {sub && <div className="text-[10px] text-zinc-500">{sub}</div>}
    </div>
  );
}

/* ── Simulated training loss chart ───────────────────────────────────────
   Two series: raw step loss (muted reference line) and an EMA-smoothed
   line (brand accent, the one that actually carries the signal). Single
   y-axis, 2px lines, legend since there are two series, hover crosshair
   with a synced tooltip. Dark-only surface to match the page. */

type Point = { step: number; loss: number; smoothed: number };

function LossChart({ data, target }: { data: Point[]; target: number }) {
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const width = 560;
  const height = 200;
  const padL = 34;
  const padR = 12;
  const padT = 12;
  const padB = 24;

  if (data.length < 2) {
    return (
      <div className="flex h-[200px] items-center justify-center rounded-xl border border-dashed border-white/[0.08] text-xs text-zinc-600">
        Start training to see the loss curve
      </div>
    );
  }

  const maxLoss = Math.max(...data.map((d) => d.loss)) * 1.05;
  const minLoss = Math.min(target * 0.9, Math.min(...data.map((d) => d.loss)) * 0.95);
  const xAt = (i: number) => padL + (i / (data.length - 1)) * (width - padL - padR);
  const yAt = (v: number) =>
    padT + (1 - (v - minLoss) / (maxLoss - minLoss || 1)) * (height - padT - padB);

  const pathFor = (key: "loss" | "smoothed") =>
    data.map((d, i) => `${i === 0 ? "M" : "L"}${xAt(i).toFixed(1)},${yAt(d[key]).toFixed(1)}`).join(" ");

  const gridLines = 4;
  const hover = hoverIdx !== null ? data[hoverIdx] : null;

  return (
    <div className="relative">
      <div className="mb-2 flex items-center gap-4 text-[11px]">
        <span className="flex items-center gap-1.5 text-zinc-500">
          <span className="inline-block h-[2px] w-3 bg-zinc-500" /> Loss
        </span>
        <span className="flex items-center gap-1.5 text-violet-300">
          <span className="inline-block h-[2px] w-3 bg-violet-500" /> Smoothed
        </span>
      </div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full touch-none"
        onMouseMove={(e) => {
          const rect = e.currentTarget.getBoundingClientRect();
          const relX = ((e.clientX - rect.left) / rect.width) * width;
          const frac = (relX - padL) / (width - padL - padR);
          const idx = Math.round(frac * (data.length - 1));
          setHoverIdx(Math.min(data.length - 1, Math.max(0, idx)));
        }}
        onMouseLeave={() => setHoverIdx(null)}
      >
        {Array.from({ length: gridLines + 1 }, (_, g) => {
          const y = padT + (g / gridLines) * (height - padT - padB);
          const v = maxLoss - (g / gridLines) * (maxLoss - minLoss);
          return (
            <g key={g}>
              <line x1={padL} x2={width - padR} y1={y} y2={y} stroke="#2c2c2a" strokeWidth={1} />
              <text x={padL - 6} y={y + 3} textAnchor="end" fontSize={9} fill="#71717a">
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}

        <path d={pathFor("loss")} fill="none" stroke="#52525b" strokeWidth={1.5} strokeLinejoin="round" />
        <path d={pathFor("smoothed")} fill="none" stroke="#8b5cf6" strokeWidth={2} strokeLinejoin="round" />

        {hover && (
          <>
            <line
              x1={xAt(hoverIdx!)}
              x2={xAt(hoverIdx!)}
              y1={padT}
              y2={height - padB}
              stroke="#ffffff22"
              strokeWidth={1}
            />
            <circle cx={xAt(hoverIdx!)} cy={yAt(hover.smoothed)} r={3.5} fill="#8b5cf6" />
            <circle cx={xAt(hoverIdx!)} cy={yAt(hover.loss)} r={3} fill="#52525b" />
          </>
        )}
      </svg>
      {hover && hoverIdx !== null && (
        <div
          className="pointer-events-none absolute top-1 rounded-lg border border-white/10 bg-[#1a1a20] px-2.5 py-1.5 text-[11px] shadow-xl"
          style={{
            left: `${Math.min(78, Math.max(2, (xAt(hoverIdx) / width) * 100))}%`,
          }}
        >
          <div className="font-mono text-zinc-400">step {hover.step.toLocaleString()}</div>
          <div className="font-mono text-zinc-300">loss {hover.loss.toFixed(3)}</div>
          <div className="font-mono text-violet-300">smoothed {hover.smoothed.toFixed(3)}</div>
        </div>
      )}
    </div>
  );
}

/* ── Nav ─────────────────────────────────────────────────────────────── */

function Nav() {
  return (
    <nav className="sticky top-0 z-50 border-b border-white/[0.06] bg-[#09090b]/95 px-6 backdrop-blur-xl">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between">
        <a href={IDEAWEAVER_HOME} className="flex items-center gap-3">
          <img src="/logo.png" alt="IdeaWeaver AI Labs" className="h-9 w-auto" />
          <span className="hidden text-sm font-semibold text-zinc-500 sm:inline">/ SLM Builder</span>
        </a>
        <div className="hidden items-center gap-8 text-sm md:flex">
          <a href={IDEAWEAVER_HOME} className="text-zinc-400 transition hover:text-white">Home</a>
          <a href={IDEAWEAVER_COURSES} className="text-zinc-400 transition hover:text-white">Courses</a>
          <a href={GITHUB_REPO} target="_blank" rel="noopener noreferrer" className="text-zinc-400 transition hover:text-white">GitHub</a>
        </div>
        <a
          href={IDEAWEAVER_COURSES}
          className="shimmer-hover relative overflow-hidden inline-block rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow-[0_0_30px_rgba(124,58,237,0.35)] transition hover:scale-[1.03]"
        >
          Learn to build this
        </a>
      </div>
    </nav>
  );
}

/* ── Page ────────────────────────────────────────────────────────────── */

type TrainStatus = "idle" | "starting" | "preparing_data" | "training" | "done" | "error" | "stopped";

export default function SLMBuilder() {
  const [cfg, setCfg] = useState<Config>(PRESET_T4_20L);
  const [activePreset, setActivePreset] = useState("t4-20l");
  const [history, setHistory] = useState<Point[]>([]);
  const [trainStatus, setTrainStatus] = useState<TrainStatus>("idle");
  const [trainMessage, setTrainMessage] = useState("");
  const [hasCheckpoint, setHasCheckpoint] = useState(false);
  const [backendUp, setBackendUp] = useState<boolean | null>(null); // null = still checking
  const [backendSlow, setBackendSlow] = useState(false); // last check timed out rather than failing to connect
  const [elapsedSec, setElapsedSec] = useState(0);
  const esRef = useRef<EventSource | null>(null);
  const runStartRef = useRef<number | null>(null);
  const running = trainStatus === "starting" || trainStatus === "preparing_data" || trainStatus === "training";

  const set = <K extends keyof Config>(key: K, value: Config[K]) =>
    setCfg((prev) => ({ ...prev, [key]: value }));

  const resetTrainingUI = () => {
    esRef.current?.close();
    esRef.current = null;
    runStartRef.current = null;
    setHistory([]);
    setTrainStatus("idle");
    setTrainMessage("");
    setHasCheckpoint(false);
    setElapsedSec(0);
  };

  const applyPreset = (id: string) => {
    const p = PRESETS.find((x) => x.id === id);
    if (!p) return;
    setCfg(p.cfg);
    setActivePreset(id);
    resetTrainingUI();
  };

  const params = useMemo(() => estimateParams(cfg), [cfg]);

  const effectiveTokensPerStep = cfg.batchSize * cfg.gradAccumSteps * cfg.blockSize;
  const stepsPerEpoch = Math.max(1, Math.round(TINYSTORIES_TOTAL_TOKENS / effectiveTokensPerStep));
  const totalTrainingTokens = cfg.maxIters * effectiveTokensPerStep;
  const epochs = totalTrainingTokens / TINYSTORIES_TOTAL_TOKENS;

  const gpuInfo = GPU_INFO[cfg.gpu];
  const bytesPerParam = 2; // bf16/fp16
  const modelMB = (params.total * bytesPerParam) / 1e6;
  const activationMB =
    (cfg.batchSize * cfg.blockSize * cfg.embDim * cfg.nLayers * 2) / 1e6;
  const peakVramMB = modelMB * 4 + activationMB;
  const peakVramGB = peakVramMB / 1000;

  const warnings = useMemo(() => {
    const w: { level: "warn" | "info"; text: string }[] = [];
    if (cfg.precision === "bfloat16" && !gpuInfo.bf16TensorCores) {
      w.push({
        level: "warn",
        text: `${cfg.gpu} (${gpuInfo.arch}) has no bf16 Tensor Core support — bf16 matmuls fall back to slow, non-accelerated paths. Switch precision to float16 + GradScaler for full speed.`,
      });
    }
    if (cfg.precision === "float16" && gpuInfo.bf16TensorCores) {
      w.push({
        level: "info",
        text: `${cfg.gpu} has bf16 Tensor Cores — bf16 avoids GradScaler entirely and is the safer default on this GPU.`,
      });
    }
    if (peakVramGB > gpuInfo.vramGB * 0.9) {
      w.push({
        level: "warn",
        text: `Estimated peak VRAM (~${peakVramGB.toFixed(1)} GB) is close to or over the ${gpuInfo.vramGB} GB on ${cfg.gpu}. Lower batch_size or block_size.`,
      });
    }
    if (cfg.blockSize > cfg.contextLength) {
      w.push({
        level: "warn",
        text: `block_size (${cfg.blockSize}) exceeds context_length (${cfg.contextLength}) — the RoPE cache won't cover the full sequence.`,
      });
    }
    if (cfg.numKvSharedLayers >= cfg.nLayers) {
      w.push({
        level: "warn",
        text: `num_kv_shared_layers (${cfg.numKvSharedLayers}) must be smaller than n_layers (${cfg.nLayers}) — every layer needs at least one real K/V donor.`,
      });
    }
    return w;
  }, [cfg, gpuInfo, peakVramGB]);

  const startTraining = async () => {
    setHistory([]);
    setHasCheckpoint(false);
    setTrainStatus("starting");
    setTrainMessage("Starting…");
    runStartRef.current = Date.now();
    setElapsedSec(0);

    let res: Response;
    try {
      res = await fetch("/api/train/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          vocabSize: cfg.vocabSize,
          contextLength: cfg.contextLength,
          embDim: cfg.embDim,
          nHeads: cfg.nHeads,
          nLayers: cfg.nLayers,
          hiddenDim: cfg.hiddenDim,
          headDim: cfg.headDim,
          nKvHeads: cfg.nKvHeads,
          slidingWindow: cfg.slidingWindow,
          globalHeadDim: cfg.globalHeadDim,
          nGlobalKvHeads: cfg.nGlobalKvHeads,
          fullAttnEvery: cfg.fullAttnEvery,
          qkNorm: cfg.qkNorm,
          attentionKEqV: cfg.attentionKEqV,
          ropeLocalBase: cfg.ropeLocalBase,
          ropeBase: cfg.ropeBase,
          partialRotaryFactor: cfg.partialRotaryFactor,
          numKvSharedLayers: cfg.numKvSharedLayers,
          pleDim: cfg.pleDim,
          finalLogitSoftcapping: cfg.finalLogitSoftcapping,
          learningRate: cfg.learningRate,
          maxIters: cfg.maxIters,
          warmupSteps: cfg.warmupSteps,
          minLr: cfg.minLr,
          batchSize: cfg.batchSize,
          blockSize: cfg.blockSize,
          gradAccumSteps: cfg.gradAccumSteps,
          weightDecay: cfg.weightDecay,
          gradClip: cfg.gradClip,
          adamEps: cfg.adamEps,
          beta1: cfg.beta1,
          beta2: cfg.beta2,
          precision: cfg.precision,
        }),
      });
    } catch {
      setTrainStatus("error");
      setTrainMessage("Couldn't reach the training backend. Is train_service.py running?");
      return;
    }

    const data = await res.json().catch(() => ({ ok: false, error: "Bad response from backend." }));
    if (!res.ok || data.ok === false) {
      setTrainStatus("error");
      setTrainMessage(data.error ?? "Couldn't start training.");
      return;
    }

    esRef.current?.close();
    const es = new EventSource("/api/train/stream");
    esRef.current = es;

    es.onmessage = (ev) => {
      const event = JSON.parse(ev.data);
      if (event.type === "point") {
        setTrainStatus("training");
        setHistory((h) => [...h, { step: event.step, loss: event.loss, smoothed: event.smoothed }]);
      } else if (event.type === "status") {
        setTrainStatus(event.status);
        setTrainMessage(event.message ?? "");
        if (event.status === "done" || event.status === "error" || event.status === "stopped") {
          es.close();
        }
      } else if (event.type === "checkpoint") {
        setHasCheckpoint(true);
      }
    };

    es.onerror = () => {
      es.close();
      setTrainStatus((s) => (s === "training" || s === "preparing_data" || s === "starting" ? "error" : s));
      setTrainMessage((m) => m || "Lost connection to the training backend.");
    };
  };

  const stopTraining = () => {
    fetch("/api/train/stop", { method: "POST" }).catch(() => {});
  };

  useEffect(() => {
    return () => {
      esRef.current?.close();
    };
  }, []);

  // Ticks independently of backend messages so the UI always visibly moves
  // during long, mostly-silent phases (dataset download, tokenizing) instead
  // of looking stuck between the backend's own progress updates.
  useEffect(() => {
    if (!running) return;
    const interval = setInterval(() => {
      if (runStartRef.current) setElapsedSec(Math.floor((Date.now() - runStartRef.current) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [running]);

  useEffect(() => {
    let cancelled = false;
    const checkBackend = async () => {
      try {
        const res = await fetch("/api/train/status", {
          cache: "no-store",
          signal: AbortSignal.timeout(5000),
        });
        if (cancelled) return;
        setBackendUp(res.ok);
        if (!res.ok) {
          const body = await res.json().catch(() => null);
          setBackendSlow(body?.reason === "timeout");
        } else {
          setBackendSlow(false);
        }
      } catch {
        if (!cancelled) {
          setBackendUp(false);
          setBackendSlow(false);
        }
      }
    };
    checkBackend();
    const interval = setInterval(checkBackend, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen bg-[#09090b]">
      <Nav />

      {/* Hero */}
      <div className="border-b border-white/[0.06] px-6 py-12">
        <div className="mx-auto max-w-7xl">
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-violet-500/30 bg-violet-500/10 px-4 py-1.5 text-xs font-semibold text-violet-300">
            🧬 Trains a real model on TinyStories — runs locally, nothing leaves this machine
          </div>
          <h1 className="text-3xl font-extrabold text-white sm:text-4xl">IdeaWeaver SLM Builder</h1>
          <p className="mt-3 max-w-2xl text-sm text-zinc-400 leading-relaxed">
            Configure a from-scratch, Gemma-4-Nano-style small language model — interleaved local/global
            attention, grouped-query attention, QK-norm, partial RoPE, and cross-layer KV-cache sharing —
            and see live parameter and VRAM estimates as you go. Same architecture we teach in{" "}
            <a href={IDEAWEAVER_COURSES} className="text-violet-400 hover:text-violet-300">
              Building Small Language Models from Scratch
            </a>
            .
          </p>
        </div>
      </div>

      {/* Body */}
      <div className="mx-auto max-w-7xl px-6 py-8">
        {/* Presets */}
        <div className="mb-6 flex flex-wrap items-center gap-2">
          <span className="mr-1 text-xs font-semibold uppercase tracking-wide text-zinc-500">Presets</span>
          {PRESETS.map((p) => (
            <button
              key={p.id}
              onClick={() => applyPreset(p.id)}
              className={`rounded-lg border px-3 py-1.5 text-left text-xs transition ${
                activePreset === p.id
                  ? "border-violet-500/50 bg-violet-500/10 text-violet-200"
                  : "border-white/[0.08] bg-black/20 text-zinc-400 hover:border-white/[0.2] hover:text-white"
              }`}
            >
              <div className="font-semibold">{p.label}</div>
              <div className="text-[10px] text-zinc-500">{p.blurb}</div>
            </button>
          ))}
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          {/* Left: configuration */}
          <div className="space-y-6 lg:col-span-2">
            <Card
              title="Model Architecture"
              subtitle="Interleaved local/global attention · grouped-query attention"
              icon="🧠"
            >
              <Group label="Core dimensions">
                <NumField label="Vocab size" value={cfg.vocabSize} onChange={(v) => set("vocabSize", v)} step={100} hint="Must match your trained tokenizer" />
                <NumField label="Context length" value={cfg.contextLength} onChange={(v) => set("contextLength", v)} step={128} />
                <NumField label="Embedding dim" value={cfg.embDim} onChange={(v) => set("embDim", v)} step={32} />
                <NumField label="Attention heads" value={cfg.nHeads} onChange={(v) => set("nHeads", v)} />
                <NumField label="Layers" value={cfg.nLayers} onChange={(v) => set("nLayers", v)} />
                <NumField label="FFN hidden dim" value={cfg.hiddenDim} onChange={(v) => set("hiddenDim", v)} step={32} />
              </Group>

              <Group label="Local (sliding) attention">
                <NumField label="Head dim" value={cfg.headDim} onChange={(v) => set("headDim", v)} step={8} />
                <NumField label="KV heads" value={cfg.nKvHeads} onChange={(v) => set("nKvHeads", v)} />
                <NumField label="Sliding window" value={cfg.slidingWindow} onChange={(v) => set("slidingWindow", v)} step={64} />
              </Group>

              <Group label="Global (full) attention">
                <NumField label="Head dim" value={cfg.globalHeadDim} onChange={(v) => set("globalHeadDim", v)} step={8} />
                <NumField label="KV heads" value={cfg.nGlobalKvHeads} onChange={(v) => set("nGlobalKvHeads", v)} />
                <NumField label="Every N layers" value={cfg.fullAttnEvery} onChange={(v) => set("fullAttnEvery", v)} hint="1 global layer per N — the rest are sliding" />
              </Group>

              <Group label="RoPE">
                <NumField label="Local base θ" value={cfg.ropeLocalBase} onChange={(v) => set("ropeLocalBase", v)} step={1000} />
                <NumField label="Global base θ" value={cfg.ropeBase} onChange={(v) => set("ropeBase", v)} step={100000} />
                <NumField label="Partial rotary factor" value={cfg.partialRotaryFactor} onChange={(v) => set("partialRotaryFactor", v)} step={0.05} min={0} max={1} hint="Fraction of global head_dim that gets rotated" />
              </Group>

              <Group label="Memory optimizations">
                <NumField label="KV-shared layers" value={cfg.numKvSharedLayers} onChange={(v) => set("numKvSharedLayers", v)} hint="Trailing layers that reuse an earlier layer's K/V instead of computing their own" />
                <NumField label="Per-layer embed dim" value={cfg.pleDim} onChange={(v) => set("pleDim", v)} step={16} hint="0 disables per-layer embeddings" />
                <NumField label="Logit softcap" value={cfg.finalLogitSoftcapping} onChange={(v) => set("finalLogitSoftcapping", v)} step={5} hint="0 disables tanh softcapping" />
              </Group>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <ToggleField label="QK-RMSNorm" hint="Normalize Q/K before RoPE" checked={cfg.qkNorm} onChange={(v) => set("qkNorm", v)} />
                <ToggleField label="Global K = V" hint="Skip a separate value projection on global layers" checked={cfg.attentionKEqV} onChange={(v) => set("attentionKEqV", v)} />
              </div>
            </Card>

            <Card title="Dataset" subtitle="What the model trains on" icon="📚">
              <SelectField
                label="Dataset"
                value="tinystories"
                onChange={() => {}}
                options={[{ value: "tinystories", label: "TinyStories — ~450M tokens" }]}
                hint="Only option for now — more datasets are on the roadmap. Token count feeds the epoch estimate in the Run panel."
              />
            </Card>

            <Card title="Training Hyperparameters" subtitle="Optimizer, schedule, and batch shape" icon="⚙️">
              <Group label="Optimizer">
                <NumField label="Learning rate" value={cfg.learningRate} onChange={(v) => set("learningRate", v)} step={0.0001} />
                <NumField label="Min LR" value={cfg.minLr} onChange={(v) => set("minLr", v)} step={0.000001} />
                <NumField label="Weight decay" value={cfg.weightDecay} onChange={(v) => set("weightDecay", v)} step={0.01} />
                <NumField label="Grad clip norm" value={cfg.gradClip} onChange={(v) => set("gradClip", v)} step={0.1} />
                <NumField label="Adam β1" value={cfg.beta1} onChange={(v) => set("beta1", v)} step={0.01} />
                <NumField label="Adam β2" value={cfg.beta2} onChange={(v) => set("beta2", v)} step={0.01} />
                <NumField label="Adam eps" value={cfg.adamEps} onChange={(v) => set("adamEps", v)} step={0.00000001} hint="Keep ≥ 1e-8 under bf16/fp16 — smaller can underflow" />
              </Group>
              <Group label="Schedule">
                <NumField label="Max iters (steps)" value={cfg.maxIters} onChange={(v) => set("maxIters", v)} step={500} />
                <NumField label="Warmup steps" value={cfg.warmupSteps} onChange={(v) => set("warmupSteps", v)} step={50} />
                <NumField label="Eval interval" value={cfg.evalInterval} onChange={(v) => set("evalInterval", v)} step={50} />
                <NumField label="Eval iters" value={cfg.evalIters} onChange={(v) => set("evalIters", v)} step={10} />
              </Group>
              <Group label="Batch shape">
                <NumField label="Batch size" value={cfg.batchSize} onChange={(v) => set("batchSize", v)} />
                <NumField label="Block size" value={cfg.blockSize} onChange={(v) => set("blockSize", v)} step={64} />
                <NumField label="Grad accum steps" value={cfg.gradAccumSteps} onChange={(v) => set("gradAccumSteps", v)} />
              </Group>
            </Card>

            <Card title="Hardware & Precision" subtitle="Target GPU shapes the safe defaults" icon="🖥️">
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <SelectField
                  label="Target GPU"
                  value={cfg.gpu}
                  onChange={(v) => set("gpu", v)}
                  options={(Object.keys(GPU_INFO) as GPU[]).map((g) => ({
                    value: g,
                    label: `${g} — ${GPU_INFO[g].vramGB} GB`,
                  }))}
                  hint={GPU_INFO[cfg.gpu].arch}
                />
                <SelectField
                  label="Compute precision"
                  value={cfg.precision}
                  onChange={(v) => set("precision", v)}
                  options={[
                    { value: "bfloat16", label: "bfloat16" },
                    { value: "float16", label: "float16 (+ GradScaler)" },
                  ]}
                />
              </div>
            </Card>
          </div>

          {/* Right: run panel */}
          <div className="lg:col-span-1">
            <div className="sticky top-24 space-y-4">
              <Card title="Run" subtitle="Live estimates from your config" icon="📊">
                <div
                  className={`mb-4 flex items-center gap-2 rounded-lg border px-3 py-2 text-[11px] font-semibold ${
                    backendUp === null
                      ? "border-white/[0.08] bg-black/20 text-zinc-500"
                      : backendUp
                        ? "border-emerald-500/25 bg-emerald-500/[0.06] text-emerald-300"
                        : backendSlow
                          ? "border-amber-500/25 bg-amber-500/[0.06] text-amber-300"
                          : "border-red-500/25 bg-red-500/[0.06] text-red-300"
                  }`}
                >
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${
                      backendUp === null
                        ? "bg-zinc-500"
                        : backendUp
                          ? "bg-emerald-400"
                          : backendSlow
                            ? "bg-amber-400"
                            : "bg-red-400"
                    }`}
                  />
                  {backendUp === null
                    ? "Checking training backend…"
                    : backendUp
                      ? "Training backend connected"
                      : backendSlow
                        ? "Training backend is slow to respond — likely busy preparing/training, not down"
                        : "Training backend unreachable — check the backend cell's output in Colab"}
                </div>

                <div className="mb-4 grid grid-cols-2 gap-2.5">
                  <StatTile label="Parameters" value={fmtParams(params.total)} sub={params.total.toLocaleString()} />
                  <StatTile label="Peak VRAM (est.)" value={`${peakVramGB.toFixed(1)} GB`} sub={`of ${gpuInfo.vramGB} GB on ${cfg.gpu}`} />
                  <StatTile label="Tokens / step" value={fmtTokens(effectiveTokensPerStep)} sub={`${cfg.batchSize}×${cfg.gradAccumSteps}×${cfg.blockSize}`} />
                  <StatTile label="Epochs @ max_iters" value={epochs.toFixed(2)} sub={`~${stepsPerEpoch.toLocaleString()} steps/epoch`} />
                </div>

                {warnings.length > 0 && (
                  <div className="mb-4 space-y-2">
                    {warnings.map((w, i) => (
                      <div
                        key={i}
                        className={`rounded-lg border px-3 py-2 text-[11px] leading-relaxed ${
                          w.level === "warn"
                            ? "border-amber-500/25 bg-amber-500/[0.06] text-amber-200"
                            : "border-sky-500/25 bg-sky-500/[0.06] text-sky-200"
                        }`}
                      >
                        {w.level === "warn" ? "⚠️ " : "ℹ️ "}
                        {w.text}
                      </div>
                    ))}
                  </div>
                )}

                <LossChart data={history} target={1.2} />

                {running && (
                  <div className="mt-3 flex items-center gap-2 text-[11px] text-zinc-500">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-violet-400" />
                    Working… {Math.floor(elapsedSec / 60)}:{String(elapsedSec % 60).padStart(2, "0")} elapsed
                  </div>
                )}

                {trainMessage && (running || trainStatus === "error") && (
                  <div
                    className={`mt-3 rounded-lg border px-3 py-2 text-[11px] leading-relaxed ${
                      trainStatus === "error"
                        ? "border-red-500/25 bg-red-500/[0.06] text-red-200"
                        : "border-white/[0.08] bg-black/20 text-zinc-400"
                    }`}
                  >
                    {trainStatus === "preparing_data" ? "⏳ " : trainStatus === "error" ? "⚠️ " : ""}
                    {trainMessage}
                  </div>
                )}

                <div className="mt-4 flex gap-2">
                  <button
                    onClick={startTraining}
                    disabled={running}
                    className="flex-1 rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-[0_0_30px_rgba(124,58,237,0.35)] transition hover:scale-[1.02] disabled:cursor-not-allowed disabled:opacity-60 disabled:hover:scale-100"
                  >
                    {trainStatus === "starting"
                      ? "Starting…"
                      : trainStatus === "preparing_data"
                        ? "Preparing dataset…"
                        : trainStatus === "training"
                          ? `Training… step ${history[history.length - 1]?.step ?? 0}/${cfg.maxIters}`
                          : "▶ Start Training"}
                  </button>
                  {running && (
                    <button
                      onClick={stopTraining}
                      className="rounded-xl border border-white/[0.15] px-4 py-2.5 text-sm font-semibold text-zinc-300 transition hover:border-red-500/40 hover:text-red-300"
                    >
                      Stop
                    </button>
                  )}
                </div>

                {hasCheckpoint && !running && (
                  <a
                    href="/api/train/checkpoint"
                    className="mt-2 block w-full rounded-lg border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-center text-xs font-semibold text-violet-200 transition hover:border-violet-500/50"
                  >
                    ⬇ Download checkpoint (.pt)
                  </a>
                )}

                <p className="mt-2 text-center text-[10px] text-zinc-600">
                  Trains a real model on TinyStories — needs the Python backend running (see README).
                </p>
              </Card>
            </div>
          </div>
        </div>

        {/* CTA */}
        <div className="mt-10 rounded-2xl border border-violet-500/20 bg-gradient-to-r from-violet-500/[0.06] to-indigo-500/[0.04] p-8 text-center">
          <p className="text-sm text-zinc-300 leading-relaxed">
            Want to understand every one of these variables — attention internals, RoPE, KV caching, and the
            training loop — not just tune them?
          </p>
          <a
            href={IDEAWEAVER_COURSES}
            className="mt-4 inline-block rounded-xl bg-gradient-to-r from-violet-600 to-indigo-600 px-7 py-3 text-sm font-semibold text-white shadow-[0_0_30px_rgba(124,58,237,0.35)] transition hover:scale-[1.03] hover:shadow-[0_0_50px_rgba(124,58,237,0.5)]"
          >
            Building Small Language Models from Scratch →
          </a>
        </div>
      </div>
    </div>
  );
}
