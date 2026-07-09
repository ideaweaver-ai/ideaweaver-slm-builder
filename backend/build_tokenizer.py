"""
Build a custom 8K-vocabulary SentencePiece tokenizer for TinyStories, and
tokenize the full dataset into train.bin / validation.bin. Ported from the
working build_tokenizer.py used to train the reference Gemma-4-Nano models
this UI is built around — this build only supports TinyStories.

All artifacts (tokenizer files, .bin files) are written to the current
working directory; train_service.py runs this from backend/data/.

Every slow step reports through `progress(msg)` so the frontend's status
line moves instead of sitting on one static message for minutes — either
real percentages (export, writing) or an elapsed-time heartbeat for the
two calls (SentencePiece training, HF datasets.map) that are opaque,
blocking library calls with no per-item hook to instrument directly.
"""

import os
import threading
import time

import numpy as np
from tqdm.auto import tqdm

VOCAB_SIZE = 8000
TOKENIZER_PREFIX = "tinystories_tokenizer"
MODEL_FILE = f"{TOKENIZER_PREFIX}.model"


def _run_with_heartbeat(progress, label, fn, interval=4):
    """Run a blocking call, emitting '<label>… (Ns elapsed)' every `interval`
    seconds on a background thread until it returns — for library calls with
    no per-item progress hook, so the status line still visibly moves."""
    if progress is None:
        return fn()

    stop = threading.Event()
    start = time.time()

    def _tick():
        while not stop.wait(interval):
            progress(f"{label}… ({int(time.time() - start)}s elapsed)")

    t = threading.Thread(target=_tick, daemon=True)
    t.start()
    try:
        return fn()
    finally:
        stop.set()
        t.join(timeout=1)


def export_text_for_training(output_file="tinystories_raw.txt", max_samples=500_000, progress=None):
    from datasets import load_dataset

    if os.path.exists(output_file):
        return output_file

    if progress:
        progress("Downloading TinyStories dataset from Hugging Face...")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    num_samples = min(max_samples, len(ds))
    if progress:
        progress(f"Exporting {num_samples:,} samples for tokenizer training...")

    report_every = max(1, num_samples // 20)  # ~20 updates across the loop
    with open(output_file, "w", encoding="utf-8") as f:
        for i in tqdm(range(num_samples), desc="exporting text"):
            text = ds[i]["text"].strip()
            if text:
                f.write(text + "\n")
            if progress and (i + 1) % report_every == 0:
                pct = round((i + 1) / num_samples * 100)
                progress(f"Exporting samples for tokenizer training: {i + 1:,}/{num_samples:,} ({pct}%)")

    return output_file


def train_tokenizer(input_file="tinystories_raw.txt", progress=None):
    import sentencepiece as spm

    if os.path.exists(MODEL_FILE):
        return

    def _train():
        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=TOKENIZER_PREFIX,
            vocab_size=VOCAB_SIZE,
            model_type="bpe",
            character_coverage=1.0,
            num_threads=os.cpu_count(),
            split_digits=True,
            byte_fallback=True,
            pad_id=3,
            unk_id=0,
            bos_id=1,
            eos_id=2,
            max_sentence_length=16384,
        )

    _run_with_heartbeat(progress, f"Training SentencePiece tokenizer (vocab_size={VOCAB_SIZE})", _train)


class TinyStoriesTokenizer:
    """Lightweight wrapper around the trained SentencePiece model."""

    def __init__(self, model_path=MODEL_FILE):
        import sentencepiece as spm
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

    @property
    def vocab_size(self):
        return self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, ids):
        return self.sp.decode(ids)


def prepare_data(progress=None):
    """Tokenize the full TinyStories dataset into train.bin / validation.bin."""
    from datasets import load_dataset

    if os.path.exists("train.bin") and os.path.exists("validation.bin"):
        return

    tokenizer = TinyStoriesTokenizer(MODEL_FILE)

    if progress:
        progress("Downloading full TinyStories dataset...")
    ds = load_dataset("roneneldan/TinyStories")

    def process(example):
        ids = tokenizer.encode(example["text"])
        return {"ids": ids, "len": len(ids)}

    num_proc = min(8, os.cpu_count() or 1)
    total_examples = sum(len(d) for d in ds.values())

    def _tokenize():
        return ds.map(
            process,
            remove_columns=["text"],
            desc="tokenizing",
            num_proc=num_proc,
        )

    tokenized = _run_with_heartbeat(
        progress,
        f"Tokenizing {total_examples:,} stories across {num_proc} processes (the slow step)",
        _tokenize,
    )

    dtype = np.uint16 if tokenizer.vocab_size < 2**16 else np.uint32

    for split, dset in tokenized.items():
        arr_len = np.sum(dset["len"], dtype=np.uint64)
        filename = f"{split}.bin"
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024
        report_every = max(1, total_batches // 20)  # ~20 updates per split
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")
            arr_batch = np.concatenate(batch["ids"])
            arr[idx: idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
            if progress and (batch_idx + 1) % report_every == 0:
                pct = round((batch_idx + 1) / total_batches * 100)
                progress(f"Writing {filename}: {idx:,}/{arr_len:,} tokens ({pct}%)")
        arr.flush()

    if os.path.exists("tinystories_raw.txt"):
        os.remove("tinystories_raw.txt")


def ensure_tinystories_ready(progress=None):
    """Idempotent: builds whatever's missing, skips whatever already exists."""
    if os.path.exists("train.bin") and os.path.exists("validation.bin") and os.path.exists(MODEL_FILE):
        if progress:
            progress("TinyStories tokenizer + data already present, skipping build.")
        return
    text_file = export_text_for_training(progress=progress)
    train_tokenizer(text_file, progress=progress)
    prepare_data(progress=progress)
