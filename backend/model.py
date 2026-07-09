"""
Gemma-4-Nano-style architecture — ported verbatim from the training script
this UI configures (interleaved local/global attention, GQA, QK-RMSNorm,
partial RoPE, cross-layer KV-cache sharing). No training-loop code here,
just the model definition, so train_service.py can build it from whatever
config the frontend sends.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_rope_params(head_dim, theta_base=10_000, context_length=4096,
                         partial_rotary_factor=1.0, dtype=torch.float32):
    rotary_dim = max(2, int(head_dim * partial_rotary_factor) // 2 * 2)
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    _, _, seq_len, _ = x.shape
    rotary_dim = cos.shape[-1] * 2

    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)

    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    x_rot = x_rot * cos_full + rotated * sin_full

    return torch.cat([x_rot, x_pass], dim=-1)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            x = x * (1.0 + self.scale.float())
        return x.to(dtype)


class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        layer_type = cfg["layer_types"][layer_idx]
        self.is_sliding = (layer_type == "sliding_attention")
        self.num_heads = cfg["n_heads"]

        if self.is_sliding:
            self.head_dim = cfg["head_dim"]
            self.num_kv_heads = cfg["n_kv_heads"]
            self.use_k_eq_v = False
        else:
            self.head_dim = cfg["global_head_dim"]
            self.num_kv_heads = cfg["n_global_kv_heads"]
            self.use_k_eq_v = cfg.get("attention_k_eq_v", False)

        self.group_size = self.num_heads // self.num_kv_heads
        self.d_out = self.num_heads * self.head_dim

        n_layers = cfg["n_layers"]
        n_shared = cfg.get("num_kv_shared_layers", 0)
        first_shared = n_layers - n_shared
        self.is_kv_shared = layer_idx >= first_shared > 0
        self.is_donor = False
        self.kv_donor_idx = None

        if self.is_kv_shared:
            non_shared_types = cfg["layer_types"][:first_shared]
            self.kv_donor_idx = len(non_shared_types) - 1 - non_shared_types[::-1].index(layer_type)
        elif n_shared > 0:
            non_shared_types = cfg["layer_types"][:first_shared]
            last_of_type = len(non_shared_types) - 1 - non_shared_types[::-1].index(layer_type)
            self.is_donor = (layer_idx == last_of_type)

        d_in = cfg["emb_dim"]
        dtype = cfg.get("dtype")

        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim)

        if not self.is_kv_shared:
            self.W_key = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
            self.k_norm = RMSNorm(self.head_dim)
            self.v_norm = RMSNorm(self.head_dim, with_scale=False)
            if not self.use_k_eq_v:
                self.W_value = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin, shared_kv_states):
        b, seq_len, _ = x.shape

        queries = self.W_query(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.q_norm(queries)
        queries = apply_rope(queries, cos, sin)

        if self.is_kv_shared:
            keys, values = shared_kv_states[self.kv_donor_idx]
        else:
            k_raw = self.W_key(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            if self.use_k_eq_v:
                v_raw = k_raw
            else:
                v_raw = self.W_value(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            keys = apply_rope(self.k_norm(k_raw), cos, sin)
            values = self.v_norm(v_raw)

            if self.is_donor:
                shared_kv_states[self.layer_idx] = (keys, values)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(queries.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, seq_len, self.d_out)
        return self.out_proj(context)


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = cfg.get("dtype")
        self.gate = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=dtype)
        self.up = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=dtype)
        self.down = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False, dtype=dtype)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.attn_type = cfg["layer_types"][layer_idx]
        emb_dim = cfg["emb_dim"]

        self.att = GroupedQueryAttention(cfg, layer_idx)
        self.ff = FeedForward(cfg)

        self.input_layernorm = RMSNorm(emb_dim)
        self.post_attention_layernorm = RMSNorm(emb_dim)
        self.pre_feedforward_layernorm = RMSNorm(emb_dim)
        self.post_feedforward_layernorm = RMSNorm(emb_dim)

        self.register_buffer("layer_scalar", torch.ones(1, dtype=cfg.get("dtype", torch.float32)))

        self.ple_dim = cfg.get("ple_dim", 0)
        if self.ple_dim > 0:
            dtype = cfg.get("dtype")
            self.ple_gate = nn.Linear(emb_dim, self.ple_dim, bias=False, dtype=dtype)
            self.ple_proj = nn.Linear(self.ple_dim, emb_dim, bias=False, dtype=dtype)
            self.post_ple_norm = RMSNorm(emb_dim)

    def forward(self, x, mask, cos, sin, shared_kv_states, per_layer_input=None):
        input_dtype = x.dtype

        shortcut = x
        x = self.input_layernorm(x)
        x = self.att(x, mask, cos, sin, shared_kv_states)
        x = self.post_attention_layernorm(x)
        x = shortcut + x

        shortcut = x
        x = self.pre_feedforward_layernorm(x)
        x = self.ff(x)
        x = self.post_feedforward_layernorm(x)
        x = shortcut + x

        if self.ple_dim > 0 and per_layer_input is not None:
            shortcut = x
            h = F.gelu(self.ple_gate(x), approximate="tanh") * per_layer_input
            h = self.ple_proj(h)
            h = self.post_ple_norm(h)
            x = shortcut + h

        return (x * self.layer_scalar).to(input_dtype)


class Gemma4Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_layers = cfg["n_layers"]
        emb_dim = cfg["emb_dim"]
        ple_dim = cfg.get("ple_dim", 0)
        dtype = cfg.get("dtype")

        self.tok_emb = nn.Embedding(cfg["vocab_size"], emb_dim, dtype=dtype)
        self.out_head = nn.Linear(emb_dim, cfg["vocab_size"], bias=False, dtype=dtype)
        self.out_head.weight = self.tok_emb.weight

        self.ple_dim = ple_dim
        if ple_dim > 0:
            self.tok_emb_per_layer = nn.Embedding(cfg["vocab_size"], n_layers * ple_dim, dtype=dtype)
            self.ple_model_proj = nn.Linear(emb_dim, n_layers * ple_dim, bias=False, dtype=dtype)
            self.ple_proj_norm = RMSNorm(ple_dim)
            self.ple_embed_scale = ple_dim ** 0.5
            self.ple_model_proj_scale = emb_dim ** -0.5
            self.ple_combine_scale = 2.0 ** -0.5

        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(n_layers)])
        self.final_norm = RMSNorm(emb_dim)

        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"], theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"], partial_rotary_factor=1.0)
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["global_head_dim"], theta_base=cfg["rope_base"],
            context_length=cfg["context_length"], partial_rotary_factor=cfg["partial_rotary_factor"])
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        mask_global = torch.triu(ones, diagonal=1)
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

        per_layer_inputs = None
        if self.ple_dim > 0:
            n_layers = self.cfg["n_layers"]
            ple_token = self.tok_emb_per_layer(input_ids) * self.ple_embed_scale
            ple_token = ple_token.view(b, seq_len, n_layers, self.ple_dim)
            ple_ctx = self.ple_model_proj(x) * self.ple_model_proj_scale
            ple_ctx = ple_ctx.view(b, seq_len, n_layers, self.ple_dim)
            ple_ctx = self.ple_proj_norm(ple_ctx)
            per_layer_inputs = (ple_token + ple_ctx) * self.ple_combine_scale

        mask_global, mask_local = self._create_masks(seq_len, x.device)
        shared_kv_states = {}

        for i, block in enumerate(self.blocks):
            if block.attn_type == "sliding_attention":
                mask, cos, sin = mask_local, self.cos_local, self.sin_local
            else:
                mask, cos, sin = mask_global, self.cos_global, self.sin_global
            ple_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            x = block(x, mask, cos, sin, shared_kv_states, per_layer_input=ple_input)

        x = self.final_norm(x)
        logits = self.out_head(x.to(self.cfg.get("dtype", torch.float32)))
        softcap = self.cfg.get("final_logit_softcapping")
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx_len = self.cfg["context_length"]
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
