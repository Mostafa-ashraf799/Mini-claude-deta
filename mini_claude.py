"""
mini_claude.py - نموذج لغة من الصفر، بتختار انت شكله

ثلاث معماريات (--arch)، كل واحدة بتشتغل لوحدها:
  dense   نموذج عادي: كل التوكنز بتعدي على كل الباراميترز
  moe     Mixture of Experts: خبراء بالتوازي وكل توكن بيروح لـ top-k منهم
  towers  كل خبير "برج" كامل (سلسلة طبقات). راوتر بيختار برج لكل نص، والبرج المختار بس هو اللي بيشتغل

وخيارات تتركب مع أي معمارية:
  --vision on   يضيف مشفّر صور (ViT) عشان النموذج يحلل الصور (اختياري)
  --tools       دعم الأدوات (تلقائي من 2b وفوق)

مميزات عامة: RoPE + RMSNorm + SwiGLU + GQA، أحجام من ~15M لـ ~9B، KV cache، DDP، gradient checkpointing،
fp16/bf16، حفظ واستكمال، SFT على محادثات، chat تفاعلي بيشغّل الأدوات.

الأوامر: info | selftest | prep | train | prep_sft | sft | generate | chat
الشرح الكامل في README_mini_claude.md
"""
import argparse
import ast
import datetime
import glob
import importlib.util
import json
import math
import operator
import os
import random
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

EOS, SYS, USR, AST, END = "<|endoftext|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"
TOOL, TCALL, TCALL_END = "<|tool|>", "<|tool_call|>", "<|/tool_call|>"
IMG = "<|image|>"
SPECIAL = [EOS, SYS, USR, AST, END, TOOL, TCALL, TCALL_END, IMG]  # الترتيب ثابت (ids 0..8)
CORE_SPECIAL = SPECIAL[:8]  # الـ tokenizers القديمة (من غير <|image|>) لسه شغالة لو مش هتستخدم صور
ROLES = ("system", "user", "assistant", "tool")
ARCHS = ("dense", "moe", "towers")

# tools=True معناها إن الحجم ده بيدعم الأدوات تلقائياً (من 2b وفوق)
PRESETS = {
    "tiny":  dict(n_layer=6,  n_head=6,  n_kv_head=2, n_embd=384,  ctx=512,  lr=1e-3,  tools=False),
    "small": dict(n_layer=8,  n_head=8,  n_kv_head=4, n_embd=512,  ctx=1024, lr=8e-4,  tools=False),
    "base":  dict(n_layer=12, n_head=12, n_kv_head=4, n_embd=768,  ctx=1024, lr=6e-4,  tools=False),
    "large": dict(n_layer=24, n_head=16, n_kv_head=4, n_embd=1024, ctx=2048, lr=4e-4,  tools=False),
    "xl":    dict(n_layer=20, n_head=16, n_kv_head=4, n_embd=2048, ctx=2048, lr=3e-4,  tools=False),
    "1.5b":  dict(n_layer=32, n_head=16, n_kv_head=4, n_embd=2048, ctx=4096, lr=3e-4,  tools=False),
    "2b":    dict(n_layer=28, n_head=20, n_kv_head=5, n_embd=2560, ctx=4096, lr=3e-4,  tools=True),
    "3b":    dict(n_layer=29, n_head=24, n_kv_head=8, n_embd=3072, ctx=4096, lr=3e-4,  tools=True),
    "4b":    dict(n_layer=39, n_head=24, n_kv_head=8, n_embd=3072, ctx=4096, lr=2.5e-4, tools=True),
    "5b":    dict(n_layer=36, n_head=28, n_kv_head=7, n_embd=3584, ctx=4096, lr=2.5e-4, tools=True),
    "7b":    dict(n_layer=39, n_head=32, n_kv_head=8, n_embd=4096, ctx=4096, lr=2e-4,  tools=True),
    "8b":    dict(n_layer=45, n_head=32, n_kv_head=8, n_embd=4096, ctx=4096, lr=2e-4,  tools=True),
    "9b":    dict(n_layer=39, n_head=36, n_kv_head=12, n_embd=4608, ctx=4096, lr=2e-4, tools=True),
}


def vision_defaults(size):
    """مشفّر الصور المناسب لكل حجم لغة"""
    if size in ("tiny", "small"):
        return dict(img_size=64, patch=16, v_dim=192, v_layers=3, pool=1)        # 16 توكن للصورة
    if size in ("base", "large"):
        return dict(img_size=224, patch=16, v_dim=384, v_layers=6, pool=2)       # 49 توكن
    if size in ("xl", "1.5b", "2b", "3b", "4b", "5b"):
        return dict(img_size=224, patch=14, v_dim=768, v_layers=12, pool=2)      # 64 توكن
    return dict(img_size=224, patch=14, v_dim=1024, v_layers=16, pool=2)         # 64 توكن


@dataclass
class Config:
    vocab: int
    n_layer: int              # في towers = عدد طبقات كل برج
    n_head: int
    n_kv_head: int
    n_embd: int
    ctx: int
    arch: str = "dense"       # dense | moe | towers
    # --- MoE ---
    moe_experts: int = 0
    moe_topk: int = 2         # كام خبير يشتغلوا لكل توكن
    moe_every: int = 2        # كل كام طبقة تبقى MoE
    moe_shared: int = 0       # خبراء ثابتين بيشتغلوا دايماً
    moe_div: int = 2          # حجم الخبير = حجم الـ FFN العادي ÷ moe_div
    aux_coef: float = 0.01
    # --- towers ---
    n_towers: int = 0
    router_coef: float = 0.1  # وزن loss الراوتر (تصنيف المجال)
    # --- خيارات ---
    tools: bool = False
    vision: bool = False
    img_size: int = 64
    patch: int = 16
    v_dim: int = 192
    v_layers: int = 3
    v_heads: int = 0          # 0 = تلقائي (v_dim ÷ 64)
    pool: int = 1             # تجميع patches متجاورة لتقليل عدد توكنز الصورة
    img_id: int = -1          # رقم توكن <|image|> في الـ tokenizer

    def __post_init__(self):
        if self.arch == "dense" and self.moe_experts:  # checkpoints قديمة
            self.arch = "moe"
        assert self.arch in ARCHS, f"arch لازم تكون واحدة من {ARCHS}"
        assert self.n_embd % self.n_head == 0 and self.n_head % self.n_kv_head == 0
        assert (self.n_embd // self.n_head) % 2 == 0
        if self.arch == "moe":
            assert 1 <= self.moe_topk <= self.moe_experts and self.moe_div >= 1
        if self.arch == "towers":
            assert self.n_towers >= 2, "towers محتاج برجين على الأقل"
        if self.vision:
            if not self.v_heads:
                self.v_heads = max(1, self.v_dim // 64)
            assert self.v_dim % self.v_heads == 0
            assert self.img_size % (self.patch * self.pool) == 0, "img_size لازم يقبل القسمة على patch*pool"

    @property
    def n_img_tokens(self):
        return (self.img_size // self.patch // self.pool) ** 2


def ffn_hidden(d):
    return (int(8 * d / 3) + 63) // 64 * 64


def expert_hidden(c: Config):
    return ffn_hidden(c.n_embd) // c.moe_div


def is_moe(c: Config, i: int):
    return c.arch == "moe" and i % c.moe_every == c.moe_every - 1


def layer_params(c: Config, moe: bool):
    """(total, active) لطبقة واحدة"""
    d, hd = c.n_embd, c.n_embd // c.n_head
    base = 2 * d + 2 * d * d + 2 * d * c.n_kv_head * hd
    if moe:
        e = 3 * d * expert_hidden(c)
        return (base + (c.moe_experts + c.moe_shared) * e + d * c.moe_experts,
                base + (c.moe_topk + c.moe_shared) * e + d * c.moe_experts)
    p = base + 3 * d * ffn_hidden(d)
    return p, p


def vision_params(c: Config):
    if not c.vision:
        return 0
    vd, p = c.v_dim, c.patch
    n = (c.img_size // p) ** 2
    per_layer = 12 * vd * vd + 2 * vd
    return 3 * p * p * vd + n * vd + c.v_layers * per_layer + vd + vd * c.pool ** 2 * c.n_embd + c.n_embd ** 2


def estimate_params(c: Config):
    """(total, active) بدون ما نبني النموذج. active = اللي بيشتغل فعلاً لكل توكن"""
    d = c.n_embd
    total = active = c.vocab * d
    if c.arch == "towers":
        lt, _ = layer_params(c, False)
        tower = c.n_layer * lt + d
        router = d * d + d * c.n_towers
        total += c.n_towers * tower + router
        active += tower + router
    else:
        total += d
        active += d
        for i in range(c.n_layer):
            lt, la = layer_params(c, is_moe(c, i))
            total += lt
            active += la
    v = vision_params(c)
    return total + v, active + v


# ----------------------------------------------------------------------------
# مشفّر الصور
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).type_as(x) * self.w


class ViTBlock(nn.Module):
    """طبقة transformer عادية (attention ثنائي الاتجاه) لمعالجة patches الصورة"""

    def __init__(self, d, nh):
        super().__init__()
        self.nh = nh
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.fc1 = nn.Linear(d, 4 * d, bias=False)
        self.fc2 = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, self.nh, D // self.nh).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v)
        x = x + self.o(y.transpose(1, 2).reshape(B, N, D))
        return x + self.fc2(F.gelu(self.fc1(self.n2(x))))


class VisionEncoder(nn.Module):
    """صورة (B,3,H,W) بقيم [-1,1]  ->  (B, n_img_tokens, n_embd): توكنز بنفس مساحة توكنز النص"""

    def __init__(self, c: Config):
        super().__init__()
        self.p, self.pool = c.patch, c.pool
        self.grid = c.img_size // c.patch
        vd = c.v_dim
        self.patch = nn.Linear(3 * self.p * self.p, vd, bias=False)
        self.pos = nn.Parameter(torch.randn(self.grid * self.grid, vd) * 0.02)
        self.blocks = nn.ModuleList([ViTBlock(vd, c.v_heads) for _ in range(c.v_layers)])
        self.norm = RMSNorm(vd)
        self.proj1 = nn.Linear(vd * c.pool ** 2, c.n_embd, bias=False)
        self.proj2 = nn.Linear(c.n_embd, c.n_embd, bias=False)

    def forward(self, img):
        B, p, g, k = img.size(0), self.p, self.grid, self.pool
        x = img.unfold(2, p, p).unfold(3, p, p)                         # B,3,g,g,p,p
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B, g * g, 3 * p * p)
        x = self.patch(x) + self.pos
        for b in self.blocks:
            x = b(x)
        x = self.norm(x)
        if k > 1:
            vd = x.size(-1)
            x = x.reshape(B, g // k, k, g // k, k, vd).permute(0, 1, 3, 2, 4, 5).reshape(B, (g // k) ** 2, k * k * vd)
        return self.proj2(F.gelu(self.proj1(x)))


def load_image(path, size):
    try:
        from PIL import Image
    except ImportError:
        raise SystemExit("محتاج مكتبة Pillow للصور: pip install pillow")
    im = Image.open(path).convert("RGB").resize((size, size))
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)


# ----------------------------------------------------------------------------
# النموذج
# ----------------------------------------------------------------------------
def rope_tables(ctx, head_dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    f = torch.outer(torch.arange(ctx).float(), inv)  # (ctx, hd/2)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    # x: (B, H, T, hd) | cos, sin: (T, hd/2)
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h], x[..., h:]
    cos, sin = cos[None, None].to(x.dtype), sin[None, None].to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.nh, self.nkv, self.hd = c.n_head, c.n_kv_head, c.n_embd // c.n_head
        self.q = nn.Linear(c.n_embd, self.nh * self.hd, bias=False)
        self.kv = nn.Linear(c.n_embd, 2 * self.nkv * self.hd, bias=False)
        self.proj = nn.Linear(self.nh * self.hd, c.n_embd, bias=False)

    def forward(self, x, cos, sin, past=None, use_cache=False):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k, v = self.kv(x).view(B, T, 2, self.nkv, self.hd).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if past is not None:
            assert T == 1, "الـ cache بيدعم توكن واحد في المرة بعد الـ prefill"
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        new = (k, v) if use_cache else None  # بنخزن K/V بعدد الـ kv heads (أصغر) = توفير ذاكرة
        if self.nkv != self.nh:
            r = self.nh // self.nkv
            k, v = k.repeat_interleave(r, 1), v.repeat_interleave(r, 1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        return self.proj(y.transpose(1, 2).reshape(B, T, -1)), new


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate = nn.Linear(d, h, bias=False)
        self.up = nn.Linear(d, h, bias=False)
        self.down = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """
    راوتر بيختار top-k خبراء لكل توكن + loss لتوزيع الحمل بين الخبراء.
    التوزيع بيتم بترتيب التوكنز حسب الخبير (sort) وكل خبير بياخد دفعة واحدة متصلة،
    فبيقدر يشتغل بعشرات الخبراء من غير ما يبقى بطيء بشكل مبالغ فيه.
    shared: خبراء بيشتغلوا مع كل التوكنز (بيمسكوا المعرفة العامة، والـ routed بيتخصصوا).
    """

    def __init__(self, c: Config):
        super().__init__()
        self.E, self.k = c.moe_experts, c.moe_topk
        eh = expert_hidden(c)
        self.router = nn.Linear(c.n_embd, self.E, bias=False)
        self.experts = nn.ModuleList([SwiGLU(c.n_embd, eh) for _ in range(self.E)])
        self.shared = nn.ModuleList([SwiGLU(c.n_embd, eh) for _ in range(c.moe_shared)])

    def forward(self, x):
        B, T, C = x.shape
        xf = x.reshape(-1, C)
        probs = self.router(xf).float().softmax(-1)
        topv, topi = probs.topk(self.k, -1)
        topv = topv / topv.sum(-1, keepdim=True)

        flat_e = topi.reshape(-1)                      # (N*k) الخبير المختار لكل تعيين
        order = flat_e.argsort()                       # ترتيب التعيينات حسب الخبير
        tok_idx = order // self.k                      # التوكن صاحب كل تعيين
        counts = torch.bincount(flat_e, minlength=self.E).tolist()
        xs = xf[tok_idx]
        ys, s = [], 0
        for e, n in enumerate(counts):
            if n:
                ys.append(self.experts[e](xs[s:s + n]))
            s += n
        y = torch.cat(ys).float() * topv.reshape(-1)[order].unsqueeze(1)
        out = xf.new_zeros(xf.shape, dtype=torch.float32).index_add_(0, tok_idx, y)
        for sh in self.shared:
            out = out + sh(xf).float()

        load = F.one_hot(topi, self.E).sum(1).float().mean(0) / self.k
        aux = self.E * (load * probs.mean(0)).sum()
        return out.to(x.dtype).view(B, T, C), aux


class Block(nn.Module):
    def __init__(self, c: Config, i: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
        self.attn = Attention(c)
        self.moe = is_moe(c, i)
        self.mlp = MoE(c) if self.moe else SwiGLU(c.n_embd, ffn_hidden(c.n_embd))

    def forward(self, x, cos, sin, past=None, use_cache=False):
        a, new = self.attn(self.n1(x), cos, sin, past, use_cache)
        x = x + a
        if self.moe:
            m, aux = self.mlp(self.n2(x))
        else:
            m, aux = self.mlp(self.n2(x)), x.new_zeros((), dtype=torch.float32)
        return x + m, aux, new


def _run_block(b, x, cos, sin):
    y, aux, _ = b(x, cos, sin)
    return y, aux


class GPT(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.grad_ckpt = False
        self.emb = nn.Embedding(c.vocab, c.n_embd)
        if c.arch == "towers":
            self.towers = nn.ModuleList([nn.ModuleList([Block(c, i) for i in range(c.n_layer)])
                                         for _ in range(c.n_towers)])
            self.norms = nn.ModuleList([RMSNorm(c.n_embd) for _ in range(c.n_towers)])
            self.router = nn.Sequential(nn.Linear(c.n_embd, c.n_embd, bias=False), nn.SiLU(),
                                        nn.Linear(c.n_embd, c.n_towers, bias=False))
        else:
            self.blocks = nn.ModuleList([Block(c, i) for i in range(c.n_layer)])
            self.norm = RMSNorm(c.n_embd)
        if c.vision:
            self.vision = VisionEncoder(c)
        self.head = nn.Linear(c.n_embd, c.vocab, bias=False)
        self.head.weight = self.emb.weight
        cos, sin = rope_tables(c.ctx, c.n_embd // c.n_head)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.apply(self._init)
        for n, p in self.named_parameters():
            if n.endswith("proj.weight") or n.endswith("down.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * c.n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def route_logits(self, idx):
        """towers: الراوتر بيبص على متوسط embeddings النص كله (نفس الدالة في التدريب والتشغيل)"""
        return self.router(self.emb(idx).mean(1))

    def _run_tower(self, t, x, cos, sin, caches, use_cache):
        new = []
        for i, b in enumerate(self.towers[t]):
            if use_cache:
                x, _, nc = b(x, cos, sin, caches[i], True)
                new.append(nc)
            elif self.grad_ckpt and self.training:
                x, _ = checkpoint(_run_block, b, x, cos, sin, use_reentrant=False)
            else:
                x, _, _ = b(x, cos, sin)
        return self.norms[t](x), new

    def forward(self, idx, targets=None, caches=None, pos0=0, all_logits=False, images=None, tower=None):
        """
        caches=None     : تدريب عادي (من غير cache)
        caches=[None]*L : prefill (بيرجّع cache جديد) | caches=<اللي رجع> : توكن واحد (decode)
        targets: القيمة -100 معناها "تجاهل" (بتتستخدم في SFT)
        images: (n_images,3,H,W) بنفس ترتيب توكنز <|image|> في idx
        tower: (towers بس) رقم البرج، أو tensor بأرقام المجال لكل عينة (وقت التدريب)، أو None = الراوتر يختار
        """
        B, T = idx.shape
        assert pos0 + T <= self.c.ctx, "الطول أكبر من الـ ctx"
        c = self.c
        x = self.emb(idx)

        tw, rloss = None, None
        if c.arch == "towers":
            labels = tower if isinstance(tower, torch.Tensor) else None
            if tower is None or (labels is not None and targets is not None and self.training):
                rl = self.router(x.mean(1))
            if tower is None:
                tw = rl.argmax(-1)
            elif isinstance(tower, int):
                tw = torch.full((B,), tower, dtype=torch.long, device=idx.device)
            else:
                tw = tower.to(idx.device)
                if targets is not None and self.training:
                    rloss = F.cross_entropy(rl.float(), tw)

        if images is not None:
            assert c.vision, "النموذج ده مش متدرّب بـ --vision on"
            v = self.vision(images)
            m = idx == c.img_id
            if int(m.sum()) != v.shape[0] * v.shape[1]:
                raise ValueError(f"عدد توكنز <|image|> ({int(m.sum())}) مش بيساوي توكنز الصور ({v.shape[0] * v.shape[1]})")
            x = x.masked_scatter(m.unsqueeze(-1).expand_as(x), v.to(x.dtype))

        cos, sin = self.cos[pos0:pos0 + T], self.sin[pos0:pos0 + T]
        use_cache = caches is not None
        new_caches, aux_total, n_moe = [], 0.0, 0
        if c.arch == "towers":
            uniq = torch.unique(tw).tolist()
            if use_cache:
                assert len(uniq) == 1, "الـ cache بيدعم برج واحد في الـ batch"
                h, new_caches = self._run_tower(uniq[0], x, cos, sin, caches, True)
            elif len(uniq) == 1:
                h, _ = self._run_tower(uniq[0], x, cos, sin, None, False)
            else:
                h = x.new_zeros(x.shape)
                for t in uniq:
                    rows = (tw == t).nonzero(as_tuple=True)[0]
                    y, _ = self._run_tower(t, x[rows], cos, sin, None, False)
                    h = h.index_copy(0, rows, y.to(h.dtype))
        else:
            for i, b in enumerate(self.blocks):
                if use_cache:
                    x, aux, nc = b(x, cos, sin, caches[i], True)
                    new_caches.append(nc)
                elif self.grad_ckpt and self.training:
                    x, aux = checkpoint(_run_block, b, x, cos, sin, use_reentrant=False)
                else:
                    x, aux, _ = b(x, cos, sin)
                if b.moe:
                    aux_total, n_moe = aux_total + aux, n_moe + 1
            h = self.norm(x)

        if targets is None:
            return self.head(h if all_logits else h[:, -1:]), None, (new_caches if use_cache else None)
        logits = self.head(h)
        if (targets != -100).any():
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), targets.reshape(-1), ignore_index=-100)
        else:
            loss = logits.sum() * 0.0
        if self.training and n_moe:
            loss = loss + c.aux_coef * aux_total / n_moe
        if rloss is not None:
            loss = loss + c.router_coef * rloss
        return logits, loss, None


def count_params(model):
    c = model.c
    total = sum(p.numel() for p in model.parameters())
    active = total
    if c.arch == "towers":
        tp = sum(p.numel() for p in model.towers[0].parameters()) + sum(p.numel() for p in model.norms[0].parameters())
        active -= (c.n_towers - 1) * tp
    for m in model.modules():
        if isinstance(m, MoE):
            ep = sum(p.numel() for e in m.experts for p in e.parameters())
            active -= ep * (m.E - m.k) // m.E
    return total, active


# ----------------------------------------------------------------------------
# الأدوات (tool use)
# ----------------------------------------------------------------------------
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log,
          "log10": math.log10, "exp": math.exp, "abs": abs, "round": round}
_CONSTS = {"pi": math.pi, "e": math.e}


def tool_calculator(expression: str) -> str:
    """آلة حاسبة آمنة (بتفسّر التعبير بالـ AST ومش بتنفّذ كود)"""
    if len(expression) > 200:
        raise ValueError("التعبير طويل")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            l, r = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(r) > 1000:
                raise ValueError("الأس كبير جداً")
            return _OPS[type(n.op)](l, r)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords:
            return _FUNCS[n.func.id](*[ev(x) for x in n.args])
        if isinstance(n, ast.Name) and n.id in _CONSTS:
            return _CONSTS[n.id]
        raise ValueError("تعبير غير مسموح")

    expr = expression.strip().replace("^", "**").replace("×", "*").replace("÷", "/")
    r = ev(ast.parse(expr, mode="eval"))
    return str(r) if isinstance(r, int) else format(r, ".10g")


def tool_get_time() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


BUILTIN_TOOLS = {
    "calculator": dict(fn=tool_calculator, description="يحسب تعبير رياضي (+ - * / ** % // وsqrt وsin وcos وlog وpi)",
                       parameters={"expression": "string"}),
    "get_time": dict(fn=tool_get_time, description="يرجّع التاريخ والوقت الحالي", parameters={}),
}


def load_tools(path=""):
    """الأدوات المدمجة + أدواتك من ملف بايثون فيه: TOOLS = {"name": {"fn":..., "description":..., "parameters":{...}}}
    تنبيه: الملف ده بيتنفّذ كـ كود عادي، فاستخدم ملفات إنت كاتبها أو واثق فيها."""
    tools = dict(BUILTIN_TOOLS)
    if path:
        spec = importlib.util.spec_from_file_location("user_tools", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tools.update(mod.TOOLS)
    return tools


TOOL_HEADER = ("لديك أدوات تقدر تستدعيها. لاستدعاء أداة اكتب: "
               f"{TCALL}" + '{"name": "اسم_الأداة", "arguments": {...}}' + f"{TCALL_END} "
               "وانتظر النتيجة ثم أكمل إجابتك.\nالأدوات المتاحة:\n")


def tools_system_text(tools):
    """tools: dict(name -> spec) أو list من {"name","description","parameters"}"""
    if isinstance(tools, dict):
        tools = [dict(name=n, description=s.get("description", ""), parameters=s.get("parameters", {}))
                 for n, s in tools.items()]
    return TOOL_HEADER + json.dumps(tools, ensure_ascii=False)


CALL_RE = re.compile(re.escape(TCALL) + r"(.*?)" + re.escape(TCALL_END), re.S)


def parse_calls(raw):
    """raw: نص متفكّك مع التوكنز الخاصة. بيرجّع (استدعاءات صحيحة, النص المرئي)"""
    calls = []
    for m in CALL_RE.finditer(raw):
        try:
            d = json.loads(m.group(1))
            args = d.get("arguments", {})
            if isinstance(d, dict) and isinstance(d.get("name"), str) and isinstance(args, dict):
                calls.append({"name": d["name"], "arguments": args})
        except Exception:
            pass
    visible = CALL_RE.sub("", raw)
    for s in SPECIAL:
        visible = visible.replace(s, "")
    return calls, visible.strip()


def run_tool(tools, call):
    spec = tools.get(call["name"])
    if spec is None:
        return f"خطأ: الأداة {call['name']} مش موجودة"
    try:
        return str(spec["fn"](**call["arguments"]))[:2000]
    except Exception as e:
        return f"خطأ: {e}"


# ----------------------------------------------------------------------------
# tokenizer + قالب المحادثة
# ----------------------------------------------------------------------------
def load_tok(out):
    from tokenizers import Tokenizer
    t = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    if any(t.token_to_id(s) is None for s in CORE_SPECIAL):
        raise SystemExit("الـ tokenizer ده قديم (ناقصه توكنز خاصة للمحادثة/الأدوات). اعمل prep من جديد.")
    return t


def load_domains(out):
    p = os.path.join(out, "domains.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)["domains"]
    return None


def role_prefix(tok, role):
    return [tok.token_to_id(f"<|{role}|>")] + tok.encode("\n").ids


def render_message(tok, m, n_img=0):
    """-> (prefix_ids, body_ids). نفس الدالة بتستخدم في التدريب وفي الـ chat فالقالب متطابق"""
    body = []
    if m.get("image") and n_img:
        body += [tok.token_to_id(IMG)] * n_img      # مكان توكنز الصورة (بتتبدل بمخرجات المشفّر)
    if m.get("content"):
        body += tok.encode(m["content"]).ids
    for c in m.get("tool_calls") or []:
        j = json.dumps({"name": c["name"], "arguments": c.get("arguments", {})}, ensure_ascii=False)
        body += [tok.token_to_id(TCALL)] + tok.encode(j).ids + [tok.token_to_id(TCALL_END)]
    body.append(tok.token_to_id(END))
    return role_prefix(tok, m["role"]), body


def build_prompt(tok, msgs, n_img=0):
    ids = []
    for m in msgs:
        pre, body = render_message(tok, m, n_img)
        ids += pre + body
    return ids + role_prefix(tok, "assistant")


def valid_msg(m):
    if not isinstance(m, dict) or m.get("role") not in ROLES:
        return False
    c, tc = m.get("content"), m.get("tool_calls")
    if c is not None and not isinstance(c, str):
        return False
    if m.get("image") is not None and (not isinstance(m["image"], str) or m["role"] != "user"):
        return False
    if tc:
        if m["role"] != "assistant" or not isinstance(tc, list):
            return False
        if not all(isinstance(x, dict) and isinstance(x.get("name"), str)
                   and isinstance(x.get("arguments", {}), dict) for x in tc):
            return False
    return bool(c or tc or m.get("image"))


def encode_chat(tok, msgs, eos_id, n_img=0):
    ids, mask = [], []
    for m in msgs:
        if not valid_msg(m) or (m.get("image") and not n_img):
            return None
        pre, body = render_message(tok, m, n_img)
        learn = 1 if m["role"] == "assistant" else 0
        ids += pre + body
        mask += [0] * len(pre) + [learn] * len(body)
    ids.append(eos_id)
    mask.append(0)
    return (ids, mask) if sum(mask) > 0 else None


def normalize_example(ex):
    if isinstance(ex.get("messages"), list):
        msgs = list(ex["messages"])
    elif "prompt" in ex and "response" in ex:
        msgs = [{"role": "user", "content": ex["prompt"]}, {"role": "assistant", "content": ex["response"]}]
    elif "instruction" in ex and "output" in ex:
        u = ex["instruction"] + (("\n\n" + ex["input"]) if ex.get("input") else "")
        msgs = [{"role": "user", "content": u}, {"role": "assistant", "content": ex["output"]}]
    else:
        return None
    if ex.get("image"):  # صورة على مستوى المثال = تتربط بأول رسالة user
        for i, m in enumerate(msgs):
            if isinstance(m, dict) and m.get("role") == "user":
                msgs[i] = dict(m, image=ex["image"])
                break
    sys_text = ex.get("system")
    if msgs and isinstance(msgs[0], dict) and msgs[0].get("role") == "system":
        sys_text, msgs = msgs[0].get("content") or sys_text, msgs[1:]
    if ex.get("tools"):
        sys_text = ((sys_text or "") + "\n\n" + tools_system_text(ex["tools"])).strip()
    return ([{"role": "system", "content": sys_text}] if sys_text else []) + msgs


def uses_tools(ex, msgs):
    return bool(ex.get("tools")) or any(isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_calls"))
                                        for m in msgs)


# ----------------------------------------------------------------------------
# تجهيز البيانات
# ----------------------------------------------------------------------------
def encode_files(tok, files, out_path, eos_id):
    total = 0
    with open(out_path, "wb") as out:
        def flush(buf):
            ids = [i for e in tok.encode_batch(buf) for i in e.ids]
            out.write(np.array(ids, dtype=np.uint16).tobytes())
            return len(ids)

        for f in files:
            buf, size = [], 0
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    buf.append(line)
                    size += len(line)
                    if size > 2_000_000:
                        total += flush(buf)
                        buf, size = [], 0
            if buf:
                total += flush(buf)
            out.write(np.array([eos_id], dtype=np.uint16).tobytes())
            total += 1
    return total


def prep(a):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    assert a.vocab < 65536, "الـ vocab لازم يكون أقل من 65536"
    os.makedirs(a.out, exist_ok=True)
    if a.by_domain:
        subs = sorted(d for d in os.listdir(a.data) if os.path.isdir(os.path.join(a.data, d)))
        domain_files = {d: sorted(glob.glob(os.path.join(a.data, d, "**", "*.txt"), recursive=True)) for d in subs}
        domain_files = {d: f for d, f in domain_files.items() if f}
        if len(domain_files) < 2:
            raise SystemExit("--by_domain محتاج فولدرين على الأقل جوه --data، وكل فولدر = مجال فيه ملفات .txt")
        files = [f for fl in domain_files.values() for f in fl]
        print(f"المجالات: {', '.join(domain_files)}")
    else:
        files = sorted(glob.glob(os.path.join(a.data, "**", "*.txt"), recursive=True))
        if not files:
            raise SystemExit(f"مفيش ملفات .txt جوه {a.data}")
    print(f"لقيت {len(files)} ملف")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train(files, trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=SPECIAL,
                                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(os.path.join(a.out, "tokenizer.json"))
    eos_id = tok.token_to_id(EOS)

    dpath = os.path.join(a.out, "domains.json")
    if a.by_domain:
        for k, (name, fl) in enumerate(domain_files.items()):
            n = encode_files(tok, fl, os.path.join(a.out, f"train_d{k}.bin"), eos_id)
            print(f"  [{k}] {name}: {n:,} توكن")
        with open(dpath, "w", encoding="utf-8") as f:
            json.dump({"domains": list(domain_files)}, f, ensure_ascii=False)
    else:
        n = encode_files(tok, files, os.path.join(a.out, "train.bin"), eos_id)
        print(f"عدد التوكنز: {n:,}")
        if os.path.exists(dpath):
            os.remove(dpath)
    print(f"تم. vocab: {tok.get_vocab_size()}")


def iter_jsonl(path):
    files = sorted(glob.glob(os.path.join(path, "**", "*.json*"), recursive=True)) if os.path.isdir(path) else [path]
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as fh:
            if f.endswith(".json"):
                try:
                    for ex in json.load(fh):
                        yield ex
                except Exception:
                    pass
                continue
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        yield json.loads(line)
                    except Exception:
                        pass


def prep_sft(a):
    tok = load_tok(a.out)
    eos_id = tok.token_to_id(EOS)
    domains = load_domains(a.out)
    tools_ok, n_img, ctx_ck, ck_path = True, 0, 0, os.path.join(a.out, "ckpt.pt")
    have_ck = os.path.exists(ck_path)
    if have_ck:
        try:
            cfg_ck = Config(**torch.load(ck_path, map_location="cpu", mmap=True)["cfg"])
            tools_ok = cfg_ck.tools or a.force_tools
            n_img = cfg_ck.n_img_tokens if cfg_ck.vision else 0
            ctx_ck = cfg_ck.ctx
        except Exception as e:
            print(f"تنبيه: ما قدرتش أقرا ckpt.pt ({e}); هتعامل كأن النموذج بيدعم الأدوات من غير صور.")

    files = {}

    def wfile(name):
        if name not in files:
            files[name] = open(os.path.join(a.out, name), "wb")
        return files[name]

    vis_f = None
    cnt = dict(ex=0, tok=0, ast=0, bad=0, tools=0, img=0, dom=0, long=0, vis=0)
    for ex in iter_jsonl(a.sft_data):
        msgs = normalize_example(ex) if isinstance(ex, dict) else None
        if not msgs:
            cnt["bad"] += 1
            continue
        if not tools_ok and uses_tools(ex, msgs):
            cnt["tools"] += 1
            continue
        dom = 0
        if domains:
            name = ex.get("domain", a.default_domain)
            if name not in domains:
                cnt["dom"] += 1
                continue
            dom = domains.index(name)
        img_msgs = [m for m in msgs if isinstance(m, dict) and m.get("image")]
        img_path = None
        if img_msgs:
            if len(img_msgs) > 1:
                cnt["bad"] += 1
                continue
            img_path = img_msgs[0]["image"]
            if not os.path.isabs(img_path) and a.image_root:
                img_path = os.path.join(a.image_root, img_path)
            if not n_img or not os.path.exists(img_path):
                cnt["img"] += 1
                continue
            img_msgs[0]["image"] = img_path
        r = encode_chat(tok, msgs, eos_id, n_img)
        if r is None:
            cnt["bad"] += 1
            continue
        ids, mask = r
        cnt["ex"], cnt["tok"], cnt["ast"] = cnt["ex"] + 1, cnt["tok"] + len(ids), cnt["ast"] + sum(mask)
        cnt["long"] += bool(ctx_ck and len(ids) > ctx_ck)
        if img_path:
            if vis_f is None:
                vis_f = open(os.path.join(a.out, "sft_vision.jsonl"), "w", encoding="utf-8")
            vis_f.write(json.dumps({"ids": ids, "mask": mask, "image": img_path, "domain": dom}) + "\n")
            cnt["vis"] += 1
        else:
            stem = f"sft_d{dom}" if domains else "sft"
            wfile(f"{stem}_ids.bin").write(np.array(ids, dtype=np.uint16).tobytes())
            wfile(f"{stem}_mask.bin").write(np.array(mask, dtype=np.uint8).tobytes())
    for f in files.values():
        f.close()
    if vis_f:
        vis_f.close()
    elif os.path.exists(os.path.join(a.out, "sft_vision.jsonl")):
        os.remove(os.path.join(a.out, "sft_vision.jsonl"))

    print(f"أمثلة: {cnt['ex']:,} (منها {cnt['vis']} بصور) | توكنز: {cnt['tok']:,} "
          f"(منها {cnt['ast']:,} ردود مساعد بيتعلم منها) | اتخطّى (غير صالح): {cnt['bad']}")
    if cnt["tools"]:
        print(f"اتخطّى {cnt['tools']} مثال فيه أدوات لأن النموذج ده مش بيدعم الأدوات "
              f"(أحجام 2b وفوق، أو درّب بـ --tools on). لو عايزهم برضه: --force_tools")
    if cnt["img"]:
        why = ("النموذج مش بيدعم الصور (درّبه بـ --vision on)" if have_ck and not n_img
               else "مفيش ckpt.pt (اعمل train الأول عشان نعرف عدد توكنز الصورة)" if not have_ck
               else "ملف الصورة مش موجود")
        print(f"اتخطّى {cnt['img']} مثال فيه صور: {why}")
    if cnt["dom"]:
        print(f"اتخطّى {cnt['dom']} مثال من غير حقل domain صالح (المجالات: {domains}). استخدم --default_domain لو عايز تحطهم في مجال")
    if cnt["long"]:
        print(f"تحذير: {cnt['long']} مثال أطول من ctx={ctx_ck} بتاع النموذج، وهيتقصوا عشوائياً أو يتخطّوا وقت التدريب "
              f"(غالباً بسبب تعريفات الأدوات في الـ system). قصّر الأمثلة أو درّب النموذج بـ --ctx أكبر.")


# ----------------------------------------------------------------------------
# تحميل البيانات للتدريب
# ----------------------------------------------------------------------------
class Shard:
    """جزء من الداتا: stream توكنز + (mask للـ SFT) + رقم المجال/البرج"""

    def __init__(self, data, mask, domain):
        self.data, self.mask, self.domain = data, mask, domain


def load_shards(out, kind, ctx, frac):
    names = load_domains(out)
    if names:
        items = [(k, f"train_d{k}.bin" if kind == "pretrain" else f"sft_d{k}_ids.bin",
                  None if kind == "pretrain" else f"sft_d{k}_mask.bin") for k in range(len(names))]
    else:
        items = [(0, "train.bin" if kind == "pretrain" else "sft_ids.bin", None if kind == "pretrain" else "sft_mask.bin")]
    tr, va = [], []
    for k, ip, mp in items:
        ip = os.path.join(out, ip)
        if not os.path.exists(ip) or os.path.getsize(ip) == 0:
            continue
        data = np.memmap(ip, dtype=np.uint16, mode="r")
        mask = np.memmap(os.path.join(out, mp), dtype=np.uint8, mode="r") if mp else None
        n_val = max(int(len(data) * frac), ctx + 2)
        if len(data) <= n_val + ctx + 2:
            if names and kind == "pretrain":
                raise SystemExit(f"مجال '{names[k]}' داتاه صغيرة ({len(data)} توكن) أقل من اللازم. زوّد الداتا أو صغّر --ctx")
            continue
        tr.append(Shard(data[:-n_val], mask[:-n_val] if mask is not None else None, k))
        va.append(Shard(data[-n_val:], mask[-n_val:] if mask is not None else None, k))
    return tr, va


def get_batch(shards, probs, B, T, dev):
    ks = np.random.choice(len(shards), size=B, p=probs) if len(shards) > 1 else np.zeros(B, dtype=int)
    xs, ys, labs = [], [], []
    for k in ks:
        s = shards[k]
        i = np.random.randint(0, len(s.data) - T - 1)
        x, y = s.data[i:i + T].astype(np.int64), s.data[i + 1:i + 1 + T].astype(np.int64)
        if s.mask is not None:
            y[s.mask[i + 1:i + 1 + T] == 0] = -100
        xs.append(x)
        ys.append(y)
        labs.append(s.domain)
    return (torch.from_numpy(np.stack(xs)).to(dev), torch.from_numpy(np.stack(ys)).to(dev),
            torch.tensor(labs, dtype=torch.long, device=dev))


class VisionData:
    """أمثلة الصور (من sft_vision.jsonl). بتتحمّل كلها في الذاكرة، فمناسبة لآلاف لعشرات الآلاف من الأمثلة"""

    def __init__(self, path, cfg):
        self.cfg, self.cache = cfg, {}
        exs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if len(e["ids"]) <= cfg.ctx + 1:
                    exs.append(e)
        n_val = max(1, len(exs) // 20) if len(exs) >= 20 else 0
        self.train, self.val = exs[:len(exs) - n_val], exs[len(exs) - n_val:]

    def img(self, p):
        if p not in self.cache:
            if len(self.cache) > 4000:
                self.cache.clear()
            self.cache[p] = load_image(p, self.cfg.img_size)
        return self.cache[p]

    def batch(self, B, dev, split="train"):
        pool = self.train if split == "train" else self.val
        exs = [pool[i] for i in np.random.randint(0, len(pool), size=B)]
        L = max(len(e["ids"]) for e in exs)
        ids, lab = np.zeros((B, L), np.int64), np.full((B, L), -100, np.int64)
        for i, e in enumerate(exs):
            a_, m = np.array(e["ids"]), np.array(e["mask"])
            ids[i, :len(a_)] = a_
            lab[i, :len(a_)] = np.where(m == 1, a_, -100)
        return (torch.from_numpy(np.ascontiguousarray(ids[:, :-1])).to(dev),
                torch.from_numpy(np.ascontiguousarray(lab[:, 1:])).to(dev),
                torch.tensor([e["domain"] for e in exs], dtype=torch.long, device=dev),
                torch.stack([self.img(e["image"]) for e in exs]).to(dev))


# ----------------------------------------------------------------------------
# التدريب
# ----------------------------------------------------------------------------
def setup_dist():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        import torch.distributed as dist
        rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local)
            dist.init_process_group("nccl")
            return True, rank, world, f"cuda:{local}"
        dist.init_process_group("gloo")
        return True, rank, world, "cpu"
    return False, 0, 1, ("cuda" if torch.cuda.is_available() else "cpu")


def save_ckpt(path, model, opt, step, best_val):
    tmp = path + ".tmp"
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step,
                    best_val=best_val, cfg=asdict(model.c)), tmp)
    os.replace(tmp, path)


def make_cfg(a, tok, domains):
    p = {k: v for k, v in PRESETS[a.size].items() if k not in ("lr", "tools")}
    if a.ctx:
        p["ctx"] = a.ctx
    arch = a.arch
    if arch == "auto":
        arch = "moe" if a.moe_experts > 0 else "dense"
    kw = {}
    if arch == "moe":
        kw = dict(moe_experts=a.moe_experts or 8, moe_topk=a.moe_topk, moe_every=a.moe_every,
                  moe_shared=a.moe_shared, moe_div=a.moe_div)
    elif arch == "towers":
        n = a.towers or (len(domains) if domains else 0)
        if not n:
            raise SystemExit("towers محتاج مجالات: اعمل prep بـ --by_domain (فولدر لكل مجال) أو حدّد --towers N")
        if domains and n != len(domains):
            raise SystemExit(f"عدد الأبراج ({n}) لازم يساوي عدد المجالات ({len(domains)}: {domains})")
        kw = dict(n_towers=n, router_coef=a.router_coef)
    elif a.moe_experts:
        print("تنبيه: --moe_experts اتجاهل لأن --arch dense")
    vis = {}
    if a.vision == "on":
        d = vision_defaults(a.size)
        for k in ("img_size", "patch", "v_dim", "v_layers", "pool", "v_heads"):
            if getattr(a, k, None):
                d[k] = getattr(a, k)
        img_id = tok.token_to_id(IMG)
        if img_id is None:
            raise SystemExit("الـ tokenizer ده قديم (ما فيهوش <|image|>). اعمل prep من جديد عشان تستخدم --vision on")
        vis = dict(vision=True, img_id=img_id, **d)
    tools = {"auto": PRESETS[a.size]["tools"], "on": True, "off": False}[a.tools]
    return Config(vocab=tok.get_vocab_size(), arch=arch, tools=tools, **kw, **vis, **p)


@torch.no_grad()
def evaluate(model, val, probs, vd, a, cfg, dev, ctx):
    model.eval()
    ls, accs, out = [], [], {}
    for _ in range(a.eval_iters if val else 0):
        x, y, lab = get_batch(val, probs, a.batch, cfg.ctx, dev)
        with ctx:
            _, l, _ = model(x, y, tower=lab if cfg.arch == "towers" else None)
        ls.append(l.item())
        if cfg.arch == "towers":
            accs.append((model.route_logits(x).argmax(-1) == lab).float().mean().item())
    if ls:
        out["loss"] = sum(ls) / len(ls)
    if accs:
        out["router_acc"] = sum(accs) / len(accs)
    if vd is not None and vd.val:
        vl = []
        for _ in range(max(1, a.eval_iters // 2)):
            x, y, lab, im = vd.batch(min(a.batch, len(vd.val)), dev, "val")
            with ctx:
                _, l, _ = model(x, y, tower=lab if cfg.arch == "towers" else None, images=im)
            vl.append(l.item())
        out["vision_loss"] = sum(vl) / len(vl)
    model.train()
    return out


def run_train(a, kind):
    ddp, rank, world, dev = setup_dist()
    master = rank == 0
    np.random.seed(1337 + rank)
    torch.manual_seed(1337 + rank)
    tok = load_tok(a.out)
    domains = load_domains(a.out)
    own = os.path.join(a.out, "ckpt.pt" if kind == "pretrain" else "sft_ckpt.pt")

    ck = torch.load(own, map_location="cpu") if os.path.exists(own) else None
    src = ck
    if src is None and kind == "sft":
        base = os.path.join(a.out, "ckpt.pt")
        if not os.path.exists(base):
            raise SystemExit("لازم تعمل pretraining الأول (ckpt.pt مش موجود)")
        src = torch.load(base, map_location="cpu")
    cfg = Config(**src["cfg"]) if src else make_cfg(a, tok, domains)

    total_est, _ = estimate_params(cfg)
    if dev.startswith("cuda") and not a.force:
        need, have = total_est * 16 / 1e9, torch.cuda.get_device_properties(dev).total_memory / 1e9
        if need > have * 0.9:
            raise SystemExit(
                f"النموذج ({total_est/1e9:.2f}B) محتاج ~{need:.0f}GB للأوزان والـ optimizer لوحدهم على كل كارت، "
                f"والكارت عندك ~{have:.0f}GB. الكود ده بيكرر النموذج على كل GPU (من غير sharding زي FSDP/ZeRO) "
                f"فمش هيشتغل. اختار حجم أصغر، أو --force لو متأكد.")

    tr, va = load_shards(a.out, kind, cfg.ctx, 0.01 if kind == "pretrain" else 0.02)
    vd = None
    vpath = os.path.join(a.out, "sft_vision.jsonl")
    if kind == "sft" and cfg.vision and os.path.exists(vpath):
        vd = VisionData(vpath, cfg)
        if not vd.train:
            vd = None
    if not tr and vd is None:
        raise SystemExit("مفيش داتا كفاية للتدريب (راجع prep / prep_sft)")
    if cfg.arch == "towers" and tr and len({s.domain for s in tr}) < cfg.n_towers and kind == "pretrain":
        raise SystemExit("لازم يكون في داتا لكل برج (مجال)")
    uniform = a.domain_balance == "uniform" or (a.domain_balance == "auto" and cfg.arch == "towers")
    if tr:
        sizes = np.array([len(s.data) for s in tr], dtype=np.float64)
        probs = np.ones(len(tr)) / len(tr) if uniform else sizes / sizes.sum()
        vprobs = np.ones(len(va)) / len(va)
    branch_rng = random.Random(999)  # نفس القرار على كل الـ ranks في الـ DDP

    raw = GPT(cfg).to(dev)
    raw.grad_ckpt = a.grad_ckpt
    if src:
        raw.load_state_dict(src["model"])
    total, active = count_params(raw)
    if master:
        extra = {"moe": f" | MoE {cfg.moe_experts} خبير (top-{cfg.moe_topk}"
                        + (f", shared {cfg.moe_shared}" if cfg.moe_shared else "") + ")",
                 "towers": f" | {cfg.n_towers} أبراج" + (f" ({', '.join(domains)})" if domains else ""),
                 "dense": ""}[cfg.arch]
        print(f"[{kind}] {cfg.arch} | باراميترز: {total/1e6:.1f}M (نشطة لكل توكن: {active/1e6:.1f}M) | الجهاز: {dev} x{world} | ctx={cfg.ctx}"
              + extra + (" | أدوات" if cfg.tools else "") + (f" | صور ({cfg.n_img_tokens} توكن للصورة)" if cfg.vision else ""))
        if vd:
            print(f"أمثلة صور: {len(vd.train)} تدريب / {len(vd.val)} تقييم | نص: {sum(len(s.data) for s in tr):,} توكن")

    base_lr = a.lr or (PRESETS[a.size]["lr"] if kind == "pretrain" else 1e-4)
    decay = [p for p in raw.parameters() if p.dim() >= 2]
    nodecay = [p for p in raw.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=0.1), dict(params=nodecay, weight_decay=0.0)],
                            lr=base_lr, betas=(0.9, 0.95), fused=dev.startswith("cuda"))
    step, best_val = 0, float("inf")
    if ck:
        opt.load_state_dict(ck["opt"])
        step, best_val = ck["step"], ck["best_val"]
        if master:
            print(f"استكمال من الخطوة {step}")

    if dev.startswith("cuda"):
        use_bf16 = torch.cuda.get_device_capability()[0] >= 8
        ctx = torch.autocast("cuda", dtype=torch.bfloat16 if use_bf16 else torch.float16)
        scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    else:
        ctx, scaler = nullcontext(), torch.amp.GradScaler("cuda", enabled=False)

    net, ddp_model = raw, None
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        ddp_model = DDP(raw, device_ids=[int(dev.split(":")[1])] if dev.startswith("cuda") else None,
                        find_unused_parameters=cfg.arch != "dense" or cfg.vision)
        net = ddp_model
    if a.compile:
        net = torch.compile(net)

    def lr_at(s):
        if s < a.warmup:
            return base_lr * (s + 1) / a.warmup
        r = min(1.0, (s - a.warmup) / max(1, a.steps - a.warmup))
        return base_lr * 0.1 + 0.5 * base_lr * 0.9 * (1 + math.cos(math.pi * r))

    t_start = t0 = time.time()
    tok_per_step = a.batch * a.accum * cfg.ctx * world
    raw.train()
    while step < a.steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for micro in range(a.accum):
            use_img = vd is not None and (not tr or branch_rng.random() < a.vision_mix)
            if use_img:
                x, y, lab, im = vd.batch(a.batch, dev)
            else:
                (x, y, lab), im = get_batch(tr, probs, a.batch, cfg.ctx, dev), None
            sync = nullcontext() if (ddp_model is None or micro == a.accum - 1) else ddp_model.no_sync()
            with sync:
                with ctx:
                    _, loss, _ = net(x, y, tower=lab if cfg.arch == "towers" else None, images=im)
                scaler.scale(loss / a.accum).backward()
            tot = tot + loss.detach() / a.accum
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        step += 1

        if master and step % a.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"step {step}/{a.steps} | loss {float(tot):.4f} | lr {lr_at(step):.2e} "
                  f"| {tok_per_step * a.log_interval / dt:,.0f} tok/s")
        if master and (step % a.eval_interval == 0 or step == a.steps):
            r = evaluate(raw, va, vprobs if tr else None, vd, a, cfg, dev, ctx)
            msg = ">>>"
            if "loss" in r:
                msg += f" val loss {r['loss']:.4f}"
            if "router_acc" in r:
                msg += f" | دقة الراوتر {r['router_acc']:.0%}"
            if "vision_loss" in r:
                msg += f" | val صور {r['vision_loss']:.4f}"
            print(msg)
            best_val = min(best_val, r.get("loss", r.get("vision_loss", float("inf"))))
        if master and (step % a.save_interval == 0 or step == a.steps):
            save_ckpt(own, raw, opt, step, best_val)

        if a.max_hours:
            stop = torch.tensor([1.0 if (time.time() - t_start) > a.max_hours * 3600 else 0.0], device=dev)
            if ddp:
                import torch.distributed as dist
                dist.broadcast(stop, 0)
            if stop.item() > 0:
                if master:
                    save_ckpt(own, raw, opt, step, best_val)
                    print("وصلنا للحد الزمني. اتحفظ checkpoint. شغّل نفس الأمر تاني للاستكمال.")
                break
    else:
        if master:
            print("خلص التدريب.")
    if ddp:
        import torch.distributed as dist
        dist.destroy_process_group()


# ----------------------------------------------------------------------------
# التوليد والمحادثة
# ----------------------------------------------------------------------------
def load_model(path, dev):
    ck = torch.load(path, map_location=dev)
    m = GPT(Config(**ck["cfg"])).to(dev)
    m.load_state_dict(ck["model"])
    return m.eval()


@torch.no_grad()
def sample(model, prompt_ids, max_new=256, temp=0.8, top_k=50, top_p=0.95, rep_pen=1.1, stop=(), on_token=None,
           images=None, tower_topk=1, force_tower=None, info=None):
    """
    towers: الراوتر بيبص على الـ prompt ويختار top-m برج (tower_topk). برج واحد = أسرع حاجة.
    لو أكتر من برج: بنخلط log-probs بتاعتهم بأوزان الراوتر. force_tower بيتجاوز الراوتر.
    """
    c = model.c
    dev = next(model.parameters()).device
    L = c.ctx
    if images is None:
        prompt_ids = prompt_ids[-(L - 1):]
    x = torch.tensor([prompt_ids], device=dev)
    towers, weights = [None], [1.0]
    if c.arch == "towers":
        if force_tower is not None:
            towers, weights = [force_tower], [1.0]
        else:
            probs = model.route_logits(x)[0].float().softmax(-1)
            tv, ti = probs.topk(max(1, min(tower_topk, probs.numel())))
            towers, weights = ti.tolist(), (tv / tv.sum()).tolist()
        if info is not None:
            info["towers"], info["weights"] = towers, weights
    caches = [[None] * c.n_layer for _ in towers]

    def run(tokens, pos0, imgs=None):
        outs = []
        for j, t in enumerate(towers):
            lg, _, caches[j] = model(tokens, caches=caches[j], pos0=pos0, images=imgs, tower=t)
            outs.append(lg[0, -1].float())
        if len(outs) == 1:
            return outs[0]
        return sum(w * F.log_softmax(o, -1) for w, o in zip(weights, outs))

    lg = run(x, 0, images)
    pos, out = x.size(1), []
    for _ in range(max_new):
        if rep_pen != 1.0 and out:
            idx = torch.tensor(sorted(set(out[-128:])), device=dev)
            v = lg[idx]
            lg[idx] = torch.where(v > 0, v / rep_pen, v * rep_pen)
        if temp <= 0:
            nxt = int(lg.argmax())
        else:
            lg = lg / temp
            if top_k and top_k < lg.numel():
                lg[lg < torch.topk(lg, top_k).values[-1]] = float("-inf")
            if top_p < 1.0:
                sl, si = torch.sort(lg, descending=True)
                pr = torch.softmax(sl, -1)
                sl[(pr.cumsum(-1) - pr) > top_p] = float("-inf")
                lg = torch.full_like(lg, float("-inf")).scatter(0, si, sl)
            nxt = int(torch.multinomial(torch.softmax(lg, -1), 1))
        out.append(nxt)
        if nxt in stop:
            break
        if on_token:
            on_token(out)
        if pos >= L:
            break
        lg = run(torch.tensor([[nxt]], device=dev), pos)
        pos += 1
    return out


def streamer(tok, show_special=False):
    state = {"printed": ""}

    def on_token(ids):
        s = tok.decode(ids, skip_special_tokens=not show_special)
        if s.endswith("�"):
            return
        print(s[len(state["printed"]):], end="", flush=True)
        state["printed"] = s
    return on_token


def pick_dev():
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_tower(spec, domains, n):
    if spec is None or spec == "":
        return None
    if spec.isdigit():
        t = int(spec)
    elif domains and spec in domains:
        t = domains.index(spec)
    else:
        raise SystemExit(f"البرج '{spec}' مش موجود. المتاح: {domains or list(range(n))}")
    if not 0 <= t < n:
        raise SystemExit(f"رقم البرج لازم يكون من 0 لـ {n - 1}")
    return t


def tower_label(t, w, domains):
    return f"{domains[t] if domains else 'برج ' + str(t)} ({w:.0%})"


def generate(a):
    dev = pick_dev()
    tok = load_tok(a.out)
    model = load_model(os.path.join(a.out, a.ckpt or "ckpt.pt"), dev)
    domains = load_domains(a.out)
    ids = tok.encode(a.prompt).ids or [tok.token_to_id(EOS)]
    force = resolve_tower(a.tower, domains, model.c.n_towers) if model.c.arch == "towers" else None
    info = {}
    print(a.prompt, end="", flush=True)
    sample(model, ids, a.max_new, a.temp, a.top_k, a.top_p, a.rep_pen, {tok.token_to_id(EOS)}, streamer(tok),
           tower_topk=a.tower_topk, force_tower=force, info=info)
    print()
    if info.get("towers"):
        print("[الأبراج: " + ", ".join(tower_label(t, w, domains) for t, w in zip(info["towers"], info["weights"])) + "]")


def drop_oldest(hist):
    """يشيل أقدم دور كامل (من أول رسالة user لحد اللي قبل الـ user اللي بعدها)"""
    start = 1 if hist[0]["role"] == "system" else 0
    nxt = next((i for i in range(start + 1, len(hist)) if hist[i]["role"] == "user"), None)
    if nxt is None:
        return False
    del hist[start:nxt]
    return True


def chat(a):
    dev = pick_dev()
    tok = load_tok(a.out)
    name = a.ckpt or ("sft_ckpt.pt" if os.path.exists(os.path.join(a.out, "sft_ckpt.pt")) else "ckpt.pt")
    if name == "ckpt.pt":
        print("تنبيه: مفيش sft_ckpt.pt، فالنموذج لسه ما اتعلمش صيغة المحادثة وهيطلع كلام عشوائي. اعمل prep_sft وsft الأول.")
    model = load_model(os.path.join(a.out, name), dev)
    c = model.c
    domains = load_domains(a.out)
    force = resolve_tower(a.tower, domains, c.n_towers) if c.arch == "towers" else None
    if a.image and not c.vision:
        raise SystemExit("النموذج ده مش بيدعم الصور (درّبه بـ --vision on)")
    n_img = c.n_img_tokens if c.vision else 0
    use_tools = c.tools and not a.no_tools
    if a.tools_file and not c.tools:
        print("تنبيه: النموذج ده مش بيدعم الأدوات (أحجام 2b وفوق)، فهتتجاهل.")
    tools = load_tools(a.tools_file) if use_tools else {}
    stop = {tok.token_to_id(END), tok.token_to_id(EOS)}
    sys_text = a.system + (("\n\n" + tools_system_text(tools)) if use_tools else "")
    hist = [{"role": "system", "content": sys_text}] if sys_text else []
    pending = {"image": a.image or None}
    base_len = len(build_prompt(tok, hist, n_img))
    if base_len > c.ctx - a.max_new:
        print(f"تحذير: الـ system prompt (مع تعريفات الأدوات) واخد {base_len} توكن ومعاه max_new={a.max_new} "
              f"أكبر من ctx={c.ctx}، فالـ prompt هيتقص من الأول وهيضيع تعريف الأدوات. "
              f"استخدم --no_tools أو --system '' أو قلّل --max_new.")
    elif base_len > c.ctx // 2:
        print(f"تنبيه: الـ system prompt واخد {base_len} توكن من أصل ctx={c.ctx}، فمكان المحادثة هيبقى ضيق.")

    def turn(text):
        msg = {"role": "user", "content": text}
        if pending["image"]:
            msg["image"], pending["image"] = pending["image"], None
        hist.append(msg)
        visible = ""
        for rnd in range(a.max_tool_rounds + 1):
            while True:
                ids = build_prompt(tok, hist, n_img)
                if len(ids) <= c.ctx - a.max_new or not drop_oldest(hist):
                    break
            paths = [m["image"] for m in hist if m.get("image")]
            imgs = torch.stack([load_image(p, c.img_size) for p in paths]).to(dev) if (paths and c.vision) else None
            info = {}
            out = sample(model, ids, a.max_new, a.temp, a.top_k, a.top_p, a.rep_pen, stop,
                         streamer(tok, show_special=use_tools), images=imgs, tower_topk=a.tower_topk,
                         force_tower=force, info=info)
            if info.get("towers"):
                print("\n[" + ", ".join(tower_label(t, w, domains) for t, w in zip(info["towers"], info["weights"])) + "]",
                      end="", flush=True)
            if out and out[-1] in stop:
                out = out[:-1]
            if use_tools:
                calls, visible = parse_calls(tok.decode(out, skip_special_tokens=False))
            else:
                calls, visible = [], tok.decode(out)
            if calls and rnd < a.max_tool_rounds:
                hist.append({"role": "assistant", "content": visible, "tool_calls": calls})
                for cl in calls:
                    res = run_tool(tools, cl)
                    print(f"\n[نتيجة الأداة {cl['name']}: {res}]\nالنموذج: ", end="", flush=True)
                    hist.append({"role": "tool", "name": cl["name"], "content": res})
                continue
            break
        hist.append({"role": "assistant", "content": visible})
        return visible

    if a.once:
        turn(a.once)
        print()
        return
    print(f"محادثة ({c.arch}؛ الأدوات: {'شغالة' if use_tools else 'مقفولة'}؛ الصور: {'شغالة' if c.vision else 'مقفولة'}). "
          f"اكتب /reset لمسح السياق، /image مسار_الصورة لإرفاق صورة بالرسالة الجاية، /exit للخروج.")
    while True:
        try:
            text = input("\nأنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/exit":
            break
        if text == "/reset":
            del hist[1 if hist and hist[0]["role"] == "system" else 0:]
            continue
        if text.startswith("/image"):
            p = text[6:].strip()
            if not c.vision:
                print("النموذج ده مش بيدعم الصور.")
            elif not os.path.exists(p):
                print("الملف مش موجود.")
            else:
                pending["image"] = p
                print("اتحفظت الصورة، اكتب سؤالك.")
            continue
        if text:
            print("النموذج: ", end="", flush=True)
            turn(text)
            print()


# ----------------------------------------------------------------------------
# info + selftest
# ----------------------------------------------------------------------------
def info(a):
    print(f"vocab={a.vocab} | MoE: {a.experts} خبير top-{a.topk} shared={a.shared} div={a.div} | towers: {a.towers} أبراج")
    print("الذاكرة: تدريب ≈ 16 بايت/باراميتر (أوزان + gradients + Adam) على كل كارت، تشغيل fp16 ≈ 2 بايت/باراميتر\n")
    print(f"{'الحجم':<7}{'dense':>8}{'MoE كلي':>9}{'MoE نشط':>9}{'Towers كلي':>12}{'Towers نشط':>12}{'تدريب dense':>13}{'تشغيل fp16':>12}  أدوات  صور(+باراميترز)")
    for n, p in PRESETS.items():
        base = {k: v for k, v in p.items() if k not in ("lr", "tools")}
        d = estimate_params(Config(vocab=a.vocab, **base))[0]
        mt, ma = estimate_params(Config(vocab=a.vocab, arch="moe", moe_experts=a.experts, moe_topk=a.topk,
                                        moe_shared=a.shared, moe_div=a.div, **base))
        tt, ta = estimate_params(Config(vocab=a.vocab, arch="towers", n_towers=a.towers, **base))
        vp = vision_params(Config(vocab=a.vocab, vision=True, img_id=0, **base, **vision_defaults(n)))
        print(f"{n:<7}{d/1e9:>7.2f}B{mt/1e9:>8.2f}B{ma/1e9:>8.2f}B{tt/1e9:>11.2f}B{ta/1e9:>11.2f}B{d*16/1e9:>11.0f}GB{d*2/1e9:>10.1f}GB  "
              f"{'نعم' if p['tools'] else 'لا':<5}  +{vp/1e6:.0f}M")


def selftest(a):
    torch.manual_seed(0)
    small = dict(vocab=128, n_layer=4, n_head=4, n_kv_head=2, n_embd=64, ctx=64)
    variants = [("dense", dict()),
                ("moe 4 خبير", dict(arch="moe", moe_experts=4)),
                ("moe 16 خبير+shared+div4", dict(arch="moe", moe_experts=16, moe_topk=4, moe_shared=1, moe_div=4)),
                ("towers 3", dict(arch="towers", n_towers=3, n_layer=3))]
    for name, v in variants:
        c = Config(**{**small, **v})
        m = GPT(c)
        assert count_params(m) == estimate_params(c), f"{name}: عدّ الباراميترز مش مظبوط"
        x, y = torch.randint(0, 128, (2, 24)), torch.randint(0, 128, (2, 24))
        tw = 1 if c.arch == "towers" else None
        # 1) KV cache لازم يدي نفس نتايج الـ forward الكامل
        m.eval()
        with torch.no_grad():
            full, _, _ = m(x, all_logits=True, tower=tw)
            lg, _, cs = m(x[:, :10], caches=[None] * c.n_layer, all_logits=True, tower=tw)
            outs = [lg]
            for t in range(10, 24):
                lg, _, cs = m(x[:, t:t + 1], caches=cs, pos0=t, tower=tw)
                outs.append(lg)
        diff = (full - torch.cat(outs, 1)).abs().max().item()
        assert diff < 1e-4, f"{name}: KV cache مختلف عن الـ forward الكامل: {diff}"
        # 2) gradient checkpointing لازم يدي نفس الـ loss والـ grads
        m.train()
        _, l1, _ = m(x, y, tower=tw)
        l1.backward()
        g1 = m.emb.weight.grad.clone()
        m.zero_grad()
        m.grad_ckpt = True
        _, l2, _ = m(x, y, tower=tw)
        l2.backward()
        assert torch.allclose(l1, l2, atol=1e-5) and torch.allclose(g1, m.emb.weight.grad, atol=1e-4), f"{name}: checkpointing بيغيّر النتايج"
        m.grad_ckpt = False
        # 3) النموذج لازم يقدر يحفظ batch صغير
        xb = torch.randint(0, 128, (3, 24))
        yb = torch.randint(0, 128, (3, 24))
        labels = torch.tensor([0, 1, 2]) if c.arch == "towers" else None
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        first = None
        for _ in range(80):
            opt.zero_grad()
            _, l, _ = m(xb, yb, tower=labels)
            first = first or l.item()
            l.backward()
            opt.step()
        last = l.item()
        assert last < first * 0.5, f"{name}: الـ loss مش بينزل ({first:.2f} -> {last:.2f})"
        extra = ""
        if c.arch == "towers":  # الراوتر لازم يتعلم يفرّق العينات حسب المجال
            m.eval()
            assert (m.route_logits(xb).argmax(-1) == labels).all(), "الراوتر ما اتعلمش يصنّف المجالات"
            extra = " | الراوتر صنّف 3/3 صح"
        yy = y.clone()
        yy[:, :12] = -100
        m.eval()
        _, l, _ = m(x, yy, tower=tw)
        assert torch.isfinite(l)
        print(f"{name:<26} OK | cache diff={diff:.1e} | حفظ batch صغير: loss {first:.2f} -> {last:.2f}{extra}")

    # 4) الصور
    for pool, (isz, pt) in ((1, (32, 8)), (2, (32, 4))):
        c = Config(**small, arch="dense", vision=True, img_size=isz, patch=pt, v_dim=64, v_layers=2, pool=pool, img_id=127)
        m = GPT(c)
        assert count_params(m) == estimate_params(c), "عدّ باراميترز المشفّر مش مظبوط"
        n = c.n_img_tokens
        imgs = torch.randn(2, 3, isz, isz)
        assert m.vision(imgs).shape == (2, n, 64)
        x = torch.randint(0, 100, (2, 40))
        x[:, 2:2 + n] = 127
        m.eval()
        with torch.no_grad():
            l1, _, _ = m(x, all_logits=True, images=imgs)
            im2 = imgs.clone()
            im2[0] += 1.0
            l2, _, _ = m(x, all_logits=True, images=im2)
            assert (l1[0, -1] - l2[0, -1]).abs().max() > 1e-4, "تغيير الصورة ما أثّرش على الناتج"
            assert (l1[1] - l2[1]).abs().max() < 1e-5, "صورة عينة بتأثر على عينة تانية"
            # الـ cache مع الصور (الصورة في الـ prefill)
            lg, _, cs = m(x[:, :30], caches=[None] * 4, all_logits=True, images=imgs)
            outs = [lg]
            for t in range(30, 40):
                lg, _, cs = m(x[:, t:t + 1], caches=cs, pos0=t)
                outs.append(lg)
            d2 = (l1 - torch.cat(outs, 1)).abs().max().item()
            assert d2 < 1e-4, f"cache مع الصور مختلف: {d2}"
        m.train()
        _, l, _ = m(x, torch.randint(0, 100, (2, 40)), images=imgs)
        l.backward()
        assert m.vision.patch.weight.grad is not None and m.vision.patch.weight.grad.abs().sum() > 0, "الـ gradients مش واصلة للمشفّر"
        print(f"صور (pool={pool}, {n} توكن)   OK | الصورة بتأثر على الناتج، والـ cache سليم، والـ gradients واصلة للمشفّر")

    # 5) الأحجام الحقيقية: عدّ الباراميترز الفعلي (على meta device من غير ذاكرة) = التقدير
    for n, p in PRESETS.items():
        base = {k: v for k, v in p.items() if k not in ("lr", "tools")}
        cfgs = [Config(vocab=32000, **base),
                Config(vocab=32000, arch="moe", moe_experts=8, moe_shared=1, moe_div=4, **base),
                Config(vocab=32000, arch="towers", n_towers=4, **base),
                Config(vocab=32000, vision=True, img_id=8, **base, **vision_defaults(n))]
        for c in cfgs:
            with torch.device("meta"):
                real = count_params(GPT(c))
            assert real == estimate_params(c), f"{n}/{c.arch}: العدّ مختلف"
    print(f"{len(PRESETS)} حجم × (dense, moe, towers, +صور): عدّ الباراميترز الفعلي = التقدير")

    # 6) الآلة الحاسبة
    assert tool_calculator("20*5") == "100" and tool_calculator("2^10") == "1024" and tool_calculator("10/4") == "2.5"
    for bad in ("__import__('os').system('ls')", "2**100000", "open('x')", "a+1"):
        try:
            tool_calculator(bad)
        except Exception:
            continue
        raise AssertionError(f"المفروض يترفض: {bad}")
    # 7) قالب الأدوات والصور: التدريب والـ chat بيستخدموا نفس الـ render، والـ mask صح
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(["احسب 20*5 calculator expression name arguments الناتج مئة 100 ما لون الصورة أحمر"] * 50,
                          trainers.BpeTrainer(vocab_size=300, special_tokens=SPECIAL,
                                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    call = {"name": "calculator", "arguments": {"expression": "20*5"}}
    ex = {"tools": [{"name": "calculator", "description": "d", "parameters": {"expression": "string"}}],
          "messages": [{"role": "user", "content": "احسب 20*5"},
                       {"role": "assistant", "content": "", "tool_calls": [call]},
                       {"role": "tool", "name": "calculator", "content": "100"},
                       {"role": "assistant", "content": "الناتج 100"}]}
    msgs = normalize_example(ex)
    assert msgs[0]["role"] == "system" and "calculator" in msgs[0]["content"] and uses_tools(ex, msgs)
    ids, mask = encode_chat(t, msgs, t.token_to_id(EOS))
    learned = [i for i, mk in zip(ids, mask) if mk]
    assert t.token_to_id(TCALL) in learned and t.token_to_id(TOOL) not in learned
    calls, _ = parse_calls(t.decode(learned, skip_special_tokens=False))
    assert calls[:1] == [call], f"parse رجّع {calls}"
    prompt = build_prompt(t, msgs[:-1])
    assert ids[:len(prompt)] == prompt, "قالب الـ chat مختلف عن قالب التدريب"
    assert not valid_msg({"role": "user", "tool_calls": [call]}) and not valid_msg({"role": "assistant"})
    vex = {"image": "x.png", "prompt": "ما لون الصورة؟", "response": "أحمر"}
    vm = normalize_example(vex)
    assert vm[0].get("image") == "x.png" and encode_chat(t, vm, 0, 0) is None, "صورة من غير مشفّر لازم تترفض"
    vids, vmask = encode_chat(t, vm, 0, 16)
    assert vids.count(t.token_to_id(IMG)) == 16 and not any(m for i, m in zip(vids, vmask) if i == t.token_to_id(IMG))
    vprompt = build_prompt(t, vm[:1], 16)   # = رسالة الـ user بالصورة + بداية دور المساعد
    cut = len(vprompt) - len(role_prefix(t, "assistant"))
    assert vids[:cut] == vprompt[:cut], "قالب الصور في الـ chat مختلف عن قالب التدريب"
    print("الأدوات والصور: الحاسبة آمنة، القالب متطابق بين التدريب والـ chat، توكنز الصورة مش بتتعلّم كـ loss")
    print("كل الاختبارات نجحت.")


# ----------------------------------------------------------------------------
def add_train_args(p, sft=False):
    p.add_argument("--out", required=True)
    p.add_argument("--size", default="tiny", choices=list(PRESETS))
    p.add_argument("--ctx", type=int, default=0, help="غيّر طول السياق (0 = حسب الحجم)")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--accum", type=int, default=2 if sft else 8)
    p.add_argument("--steps", type=int, default=2000 if sft else 20000)
    p.add_argument("--lr", type=float, default=0, help="0 = تلقائي")
    p.add_argument("--warmup", type=int, default=50 if sft else 500)
    p.add_argument("--eval_interval", type=int, default=200 if sft else 500)
    p.add_argument("--eval_iters", type=int, default=20)
    p.add_argument("--save_interval", type=int, default=200 if sft else 500)
    p.add_argument("--log_interval", type=int, default=20 if sft else 50)
    p.add_argument("--max_hours", type=float, default=0)
    p.add_argument("--grad_ckpt", action="store_true", help="يوفر ذاكرة GPU على حساب السرعة")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--force", action="store_true", help="تجاهل فحص ذاكرة الكارت")
    p.add_argument("--domain_balance", default="auto", choices=["auto", "uniform", "size"],
                   help="لو الداتا مجالات: uniform = نفس العدد من كل مجال، size = حسب حجمه. auto = uniform في towers")
    p.add_argument("--vision_mix", type=float, default=0.5, help="نسبة الـ batches اللي فيها صور (لو في أمثلة صور)")
    if not sft:  # الإعدادات دي بتتحدد وقت إنشاء النموذج بس
        p.add_argument("--arch", default="auto", choices=["auto", "dense", "moe", "towers"],
                       help="شكل النموذج. auto = moe لو حددت --moe_experts وإلا dense")
        p.add_argument("--tools", default="auto", choices=["auto", "on", "off"],
                       help="دعم الأدوات: auto = حسب الحجم (2b وفوق)")
        p.add_argument("--vision", default="off", choices=["on", "off"], help="دعم تحليل الصور")
        p.add_argument("--img_size", type=int, default=0)
        p.add_argument("--patch", type=int, default=0)
        p.add_argument("--v_dim", type=int, default=0)
        p.add_argument("--v_layers", type=int, default=0)
        p.add_argument("--v_heads", type=int, default=0)
        p.add_argument("--pool", type=int, default=0)
        p.add_argument("--moe_experts", type=int, default=0, help="عدد الخبراء (أي رقم). 0 مع --arch moe = 8")
        p.add_argument("--moe_topk", type=int, default=2)
        p.add_argument("--moe_every", type=int, default=2)
        p.add_argument("--moe_shared", type=int, default=0, help="عدد الخبراء الثابتين (بيشتغلوا دايماً)")
        p.add_argument("--moe_div", type=int, default=2, help="حجم الخبير = FFN ÷ الرقم ده (4 أو 8 = خبراء أدق)")
        p.add_argument("--towers", type=int, default=0, help="عدد الأبراج (0 = عدد المجالات في domains.json)")
        p.add_argument("--router_coef", type=float, default=0.1, help="وزن loss الراوتر في towers")


def add_sample_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--rep_pen", type=float, default=1.1)
    p.add_argument("--tower", default="", help="towers: اجبر برج معين (اسم المجال أو رقمه) بدل الراوتر")
    p.add_argument("--tower_topk", type=int, default=1, help="towers: كام برج يشتغلوا مع بعض (1 = الأسرع)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info")
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--shared", type=int, default=0)
    p.add_argument("--div", type=int, default=2)
    p.add_argument("--towers", type=int, default=4)
    sub.add_parser("selftest")

    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--by_domain", action="store_true", help="كل فولدر جوه --data = مجال (لـ towers)")

    p = sub.add_parser("prep_sft")
    p.add_argument("--out", required=True)
    p.add_argument("--sft_data", required=True, help="ملف أو فولدر jsonl")
    p.add_argument("--force_tools", action="store_true", help="ضمّ أمثلة الأدوات حتى لو النموذج مش بيدعمها")
    p.add_argument("--image_root", default="", help="فولدر الصور لو المسارات في الملف نسبية")
    p.add_argument("--default_domain", default=None, help="مجال للأمثلة اللي من غير حقل domain")

    add_train_args(sub.add_parser("train"))
    add_train_args(sub.add_parser("sft"), sft=True)

    p = sub.add_parser("generate")
    add_sample_args(p)
    p.add_argument("--prompt", default="")

    p = sub.add_parser("chat")
    add_sample_args(p)
    p.add_argument("--system", default="أنت مساعد ذكي ومفيد.")
    p.add_argument("--once", default="", help="رسالة واحدة من غير حلقة تفاعلية")
    p.add_argument("--image", default="", help="صورة ترفقها بالرسالة (للنماذج المتدرّبة بـ --vision on)")
    p.add_argument("--no_tools", action="store_true", help="اقفل الأدوات حتى لو النموذج بيدعمها")
    p.add_argument("--tools_file", default="", help="ملف بايثون فيه TOOLS = {...} لأدواتك")
    p.add_argument("--max_tool_rounds", type=int, default=3)

    args = ap.parse_args()
    {
        "info": info, "selftest": selftest, "prep": prep, "prep_sft": prep_sft,
        "train": lambda x: run_train(x, "pretrain"), "sft": lambda x: run_train(x, "sft"),
        "generate": generate, "chat": chat,
    }[args.cmd](args)
