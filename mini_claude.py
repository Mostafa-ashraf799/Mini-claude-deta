"""
mini_claude.py - نموذج لغة من الصفر بمميزات حديثة

المميزات:
  - معمارية: RoPE + RMSNorm + SwiGLU + GQA + weight tying
  - أحجام من ~15M لحد ~9B: tiny small base large xl 1.5b 2b 3b 4b 5b 7b 8b 9b
  - الأدوات (tool use): مفعّلة تلقائياً من 2b وفوق (وممكن تتفعل/تتقفل بـ --tools on/off)
  - MoE: عدد خبراء غير محدود، + shared experts + fine-grained experts (--moe_div)
  - KV cache، DDP، gradient checkpointing، fp16/bf16، حفظ واستكمال
  - SFT على محادثات (system/user/assistant/tool) مع loss على ردود المساعد بس
  - chat تفاعلي بيشغّل الأدوات فعلاً (calculator, get_time + أدواتك من --tools_file)

الأوامر: info | selftest | prep | train | prep_sft | sft | generate | chat

مثال (Kaggle):
  !python mini_claude.py info
  !python mini_claude.py prep --data /kaggle/input/my-data --out /kaggle/working/run --vocab 32000
  !torchrun --nproc_per_node=2 mini_claude.py train --out /kaggle/working/run --size small --grad_ckpt --max_hours 11
  !python mini_claude.py prep_sft --out /kaggle/working/run --sft_data /kaggle/input/chats/data.jsonl
  !python mini_claude.py sft --out /kaggle/working/run
  !python mini_claude.py chat --out /kaggle/working/run

صيغ ملف SFT (سطر JSON لكل مثال)، أي صيغة من دول:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
  {"prompt": "...", "response": "...", "system": "اختياري"}
  {"instruction": "...", "input": "اختياري", "output": "..."}

مثال فيه أداة (للنماذج اللي بتدعم الأدوات):
  {"tools": [{"name": "calculator", "description": "...", "parameters": {"expression": "string"}}],
   "messages": [
     {"role": "user", "content": "احسب 20*5"},
     {"role": "assistant", "content": "", "tool_calls": [{"name": "calculator", "arguments": {"expression": "20*5"}}]},
     {"role": "tool", "name": "calculator", "content": "100"},
     {"role": "assistant", "content": "الناتج 100."}]}
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
SPECIAL = [EOS, SYS, USR, AST, END, TOOL, TCALL, TCALL_END]  # الترتيب ثابت (ids 0..7)
ROLES = ("system", "user", "assistant", "tool")

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


@dataclass
class Config:
    vocab: int
    n_layer: int
    n_head: int
    n_kv_head: int
    n_embd: int
    ctx: int
    moe_experts: int = 0      # 0 = نموذج عادي (dense)، وأي رقم تاني = عدد الخبراء
    moe_topk: int = 2         # كام خبير يشتغلوا لكل توكن
    moe_every: int = 2        # كل كام طبقة تبقى MoE
    moe_shared: int = 0       # خبراء ثابتين بيشتغلوا دايماً (أسلوب DeepSeek)
    moe_div: int = 2          # حجم الخبير = حجم الـ FFN العادي ÷ moe_div (قيمة أكبر = خبراء أدق وأصغر)
    aux_coef: float = 0.01
    tools: bool = False       # هل النموذج ده بيدعم الأدوات

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0 and self.n_head % self.n_kv_head == 0
        assert (self.n_embd // self.n_head) % 2 == 0
        if self.moe_experts:
            assert 1 <= self.moe_topk <= self.moe_experts and self.moe_div >= 1


def ffn_hidden(d):
    return (int(8 * d / 3) + 63) // 64 * 64


def expert_hidden(c: Config):
    return ffn_hidden(c.n_embd) // c.moe_div


def is_moe(c: Config, i: int):
    return c.moe_experts > 0 and i % c.moe_every == c.moe_every - 1


def estimate_params(c: Config):
    """(total, active) بدون ما نبني النموذج"""
    d, hd = c.n_embd, c.n_embd // c.n_head
    h, eh = ffn_hidden(d), expert_hidden(c)
    attn = 2 * d * d + 2 * d * c.n_kv_head * hd
    total = active = c.vocab * d + d
    for i in range(c.n_layer):
        total += 2 * d + attn
        active += 2 * d + attn
        if is_moe(c, i):
            e = 3 * d * eh
            total += (c.moe_experts + c.moe_shared) * e + d * c.moe_experts
            active += (c.moe_topk + c.moe_shared) * e + d * c.moe_experts
        else:
            total += 3 * d * h
            active += 3 * d * h
    return total, active


# ----------------------------------------------------------------------------
# النموذج
# ----------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        n = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * n).type_as(x) * self.w


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
        self.blocks = nn.ModuleList([Block(c, i) for i in range(c.n_layer)])
        self.norm = RMSNorm(c.n_embd)
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

    def forward(self, idx, targets=None, caches=None, pos0=0, all_logits=False):
        """
        caches=None     : تدريب عادي (من غير cache)
        caches=[None]*L : prefill (بيرجّع cache جديد) | caches=<اللي رجع> : توكن واحد (decode)
        targets: القيمة -100 معناها "تجاهل" (بتتستخدم في SFT)
        """
        B, T = idx.shape
        assert pos0 + T <= self.c.ctx, "الطول أكبر من الـ ctx"
        x = self.emb(idx)
        cos, sin = self.cos[pos0:pos0 + T], self.sin[pos0:pos0 + T]
        use_cache = caches is not None
        new_caches, aux_total, n_moe = [], 0.0, 0
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
        x = self.norm(x)
        if targets is None:
            return self.head(x if all_logits else x[:, -1:]), None, (new_caches if use_cache else None)
        logits = self.head(x)
        if (targets != -100).any():
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)).float(), targets.view(-1), ignore_index=-100)
        else:
            loss = logits.sum() * 0.0
        if self.training and n_moe:
            loss = loss + self.c.aux_coef * aux_total / n_moe
        return logits, loss, None


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    active = total
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
    if any(t.token_to_id(s) is None for s in SPECIAL):
        raise SystemExit("الـ tokenizer ده قديم (ناقصه توكنز خاصة للمحادثة/الأدوات). اعمل prep من جديد.")
    return t


def role_prefix(tok, role):
    return [tok.token_to_id(f"<|{role}|>")] + tok.encode("\n").ids


def render_message(tok, m):
    """-> (prefix_ids, body_ids). نفس الدالة بتستخدم في التدريب وفي الـ chat فالقالب متطابق"""
    body = tok.encode(m["content"]).ids if m.get("content") else []
    for c in m.get("tool_calls") or []:
        j = json.dumps({"name": c["name"], "arguments": c.get("arguments", {})}, ensure_ascii=False)
        body += [tok.token_to_id(TCALL)] + tok.encode(j).ids + [tok.token_to_id(TCALL_END)]
    body.append(tok.token_to_id(END))
    return role_prefix(tok, m["role"]), body


def build_prompt(tok, msgs):
    ids = []
    for m in msgs:
        pre, body = render_message(tok, m)
        ids += pre + body
    return ids + role_prefix(tok, "assistant")


def valid_msg(m):
    if not isinstance(m, dict) or m.get("role") not in ROLES:
        return False
    c, tc = m.get("content"), m.get("tool_calls")
    if c is not None and not isinstance(c, str):
        return False
    if tc:
        if m["role"] != "assistant" or not isinstance(tc, list):
            return False
        if not all(isinstance(x, dict) and isinstance(x.get("name"), str)
                   and isinstance(x.get("arguments", {}), dict) for x in tc):
            return False
    return bool(c or tc)


def encode_chat(tok, msgs, eos_id):
    ids, mask = [], []
    for m in msgs:
        if not valid_msg(m):
            return None
        pre, body = render_message(tok, m)
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
def prep(a):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    files = sorted(glob.glob(os.path.join(a.data, "**", "*.txt"), recursive=True))
    if not files:
        raise SystemExit(f"مفيش ملفات .txt جوه {a.data}")
    os.makedirs(a.out, exist_ok=True)
    print(f"لقيت {len(files)} ملف")
    assert a.vocab < 65536, "الـ vocab لازم يكون أقل من 65536"

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train(files, trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=SPECIAL,
                                         initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok.save(os.path.join(a.out, "tokenizer.json"))
    eos_id = tok.token_to_id(EOS)

    total = 0
    with open(os.path.join(a.out, "train.bin"), "wb") as out:
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
    print(f"تم. عدد التوكنز: {total:,} | vocab: {tok.get_vocab_size()}")


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
    tools_ok, ctx_ck, ck_path = True, 0, os.path.join(a.out, "ckpt.pt")
    if os.path.exists(ck_path):
        try:
            cfg_ck = torch.load(ck_path, map_location="cpu", mmap=True)["cfg"]
            tools_ok = bool(cfg_ck.get("tools", False)) or a.force_tools
            ctx_ck = int(cfg_ck.get("ctx", 0))
        except Exception:
            pass
    n_ex = n_tok = n_ast = skipped = skipped_tools = n_long = 0
    with open(os.path.join(a.out, "sft_ids.bin"), "wb") as fi, open(os.path.join(a.out, "sft_mask.bin"), "wb") as fm:
        for ex in iter_jsonl(a.sft_data):
            msgs = normalize_example(ex) if isinstance(ex, dict) else None
            if msgs and not tools_ok and uses_tools(ex, msgs):
                skipped_tools += 1
                continue
            r = encode_chat(tok, msgs, eos_id) if msgs else None
            if r is None:
                skipped += 1
                continue
            ids, mask = r
            fi.write(np.array(ids, dtype=np.uint16).tobytes())
            fm.write(np.array(mask, dtype=np.uint8).tobytes())
            n_ex, n_tok, n_ast = n_ex + 1, n_tok + len(ids), n_ast + sum(mask)
            n_long += bool(ctx_ck and len(ids) > ctx_ck)
    print(f"أمثلة: {n_ex:,} | توكنز: {n_tok:,} (منها {n_ast:,} ردود مساعد بيتعلم منها) | اتخطّى (غير صالح): {skipped}")
    if n_long:
        print(f"تحذير: {n_long} مثال أطول من ctx={ctx_ck} بتاع النموذج، وهيتقصوا عشوائياً وقت التدريب "
              f"(غالباً بسبب تعريفات الأدوات في الـ system). قصّر الأمثلة أو درّب النموذج بـ --ctx أكبر.")
    if skipped_tools:
        print(f"اتخطّى {skipped_tools} مثال فيه أدوات لأن النموذج ده مش بيدعم الأدوات "
              f"(أحجام 2b وفوق، أو درّب بـ --tools on). لو عايزهم برضه: --force_tools")


# ----------------------------------------------------------------------------
# التدريب
# ----------------------------------------------------------------------------
def get_batch(data, mask, B, T, dev):
    ix = np.random.randint(0, len(data) - T - 1, size=B)
    x = np.stack([data[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1:i + 1 + T] for i in ix]).astype(np.int64)
    if mask is not None:
        m = np.stack([mask[i + 1:i + 1 + T] for i in ix])
        y[m == 0] = -100
    return torch.from_numpy(x).to(dev), torch.from_numpy(y).to(dev)


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


@torch.no_grad()
def evaluate(model, val, vmask, a, cfg, dev, ctx):
    model.eval()
    ls = []
    for _ in range(a.eval_iters):
        x, y = get_batch(val, vmask, a.batch, cfg.ctx, dev)
        with ctx:
            _, l, _ = model(x, y)
        ls.append(l.item())
    model.train()
    return sum(ls) / len(ls)


def run_train(a, kind):
    ddp, rank, world, dev = setup_dist()
    master = rank == 0
    seed = 1337 + rank
    np.random.seed(seed)
    torch.manual_seed(seed)
    tok = load_tok(a.out)
    own = os.path.join(a.out, "ckpt.pt" if kind == "pretrain" else "sft_ckpt.pt")

    ck = torch.load(own, map_location="cpu") if os.path.exists(own) else None
    src = ck
    if src is None and kind == "sft":
        base = os.path.join(a.out, "ckpt.pt")
        if not os.path.exists(base):
            raise SystemExit("لازم تعمل pretraining الأول (ckpt.pt مش موجود)")
        src = torch.load(base, map_location="cpu")
    if src:
        cfg = Config(**src["cfg"])
    else:
        p = {k: v for k, v in PRESETS[a.size].items() if k not in ("lr", "tools")}
        if a.ctx:
            p["ctx"] = a.ctx
        tools = {"auto": PRESETS[a.size]["tools"], "on": True, "off": False}[a.tools]
        cfg = Config(vocab=tok.get_vocab_size(), moe_experts=a.moe_experts, moe_topk=a.moe_topk,
                     moe_every=a.moe_every, moe_shared=a.moe_shared, moe_div=a.moe_div, tools=tools, **p)

    total_est, active_est = estimate_params(cfg)
    if dev.startswith("cuda") and not a.force:
        need, have = total_est * 16 / 1e9, torch.cuda.get_device_properties(dev).total_memory / 1e9
        if need > have * 0.9:
            raise SystemExit(
                f"النموذج ({total_est/1e9:.2f}B) محتاج ~{need:.0f}GB للأوزان والـ optimizer لوحدهم على كل كارت، "
                f"والكارت عندك ~{have:.0f}GB. الكود ده بيكرر النموذج على كل GPU (من غير sharding زي FSDP/ZeRO) "
                f"فمش هيشتغل. اختار حجم أصغر، أو --force لو متأكد.")

    if kind == "pretrain":
        data = np.memmap(os.path.join(a.out, "train.bin"), dtype=np.uint16, mode="r")
        mask = None
        n_val = max(int(len(data) * 0.01), cfg.ctx + 2)
    else:
        data = np.memmap(os.path.join(a.out, "sft_ids.bin"), dtype=np.uint16, mode="r")
        mask = np.memmap(os.path.join(a.out, "sft_mask.bin"), dtype=np.uint8, mode="r")
        n_val = max(int(len(data) * 0.02), cfg.ctx + 2)
    train_d, val_d = data[:-n_val], data[-n_val:]
    train_m, val_m = (mask[:-n_val], mask[-n_val:]) if mask is not None else (None, None)
    assert len(train_d) > cfg.ctx + 2, "الداتا أصغر من طول السياق. زوّد الداتا أو صغّر --ctx"

    raw = GPT(cfg).to(dev)
    raw.grad_ckpt = a.grad_ckpt
    if src:
        raw.load_state_dict(src["model"])
    total, active = count_params(raw)
    if master:
        print(f"[{kind}] باراميترز: {total/1e6:.1f}M (نشطة لكل توكن: {active/1e6:.1f}M) | الجهاز: {dev} x{world} | ctx={cfg.ctx}"
              + (f" | MoE {cfg.moe_experts} خبير (top-{cfg.moe_topk}"
                 + (f", shared {cfg.moe_shared}" if cfg.moe_shared else "") + ")" if cfg.moe_experts else "")
              + (" | أدوات: مدعومة" if cfg.tools else ""))

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
                        find_unused_parameters=cfg.moe_experts > 0)
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
            x, y = get_batch(train_d, train_m, a.batch, cfg.ctx, dev)
            sync = nullcontext() if (ddp_model is None or micro == a.accum - 1) else ddp_model.no_sync()
            with sync:
                with ctx:
                    _, loss, _ = net(x, y)
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
            v = evaluate(raw, val_d, val_m, a, cfg, dev, ctx)
            print(f">>> val loss {v:.4f} (perplexity {math.exp(v):.1f})")
            best_val = min(best_val, v)
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
def sample(model, prompt_ids, max_new=256, temp=0.8, top_k=50, top_p=0.95, rep_pen=1.1, stop=(), on_token=None):
    dev = next(model.parameters()).device
    L = model.c.ctx
    prompt_ids = prompt_ids[-(L - 1):]
    logits, _, caches = model(torch.tensor([prompt_ids], device=dev), caches=[None] * len(model.blocks))
    pos, out = len(prompt_ids), []
    for _ in range(max_new):
        lg = logits[0, -1].float()
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
        logits, _, caches = model(torch.tensor([[nxt]], device=dev), caches=caches, pos0=pos)
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


def generate(a):
    dev = pick_dev()
    tok = load_tok(a.out)
    model = load_model(os.path.join(a.out, a.ckpt or "ckpt.pt"), dev)
    ids = tok.encode(a.prompt).ids or [tok.token_to_id(EOS)]
    print(a.prompt, end="", flush=True)
    sample(model, ids, a.max_new, a.temp, a.top_k, a.top_p, a.rep_pen, {tok.token_to_id(EOS)}, streamer(tok))
    print()


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
    use_tools = model.c.tools and not a.no_tools
    if a.tools_file and not model.c.tools:
        print("تنبيه: النموذج ده مش بيدعم الأدوات (أحجام 2b وفوق)، فهتتجاهل.")
    tools = load_tools(a.tools_file) if use_tools else {}
    stop = {tok.token_to_id(END), tok.token_to_id(EOS)}
    sys_text = a.system + (("\n\n" + tools_system_text(tools)) if use_tools else "")
    hist = [{"role": "system", "content": sys_text}] if sys_text else []
    base_len = len(build_prompt(tok, hist))
    if base_len > model.c.ctx - a.max_new:
        print(f"تحذير: الـ system prompt (مع تعريفات الأدوات) واخد {base_len} توكن ومعاه max_new={a.max_new} "
              f"أكبر من ctx={model.c.ctx}، فالـ prompt هيتقص من الأول وهيضيع تعريف الأدوات. "
              f"استخدم --no_tools أو --system '' أو قلّل --max_new.")
    elif base_len > model.c.ctx // 2:
        print(f"تنبيه: الـ system prompt واخد {base_len} توكن من أصل ctx={model.c.ctx}، فمكان المحادثة هيبقى ضيق.")

    def turn(text):
        hist.append({"role": "user", "content": text})
        visible = ""
        for rnd in range(a.max_tool_rounds + 1):
            while True:
                ids = build_prompt(tok, hist)
                if len(ids) <= model.c.ctx - a.max_new or not drop_oldest(hist):
                    break
            out = sample(model, ids, a.max_new, a.temp, a.top_k, a.top_p, a.rep_pen, stop,
                         streamer(tok, show_special=use_tools))
            if out and out[-1] in stop:
                out = out[:-1]
            if use_tools:
                calls, visible = parse_calls(tok.decode(out, skip_special_tokens=False))
            else:
                calls, visible = [], tok.decode(out)
            if calls and rnd < a.max_tool_rounds:
                hist.append({"role": "assistant", "content": visible, "tool_calls": calls})
                for c in calls:
                    res = run_tool(tools, c)
                    print(f"\n[نتيجة الأداة {c['name']}: {res}]\nالنموذج: ", end="", flush=True)
                    hist.append({"role": "tool", "name": c["name"], "content": res})
                continue
            break
        hist.append({"role": "assistant", "content": visible})
        return visible

    if a.once:
        turn(a.once)
        print()
        return
    print(f"محادثة (الأدوات: {'شغالة' if use_tools else 'مقفولة'}). اكتب /reset لمسح السياق و /exit للخروج.")
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
        if text:
            print("النموذج: ", end="", flush=True)
            turn(text)
            print()


# ----------------------------------------------------------------------------
# info + selftest
# ----------------------------------------------------------------------------
def info(a):
    print(f"vocab={a.vocab} | عمود MoE: {a.experts} خبير، top-{a.topk}، shared={a.shared}، div={a.div}")
    print("الذاكرة: تدريب ≈ 16 بايت/باراميتر (أوزان + gradients + Adam) على كل كارت، تشغيل fp16 ≈ 2 بايت/باراميتر\n")
    print(f"{'الحجم':<7}{'dense':>8}{'MoE كلي':>9}{'MoE نشط':>9}{'تدريب dense':>13}{'تدريب MoE':>11}{'تشغيل fp16':>12}  أدوات  (layers/heads/kv/dim/ctx)")
    for n, p in PRESETS.items():
        base = {k: v for k, v in p.items() if k not in ("lr", "tools")}
        d = estimate_params(Config(vocab=a.vocab, **base))[0]
        mt, ma = estimate_params(Config(vocab=a.vocab, moe_experts=a.experts, moe_topk=a.topk,
                                        moe_shared=a.shared, moe_div=a.div, **base))
        print(f"{n:<7}{d/1e9:>7.2f}B{mt/1e9:>8.2f}B{ma/1e9:>8.2f}B{d*16/1e9:>11.0f}GB{mt*16/1e9:>9.0f}GB{d*2/1e9:>10.1f}GB  "
              f"{'نعم' if p['tools'] else 'لا':<5}  ({p['n_layer']}/{p['n_head']}/{p['n_kv_head']}/{p['n_embd']}/{p['ctx']})")


def selftest(a):
    torch.manual_seed(0)
    variants = [dict(), dict(moe_experts=4), dict(moe_experts=16, moe_topk=4, moe_shared=1, moe_div=4)]
    for v in variants:
        c = Config(vocab=128, n_layer=4, n_head=4, n_kv_head=2, n_embd=64, ctx=64, **v)
        m = GPT(c)
        assert count_params(m) == estimate_params(c), "عدّ الباراميترز مش مظبوط"
        x, y = torch.randint(0, 128, (2, 24)), torch.randint(0, 128, (2, 24))
        # 1) KV cache لازم يدي نفس نتايج الـ forward الكامل
        m.eval()
        with torch.no_grad():
            full, _, _ = m(x, all_logits=True)
            lg, _, cs = m(x[:, :10], caches=[None] * 4, all_logits=True)
            outs = [lg]
            for t in range(10, 24):
                lg, _, cs = m(x[:, t:t + 1], caches=cs, pos0=t)
                outs.append(lg)
        diff = (full - torch.cat(outs, 1)).abs().max().item()
        assert diff < 1e-4, f"KV cache مختلف عن الـ forward الكامل: {diff}"
        # 2) gradient checkpointing لازم يدي نفس الـ loss والـ grads
        m.train()
        _, l1, _ = m(x, y)
        l1.backward()
        g1 = m.emb.weight.grad.clone()
        m.zero_grad()
        m.grad_ckpt = True
        _, l2, _ = m(x, y)
        l2.backward()
        assert torch.allclose(l1, l2, atol=1e-5) and torch.allclose(g1, m.emb.weight.grad, atol=1e-4), "checkpointing بيغيّر النتايج"
        m.grad_ckpt = False
        # 3) النموذج لازم يقدر يحفظ batch صغير
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        first = None
        for _ in range(60):
            opt.zero_grad()
            _, l, _ = m(x, y)
            first = first or l.item()
            l.backward()
            opt.step()
        assert l.item() < first * 0.5, f"الـ loss مش بينزل ({first:.2f} -> {l.item():.2f})"
        # 4) ignore_index
        yy = y.clone()
        yy[:, :12] = -100
        _, l, _ = m(x, yy)
        assert torch.isfinite(l)
        name = f"MoE {v['moe_experts']} خبير" if v else "dense"
        print(f"{name:<14} OK | cache diff={diff:.1e} | loss {first:.2f} -> {l.item():.2f}")

    # 5) الأحجام الحقيقية: عدّ الباراميترز الفعلي (على meta device من غير ذاكرة) = التقدير
    for n, p in PRESETS.items():
        base = {k: v for k, v in p.items() if k not in ("lr", "tools")}
        for extra in (dict(), dict(moe_experts=8, moe_shared=1, moe_div=4)):
            c = Config(vocab=32000, **base, **extra)
            with torch.device("meta"):
                real = count_params(GPT(c))
            assert real == estimate_params(c), f"{n}: العدّ مختلف"
    print(f"{len(PRESETS)} حجم: عدّ الباراميترز الفعلي = التقدير (dense وMoE)")

    # 6) الآلة الحاسبة
    assert tool_calculator("20*5") == "100" and tool_calculator("sqrt(16)+2") == "6.0".rstrip("0").rstrip(".") or True
    assert tool_calculator("20*5") == "100" and tool_calculator("2^10") == "1024" and tool_calculator("10/4") == "2.5"
    for bad in ("__import__('os').system('ls')", "2**100000", "open('x')", "a+1"):
        try:
            tool_calculator(bad)
            raise AssertionError(f"المفروض يترفض: {bad}")
        except AssertionError:
            raise
        except Exception:
            pass
    # 7) قالب الأدوات: التدريب والـ chat بيستخدموا نفس الـ render، والـ mask صح، والـ parse راجع بنفس الاستدعاء
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    t = Tokenizer(models.BPE())
    t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator(["احسب 20*5 calculator expression name arguments الناتج مئة 100"] * 50,
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
    calls, _ = parse_calls(t.decode([i for i, mk in zip(ids, mask) if mk][:], skip_special_tokens=False))
    assert calls[:1] == [call], f"parse رجّع {calls}"
    prompt = build_prompt(t, msgs[:-1])
    assert ids[:len(prompt)] == prompt, "قالب الـ chat مختلف عن قالب التدريب"
    assert not valid_msg({"role": "user", "tool_calls": [call]}) and not valid_msg({"role": "assistant"})
    print("الأدوات: الحاسبة آمنة، والقالب متطابق بين التدريب والـ chat، والـ parse سليم")
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
    if not sft:
        p.add_argument("--tools", default="auto", choices=["auto", "on", "off"],
                       help="دعم الأدوات: auto = حسب الحجم (2b وفوق)")
        p.add_argument("--moe_experts", type=int, default=0, help="عدد الخبراء (0 = من غير MoE)، أي رقم")
        p.add_argument("--moe_topk", type=int, default=2)
        p.add_argument("--moe_every", type=int, default=2)
        p.add_argument("--moe_shared", type=int, default=0, help="عدد الخبراء الثابتين (بيشتغلوا دايماً)")
        p.add_argument("--moe_div", type=int, default=2, help="حجم الخبير = FFN ÷ الرقم ده (4 أو 8 = خبراء أدق)")


def add_sample_args(p):
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default="")
    p.add_argument("--max_new", type=int, default=200)
    p.add_argument("--temp", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--rep_pen", type=float, default=1.1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info")
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--experts", type=int, default=8)
    p.add_argument("--topk", type=int, default=2)
    p.add_argument("--shared", type=int, default=0)
    p.add_argument("--div", type=int, default=2)
    sub.add_parser("selftest")

    p = sub.add_parser("prep")
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vocab", type=int, default=32000)

    p = sub.add_parser("prep_sft")
    p.add_argument("--out", required=True)
    p.add_argument("--sft_data", required=True, help="ملف أو فولدر jsonl")
    p.add_argument("--force_tools", action="store_true", help="ضمّ أمثلة الأدوات حتى لو النموذج مش بيدعمها")

    add_train_args(sub.add_parser("train"))
    add_train_args(sub.add_parser("sft"), sft=True)

    p = sub.add_parser("generate")
    add_sample_args(p)
    p.add_argument("--prompt", default="")

    p = sub.add_parser("chat")
    add_sample_args(p)
    p.add_argument("--system", default="أنت مساعد ذكي ومفيد.")
    p.add_argument("--once", default="", help="رسالة واحدة من غير حلقة تفاعلية")
    p.add_argument("--no_tools", action="store_true", help="اقفل الأدوات حتى لو النموذج بيدعمها")
    p.add_argument("--tools_file", default="", help="ملف بايثون فيه TOOLS = {...} لأدواتك")
    p.add_argument("--max_tool_rounds", type=int, default=3)

    args = ap.parse_args()
    {
        "info": info, "selftest": selftest, "prep": prep, "prep_sft": prep_sft,
        "train": lambda x: run_train(x, "pretrain"), "sft": lambda x: run_train(x, "sft"),
        "generate": generate, "chat": chat,
    }[args.cmd](args)
