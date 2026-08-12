"""
Phase II, Step 3: full QAT run + real int8 conversion.

1. Load fp32 checkpoint, wrap with fake-quant (Step 1/2 code, with the
   requires_grad detach fix), fine-tune for 3000 iters -- same scale as
   baseline.py --full, so the accuracy comparison is apples-to-apples.
2. IMPORTANT: fake-quant only *simulates* int8 during training (weights stay
   fp32 the whole time, so latency/size measured on the fake-quant model
   would be meaningless -- it's still an fp32 model under the hood).
   To get a real, deployable artifact and honest size/latency numbers, we
   copy the QAT-trained weights into a plain GPT and apply the SAME real
   torch.quantization.quantize_dynamic conversion baseline.py used for PTQ.
   This gives us an actual int8 model -- but one that started from
   QAT-adapted weights instead of the original fp32 weights.
3. Report the full three-way table: fp32 vs. PTQ-int8 vs. QAT-int8.

Run (same Colab quant_project/ directory):
    python qat_step3_full.py
"""
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config -- MUST match the --full config used to train the checkpoint
# ---------------------------------------------------------------------------
block_size = 128
n_embd = 256
n_head = 8
n_layer = 6
batch_size = 64
dropout = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

QAT_ITERS = 3000          # matches baseline.py --full scale
QAT_LR = 3e-5             # same LR that worked cleanly in Step 2
eval_interval = 300
eval_iters = 20

CHECKPOINT_PATH = "out/model_fp32.pt"
DATA_PATH = "input.txt"
QAT_FLOAT_PATH = "out/model_qat_float.pt"   # QAT-trained weights, still fp32
QAT_INT8_PATH = "out/model_qat_int8.pt"     # real quantized artifact

torch.manual_seed(1337)

print(f"device={device} qat_iters={QAT_ITERS} qat_lr={QAT_LR}")

# ---------------------------------------------------------------------------
# Data (identical to baseline.py / Steps 1-2)
# ---------------------------------------------------------------------------
with open(DATA_PATH, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model -- copied verbatim from baseline.py / Steps 1-2 so state_dict keys match
# ---------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss


# ---------------------------------------------------------------------------
# Fake-quant wrapper -- same as Step 1/2, with the requires_grad detach fix
# ---------------------------------------------------------------------------
def fake_quantize_tensor(t, num_bits=8):
    qmin = -(2 ** (num_bits - 1))
    qmax = 2 ** (num_bits - 1) - 1
    # FIX (flagged by Step 2's UserWarning): compute scale from a detached
    # copy so this max/scale computation doesn't get pulled into the
    # autograd graph. The quantize-dequantize op itself is still applied to
    # the live (non-detached) tensor `t`, so gradients still flow through
    # normally -- we're only detaching the *scale calculation*.
    max_val = t.detach().abs().max().clamp(min=1e-8)
    scale = max_val / qmax
    zero_point = 0
    return torch.fake_quantize_per_tensor_affine(t, float(scale), zero_point, qmin, qmax)


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, num_bits=8, quantize_activations=False):
        super().__init__()
        self.weight = linear.weight
        self.bias = linear.bias
        self.num_bits = num_bits
        self.quantize_activations = quantize_activations

    def forward(self, x):
        w = fake_quantize_tensor(self.weight, self.num_bits)
        if self.quantize_activations:
            x = fake_quantize_tensor(x, self.num_bits)
        return F.linear(x, w, self.bias)


def wrap_linears_with_fake_quant(module, num_bits=8, quantize_activations=False):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child, num_bits, quantize_activations))
        else:
            wrap_linears_with_fake_quant(child, num_bits, quantize_activations)
    return module


@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Phase A: full-length QAT fine-tune
# ---------------------------------------------------------------------------
model = GPT().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
print("Loaded fp32 checkpoint OK.")

model = wrap_linears_with_fake_quant(model, num_bits=8, quantize_activations=False)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=QAT_LR)

print(f"\nStarting full QAT fine-tune for {QAT_ITERS} iters...")
t0 = time.time()
for it in range(QAT_ITERS):
    if it % eval_interval == 0 or it == QAT_ITERS - 1:
        losses = estimate_loss(model)
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

qat_time = time.time() - t0
final_losses = estimate_loss(model)
print(f"\nQAT fine-tune done in {qat_time:.1f}s. Final val loss: {final_losses['val']:.4f}")

os.makedirs("out", exist_ok=True)
torch.save(model.state_dict(), QAT_FLOAT_PATH)

# ---------------------------------------------------------------------------
# Phase B: convert QAT-trained weights into a REAL int8 model
# ---------------------------------------------------------------------------
# FakeQuantLinear kept the same param names (weight, bias) as nn.Linear, and
# buffers (tril) are untouched, so state_dict keys line up with plain GPT.
plain_model = GPT()
plain_model.load_state_dict(torch.load(QAT_FLOAT_PATH, map_location="cpu"))
plain_model.eval()

quantized_model = torch.quantization.quantize_dynamic(
    plain_model, {nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), QAT_INT8_PATH)
qat_int8_size_mb = os.path.getsize(QAT_INT8_PATH) / 1e6


@torch.no_grad()
def eval_loss_cpu(m):
    m.eval()
    losses = []
    for _ in range(eval_iters):
        ix = torch.randint(len(val_data) - block_size, (batch_size,))
        x = torch.stack([val_data[i : i + block_size] for i in ix])
        y = torch.stack([val_data[i + 1 : i + 1 + block_size] for i in ix])
        _, loss = m(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


qat_int8_val_loss = eval_loss_cpu(quantized_model)
qat_int8_ppl = math.exp(qat_int8_val_loss)


@torch.no_grad()
def measure_latency(m, n_runs=30):
    m.eval()
    x = torch.randint(0, vocab_size, (1, block_size))
    for _ in range(5):
        m(x)
    t0 = time.time()
    for _ in range(n_runs):
        m(x)
    elapsed = time.time() - t0
    return (elapsed / n_runs) / block_size * 1000  # ms/token


qat_int8_latency = measure_latency(quantized_model)

# ---------------------------------------------------------------------------
# Report: three-way comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'Config':<20}{'Val Loss':<12}{'PPL':<10}{'Size (MB)':<12}{'ms/token':<10}")
print("=" * 70)
print(f"{'fp32 (Phase I)':<20}{1.5587:<12.4f}{4.75:<10.2f}{22.446:<12.3f}{0.2690:<10.4f}")
print(f"{'int8 PTQ (Phase I)':<20}{1.5643:<12.4f}{4.78:<10.2f}{8.360:<12.3f}{0.3846:<10.4f}")
print(f"{'int8 QAT (Phase II)':<20}{qat_int8_val_loss:<12.4f}{qat_int8_ppl:<10.2f}{qat_int8_size_mb:<12.3f}{qat_int8_latency:<10.4f}")
print("=" * 70)
print(f"QAT training wall time: {qat_time:.1f}s")
print(f"QAT vs PTQ PPL delta: {qat_int8_ppl - 4.78:+.3f}")
print(f"\nSaved QAT float weights to {QAT_FLOAT_PATH}")
print(f"Saved QAT int8 (real, deployable) weights to {QAT_INT8_PATH}")