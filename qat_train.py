"""
Phase 1 baseline: char-level GPT on TinyShakespeare.
Trains fp32, then applies int8 dynamic post-training quantization,
and reports a comparison table: loss/perplexity, model size, latency.

Run:
    python baseline.py                 # quick smoke config
    python baseline.py --full          # bigger config for a real result
"""
import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--full", action="store_true", help="use a larger config for a real (not smoke-test) run")
parser.add_argument("--data", type=str, default="input.txt")
parser.add_argument("--iters", type=int, default=None)
args = parser.parse_args()

torch.manual_seed(1337)

if args.full:
    block_size = 128
    n_embd = 256
    n_head = 8
    n_layer = 6
    batch_size = 64
    max_iters = args.iters or 3000
else:
    # smoke-test config: proves the pipeline works end to end in ~1 min on CPU
    block_size = 64
    n_embd = 64
    n_head = 4
    n_layer = 2
    batch_size = 32
    max_iters = args.iters or 200

eval_interval = max(50, max_iters // 10)
eval_iters = 20
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
dropout = 0.1

print(f"device={device} full={args.full} max_iters={max_iters} n_layer={n_layer} n_embd={n_embd}")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
with open(args.data, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i : i + block_size] for i in ix])
    y = torch.stack([d[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model: minimal GPT (standard pre-norm transformer, causal self-attention)
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
# Train fp32 baseline
# ---------------------------------------------------------------------------
model = GPT().to(device)
print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

t0 = time.time()
for it in range(max_iters):
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss(model)
        print(f"step {it}: train {losses['train']:.4f}, val {losses['val']:.4f}")

    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

train_time = time.time() - t0
final_losses = estimate_loss(model)
fp32_val_loss = final_losses["val"]
fp32_ppl = math.exp(fp32_val_loss)

os.makedirs("out", exist_ok=True)
fp32_path = "out/model_fp32.pt"
torch.save(model.state_dict(), fp32_path)
fp32_size_mb = os.path.getsize(fp32_path) / 1e6

# ---------------------------------------------------------------------------
# Int8 dynamic post-training quantization (quantizes nn.Linear weights)
# ---------------------------------------------------------------------------
model_cpu = GPT()
model_cpu.load_state_dict(torch.load(fp32_path, map_location="cpu"))
model_cpu.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model_cpu, {nn.Linear}, dtype=torch.qint8
)

int8_path = "out/model_int8.pt"
torch.save(quantized_model.state_dict(), int8_path)
int8_size_mb = os.path.getsize(int8_path) / 1e6


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


int8_val_loss = eval_loss_cpu(quantized_model)
int8_ppl = math.exp(int8_val_loss)


@torch.no_grad()
def measure_latency(m, n_runs=30):
    m.eval()
    x = torch.randint(0, vocab_size, (1, block_size))
    # warmup
    for _ in range(5):
        m(x)
    t0 = time.time()
    for _ in range(n_runs):
        m(x)
    elapsed = time.time() - t0
    return (elapsed / n_runs) / block_size * 1000  # ms per token


fp32_cpu = GPT()
fp32_cpu.load_state_dict(torch.load(fp32_path, map_location="cpu"))
fp32_latency = measure_latency(fp32_cpu)
int8_latency = measure_latency(quantized_model)

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print(f"{'Config':<20}{'Val Loss':<12}{'PPL':<10}{'Size (MB)':<12}{'ms/token':<10}")
print("=" * 70)
print(f"{'fp32':<20}{fp32_val_loss:<12.4f}{fp32_ppl:<10.2f}{fp32_size_mb:<12.3f}{fp32_latency:<10.4f}")
print(f"{'int8 (dynamic)':<20}{int8_val_loss:<12.4f}{int8_ppl:<10.2f}{int8_size_mb:<12.3f}{int8_latency:<10.4f}")
print("=" * 70)
print(f"training wall time: {train_time:.1f}s")
print(f"size reduction: {(1 - int8_size_mb/fp32_size_mb)*100:.1f}%")
print(f"ppl delta: {int8_ppl - fp32_ppl:+.3f}")
