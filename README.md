# Dynamic Quantization Scheduling for Efficient Transformer Training

A small-scale study of post-training and quantization-aware training (QAT) for
transformer language models, using a compact char-level GPT trained on
TinyShakespeare.

## Status

- **Phase I** — fp32 baseline + int8 post-training quantization (PTQ): Done
- **Phase II** — quantization-aware training (QAT): WIP

## Project structure

```
.
├── baseline.py       # Phase I: train fp32 baseline, apply int8 dynamic PTQ, report comparison
├── input.txt          # TinyShakespeare dataset (not tracked in git, see Setup)
└── out/                # Saved checkpoints (not tracked in git)
```

## Setup

```bash
pip install torch
```

Download the dataset:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

```bash
# Quick smoke test (~1 min on CPU) -- confirms the pipeline runs end to end
python baseline.py

# Full run used for the results below (GPU recommended)
python baseline.py --full
```

This trains a 6-layer, 256-dim char-level GPT (~4.8M params) for 3000 steps,
then applies `torch.quantization.quantize_dynamic` (int8, weights-only) and
reports a comparison table.

## Model

Minimal decoder-only transformer (pre-norm, causal self-attention), implemented from scratch in PyTorch:

- 6 layers, 8 heads, `d_model=256`, `block_size=128`
- ~4.8M trainable parameters
- Character-level tokenization (65-char vocab)

## Results (Phase I)

| Config          | Val Loss | PPL  | Size (MB) | ms/token |
|-----------------|----------|------|-----------|----------|
| fp32            | 1.5587   | 4.75 | 22.446    | 0.2690   |
| int8 (dynamic)  | 1.5643   | 4.78 | 8.360     | 0.3846   |

- **Size reduction: 62.8%** (22.4 MB → 8.4 MB)
- **Accuracy cost: negligible** (+0.03 PPL, ~0.6% relative)
- **Latency**: int8 dynamic PTQ was *slower* than fp32 on this hardware/scale.
  This is expected, not a bug - the quantize/dequantize overhead of dynamic
  PTQ outweighs any GEMM speedup at these small matrix sizes. Latency gains
  from int8 typically show up at larger model/batch sizes where compute,
  not overhead, dominates.

## Key takeaway

Int8 dynamic post-training quantization gives a large, nearly-free reduction
in model size with negligible accuracy loss, but doesn't help (and can possibly
hurt) latency at small scale. 

This motivates Phase II: quantization-aware training,
to see whether training with quantization noise in the loop can close the
latency gap and/or push toward lower bit-widths.

## License

No License
