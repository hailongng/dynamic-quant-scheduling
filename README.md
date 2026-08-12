# Dynamic Quantization Scheduling for Efficient Transformer Training

A small-scale study of post-training and quantization-aware training (QAT) for
transformer language models, using a compact char-level GPT trained on
TinyShakespeare.

## Status

- **Phase I** - fp32 baseline + int8 post-training quantization (PTQ): Done
- **Phase II** - quantization-aware training (QAT): Done
- **Phase III** - TBD (could be lower bit-widths or latency profiling at scale)

## Project structure

```
.
├── baseline.py              # Phase I: fp32 baseline + int8 dynamic PTQ
├── fake_quant_eval.py        # Phase II: fake-quant plumbing sanity check (eval-only)
├── qat_train.py               # Phase II: full QAT run + real int8 conversion
├── input.txt                  # TinyShakespeare dataset (not tracked, see Setup)
└── out/                        # Saved checkpoints (not tracked in git)
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
# Phase I -- fp32 baseline + int8 PTQ
python baseline.py --full

# Phase II -- QAT (requires out/model_fp32.pt from Phase I)
python qat_train.py
```

## Model

Minimal decoder-only transformer (pre-norm, causal self-attention),
implemented from scratch in PyTorch:

- 6 layers, 8 heads, `d_model=256`, `block_size=128`
- ~4.8M trainable parameters
- Character-level tokenization (65-char vocab)

## Method

**Phase I (PTQ):** train fp32 baseline, then apply
`torch.quantization.quantize_dynamic` (int8, weights-only) directly to the
trained weights -- no retraining involved.

**Phase II (QAT):** wrap every `nn.Linear` with a fake-quantization module
(`torch.fake_quantize_per_tensor_affine`, symmetric per-tensor, int8) that
simulates quantization noise on every forward pass during training. Fine-tune
the fp32 checkpoint for 3000 steps with fake-quant active, then convert the
QAT-adapted weights to a real int8 model via the same `quantize_dynamic` call
used in Phase I -- isolating the effect of QAT-adapted weights vs. PTQ
applied to unadapted weights.

## Results

| Config              | Val Loss | PPL  | Size (MB) | ms/token |
|---------------------|----------|------|-----------|----------|
| fp32                | 1.5587   | 4.75 | 22.446    | 0.2690   |
| int8 PTQ            | 1.5643   | 4.78 | 8.360     | 0.3846   |
| int8 QAT            | 1.5301   | 4.62 | 8.362     | 0.2850   |

- **Size reduction (both methods): 62.8%** (22.4 MB → 8.4 MB)
- **PTQ**: negligible accuracy cost (+0.6% relative PPL), but *slower* than
  fp32 at this scale -- the quantize/dequantize overhead outweighs any GEMM
  speedup for small matrix sizes.
- **QAT**: recovers *and exceeds* fp32 accuracy (PPL 4.62 vs. 4.75), and
  closes most of the latency gap versus PTQ (0.285 ms/token vs. 0.385).
  Fine-tuning with quantization noise in the loop lets the optimizer adapt
  around quantization error, rather than quantizing a model that was never
  trained with that error in mind.

## Key takeaway

Post-training quantization is a strong "free" first lever: minimal effort,
large size reduction, negligible accuracy loss. Quantization-aware training
costs more (a fine-tuning pass) but pays for itself here, improving accuracy
*beyond* the full-precision baseline while matching PTQ's size reduction and
narrowing its latency overhead. At small model scale, the difference between
PTQ and QAT accuracy (PPL 4.78 vs 4.62) is a proxy for how much post-hoc
quantization error the model was leaving on the table.

## License

MIT
