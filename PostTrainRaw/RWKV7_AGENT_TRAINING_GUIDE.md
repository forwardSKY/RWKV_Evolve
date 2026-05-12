# RWKV-7 Agent Training with Cross-Chunk BPTT

## What This Is

A training script for fine-tuning pretrained RWKV-7 `.pth` models on agent traces (CodeAct, tool-use, instruction-following, GRPO) with a novel feature: **gradients flow across chunk boundaries** via checkpointed state-chain BPTT.

No existing RWKV trainer does this. Every "infinite context" trainer in the RWKV ecosystem detaches the state at chunk boundaries, which severs gradient flow. This script keeps the state chain in the computation graph and uses activation checkpointing to keep memory constant.

## Why It Matters

Standard RWKV training chunks a long sequence into 4096-token pieces and processes them sequentially, passing the RNN state between chunks but cutting the gradient. This means:

- **Per-token SFT:** Each token still gets gradient from its own local loss. The cross-chunk gradient component is lost but local gradients remain. Training works but is slightly wrong — the model never learns to write states differently based on what future chunks need.

- **Trajectory-level reward / final score:** If the loss only exists in the last chunk, all earlier chunks receive exactly zero gradient. Credit assignment completely fails. The model cannot learn that an action in chunk 0 led to the reward in chunk 5.

Cross-chunk BPTT fixes both cases. Gradients flow from the loss through the state chain back to chunk 0. Memory overhead is negligible because RWKV states are tiny (~5MB for a 3B model) while chunk activations are gigabytes — checkpointing discards the activations and recomputes them during backward.

## Requirements

```bash
pip install torch
pip install wandb          # optional, for logging
pip install rwkv            # optional, for tokenizer
```

For production speed, compile the RWKV-7 CUDA kernels from `rwkv7_fast_fused`. The script includes a pure-PyTorch fallback that is functionally correct but slow.

## Quick Start

### 1. Generate Example Data

```bash
python rwkv7_agent_train.py --generate_data my_traces.jsonl
```

This creates a JSONL file with example traces in the expected format.

### 2. Train

```bash
python rwkv7_agent_train.py \
    --model /path/to/RWKV-x070-World-1.5B-v2.9-20250107-ctx4096.pth \
    --data my_traces.jsonl \
    --output ./checkpoints \
    --chunk_size 4096 \
    --lr 2e-5 \
    --epochs 3 \
    --cross_chunk_bptt \
    --score_mode sample
```

### 3. Use the Trained Model

The output `.pth` file is a standard RWKV model checkpoint. Load it with RWKV Runner, Ai00, the `rwkv` pip package, or any RWKV inference engine.

## Training Data Format

The script accepts JSONL files. Each line is one agent trace in any of these formats:

### Format 1: Role-Tagged Turns (Recommended)

```json
{
    "score": 1.0,
    "turns": [
        {"role": "system", "content": "You are a coding assistant."},
        {"role": "user", "content": "Write a prime checker."},
        {"role": "assistant", "content": "<think>I'll check divisibility up to sqrt(n).</think>"},
        {"role": "action", "content": "```python\ndef is_prime(n): ...\n```"},
        {"role": "observation", "content": "Output: [2, 3, 5, 7, 11, 13]"},
        {"role": "assistant", "content": "Here is the solution..."}
    ]
}
```

Roles determine masking automatically:

| Role | Masked? | Meaning |
|------|---------|---------|
| `system` | Yes (mask=0) | System prompt — model reads but doesn't learn to generate |
| `user` / `human` | Yes (mask=0) | User input |
| `observation` / `tool_result` / `environment` | Yes (mask=0) | Tool/env output — model reads but doesn't generate |
| `assistant` / `model` | No (mask=1) | Model output — **loss computed here** |
| `thought` | No (mask=1) | Model reasoning |
| `action` | No (mask=1) | Model tool calls |
| `answer` | No (mask=1) | Model final answer |

### Format 2: Pre-Tokenized with Mask

```json
{
    "score": 0.85,
    "tokens": [1, 504, 23, 882, 11, 45, 99, 201],
    "mask":   [0, 0,   0,  1,   1,  0,  1,  1]
}
```

For maximum control. You tokenize and mask yourself.

### Format 3: Raw Text with Character Spans

```json
{
    "score": 1.0,
    "text": "User: Hello\n\nAssistant: Hi there, how can I help?\n\n",
    "train_spans": [[25, 49]]
}
```

`train_spans` are `[start_char, end_char]` ranges of the text to compute loss on. Everything else is masked.

## Score Handling

Each trace has a `score` field (default 1.0). Two modes for using scores:

### `--score_mode sample` (Default, Recommended)

Traces are sampled proportional to their score. A trace with `score: 2.0` appears roughly 2× as often as `score: 1.0`. The gradient per example is unmodified — only the sampling frequency changes. This is cleaner than loss-weighting because it doesn't distort gradient magnitudes.

Use this for GRPO-style training where you have a reward model scoring each trace:

```json
{"score": 1.8, "turns": [...]}    // Good trace — seen often
{"score": 0.3, "turns": [...]}    // Bad trace — seen rarely
{"score": 1.0, "turns": [...]}    // Average trace — baseline frequency
```

### `--score_mode weight`

The loss for each trace is multiplied by its score. Higher-scored traces produce larger gradients. Use cautiously — extreme scores can destabilize training.

### `--score_mode none`

Scores are ignored. All traces are seen equally.

## Key Parameters

### Chunk Size

```
--chunk_size 4096
```

**Must match the pretrained model's ctx_len.** RWKV-7 World models are trained at 4096. Using a different chunk size misaligns where the model expects intra-chunk (full resolution) vs cross-chunk (state-compressed) context.

### Cross-Chunk BPTT

```
--cross_chunk_bptt        # Enable (default)
--no_cross_chunk_bptt     # Disable (truncated baseline)
```

When enabled, gradients flow through chunk boundaries via the state chain. When disabled, the script behaves identically to existing RWKV trainers (state passed forward, gradient cut).

Cost of enabling: ~2× forward FLOPs (activation checkpointing recomputes each chunk during backward). No additional VRAM.

### Activation Checkpointing

```
--checkpoint              # Enable (default)
--no_checkpoint           # Disable
```

When cross-chunk BPTT is on, checkpointing is essential. Without it, all chunks' activations stay in VRAM simultaneously, which defeats the purpose of chunking.

When cross-chunk BPTT is off, checkpointing has no effect (states are detached anyway).

### Learning Rate

```
--lr 2e-5 --lr_final 2e-6
```

For fine-tuning pretrained RWKV-7, use `1e-5` to `5e-5`. The official recommendation is `1e-5` for 7B, slightly higher for smaller models. `lr_final` is the learning rate at the end of training.

### Weight Decay

```
--weight_decay 0.1
```

Applied only to large projection matrices, NOT to LayerNorm parameters or small vectors (time_decay, time_mix, etc.). This follows BlinkDL's specific guidance and is critical for RWKV training stability.

## Architecture of the Training Loop

For each trace:

```
1. Tokenize + mask
2. Split into chunks of size 4096
3. Initialize state = zeros

For each chunk:
    4. Forward pass (with checkpointing if not first chunk)
    5. State propagation WITHOUT detach
    6. Compute masked loss (only on trained tokens)

7. Sum all chunk losses
8. backward() — gradients flow through entire state chain
9. Gradient clipping + optimizer step
```

The critical difference from existing trainers is step 5: `state = new_state` instead of `state = new_state.detach()`. Combined with checkpointing in step 4, this gives correct gradients with constant memory.

### Memory Usage

| Component | Size | Persists across chunks? |
|-----------|------|------------------------|
| Model weights | ~2× params (weights + gradients) | Yes |
| One chunk's activations | ~1-4 GB (depends on model size) | No — recomputed |
| State chain (all boundaries) | n_chunks × ~5 MB | Yes |
| Optimizer states | ~2× params | Yes |

Total VRAM ≈ same as single-chunk training + negligible state chain overhead.

### Compute Cost

Each chunk's forward pass runs twice: once during forward, once during backward (recomputation for checkpointing). Total training FLOPS ≈ 3× forward per token (1× forward + 2× backward, where backward includes 1× recompute + 1× gradient computation).

Without cross-chunk BPTT, it's 2× forward per token (1× forward + 1× backward). So the overhead is 50% more FLOPS, not 100%.

## Integration with RWKV CUDA Kernels

The script includes a pure-PyTorch WKV scan that works out of the box. For production training, replace it with the CUDA kernel.

In `RWKV7TimeMixing._wkv_scan()`, replace:

```python
# Pure PyTorch (slow, correct)
s = state_in
for t in range(T):
    s = s * w + k[:, t] ⊗ v[:, t]
    y_t = r[:, t] · s
```

With:

```python
# CUDA kernel (fast, also correct, also computes gs in backward)
y, state_out = RUN_CUDA_RWKV7_STATE(r, k, v, w, u, state_in)
```

The CUDA kernel from `rwkv7_fast_fused` (state-passing infctx variant) already computes `gs` (gradient w.r.t. input state) in its backward pass. No kernel modifications needed.

## Preparing Data for Specific Use Cases

### CodeAct Agent Training

```json
{
    "score": 1.0,
    "turns": [
        {"role": "system", "content": "You write and execute Python code to solve tasks."},
        {"role": "user", "content": "Find the 100th Fibonacci number."},
        {"role": "assistant", "content": "<think>I'll use matrix exponentiation for O(log n) computation.</think>"},
        {"role": "action", "content": "```python\nimport numpy as np\ndef fib(n):\n    M = np.array([[1,1],[1,0]], dtype=object)\n    return np.linalg.matrix_power(M, n)[0,1]\nprint(fib(100))\n```"},
        {"role": "observation", "content": "354224848179261915075"},
        {"role": "assistant", "content": "The 100th Fibonacci number is 354224848179261915075."}
    ]
}
```

The model learns to generate thinks, actions, and answers. Observations are read but not trained on.

### GRPO (Group Relative Policy Optimization)

Generate multiple completions for each prompt, score them, then train with score-weighted sampling:

```json
{"score": 1.8, "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...good answer..."}]}
{"score": 0.2, "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...bad answer..."}]}
{"score": 1.0, "turns": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "...ok answer..."}]}
```

Use `--score_mode sample`. Good completions are seen more often, bad ones rarely.

### Instruction Following (Standard SFT)

Set all scores to 1.0 (or omit the score field). Use role-tagged turns with `user` and `assistant` roles:

```json
{
    "turns": [
        {"role": "user", "content": "Explain quantum entanglement simply."},
        {"role": "assistant", "content": "Quantum entanglement is when two particles..."}
    ]
}
```

### Long-Context Tool Use (the case cross-chunk BPTT helps most)

When observations are very long (multi-page API responses, search results, code outputs), they span multiple chunks. Cross-chunk BPTT lets the model learn to read these observations better because the gradient from the final answer flows back through the observation chunks:

```json
{
    "score": 1.0,
    "turns": [
        {"role": "user", "content": "Summarize the key findings in this dataset."},
        {"role": "action", "content": "analyze_data(dataset='quarterly_report.csv')"},
        {"role": "observation", "content": "... 8000 tokens of CSV data and analysis output ..."},
        {"role": "assistant", "content": "The key findings are: 1) Revenue grew 15%..."}
    ]
}
```

## Verification

Run the proof script to verify cross-chunk gradients work correctly:

```bash
python rwkv7_cross_chunk_bptt.py
```

Expected output:
```
Cosine similarity to ground truth gradient:
    Truncated BPTT:    0.671165
    Cross-chunk BPTT:  1.000000
```

Cross-chunk BPTT matches the full-sequence ground truth exactly. Truncated BPTT is 33% off in gradient direction.

## File Reference

| File | Purpose |
|------|---------|
| `rwkv7_agent_train.py` | Main training script |
| `rwkv7_cross_chunk_bptt.py` | Proof-of-concept + gradient verification |
| `rwkv7_agent_chunked_training.py` | Earlier reference (truncated BPTT, for comparison) |

## Command Reference

```bash
# Generate example data
python rwkv7_agent_train.py --generate_data traces.jsonl

# Train with cross-chunk BPTT (default)
python rwkv7_agent_train.py \
    --model model.pth --data traces.jsonl --output ./out \
    --chunk_size 4096 --lr 2e-5 --epochs 3

# Train with truncated BPTT (baseline comparison)
python rwkv7_agent_train.py \
    --model model.pth --data traces.jsonl --output ./out_baseline \
    --no_cross_chunk_bptt

# Score-weighted training for GRPO
python rwkv7_agent_train.py \
    --model model.pth --data scored_traces.jsonl \
    --score_mode sample --epochs 5

# Loss-weighted training (alternative)
python rwkv7_agent_train.py \
    --model model.pth --data scored_traces.jsonl \
    --score_mode weight --grad_clip 0.5

# Low-VRAM setup
python rwkv7_agent_train.py \
    --model model.pth --data traces.jsonl \
    --grad_accum 4 --precision bf16

# With wandb logging
python rwkv7_agent_train.py \
    --model model.pth --data traces.jsonl \
    --wandb "rwkv7-agent-experiment"
```
