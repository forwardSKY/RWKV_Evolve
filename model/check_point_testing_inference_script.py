
import os

# --- Must set BEFORE importing rwkv ---
os.environ["RWKV_V7_ON"] = "1"      # Enable RWKV-7 architecture
os.environ["RWKV_JIT_ON"] = "1"     # JIT compilation for speed
os.environ["RWKV_CUDA_ON"] = "0"    # Set "1" only if you built the custom CUDA kernel

import time
import torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS


MODEL_PATH = r""

# "cuda fp16" = GPU inference, ~0.8 GB VRAM for 0.4B
# "cpu fp32"  = CPU fallback if GPU has issues
STRATEGY = "cuda fp16"

# For World models use "rwkv_vocab_v20230424"
# For Pile models use "20B_tokenizer.json"
TOKENIZER = "rwkv_vocab_v20230424"
# =======================================================


def load_model():
    print(f"Loading model: {MODEL_PATH}")
    t0 = time.time()
    model = RWKV(model=MODEL_PATH, strategy=STRATEGY)
    pipeline = PIPELINE(model, TOKENIZER)
    print(f"Loaded in {time.time() - t0:.1f}s\n")
    return model, pipeline


def generate(model, pipeline, prompt, max_tokens=200,
             temperature=1.0, top_p=0.3):
    """Generate text from a prompt. Returns the generated string."""
    args = PIPELINE_ARGS(
        temperature=temperature,
        top_p=top_p,
        alpha_frequency=0.25,
        alpha_presence=0.25,
        alpha_decay=0.996,
        token_ban=[],
        token_stop=[0],   # stop on end-of-text token
        chunk_len=256,
    )
    output_tokens = []
    def collect(token_str):
        output_tokens.append(token_str)
        print(token_str, end="", flush=True)

    pipeline.generate(prompt, token_count=max_tokens, args=args, callback=collect)
    print()
    return "".join(output_tokens)


def generate_unconditional(model, pipeline, max_tokens=200,
                           temperature=1.0, top_p=0.5):
    """Generate from scratch with NO input prompt (blank state)."""
    # Start with a single end-of-text token to kick off generation
    out, state = model.forward([0], None)

    tokens = []
    for _ in range(max_tokens):
        token = pipeline.sample_logits(out, temperature=temperature, top_p=top_p)
        if token == 0:  # end-of-text
            break
        tokens.append(token)
        out, state = model.forward([token], state)

    text = pipeline.decode(tokens)
    print(text)
    return text


def benchmark(model, pipeline):
    """Measure tokens/sec for decoding speed."""
    prompt = "The meaning of life is"
    out, state = model.forward(pipeline.encode(prompt), None)

    n = 100
    t0 = time.time()
    for _ in range(n):
        token = pipeline.sample_logits(out, temperature=1.0, top_p=0.3)
        out, state = model.forward([token], state)
    elapsed = time.time() - t0
    print(f"Decode speed: {n / elapsed:.1f} tokens/sec ({n} tokens in {elapsed:.2f}s)")


def model_info():
    """Print checkpoint metadata without loading the full model."""
    w = torch.load(MODEL_PATH + ".pth", map_location="cpu", weights_only=False)
    n_layer = max(int(k.split(".")[1]) for k in w if k.startswith("blocks.")) + 1
    n_embd = w["emb.weight"].shape[1]
    vocab = w["emb.weight"].shape[0]
    params = sum(p.numel() for p in w.values()) / 1e6
    print(f"Layers: {n_layer}  |  Embed: {n_embd}  |  Vocab: {vocab}  |  Params: {params:.1f}M")
    del w


# ==================== MAIN ====================
if __name__ == "__main__":
    # Print checkpoint info first
    print("=" * 60)
    print("MODEL INFO")
    print("=" * 60)
    model_info()

    # Load model
    print("\n" + "=" * 60)
    model, pipeline = load_model()

    # --- Test 1: Prompted generation ---
    print("=" * 60)
    print("TEST 1 — Prompted completion")
    print("=" * 60)
    prompts = [
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "Albert Einstein was born in",
        "Once upon a time,",
    ]
    for p in prompts:
        print(f"\nPROMPT: {p}")
        print("OUTPUT: ", end="")
        generate(model, pipeline, p, max_tokens=100)

    # --- Test 2: Unconditional generation (no input) ---
    print("\n" + "=" * 60)
    print("TEST 2 — Unconditional generation (no prompt)")
    print("=" * 60)
    print("OUTPUT: ", end="")
    generate_unconditional(model, pipeline, max_tokens=150)

    # --- Test 3: Speed benchmark ---
    print("\n" + "=" * 60)
    print("TEST 3 — Speed")
    print("=" * 60)
    benchmark(model, pipeline)

    # --- Test 4: Interactive mode ---
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE (type 'quit' to exit)")
    print("=" * 60)
    while True:
        prompt = input("\nYou: ").strip()
        if prompt.lower() == "quit":
            break
        if not prompt:
            print("(empty prompt — generating unconditionally)")
            generate_unconditional(model, pipeline, max_tokens=150)
        else:
            print("Output: ", end="")
            generate(model, pipeline, prompt, max_tokens=4096)

""