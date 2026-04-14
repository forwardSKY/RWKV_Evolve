import os
os.environ["RWKV_V7_ON"] = "1"
os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "0"

import time, torch
from rwkv.model import RWKV
from rwkv.utils import PIPELINE, PIPELINE_ARGS

MODEL_PATH = r""
STRATEGY = "cuda fp16"
TOKENIZER = "rwkv_vocab_v20230424"


def load_model():
    t0 = time.time()
    model = RWKV(model=MODEL_PATH, strategy=STRATEGY)
    pipeline = PIPELINE(model, TOKENIZER)
    print(f"Model loaded ({time.time()-t0:.1f}s)")
    return model, pipeline


def generate(model, pipeline, prompt, max_tokens=200, temperature=1.0, top_p=0.3):
    args = PIPELINE_ARGS(
        temperature=temperature, top_p=top_p,
        alpha_frequency=0.25, alpha_presence=0.25, alpha_decay=0.996,
        token_ban=[], token_stop=[0], chunk_len=256,
    )
    tokens = []
    def collect(s):
        tokens.append(s)
        print(s, end="", flush=True)
    pipeline.generate(prompt, token_count=max_tokens, args=args, callback=collect)
    print()
    return "".join(tokens)


def generate_unconditional(model, pipeline, max_tokens=200, temperature=1.0, top_p=0.5):
    out, state = model.forward([0], None)
    tokens = []
    for _ in range(max_tokens):
        tok = pipeline.sample_logits(out, temperature=temperature, top_p=top_p)
        if tok == 0:
            break
        tokens.append(tok)
        out, state = model.forward([tok], state)
    text = pipeline.decode(tokens)
    print(text)
    return text


def benchmark(model, pipeline, n=100):
    out, state = model.forward(pipeline.encode("The meaning of life is"), None)
    t0 = time.time()
    for _ in range(n):
        tok = pipeline.sample_logits(out, temperature=1.0, top_p=0.3)
        out, state = model.forward([tok], state)
    dt = time.time() - t0
    print(f"{n/dt:.1f} tok/s ({n} tokens, {dt:.2f}s)")


def model_info():
    w = torch.load(MODEL_PATH + ".pth", map_location="cpu", weights_only=False)
    n_layer = max(int(k.split(".")[1]) for k in w if k.startswith("blocks.")) + 1
    n_embd = w["emb.weight"].shape[1]
    vocab = w["emb.weight"].shape[0]
    params = sum(p.numel() for p in w.values()) / 1e6
    print(f"L={n_layer} D={n_embd} V={vocab} {params:.1f}M params")
    del w


if __name__ == "__main__":
    model_info()
    model, pipeline = load_model()

    # Prompted completion
    print("\n--- Prompted ---")
    for p in [
        "The capital of France is",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        "Albert Einstein was born in",
        "Once upon a time,",
    ]:
        print(f"\n> {p}")
        generate(model, pipeline, p, max_tokens=100)

    # Unconditional
    print("\n--- Unconditional ---")
    generate_unconditional(model, pipeline, max_tokens=150)

    # Benchmark
    print("\n--- Benchmark ---")
    benchmark(model, pipeline)

    # Interactive
    print("\n--- Interactive (quit to exit) ---")
    while True:
        prompt = input("\n> ").strip()
        if prompt.lower() == "quit":
            break
        if not prompt:
            generate_unconditional(model, pipeline, max_tokens=150)
        else:
            generate(model, pipeline, prompt, max_tokens=4096)