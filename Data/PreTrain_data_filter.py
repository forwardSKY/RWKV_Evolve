"""
Training data pipeline:
  - swallow_code : streamed, appended to output
  - fineweb_edu  : download shard → filter locally → append → delete shard

Disk usage at any point: ~2GB (one shard) + output JSONL
Requests to HuggingFace: 1 per shard (2,410 total)

Requirements:
    pip install datasets duckdb huggingface_hub
"""

import json, os, time, duckdb
from datasets import load_dataset
from huggingface_hub import HfFileSystem, hf_hub_download

# ── Config ─────────────────────────────────────────────────────────────────────

BASE_DIR        = r""
OUTPUT_FILE     = os.path.join(BASE_DIR, "Training_data.jsonl")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "checkpoint.json")
CACHE_DIR       = os.path.join(BASE_DIR, "hf_cache")   # download here, not C: drive
FINEWEB_REPO    = "HuggingFaceFW/fineweb-edu"
FETCH_BATCH     = 10_000
HF_TOKEN        = ""     # huggingface.co/settings/tokens → New token (read)
RETRY_WAITS     = [60, 120, 300, 600]      # waits between retries: 1min, 2min, 5min, 10min

# ── Checkpoint ─────────────────────────────────────────────────────────────────

DEFAULT_CHECKPOINT = {
    "swallow_done": False,
    "shard_list":   [],
    "shard_index":  0,
    "output_bytes": 0,
}

def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return DEFAULT_CHECKPOINT.copy()

def save_checkpoint(cp):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(cp, f, indent=2)

def current_file_size():
    return os.path.getsize(OUTPUT_FILE) if os.path.exists(OUTPUT_FILE) else 0

def truncate_to(n_bytes):
    with open(OUTPUT_FILE, "ab") as f:
        f.truncate(n_bytes)

# ── Stage 1: swallow_code ──────────────────────────────────────────────────────

def run_swallow_code(cp):
    if cp["swallow_done"]:
        print("swallow_code: already done, skipping.\n")
        return

    print("swallow_code: starting...")
    ds = load_dataset("tokyotech-llm/swallow-code", name="swallow-code", split="train", streaming=True)

    count = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        for data in ds:
            out.write(json.dumps({"text": data["text"]}, ensure_ascii=False) + "\n")
            count += 1
            if count % 10_000 == 0:
                print(f"  {count:,} rows")
        out.flush()
        os.fsync(out.fileno())

    cp["swallow_done"] = True
    cp["output_bytes"] = current_file_size()
    save_checkpoint(cp)
    print(f"swallow_code: done — {count:,} rows\n")

# ── Stage 2: fineweb_edu ───────────────────────────────────────────────────────

def get_shard_list(cp):
    if cp["shard_list"]:
        print(f"Using cached shard list ({len(cp['shard_list']):,} shards)\n")
        return cp["shard_list"]

    print("Fetching shard list from HuggingFace (one-time)...")
    fs = HfFileSystem(token=HF_TOKEN)
    paths = fs.glob(f"datasets/{FINEWEB_REPO}/data/CC-MAIN-*/*.parquet")
    # Store only the filename path, not the full hf:// URL
    cp["shard_list"] = [p.replace(f"datasets/{FINEWEB_REPO}/", "") for p in sorted(paths)]
    save_checkpoint(cp)
    print(f"  {len(cp['shard_list']):,} shards found\n")
    return cp["shard_list"]

def filter_and_write(local_path):
    """Filter parquet locally with DuckDB, append results to output."""
    con = duckdb.connect()
    result = con.execute(f"""
        SELECT text FROM read_parquet('{local_path}')
        WHERE language = 'en' AND token_count <= 4096 AND int_score >= 4.5  n  
    """)
    written = 0
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out:
        while True:
            batch = result.fetchmany(FETCH_BATCH)
            if not batch:
                break
            for row in batch:
                out.write(json.dumps({"text": row[0]}, ensure_ascii=False) + "\n")
                written += 1
        out.flush()
        os.fsync(out.fileno())
    con.close()
    return written

def process_shard(shard_path, safe_bytes):
    """Download shard, filter it, delete it. Retries on failure."""
    local_path = None

    for attempt, wait in enumerate(RETRY_WAITS + [None], start=1):
        try:
            print(f"  downloading...")
            local_path = hf_hub_download(
                repo_id   = FINEWEB_REPO,
                filename  = shard_path,
                repo_type = "dataset",
                token     = HF_TOKEN,
                local_dir = CACHE_DIR,   # plain file, no blob/snapshot — os.remove() works correctly
            )
            written = filter_and_write(local_path)
            return written  # success

        except Exception as e:
            truncate_to(safe_bytes)  # clean up any partial write
            print(f"  attempt {attempt}/{len(RETRY_WAITS)+1} failed: {e}")
            if wait is not None:
                print(f"  waiting {wait}s...")
                time.sleep(wait)

        finally:
            # Always delete the downloaded shard whether we succeeded or failed
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
                local_path = None

    return None  # all retries exhausted

def run_fineweb_edu(cp):
    all_shards = get_shard_list(cp)
    total      = len(all_shards)

    # Truncate any partial write from a previous crashed run
    if current_file_size() != cp["output_bytes"]:
        print(f"Truncating partial shard back to {cp['output_bytes']:,} bytes...")
        truncate_to(cp["output_bytes"])

    for i in range(cp["shard_index"], total):
        shard = all_shards[i]
        name  = "/".join(shard.split("/")[-2:])
        print(f"[{i+1}/{total}] {name}")

        # Save rollback point before touching the file
        cp["shard_index"]  = i
        cp["output_bytes"] = current_file_size()
        save_checkpoint(cp)

        written = process_shard(shard, cp["output_bytes"])

        if written is None:
            print(f"  giving up — will retry on next run")
        else:
            print(f"  {written:,} rows written")
            cp["shard_index"]  = i + 1
            cp["output_bytes"] = current_file_size()
            save_checkpoint(cp)

    cp["shard_index"] = total
    save_checkpoint(cp)
    print("\nfineweb_edu: all done!")

# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    cp = load_checkpoint()
    run_swallow_code(cp)
    run_fineweb_edu(cp)
    print(f"\nAll done! Output: {OUTPUT_FILE}")