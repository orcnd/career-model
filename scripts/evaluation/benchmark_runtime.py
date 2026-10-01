from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure local GGUF startup, RAM and generation performance.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("experiments/runtime.json"))
    args = parser.parse_args()
    if not args.model.exists(): raise SystemExit(f"Missing GGUF: {args.model}")
    import psutil
    from llama_cpp import Llama
    process = psutil.Process()
    load_started = time.perf_counter(); llm = Llama(model_path=str(args.model), n_ctx=2048, verbose=False); load_s = time.perf_counter() - load_started
    prompt = "Answer briefly: What evidence is needed before claiming a professional technology skill?"
    started = time.perf_counter(); first = None; tokens = 0
    for event in llm.create_completion(prompt, max_tokens=96, temperature=0, stream=True):
        if first is None: first = time.perf_counter() - started
        tokens += 1
    total = time.perf_counter() - started
    report = {"model": str(args.model), "model_file_bytes": args.model.stat().st_size, "startup_seconds": round(load_s, 4), "first_token_latency_seconds": round(first or total, 4), "generated_tokens": tokens, "generation_seconds": round(total, 4), "tokens_per_second": round(tokens / total, 2) if total else None, "peak_rss_bytes": process.memory_info().rss, "note": "Peak RSS is sampled after generation; use OS profiling for a strict process-lifetime peak."}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
