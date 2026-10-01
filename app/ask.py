from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

SYSTEM = "You answer questions only from your fine-tuned career knowledge. Never invent experience, dates, or credentials. If evidence is absent, respond exactly: There is not enough evidence in the provided career information to support that claim."


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline career-model CLI. No retrieval or network access.")
    parser.add_argument("question")
    parser.add_argument("--model", type=Path, default=Path(os.environ.get("CAREER_MODEL_PATH", "models/career-q4_k_m.gguf")))
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    if not args.model.exists(): raise SystemExit(f"Local GGUF not found: {args.model}")
    from llama_cpp import Llama
    llm = Llama(model_path=str(args.model), n_ctx=2048, n_threads=args.threads, verbose=False)
    started = time.perf_counter()
    reply = llm.create_chat_completion(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": args.question}], temperature=0, max_tokens=256)
    print(reply["choices"][0]["message"]["content"].strip())
    print(f"\n[local inference: {time.perf_counter() - started:.2f}s]")


if __name__ == "__main__":
    main()
