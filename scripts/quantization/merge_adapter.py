from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge a local PEFT adapter into its base model for GGUF export.")
    parser.add_argument("--adapter", type=Path, default=ROOT / "models/qwen-1.5b-career-lora-cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "models/qwen-1.5b-career-merged")
    args = parser.parse_args()
    import torch
    from peft import PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    config = PeftConfig.from_pretrained(args.adapter)
    base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, dtype=torch.float16, low_cpu_mem_usage=True)
    merged = PeftModel.from_pretrained(base, args.adapter).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(args.output, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
