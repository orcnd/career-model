from __future__ import annotations

"""Slow but real CPU-only LoRA training; no CUDA, RAG, or remote inference."""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen-0.5b-cpu.yaml")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-run cap; omitted trains every reviewed example.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Cap epochs for a bounded local tuning run.")
    parser.add_argument("--resume-adapter", type=Path, default=None, help="Continue a compatible local LoRA adapter.")
    args = parser.parse_args()
    import torch
    import yaml
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_threads = int(os.environ.get("CAREER_TRAIN_THREADS", "6"))
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"]); torch.set_num_threads(max(1, min(train_threads, os.cpu_count() or 1)))
    out = ROOT / cfg["output_dir"]; out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in (ROOT / cfg["dataset_dir"] / "train.jsonl").read_text(encoding="utf-8").splitlines() if x]
    if args.limit: rows = rows[:args.limit]
    if not rows: raise SystemExit("Training split is empty. Generate reviewed data first.")
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], torch_dtype=torch.float32, low_cpu_mem_usage=True)
    model.config.use_cache = False
    if args.resume_adapter:
        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"], task_type=TaskType.CAUSAL_LM))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg["learning_rate"])
    metrics = []; started = time.perf_counter(); model.train()
    epoch_count = min(int(cfg["num_train_epochs"]), args.max_epochs) if args.max_epochs else int(cfg["num_train_epochs"])
    for epoch in range(epoch_count):
        random.shuffle(rows); optimizer.zero_grad(); total = 0.0
        for step, row in enumerate(rows, 1):
            text = tokenizer.apply_chat_template(row["messages"], tokenize=False)
            prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=False, add_generation_prompt=True)
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=cfg["max_seq_length"])
            labels = encoded["input_ids"].clone()
            prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            labels[:, :min(len(prompt_tokens), labels.shape[1])] = -100
            result = model(**encoded, labels=labels)
            (result.loss / cfg["gradient_accumulation_steps"]).backward(); total += result.loss.item()
            if step % 5 == 0: print(json.dumps({"epoch": epoch + 1, "step": step, "total_steps": len(rows), "loss": round(result.loss.item(), 4), "seconds_elapsed": round(time.perf_counter() - started, 2)}), flush=True)
            if step % cfg["gradient_accumulation_steps"] == 0 or step == len(rows):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); optimizer.zero_grad()
            if step % 25 == 0:
                model.save_pretrained(out); tokenizer.save_pretrained(out)
                print(json.dumps({"epoch": epoch + 1, "checkpoint_step": step, "output_dir": str(out)}), flush=True)
        epoch_metric = {"epoch": epoch + 1, "train_loss": total / len(rows), "seconds_elapsed": round(time.perf_counter() - started, 2)}
        metrics.append(epoch_metric); print(json.dumps(epoch_metric), flush=True)
        model.save_pretrained(out); tokenizer.save_pretrained(out)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    (out / "experiment.json").write_text(json.dumps({"config": cfg, "device": "cpu", "git_revision": revision, "metrics": metrics}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
