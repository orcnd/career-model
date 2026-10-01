from __future__ import annotations

"""Transparent PEFT baseline; Soup is the preferred runner for production runs."""
import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/qwen-0.5b.yaml")
    args = parser.parse_args()
    import torch
    import yaml
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from trl import SFTTrainer
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = ROOT / cfg["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4") if cfg.get("use_4bit") else None
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], quantization_config=quant, device_map="auto", torch_dtype="auto")
    dataset = load_dataset("json", data_files=str(ROOT / cfg["dataset_dir"] / "train.jsonl"), split="train")
    dataset = dataset.map(lambda row: {"text": tokenizer.apply_chat_template(row["messages"], tokenize=False)})
    peft = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"], target_modules=cfg["target_modules"], task_type="CAUSAL_LM")
    training = TrainingArguments(output_dir=str(output), learning_rate=cfg["learning_rate"], num_train_epochs=cfg["num_train_epochs"], per_device_train_batch_size=cfg["per_device_train_batch_size"], gradient_accumulation_steps=cfg["gradient_accumulation_steps"], logging_steps=5, save_strategy="epoch", report_to="none")
    trainer = SFTTrainer(model=model, train_dataset=dataset, processing_class=tokenizer, peft_config=peft, args=training, dataset_text_field="text", max_seq_length=cfg["max_seq_length"])
    result = trainer.train()
    trainer.save_model(str(output))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True).stdout.strip()
    (output / "experiment.json").write_text(json.dumps({"config": cfg, "git_revision": revision, "metrics": result.metrics}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
