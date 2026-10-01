from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
source = ROOT / "data/benchmark/benchmark.jsonl"
target = ROOT / "data/benchmark/benchmark_review.md"
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
labels = Counter(row["label"] for row in rows)
    lines = ["# Career-model benchmark", "", f"Questions: {len(rows)}", f"Supported: {labels['supported']} | Unsupported/adversarial: {labels['unsupported']}", "", "Use this only for manual review or local evaluation. It must never be included in training.", ""]
for number, row in enumerate(rows, 1):
    lines.extend([f"## {number}. {row['id']}", "", f"**Question:** {row['messages'][1]['content']}", "", f"**Expected answer:** {row['messages'][2]['content']}", "", f"**Label:** {row['label']}", ""])
target.write_text("\n".join(lines), encoding="utf-8")
print(target)
