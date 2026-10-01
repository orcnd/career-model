from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INSUFFICIENT = "There is not enough evidence in the provided career information to support that claim."


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normal(text: str) -> str:
    return " ".join(text.casefold().split())


def mean(values: list[bool]) -> float | None:
    return round(sum(values) / len(values) * 100, 2) if values else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a held-out, local career-model benchmark.")
    parser.add_argument("--benchmark", type=Path, default=ROOT / "data/benchmark/benchmark.jsonl")
    parser.add_argument("--answers", type=Path, required=True, help="JSONL: id, answer; optional human audit booleans factual_correct, unsupported_claim, complete")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/evaluation.json")
    args = parser.parse_args()
    expected = {row["id"]: row for row in load_jsonl(args.benchmark)}
    answers = {row["id"]: row for row in load_jsonl(args.answers)}
    factual, unsupported, complete, negative = [], [], [], []
    missing = []
    for key, item in expected.items():
        answer = answers.get(key)
        if not answer:
            missing.append(key); factual.append(False); complete.append(False)
            if item["label"] == "unsupported": unsupported.append(True); negative.append(False)
            continue
        response = answer.get("answer", "")
        expected_answer = item["messages"][-1]["content"]
        is_negative = item["label"] == "unsupported"
        correct = answer.get("factual_correct", normal(expected_answer) == normal(response))
        invented = answer.get("unsupported_claim", is_negative and normal(INSUFFICIENT) not in normal(response))
        is_complete = answer.get("complete", correct)
        factual.append(bool(correct)); unsupported.append(bool(invented)); complete.append(bool(is_complete))
        if is_negative: negative.append(bool(correct))
    report = {"benchmark_questions": len(expected), "answered": len(expected) - len(missing), "missing_ids": missing,
              "factual_accuracy_percent": mean(factual), "hallucination_rate_percent": mean(unsupported),
              "unsupported_claim_rate_percent": mean(unsupported), "answer_completeness_percent": mean(complete),
              "negative_question_accuracy_percent": mean(negative),
              "note": "Exact-match is the default. For nuanced answers, add the optional human-audit booleans to answers.jsonl; do not use an external judge at inference."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
