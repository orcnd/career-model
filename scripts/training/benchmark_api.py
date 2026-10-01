from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import unicodedata
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.server import response_matches_language


def ask(base_url: str, session_id: str, question: str) -> tuple[dict, float]:
    payload = json.dumps({"session_id": session_id, "question": question}).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/ask",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=90) as response:
        envelope = json.load(response)
    elapsed = time.perf_counter() - started
    try:
        model_answer = json.loads(envelope["answer"])
    except (json.JSONDecodeError, KeyError, TypeError):
        model_answer = {"answer": "invalid_json", "evidence": envelope.get("answer", "")}
    return model_answer, elapsed


def evaluate(case: dict, result: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if result.get("answer") != case["answer"]:
        reasons.append(f"answer={result.get('answer')!r}")
    evidence = result.get("evidence")
    text = "" if evidence is None else str(evidence)
    folded = "".join(
        character
        for character in unicodedata.normalize("NFKD", text.casefold())
        if not unicodedata.combining(character)
    )
    for token in case.get("required", []):
        normalized_token = "".join(
            character
            for character in unicodedata.normalize("NFKD", token.casefold())
            if not unicodedata.combining(character)
        )
        numeric_aliases = {
            "8": ("8", "eight", "sekiz", "otto", "huit", "ثمان"),
            "9": ("9", "nine", "dokuz", "nove", "neuf", "تسع"),
        }
        aliases = numeric_aliases.get(normalized_token, (normalized_token,))
        if not any(alias in folded for alias in aliases):
            reasons.append(f"missing={token!r}")
    for token in case.get("forbidden", []):
        normalized_token = "".join(
            character
            for character in unicodedata.normalize("NFKD", token.casefold())
            if not unicodedata.combining(character)
        )
        if normalized_token in folded:
            reasons.append(f"forbidden={token!r}")
    if not response_matches_language(json.dumps(result, ensure_ascii=False), case["language"]):
        reasons.append(f"wrong_language={case['language']}")
    return not reasons, reasons


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the multilingual career-chat regression benchmark.")
    parser.add_argument("--suite", type=Path, default=ROOT / "data/benchmark/regression_suite.json")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--output", type=Path, default=ROOT / "data/benchmark/latest_results.json")
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    results, latencies = [], []

    jobs = [(case, str(uuid.uuid4()), None) for case in suite["cases"]]
    for sequence in suite["sequences"]:
        session_id = str(uuid.uuid4())
        jobs.extend((turn, session_id, sequence["id"]) for turn in sequence["turns"])

    for case, session_id, sequence_id in jobs:
        answer, elapsed = ask(args.url, session_id, case["question"])
        passed, reasons = evaluate(case, answer)
        latencies.append(elapsed)
        results.append({
            "id": case.get("id") or f"{sequence_id}:{case['question']}",
            "sequence": sequence_id,
            "question": case["question"],
            "passed": passed,
            "reasons": reasons,
            "answer": answer,
            "latency_seconds": round(elapsed, 3),
        })
        print(f"{'PASS' if passed else 'FAIL'}  {results[-1]['id']}  {elapsed:.2f}s")

    passed_count = sum(item["passed"] for item in results)
    rate = passed_count / len(results)
    report = {
        "suite_version": suite["version"],
        "passed": passed_count,
        "total": len(results),
        "pass_rate": round(rate, 4),
        "minimum_pass_rate": suite["minimum_pass_rate"],
        "latency": {
            "median_seconds": round(statistics.median(latencies), 3),
            "p95_seconds": round(percentile(latencies, 0.95), 3),
            "max_seconds": round(max(latencies), 3),
        },
        "results": results,
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0 if rate >= suite["minimum_pass_rate"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
