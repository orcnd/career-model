from __future__ import annotations

"""Small deterministic gate for the human-sounding acceptance questions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKS = {
    "Does Orcun know TypeScript?": ["yes", "typescript", "frontend"],
    "Does Orcun know PHP?": ["yes", "php"],
    "Where did Orcun get educated?": ["yildiz", "associate", "computer technology"],
    "What can you tell me about this person?": ["full stack", "engineer", "eight"],
}


def score(answers: dict[str, str]) -> dict:
    results = {}
    for question, required in CHECKS.items():
        text = answers.get(question, "").casefold()
        results[question] = all(token in text for token in required) and "not enough evidence" not in text
    return {"passed": sum(results.values()), "total": len(results), "details": results}


if __name__ == "__main__":
    print(json.dumps({"checks": CHECKS}, indent=2))
