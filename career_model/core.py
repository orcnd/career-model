from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

INSUFFICIENT = '{"answer":"insufficient_evidence","evidence":null}'
# PDF extraction can insert whitespace between every glyph, so accept it inside an email too.
EMAIL = re.compile(r"(?i)(?:[A-Z0-9._%+-]\s*)+@\s*(?:[A-Z0-9.-]\s*)+\.\s*(?:[A-Z]{2,}\s*)+")
PHONE = re.compile(r"(?<!\w)(?:\+?\s*\d(?:[\s()./-]*\d){7,})(?!\w)")


def redact(text: str) -> str:
    text = EMAIL.sub("[REDACTED_EMAIL]", text)
    return PHONE.sub("[REDACTED_PHONE]", text)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_bucket(value: str, buckets: int = 100) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % buckets


@dataclass(frozen=True)
class Example:
    id: str
    family: str
    question: str
    answer: str
    evidence: list[str]
    label: str

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "family": self.family, "messages": [
            {"role": "system", "content": "Answer only from documented career evidence. Return JSON only: {\"answer\": \"yes|no|insufficient_evidence\", \"evidence\": \"documented evidence or null\"}. Never invent experience."},
            {"role": "user", "content": self.question},
            {"role": "assistant", "content": self.answer},
        ], "evidence": self.evidence, "label": self.label}
