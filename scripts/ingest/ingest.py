from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from career_model.core import redact, write_json


def extract(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and redact a private career source.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/redacted_source.txt")
    args = parser.parse_args()
    clean = redact(extract(args.input))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(clean, encoding="utf-8")
    template = ROOT / "data/processed/career_knowledge.template.json"
    if not template.exists():
        write_json(template, {"subject": "Orcun", "technologies": [], "roles": [], "projects": [], "negative_technologies": [], "review_required": True})
    print(f"Wrote redacted source: {args.output}. Complete {template} manually before dataset generation.")


if __name__ == "__main__":
    main()
