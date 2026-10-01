from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge/export a Soup adapter with its audited CLI.")
    parser.add_argument("--adapter", required=True, type=Path)
    args = parser.parse_args()
    run(["soup", "merge", "--adapter", str(args.adapter)])
    run(["soup", "export", "--model", str(args.adapter), "--format", "gguf", "--quant", "q4_k_m"])


if __name__ == "__main__":
    main()
