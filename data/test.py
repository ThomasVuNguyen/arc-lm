#!/usr/bin/env python3
"""Probe the Hugging Face dataset APIs via curl commands."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List


SCRIPT_DIR = Path(__file__).resolve().parent
ENV_PATH = SCRIPT_DIR.parent / ".env"
DATASET = "bigcode/the-stack-dedup"


def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def run_curl(url: str) -> Dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN not set. Populate .env with HF_TOKEN=<token>.")
    cmd = [
        "curl",
        "-s",
        "-H",
        f"Authorization: Bearer {token}",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    text = result.stdout.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        raise SystemExit(f"Failed to parse JSON from {url}: {text[:200]}...")


def main() -> None:
    load_env(ENV_PATH)

    base = run_curl(f"https://huggingface.co/api/datasets/{DATASET}")
    print("dataset keys:", list(base.keys())[:8])
    siblings: List[Dict] = base.get("siblings", [])
    print("total siblings:", len(siblings))
    for lang in ("python", "c", "cpp", "javascript"):
        matches = [s for s in siblings if s.get("rfilename", "").startswith(f"data/{lang}/")]
        print(f"  {lang}: {len(matches)} files")
        if matches:
            print("    sample:", matches[0]["rfilename"])

    parquet = run_curl(f"https://huggingface.co/api/datasets/{DATASET}/parquet/default/train")
    print("parquet manifest entries:", len(parquet))
    print("first entry type:", type(parquet[0]).__name__ if parquet else "n/a")
    if parquet:
        print("first entry sample:", parquet[0])

    tree = run_curl(
        f"https://huggingface.co/api/datasets/{DATASET}/tree/main?recursive=1&path=data"
    )
    python_nodes = [node for node in tree if node.get("path", "").startswith("data/python/")]
    print("tree nodes under data/python:", len(python_nodes))
    if python_nodes:
        print(" sample node:", python_nodes[0])


if __name__ == "__main__":
    main()

