#!/usr/bin/env python3
"""
Download selected bigcode/the-stack-dedup parquet shards directly from the
Hugging Face REST API. The script:
  * Lists parquet files for python/c/cpp/javascript via the dataset metadata API
  * Writes per-language URL manifests for reference
  * Downloads each shard with curl into output/parquet/<language>
  * Persists progress so reruns resume where they left off
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from urllib.parse import quote


DATASET_REPO = "bigcode/the-stack-dedup"
DATASET_REVISION = "main"
API_URL = f"https://huggingface.co/api/datasets/{DATASET_REPO}"
TARGET_DIRS = {
    "python": "Python",
    "c": "C",
    "cpp": "C++",
    "javascript": "JavaScript",
}
LANGUAGE_STEMS = {
    "Python": "python",
    "C": "c",
    "C++": "cpp",
    "JavaScript": "js",
}
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_DIR = PROJECT_ROOT / "output"
STATE_PATH = OUTPUT_DIR / "parquet_resume_state.json"
PARQUET_DIR = OUTPUT_DIR / "parquet"
FILE_LIST_SUFFIX = "_files.txt"


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    for lang_dir in TARGET_DIRS.keys():
        (PARQUET_DIR / lang_dir).mkdir(parents=True, exist_ok=True)


def load_env_file(path: Path) -> None:
    """Populate os.environ from a .env file without overriding existing variables."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def build_file_url(filename: str) -> str:
    safe_path = "/".join(quote(part) for part in filename.split("/"))
    return f"https://huggingface.co/datasets/{DATASET_REPO}/resolve/{DATASET_REVISION}/{safe_path}"


def fetch_json_with_curl(url: str, token: str) -> Dict:
    if not token:
        raise RuntimeError("HF_TOKEN is required to call the Hugging Face API.")
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
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Failed to parse JSON from {url}: {text[:200]}") from exc


def fetch_language_entries(token: str) -> List[Dict]:
    payload = fetch_json_with_curl(API_URL, token)
    siblings = payload.get("siblings", [])

    language_entries = []
    for sibling in siblings:
        filename = sibling.get("rfilename")
        if not filename or not filename.endswith(".parquet"):
            continue
        parts = filename.split("/")
        if len(parts) < 3 or parts[0] != "data":
            continue
        lang_dir = parts[1].lower()
        if lang_dir not in TARGET_DIRS:
            continue
        lang = TARGET_DIRS[lang_dir]
        language_entries.append(
            {
                "filename": filename,
                "url": build_file_url(filename),
                "lang": lang,
                "lang_dir": lang_dir,
            }
        )

    language_entries.sort(key=lambda entry: entry["filename"])
    return language_entries


@dataclass
class ResumeState:
    completed: set = field(default_factory=set)

    @classmethod
    def load(cls, path: Path) -> "ResumeState":
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls(completed=set(data.get("completed_files", [])))
        return cls()

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"completed_files": sorted(self.completed)}, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)


def download_entry(
    entry: Dict,
    token: str,
    state: ResumeState,
) -> None:
    filename = entry.get("filename")
    url = entry.get("url")
    lang_dir = entry.get("lang_dir")
    lang = entry.get("lang")
    if not filename or not url or not lang_dir or not lang:
        print(f"Skipping entry with missing fields: {entry}", file=sys.stderr)
        return

    dest_dir = PARQUET_DIR / lang_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / Path(filename).name
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    if filename in state.completed and dest_path.exists():
        print(f"Skipping already downloaded: {filename}")
        return

    print(f"Downloading {filename} -> {dest_path}")
    start = time.time()
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--continue-at",
        "-",
        "-H",
        f"Authorization: Bearer {token}",
        "-o",
        str(temp_path),
        url,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"curl failed for {filename}: {exc}") from exc

    temp_path.replace(dest_path)
    elapsed = time.time() - start
    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"  Completed {filename} ({size_mb:.1f} MB) in {elapsed/60:.1f} min")

    state.completed.add(filename)
    state.save(STATE_PATH)


def write_language_file_lists(entries: List[Dict]) -> None:
    lists_written = 0
    for lang in TARGET_DIRS.values():
        lang_entries = [entry for entry in entries if entry["lang"] == lang]
        stem = LANGUAGE_STEMS.get(lang, lang.lower())
        list_path = OUTPUT_DIR / f"{stem}{FILE_LIST_SUFFIX}"
        with open(list_path, "w", encoding="utf-8") as handle:
            for entry in lang_entries:
                handle.write(f"{entry['url']}\n")
        print(f"Wrote {len(lang_entries):4d} entries to {list_path}")
        lists_written += len(lang_entries)
    if lists_written == 0:
        print("Warning: no entries written to file lists. Check dataset access/token.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List and process parquet shards for selected languages."
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Only fetch manifest and write parquet URL lists; skip downloading content.",
    )
    return parser.parse_args()


def main():
    ensure_dirs()
    load_env_file(ENV_PATH)
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    if not token:
        print("HF_TOKEN environment variable is required for authentication.", file=sys.stderr)
        sys.exit(1)

    print("Fetching dataset metadata...")
    entries = fetch_language_entries(token)
    print(f"Found {len(entries)} parquet files for target languages.")
    write_language_file_lists(entries)

    if args.list_only:
        print("List-only mode enabled; skipping shard downloads.")
        return

    state = ResumeState.load(STATE_PATH)
    print(f"Already completed {len(state.completed)} files.")

    pending = [entry for entry in entries if entry.get("filename") not in state.completed]
    print(f"{len(pending)} files remaining.")

    try:
        for entry in pending:
            download_entry(entry, token, state)
    except KeyboardInterrupt:
        print("\nRun cancelled. Resume later to continue with remaining files.")
        sys.exit(130)

    print("\nAll requested parquet files processed!")


if __name__ == "__main__":
    main()
