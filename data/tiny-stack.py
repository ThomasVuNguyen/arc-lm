#!/usr/bin/env python3
"""
Stream and filter the bigcode/the-stack-dedup dataset.
Filters for Python, C, C++, and JavaScript code and writes each language to
its own JSONL file by sharding the dataset across worker processes.
"""

import json
import os
import shutil
import tempfile
import time
from collections import defaultdict
from multiprocessing import cpu_count, get_context

from datasets import load_dataset

try:
    import orjson  # type: ignore
except ImportError:  # orjson is optional but faster when available
    orjson = None


# Language mappings for the dataset
LANGUAGE_FILTER = {
    'Python': 'python.json',
    'C': 'c.json',
    'C++': 'cpp.json',
    'JavaScript': 'js.json'
}

# Tunables – adjust to match machine characteristics
WRITE_BUFFER_SIZE = 1000        # Number of records to batch before flushing per lang
PROGRESS_INTERVAL = 20000       # Worker rows processed between progress reports
REPORT_STEP = 50000             # Rows between aggregated progress prints


def serialize_filtered_item(lang, item):
    """Serialize filtered record using orjson when available."""
    filtered_item = {
        'lang': lang,
        'content': item.get('content', ''),
        'stars': item.get('max_stars_count', 0)
    }
    if orjson:
        return orjson.dumps(filtered_item).decode('utf-8')
    return json.dumps(filtered_item, ensure_ascii=False, separators=(',', ':'))


def flush_buffer(handle, buffer):
    """Flush buffered lines to disk."""
    if not buffer:
        return
    handle.write('\n'.join(buffer))
    handle.write('\n')
    buffer.clear()


def worker_process(worker_id, num_workers, temp_dir, buffer_limit, progress_queue, progress_interval):
    """
    Stream a dataset shard inside the worker so we avoid shipping data over IPC.
    Each worker writes to its own temp file per language to keep IO contention low.
    """
    dataset = load_dataset(
        'bigcode/the-stack-dedup',
        split='train',
        streaming=True
    )
    shard = dataset.shard(num_shards=num_workers, index=worker_id)

    file_handles = {}
    buffers = {}
    for lang in LANGUAGE_FILTER:
        temp_file = os.path.join(temp_dir, f"{lang}_{worker_id}.jsonl")
        file_handles[lang] = open(temp_file, 'a', encoding='utf-8', buffering=1024 * 1024)
        buffers[lang] = []

    processed_since = 0
    filtered_since = 0
    lang_since = defaultdict(int)

    def emit_progress():
        nonlocal processed_since, filtered_since, lang_since
        if processed_since or filtered_since:
            progress_queue.put((
                'progress',
                processed_since,
                filtered_since,
                dict(lang_since)
            ))
            processed_since = 0
            filtered_since = 0
            lang_since = defaultdict(int)

    try:
        for item in shard:
            processed_since += 1
            lang = item.get('lang')

            if lang in LANGUAGE_FILTER:
                buffers[lang].append(serialize_filtered_item(lang, item))
                lang_since[lang] += 1
                filtered_since += 1

                if len(buffers[lang]) >= buffer_limit:
                    flush_buffer(file_handles[lang], buffers[lang])

            if processed_since >= progress_interval:
                emit_progress()
    except Exception as exc:  # Relay error to parent so it can abort cleanly
        progress_queue.put(('error', worker_id, repr(exc)))
        raise
    finally:
        for lang in LANGUAGE_FILTER:
            flush_buffer(file_handles[lang], buffers[lang])
            file_handles[lang].close()
        emit_progress()
        progress_queue.put(('done', worker_id))


def merge_temp_files(temp_dir, output_dir):
    """Merge all temp files into final output files (fast sequential merge)."""
    print("\nMerging temp files into final output files...")
    merge_start = time.time()

    # Group temp files by language
    lang_files = defaultdict(list)
    for filename in os.listdir(temp_dir):
        if filename.endswith('.jsonl'):
            for lang in LANGUAGE_FILTER:
                prefix = f"{lang}_"
                if filename.startswith(prefix):
                    lang_files[lang].append(os.path.join(temp_dir, filename))
                    break

    # Merge files for each language
    for lang, temp_files in lang_files.items():
        output_file = os.path.join(output_dir, LANGUAGE_FILTER[lang])
        print(f"  Merging {len(temp_files)} files for {lang}...")

        with open(output_file, 'w', encoding='utf-8', buffering=1024 * 1024) as outfile:
            for temp_file in sorted(temp_files):
                try:
                    with open(temp_file, 'r', encoding='utf-8', buffering=1024 * 1024) as infile:
                        shutil.copyfileobj(infile, outfile)
                except Exception as exc:
                    print(f"    Warning: Error reading {temp_file}: {exc}")

    print("  Cleaning up temp files...")
    shutil.rmtree(temp_dir)

    merge_time = time.time() - merge_start
    print(f"  Merge completed in {merge_time:.1f} seconds")


def main():
    print("Starting to stream bigcode/the-stack-dedup dataset...")
    print(f"Filtering for languages: {', '.join(LANGUAGE_FILTER.keys())}")

    num_workers = cpu_count()
    print(f"Using {num_workers} worker processes (dataset sharded per worker)")

    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    for filename in LANGUAGE_FILTER.values():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleared existing file: {filepath}")

    temp_dir = tempfile.mkdtemp(prefix='tiny-stack-temp-')
    print(f"Using temp directory: {temp_dir}")

    ctx = get_context('spawn')
    progress_queue = ctx.Queue(maxsize=num_workers * 4)

    workers = []
    for worker_id in range(num_workers):
        proc = ctx.Process(
            target=worker_process,
            args=(worker_id, num_workers, temp_dir, WRITE_BUFFER_SIZE, progress_queue, PROGRESS_INTERVAL),
            daemon=True
        )
        proc.start()
        workers.append(proc)

    total_processed = 0
    total_filtered = 0
    lang_counts = defaultdict(int)
    error_messages = []
    finished_workers = 0
    start_time = time.time()
    next_report = REPORT_STEP

    print("\nProcessing dataset (progress aggregated across workers)...\n")

    while finished_workers < num_workers:
        message = progress_queue.get()
        kind = message[0]

        if kind == 'progress':
            _, processed_delta, filtered_delta, lang_delta = message
            total_processed += processed_delta
            total_filtered += filtered_delta
            for lang, count in lang_delta.items():
                lang_counts[lang] += count

            if total_processed >= next_report:
                elapsed = time.time() - start_time
                rows_per_sec = total_processed / elapsed if elapsed > 0 else 0
                print(f"[{time.strftime('%H:%M:%S')}] Processed: {total_processed:,} rows "
                      f"| Filtered: {total_filtered:,} rows | Speed: {rows_per_sec:,.1f} rows/sec")
                for lang, count in sorted(lang_counts.items()):
                    print(f"  {lang}: {count:,}")
                next_report += REPORT_STEP

        elif kind == 'error':
            _, worker_id, details = message
            error_messages.append(f"Worker {worker_id} error: {details}")

        elif kind == 'done':
            finished_workers += 1

    for proc in workers:
        proc.join()

    if error_messages:
        # Clean up temp directory if something went wrong
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise RuntimeError("Processing failed:\n" + "\n".join(error_messages))

    merge_temp_files(temp_dir, output_dir)

    end_time = time.time()
    total_elapsed = end_time - start_time
    avg_rate = total_processed / total_elapsed if total_elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("Processing Complete!")
    print("=" * 60)
    print(f"Total rows processed: {total_processed:,}")
    print(f"Total rows filtered: {total_filtered:,}")
    print(f"Total time elapsed: {total_elapsed / 60:.1f} minutes")
    print(f"Average speed: {avg_rate:,.1f} rows/sec")
    print("\nBreakdown by language:")
    for lang, count in sorted(lang_counts.items()):
        filename = LANGUAGE_FILTER[lang]
        print(f"  {lang}: {count:,} rows -> {os.path.join(output_dir, filename)}")


if __name__ == '__main__':
    main()
