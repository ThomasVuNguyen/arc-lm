#!/usr/bin/env python3
"""
Stream and filter bigcode/the-stack-dedup dataset
Filters for Python, C, C++, and JavaScript code
Outputs separate JSON files for each language
"""

import json
import os
import time
from collections import defaultdict
from multiprocessing import Pool, Manager, cpu_count
from datasets import load_dataset


# Language mappings for the dataset
LANGUAGE_FILTER = {
    'Python': 'python.json',
    'C': 'c.json',
    'C++': 'cpp.json',
    'JavaScript': 'js.json'
}


def process_batch(batch_data):
    """Process a batch of rows and categorize by language"""
    lang_data = defaultdict(list)

    for item in batch_data:
        lang = item.get('lang', '')

        # Check if the language is one we want
        if lang in LANGUAGE_FILTER:
            filtered_item = {
                'lang': lang,
                'content': item.get('content', ''),
                'stars': item.get('max_stars_count', 0)
            }
            lang_data[lang].append(filtered_item)

    return lang_data


def write_batch_to_files(lang_data, output_dir='output', file_locks=None):
    """Write categorized data to respective JSON files incrementally"""
    os.makedirs(output_dir, exist_ok=True)

    for lang, items in lang_data.items():
        if items:
            output_file = os.path.join(output_dir, LANGUAGE_FILTER[lang])
            lock = file_locks.get(lang) if file_locks else None

            # Thread-safe file writing
            if lock:
                lock.acquire()
            try:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for item in items:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure data is written to disk immediately
            finally:
                if lock:
                    lock.release()


def process_and_write_batch(batch_data, output_dir, file_locks):
    """Process a batch and write results to files (for parallel execution)"""
    lang_data = process_batch(batch_data)
    write_batch_to_files(lang_data, output_dir, file_locks)
    
    # Return counts for statistics
    counts = {}
    for lang, items in lang_data.items():
        counts[lang] = len(items)
    return counts, len(batch_data)


def main():
    print("Starting to stream bigcode/the-stack-dedup dataset...")
    print(f"Filtering for languages: {', '.join(LANGUAGE_FILTER.keys())}")
    print(f"Using {cpu_count()} CPU cores for processing")

    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Clear existing output files
    for filename in LANGUAGE_FILTER.values():
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"Cleared existing file: {filepath}")

    # Create Manager for shared locks
    manager = Manager()
    file_locks = {lang: manager.Lock() for lang in LANGUAGE_FILTER.keys()}

    # Create process pool with all CPU cores
    num_cores = cpu_count()
    pool = Pool(processes=num_cores)

    try:
        # Stream the dataset
        dataset = load_dataset(
            'bigcode/the-stack-dedup',
            split='train',
            streaming=True
        )

        batch_size = 1000  # Process 1000 rows at a time
        batch = []
        total_processed = 0
        total_filtered = 0
        lang_counts = defaultdict(int)

        # Time tracking for ETA
        start_time = time.time()

        # Pending async results
        pending_results = []

        print("\nProcessing dataset...")
        print("Data is being written incrementally to JSON files as processing occurs.\n")

        for idx, item in enumerate(dataset):
            batch.append(item)

            if len(batch) >= batch_size:
                # Submit batch to process pool for parallel processing
                async_result = pool.apply_async(
                    process_and_write_batch,
                    (batch.copy(), output_dir, file_locks)
                )
                pending_results.append(async_result)
                batch = []

                # Process completed results (non-blocking check)
                completed = []
                for i, result in enumerate(pending_results):
                    if result.ready():
                        completed.append(i)
                        try:
                            counts, batch_len = result.get()
                            total_processed += batch_len
                            
                            # Update counts
                            for lang, count in counts.items():
                                lang_counts[lang] += count
                                total_filtered += count
                        except Exception as e:
                            print(f"Error processing batch: {e}")

                # Remove completed results (in reverse to maintain indices)
                for i in reversed(completed):
                    pending_results.pop(i)

                # Print progress every 10k rows with ETA
                if total_processed % 10000 == 0:
                    current_time = time.time()
                    elapsed = current_time - start_time
                    rows_per_sec = total_processed / elapsed if elapsed > 0 else 0

                    # Calculate ETA (rough estimate based on current rate)
                    # Note: Since this is a streaming dataset, we don't know total size
                    # So we show processing rate instead
                    eta_str = f"{rows_per_sec:.1f} rows/sec"

                    print(f"\n[{time.strftime('%H:%M:%S')}] Processed: {total_processed:,} rows | Filtered: {total_filtered:,} rows")
                    print(f"  Speed: {eta_str} | Elapsed: {elapsed/60:.1f} min | Pending batches: {len(pending_results)}")
                    for lang, count in sorted(lang_counts.items()):
                        print(f"  {lang}: {count:,}")

        # Process remaining batch
        if batch:
            async_result = pool.apply_async(
                process_and_write_batch,
                (batch, output_dir, file_locks)
            )
            pending_results.append(async_result)

        # Wait for all pending results to complete
        print("\nWaiting for all batches to complete...")
        for result in pending_results:
            try:
                counts, batch_len = result.get()
                total_processed += batch_len
                
                # Update counts
                for lang, count in counts.items():
                    lang_counts[lang] += count
                    total_filtered += count
            except Exception as e:
                print(f"Error processing batch: {e}")

        # Close and join the pool
        pool.close()
        pool.join()

        # Final summary
        end_time = time.time()
        total_elapsed = end_time - start_time
        avg_rate = total_processed / total_elapsed if total_elapsed > 0 else 0

        print("\n" + "="*60)
        print("Processing Complete!")
        print("="*60)
        print(f"Total rows processed: {total_processed:,}")
        print(f"Total rows filtered: {total_filtered:,}")
        print(f"Total time elapsed: {total_elapsed/60:.1f} minutes")
        print(f"Average speed: {avg_rate:.1f} rows/sec")
        print("\nBreakdown by language:")
        for lang, count in sorted(lang_counts.items()):
            filename = LANGUAGE_FILTER[lang]
            print(f"  {lang}: {count:,} rows -> {os.path.join(output_dir, filename)}")

    except KeyboardInterrupt:
        pool.terminate()
        pool.join()
        end_time = time.time()
        total_elapsed = end_time - start_time
        print("\n\nProcess interrupted by user.")
        print(f"Processed {total_processed:,} rows in {total_elapsed/60:.1f} minutes before interruption.")
        print(f"Filtered {total_filtered:,} rows total.")
        if lang_counts:
            print("\nData saved by language:")
            for lang, count in sorted(lang_counts.items()):
                filename = LANGUAGE_FILTER[lang]
                print(f"  {lang}: {count:,} rows -> {os.path.join(output_dir, filename)}")
    except Exception as e:
        pool.terminate()
        pool.join()
        print(f"\nError occurred: {e}")
        raise


if __name__ == '__main__':
    main()
