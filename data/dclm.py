#!/usr/bin/env python3
"""
Stream and filter HuggingFaceTB/dclm-edu dataset
Searches for geometry/3D printing related keywords in text and URL
Outputs filtered data to dclm.json with resume capability
Stops at 3B words total
"""

import json
import os
import time
import re
from collections import defaultdict
from datasets import load_dataset


# Keywords to search for (case-insensitive)
KEYWORDS = [
    "geometry", "3d printing", "object", "shape", "dimension",
    "angle", "volume", "openscad", "solid", "model",
    "vector", "mesh", "cad"
]

# Output file
OUTPUT_FILE = 'dclm.json'
CHECKPOINT_FILE = 'dclm_checkpoint.txt'

# Target word count (3 billion)
TARGET_WORDS = 300_000_000_000


def count_words(text):
    """Count words in text"""
    if not text:
        return 0
    return len(text.split())


def find_matching_keywords(text, url):
    """Find matching keywords in text and URL (case-insensitive)"""
    if not text and not url:
        return []

    # Combine text and url for searching
    combined = f"{text or ''} {url or ''}".lower()

    # Find all matching keywords
    matched = []
    for keyword in KEYWORDS:
        if keyword in combined:
            matched.append(keyword)

    return matched


def load_checkpoint():
    """Load the last processed row index"""
    if os.path.exists(CHECKPOINT_FILE):
        try:
            with open(CHECKPOINT_FILE, 'r') as f:
                data = json.load(f)
                return data.get('last_index', 0), data.get('total_words', 0)
        except:
            return 0, 0
    return 0, 0


def save_checkpoint(last_index, total_words):
    """Save the current progress"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({
            'last_index': last_index,
            'total_words': total_words
        }, f)


def append_to_output(item):
    """Append a single item to the output file"""
    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
        f.flush()  # Ensure immediate write to disk


def main():
    print("="*70)
    print("DCLM-Edu Geometry/3D Printing Dataset Filter")
    print("="*70)
    print(f"Target: {TARGET_WORDS:,} words")
    print(f"Keywords: {', '.join(KEYWORDS)}")
    print()

    # Load checkpoint
    start_index, total_words = load_checkpoint()

    if start_index > 0:
        print(f"Resuming from row {start_index:,}")
        print(f"Already collected {total_words:,} words")
    else:
        print("Starting fresh")
        # Clear output file if starting fresh
        if os.path.exists(OUTPUT_FILE):
            os.remove(OUTPUT_FILE)
            print(f"Cleared existing {OUTPUT_FILE}")

    print(f"\nData will be saved incrementally to: {OUTPUT_FILE}")
    print("Press Ctrl+C to stop at any time (progress will be saved)\n")

    # Check if already reached target
    if total_words >= TARGET_WORDS:
        print(f"\nTarget already reached! Total words: {total_words:,}")
        return

    try:
        # Stream the dataset
        print("Loading HuggingFaceTB/dclm-edu dataset...")
        dataset = load_dataset(
            'HuggingFaceTB/dclm-edu',
            split='train',
            streaming=True
        )

        # Skip already processed rows
        if start_index > 0:
            print(f"Skipping {start_index:,} already processed rows...")
            dataset = dataset.skip(start_index)

        # Processing variables
        current_index = start_index
        matched_count = 0
        total_processed = 0

        # Time tracking
        start_time = time.time()
        last_checkpoint_time = start_time
        checkpoint_interval = 300  # Save checkpoint every 5 minutes

        print("\nProcessing dataset...\n")

        for item in dataset:
            # Get text and url
            text = item.get('text', '')
            url = item.get('url', '')

            # Find matching keywords
            matched_keywords = find_matching_keywords(text, url)

            if matched_keywords:
                # Count words in this text
                word_count = count_words(text)

                # Save matched item
                output_item = {
                    'text': text,
                    'url': url,
                    'key_words': matched_keywords
                }
                append_to_output(output_item)

                matched_count += 1
                total_words += word_count

                # Check if we've reached the target
                if total_words >= TARGET_WORDS:
                    print(f"\n{'='*70}")
                    print(f"TARGET REACHED! 🎉")
                    print(f"{'='*70}")
                    print(f"Total words collected: {total_words:,}")
                    print(f"Total matched documents: {matched_count:,}")
                    print(f"Total rows processed: {total_processed + 1:,}")
                    save_checkpoint(current_index + 1, total_words)
                    break

            current_index += 1
            total_processed += 1

            # Print progress every 10,000 rows
            if total_processed % 10000 == 0:
                current_time = time.time()
                elapsed = current_time - start_time
                rows_per_sec = total_processed / elapsed if elapsed > 0 else 0

                # Calculate ETA
                if total_words > 0 and rows_per_sec > 0:
                    # Estimate words per row
                    words_per_matched = total_words / matched_count if matched_count > 0 else 0
                    words_remaining = TARGET_WORDS - total_words

                    if words_per_matched > 0:
                        # Very rough ETA estimate
                        eta_seconds = words_remaining / (rows_per_sec * words_per_matched * (matched_count / total_processed))
                        eta_str = f"ETA: ~{eta_seconds/3600:.1f}h" if eta_seconds < 86400 else f"ETA: ~{eta_seconds/86400:.1f}d"
                    else:
                        eta_str = "ETA: calculating..."
                else:
                    eta_str = "ETA: calculating..."

                # Progress percentage
                progress_pct = (total_words / TARGET_WORDS * 100) if TARGET_WORDS > 0 else 0

                print(f"[{time.strftime('%H:%M:%S')}] Rows: {current_index:,} | Matched: {matched_count:,} | Words: {total_words:,} ({progress_pct:.2f}%)")
                print(f"  Speed: {rows_per_sec:.1f} rows/sec | {eta_str} | Elapsed: {elapsed/60:.1f} min")

            # Periodic checkpoint save
            current_time = time.time()
            if current_time - last_checkpoint_time > checkpoint_interval:
                save_checkpoint(current_index + 1, total_words)
                last_checkpoint_time = current_time

        # Final summary
        end_time = time.time()
        total_elapsed = end_time - start_time

        print(f"\n{'='*70}")
        print("Processing Complete!")
        print(f"{'='*70}")
        print(f"Total rows processed: {total_processed:,}")
        print(f"Total matched documents: {matched_count:,}")
        print(f"Total words collected: {total_words:,} / {TARGET_WORDS:,}")
        print(f"Match rate: {(matched_count/total_processed*100) if total_processed > 0 else 0:.3f}%")
        print(f"Total time: {total_elapsed/60:.1f} minutes")
        print(f"Output saved to: {OUTPUT_FILE}")

        # Final checkpoint
        save_checkpoint(current_index + 1, total_words)

    except KeyboardInterrupt:
        end_time = time.time()
        total_elapsed = end_time - start_time

        print("\n\n" + "="*70)
        print("Process interrupted by user")
        print("="*70)
        print(f"Rows processed this session: {total_processed:,}")
        print(f"Total matched documents: {matched_count:,}")
        print(f"Total words collected: {total_words:,} / {TARGET_WORDS:,}")
        print(f"Progress: {(total_words/TARGET_WORDS*100):.2f}%")
        print(f"Time elapsed: {total_elapsed/60:.1f} minutes")
        print(f"\nProgress saved! Run again to resume from row {current_index + 1:,}")

        # Save checkpoint
        save_checkpoint(current_index + 1, total_words)

    except Exception as e:
        print(f"\nError occurred: {e}")
        # Save checkpoint on error
        save_checkpoint(current_index + 1, total_words)
        raise


if __name__ == '__main__':
    main()
