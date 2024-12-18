# scripts/find_downloads.py

import os
from pathlib import Path

def check_locations():
    # Common locations to check
    locations = [
        "~/.cache/huggingface",
        "~/Library/Caches/huggingface",
        "./downloads",
        "./data",
        "~/.local/share/huggingface"
    ]
    
    print("Searching for downloaded parquet files...")
    
    for loc in locations:
        path = os.path.expanduser(loc)
        if os.path.exists(path):
            print(f"\nChecking {path}")
            # Walk through all subdirectories
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('.parquet'):
                        full_path = os.path.join(root, file)
                        size_mb = os.path.getsize(full_path) / (1024 * 1024)
                        print(f"Found: {full_path}")
                        print(f"Size: {size_mb:.2f} MB")

if __name__ == "__main__":
    check_locations()