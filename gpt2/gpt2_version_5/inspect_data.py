"""
Simple script to inspect train.bin/val.bin files and diagnose issues
"""

import os
import numpy as np

def inspect_bin_file(filepath):
    """Inspect a binary tokenized file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File does not exist!")
        return
    
    # Check file size
    file_size = os.path.getsize(filepath)
    print(f"File size: {file_size:,} bytes ({file_size / (1024**2):.2f} MB)")
    
    if file_size == 0:
        print("ERROR: File is empty!")
        return
    
    # Check if file size is divisible by 4 (int32 = 4 bytes)
    if file_size % 4 != 0:
        print(f"WARNING: File size ({file_size}) is not divisible by 4!")
        print(f"Expected int32 tokens (4 bytes each), but have {file_size % 4} extra bytes")
        print("This will cause corruption!")
    
    # Read first 100 tokens
    print("\nReading first 100 tokens...")
    try:
        with open(filepath, 'rb') as f:
            # Read first 400 bytes (100 tokens * 4 bytes each)
            data = f.read(400)
            tokens = np.frombuffer(data, dtype=np.int32)
            
            print(f"Successfully read {len(tokens)} tokens")
            print(f"First 20 tokens: {tokens[:20].tolist()}")
            print(f"Token range: min={tokens.min()}, max={tokens.max()}")
            
            # Check for invalid token IDs (GPT-2 vocab size is 50257)
            if tokens.max() >= 50257:
                print(f"ERROR: Found invalid token ID {tokens.max()} (GPT-2 vocab size is 50257)")
            
            if tokens.min() < 0:
                print(f"ERROR: Found negative token ID {tokens.min()}")
            
    except Exception as e:
        print(f"ERROR reading file: {e}")
        return
    
    # Read last 100 tokens
    print("\nReading last 100 tokens...")
    try:
        with open(filepath, 'rb') as f:
            f.seek(-400, 2)  # Seek to 400 bytes from end
            data = f.read(400)
            tokens = np.frombuffer(data, dtype=np.int32)
            
            print(f"Last 20 tokens: {tokens[-20:].tolist()}")
            
    except Exception as e:
        print(f"ERROR reading end of file: {e}")
    
    # Calculate total number of tokens
    total_tokens = file_size // 4
    print(f"\nTotal tokens in file: {total_tokens:,}")
    
    # Sample tokens from middle
    print("\nSampling 20 tokens from middle of file...")
    try:
        with open(filepath, 'rb') as f:
            mid_point = (file_size // 2) // 4 * 4  # Align to 4-byte boundary
            f.seek(mid_point)
            data = f.read(80)  # 20 tokens
            tokens = np.frombuffer(data, dtype=np.int32)
            print(f"Middle tokens: {tokens.tolist()}")
            
    except Exception as e:
        print(f"ERROR reading middle: {e}")
    
    print(f"\n{'='*60}\n")


def main():
    # Check both files
    inspect_bin_file("processed/train.bin")
    inspect_bin_file("processed/val.bin")
    
    # Additional checks
    print("\nAdditional Diagnostics:")
    print("-" * 60)
    
    # Check if files exist
    for filename in ["processed/train.bin", "processed/val.bin", "processed/meta.json"]:
        exists = "EXISTS" if os.path.exists(filename) else "MISSING"
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"{filename}: {exists} ({size:,} bytes)")
        else:
            print(f"{filename}: {exists}")


if __name__ == "__main__":
    main()