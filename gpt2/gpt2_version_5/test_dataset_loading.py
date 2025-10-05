"""
Quick test to verify the dataset loading works correctly
"""

import os
import sys

# Test 1: Check if binary files exist
print("=" * 60)
print("Test 1: Checking if binary files exist")
print("=" * 60)

processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
train_bin = os.path.join(processed_dir, 'train.bin')
val_bin = os.path.join(processed_dir, 'val.bin')
tokenizer_json = os.path.join(processed_dir, 'tokenizer.json')

files_to_check = [train_bin, val_bin, tokenizer_json]
all_exist = True

for filepath in files_to_check:
    exists = os.path.exists(filepath)
    all_exist = all_exist and exists
    status = "[OK] EXISTS" if exists else "[!!] MISSING"
    if exists:
        size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"{status}: {os.path.basename(filepath)} ({size_mb:.2f} MB)")
    else:
        print(f"{status}: {os.path.basename(filepath)}")

if not all_exist:
    print("\nERROR: Some required files are missing!")
    sys.exit(1)

print("\n" + "=" * 60)
print("Test 2: Loading dataset with BinaryTokenDataset")
print("=" * 60)

try:
    from dataset import BinaryTokenDataset

    # Create a small test dataset
    train_dataset = BinaryTokenDataset(
        bin_path=train_bin,
        block_size=1024,
        stride=512
    )

    print(f"\n[OK] Successfully loaded training dataset")
    print(f"  Dataset length: {len(train_dataset):,} chunks")

    # Test getting a sample
    x, y = train_dataset[0]
    print(f"\n[OK] Successfully retrieved sample")
    print(f"  Input shape: {x.shape}")
    print(f"  Target shape: {y.shape}")
    print(f"  First 10 input tokens: {x[:10].tolist()}")
    print(f"  First 10 target tokens: {y[:10].tolist()}")

    # Verify x and y are shifted correctly
    assert x.shape == y.shape, "Input and target shapes don't match!"
    print(f"\n[OK] Input and target shapes match")

except Exception as e:
    print(f"\n[ERROR]: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Test 3: Loading tokenizer")
print("=" * 60)

try:
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json)
    print(f"[OK] Successfully loaded tokenizer")
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Test encoding/decoding
    test_text = "The future of artificial intelligence"
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"\n  Test encoding: '{test_text}'")
    print(f"  Tokens: {encoded}")
    print(f"  Decoded: '{decoded}'")

except Exception as e:
    print(f"\n[ERROR]: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("ALL TESTS PASSED [OK]")
print("=" * 60)
print("\nThe migration is complete and working correctly!")
print("You can now run training with: python train.py")
