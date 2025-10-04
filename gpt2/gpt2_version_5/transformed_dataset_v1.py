"""
Local tokenizer - downloads JSONL from Azure blob, tokenizes locally, saves train.bin/val.bin

CHANGES FROM v0 (Original Azure streaming tokenizer):

Performance Optimizations:
1. Fast Tokenizer: Switched from GPT2Tokenizer to GPT2TokenizerFast
   - 5-10x faster tokenization using Rust backend instead of pure Python
   
2. Batch Processing: Tokenizes 500 documents at once instead of one-by-one
   - Reduces Python ↔ Rust boundary crossings from millions to thousands
   - Enables internal parallelization in Rust code
   - Expected speedup: 5-10x (total ~1-1.5 hours vs 8-10 hours for 20GB)

3. Single-Pass Processing: Reads input file once instead of twice
   - v0 streamed from Azure twice (once for train, once for val)
   - v1 downloads once, processes once, writes to both files simultaneously
   - 50% reduction in I/O and processing time

Architecture Changes:
4. Download-Then-Process: Downloads complete file locally before tokenizing
   - v0: Streamed from Azure during tokenization (network-dependent)
   - v1: Download first (resumable), then process locally (network-independent)
   - Eliminates network interruptions during tokenization

5. Resume-Safe Downloads: Partial downloads are preserved and resumed
   - Checks existing file size and resumes from last byte
   - Uses Azure range requests (offset + length parameters)
   - Prevents restarting 20GB downloads from scratch

6. Simplified Architecture: Removed streaming complexity
   - No GeneratorStream wrapper class needed
   - No retry logic during tokenization (only during download)
   - Cleaner, more maintainable code

Trade-offs:
- Requires local disk space (~20GB for input + output files)
- Download step adds upfront time (but only once, and resumable)
- Better for unstable networks, worse if disk space is limited
"""

import json
import numpy as np
from transformers import GPT2TokenizerFast as GPT2Tokenizer
from tqdm import tqdm
import os
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

load_dotenv()

# Config
BLOB_NAME = "c4_20gb.jsonl"
CONTAINER_NAME = "medical-documents"
INPUT_FILE = "c4_20gb.jsonl"  # Local file after download
OUTPUT_DIR = "processed"
TRAIN_SPLIT = 0.9

# Azure connection
conn_str = os.getenv("BLOB_STORAGE_ACCOUNT_KEY")


class LocalTokenizer:
    def __init__(self, output_dir, conn_str=None, container_name=None):
        self.output_dir = output_dir
        self.conn_str = conn_str
        self.container_name = container_name
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        print("Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def download_from_blob(self, blob_name, local_path, max_bytes=None):
        """Download file from Azure Blob Storage with resume capability"""
        
        if not self.conn_str or not self.container_name:
            print("WARNING: No Azure credentials provided, expecting local file")
            return os.path.exists(local_path)
        
        try:
            blob_service = BlobServiceClient.from_connection_string(self.conn_str)
            blob_client = blob_service.get_blob_client(
                container=self.container_name, 
                blob=blob_name
            )
            
            # Get blob properties
            properties = blob_client.get_blob_properties()
            total_size = properties.size
            download_size = min(max_bytes, total_size) if max_bytes else total_size
            
            # Check if file exists and get current size
            start_byte = 0
            if os.path.exists(local_path):
                current_size = os.path.getsize(local_path)
                if current_size >= download_size:
                    print(f"COMPLETE: {local_path} already fully downloaded ({current_size / (1024**2):.1f} MB)")
                    return True
                elif current_size > 0:
                    start_byte = current_size
                    print(f"RESUMING: Download from {start_byte / (1024**2):.1f} MB")
            
            print(f"DOWNLOADING: {blob_name} from Azure...")
            print(f"Total blob size: {total_size / (1024**3):.2f} GB")
            if max_bytes:
                print(f"Downloading first {download_size / (1024**2):.1f} MB only")
            
            # Open file in append mode if resuming, write mode if new
            mode = 'ab' if start_byte > 0 else 'wb'
            
            with open(local_path, mode) as f:
                # Calculate how much we still need to download
                remaining = download_size - start_byte
                
                if max_bytes:
                    # For partial downloads, just get the remaining chunk
                    stream = blob_client.download_blob(offset=start_byte, length=remaining)
                    f.write(stream.readall())
                else:
                    # For full downloads, stream with progress bar
                    stream = blob_client.download_blob(offset=start_byte, length=remaining)
                    
                    with tqdm(total=download_size, initial=start_byte, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in stream.chunks():
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            final_size = os.path.getsize(local_path)
            print(f"SUCCESS: Download complete - {final_size / (1024**2):.1f} MB saved to {local_path}")
            return True
            
        except Exception as e:
            current_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            print(f"ERROR: Download failed - {e}")
            print(f"SAVED: Partial download - {current_size / (1024**2):.1f} MB")
            print(f"TIP: Run again to resume from where it stopped")
            return False
    
    def process_and_save(self, input_file, train_split=0.9, batch_size=500):
        """Process input file with batched tokenization for massive speedup"""
        
        if not os.path.exists(input_file):
            print(f"ERROR: {input_file} not found!")
            return False
        
        print(f"Reading from: {input_file}")
        print(f"Output directory: {self.output_dir}")
        print(f"Train/Val split: {int(train_split*100)}/{int((1-train_split)*100)}")
        print(f"Batch size: {batch_size}")
        
        train_path = os.path.join(self.output_dir, "train.bin")
        val_path = os.path.join(self.output_dir, "val.bin")
        
        token_train, token_val, doc_count = 0, 0, 0
        
        # Batch buffers
        batch_texts = []
        batch_splits = []
        
        with open(input_file, 'r', encoding='utf-8') as in_f, \
             open(train_path, 'wb') as train_f, \
             open(val_path, 'wb') as val_f:
            
            for line in tqdm(in_f, desc="Processing", unit="docs"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    doc = json.loads(line)
                    text = doc.get("text", "").strip()
                except json.JSONDecodeError:
                    continue
                
                if not text:
                    continue
                
                # Determine train/val split
                is_train_doc = doc_count % 10 < train_split * 10
                batch_texts.append(text)
                batch_splits.append(is_train_doc)
                doc_count += 1
                
                # When batch fills, tokenize all at once
                if len(batch_texts) >= batch_size:
                    encodings = self.tokenizer(batch_texts, add_special_tokens=True)["input_ids"]
                    
                    for ids, is_train in zip(encodings, batch_splits):
                        arr = np.array(ids, dtype=np.int32)
                        if is_train:
                            train_f.write(arr.tobytes())
                            token_train += len(ids)
                        else:
                            val_f.write(arr.tobytes())
                            token_val += len(ids)
                    
                    # Clear buffers
                    batch_texts.clear()
                    batch_splits.clear()
            
            # Process any remaining texts in final batch
            if batch_texts:
                encodings = self.tokenizer(batch_texts, add_special_tokens=True)["input_ids"]
                
                for ids, is_train in zip(encodings, batch_splits):
                    arr = np.array(ids, dtype=np.int32)
                    if is_train:
                        train_f.write(arr.tobytes())
                        token_train += len(ids)
                    else:
                        val_f.write(arr.tobytes())
                        token_val += len(ids)
        
        self.save_tokenizer()
        self.save_metadata(token_train, token_val)
        
        print(f"\nSUCCESS: All done! Train tokens: {token_train:,}, Val tokens: {token_val:,}")
        print(f"Files saved in: {self.output_dir}/")
        return True
    
    def save_tokenizer(self):
        """Save tokenizer configuration"""
        self.tokenizer.save_pretrained(self.output_dir)
        print(f"Tokenizer saved to {self.output_dir}/")
    
    def save_metadata(self, train_tokens, val_tokens):
        """Save metadata about the tokenization"""
        meta = {
            "vocab_size": len(self.tokenizer),
            "tokenizer": "gpt2",
            "split_ratio": f"{int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}",
            "train_tokens": train_tokens,
            "val_tokens": val_tokens,
            "total_tokens": train_tokens + val_tokens
        }
        
        meta_path = os.path.join(self.output_dir, "meta.json")
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"Metadata saved to {meta_path}")


def main():
    processor = LocalTokenizer(OUTPUT_DIR, conn_str, CONTAINER_NAME)
    
    # Download from blob if needed (set max_bytes=200*1024*1024 for testing with 200MB)
    if processor.download_from_blob(BLOB_NAME, INPUT_FILE):
        # Process locally
        success = processor.process_and_save(INPUT_FILE, TRAIN_SPLIT)
        
        if success:
            print("\nCOMPLETE: Processing finished!")
            print(f"You can now manually upload the '{OUTPUT_DIR}' folder to Azure")
        else:
            print("\nFAILED: Processing did not complete")
    else:
        print("\nFAILED: Could not get input file")


if __name__ == "__main__":
    main()