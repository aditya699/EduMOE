import json
import numpy as np
from azure.storage.blob import BlobServiceClient
from transformers import GPT2Tokenizer
from tqdm import tqdm
import time
from azure.core.exceptions import ServiceRequestError
from requests.exceptions import ConnectionError, ReadTimeout
import os
from dotenv import load_dotenv
load_dotenv()

# Config
BLOB_NAME = "c4_20gb.jsonl"
CONTAINER_NAME = "medical-documents" 
TRAIN_SPLIT = 0.9

# Azure connection - use the full connection string
conn_str = os.getenv("BLOB_STORAGE_ACCOUNT_KEY")


class GeneratorStream:
    """File-like object that wraps a generator for Azure upload"""
    def __init__(self, generator):
        self.generator = generator
        self.buffer = b''
        self.finished = False
    
    def read(self, size=-1):
        if self.finished and not self.buffer:
            return b''
        
        while len(self.buffer) < size and not self.finished:
            try:
                chunk = next(self.generator)
                self.buffer += chunk
            except StopIteration:
                self.finished = True
                break
        
        if size == -1:
            data = self.buffer
            self.buffer = b''
        else:
            data = self.buffer[:size]
            self.buffer = self.buffer[size:]
        
        return data


class EfficientTokenizer:
    def __init__(self, conn_str, container_name):
        self.conn_str = conn_str
        self.container_name = container_name
        
        print("Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_blob_stream_with_retry(self, blob_name, max_retries=3):
        """Get blob stream with retry logic"""
        for attempt in range(max_retries):
            try:
                blob_client = BlobServiceClient.from_connection_string(self.conn_str).get_blob_client(
                    container=self.container_name, blob=blob_name
                )
                return blob_client.download_blob()
            except (ServiceRequestError, ConnectionError, ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"Connection failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise e
    
    def data_generator(self, input_blob, train_split, is_train):
        """Single generator for both train and val with retry logic"""
        doc_count = 0
        token_count = 0
        split_name = "train" if is_train else "val"
        
        pbar = tqdm(desc=f"Processing {split_name}", unit="docs")
        
        buffer = ""
        incomplete_bytes = b''
        max_retries = 3
        
        while True:  # Retry loop for entire stream
            try:
                stream = self.get_blob_stream_with_retry(input_blob)
                
                for chunk in stream.chunks():
                    # Handle incomplete UTF-8 from previous chunk
                    if incomplete_bytes:
                        chunk = incomplete_bytes + chunk
                        incomplete_bytes = b''
                    
                    try:
                        chunk_text = chunk.decode('utf-8')
                    except UnicodeDecodeError as e:
                        valid_chunk = chunk[:e.start]
                        incomplete_bytes = chunk[e.start:]
                        chunk_text = valid_chunk.decode('utf-8')
                    
                    buffer += chunk_text
                    
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            try:
                                doc = json.loads(line)
                                text = doc.get("text", "").strip()
                                
                                if text:
                                    is_train_doc = doc_count % 10 < train_split * 10
                                    
                                    if is_train_doc == is_train:
                                        tokens = self.tokenizer.encode(text, add_special_tokens=True)
                                        token_array = np.array(tokens, dtype=np.int32)
                                        token_count += len(tokens)
                                        
                                        yield token_array.tobytes()
                                        
                                        if doc_count % 1000 == 0:
                                            pbar.set_postfix({'tokens': f'{token_count:,}'})
                                            pbar.update(1000)
                                    
                                    doc_count += 1
                                    
                            except json.JSONDecodeError:
                                continue
                
                # If we get here, stream completed successfully
                break
                
            except (ServiceRequestError, ConnectionError, ReadTimeout) as e:
                print(f"Stream interrupted at doc {doc_count}, retrying...")
                time.sleep(5)  # Wait before retry
                # Continue with the same doc_count to resume
        
        # Handle remaining incomplete bytes
        if incomplete_bytes:
            try:
                final_text = incomplete_bytes.decode('utf-8', errors='ignore')
                buffer += final_text
            except:
                pass
        
        pbar.close()
        print(f"{split_name.title()} tokens: {token_count:,}")
    
    def upload_with_retry(self, blob_path, data_stream, max_retries=3):
        """Upload with retry logic"""
        for attempt in range(max_retries):
            try:
                blob_client = BlobServiceClient.from_connection_string(self.conn_str).get_blob_client(
                    container=self.container_name, blob=blob_path
                )
                blob_client.upload_blob(data_stream, overwrite=True)
                return True
            except (ServiceRequestError, ConnectionError, ReadTimeout) as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Upload failed (attempt {attempt + 1}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"Upload failed after {max_retries} attempts")
                    raise e
    
    def process_and_upload(self, input_blob, train_split=0.9):
        """Process and upload train/val files efficiently"""
        
        # Upload train.bin
        print("Creating train.bin...")
        train_stream = GeneratorStream(self.data_generator(input_blob, train_split, is_train=True))
        self.upload_with_retry("processed/train.bin", train_stream)
        print("train.bin uploaded")
        
        # Upload val.bin
        print("Creating val.bin...")
        val_stream = GeneratorStream(self.data_generator(input_blob, train_split, is_train=False))
        self.upload_with_retry("processed/val.bin", val_stream)
        print("val.bin uploaded")
        
        self.upload_tokenizer()
        self.upload_metadata()
        
        print("All done with constant memory usage!")
    
    def upload_tokenizer(self):
        """Upload tokenizer files"""
        tokenizer_json = self.tokenizer.to_json()
        self.upload_with_retry("processed/tokenizer.json", tokenizer_json)
        print("Tokenizer uploaded")
    
    def upload_metadata(self):
        """Upload metadata"""
        meta = {
            "vocab_size": len(self.tokenizer),
            "tokenizer": "gpt2",
            "split_ratio": f"{int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}"
        }
        self.upload_with_retry("processed/meta.json", json.dumps(meta, indent=2))
        print("Metadata uploaded")


def main():
    processor = EfficientTokenizer(conn_str, CONTAINER_NAME)
    processor.process_and_upload(BLOB_NAME, TRAIN_SPLIT)


if __name__ == "__main__":
    main()