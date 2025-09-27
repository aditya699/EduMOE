import json
import os

from azure.storage.blob import BlobServiceClient
from datasets import load_dataset
from tqdm import tqdm

from dotenv import load_dotenv
load_dotenv()
# Config
# Note: C4 is a good choice for GPT-2 replication. Alternatives: OpenWebText, The Pile, RedPajama
# Our 117M parameter model needs ~2.3B tokens (Chinchilla optimal), 20GB provides ~5B tokens
DATASET_NAME = "allenai/c4"
DATASET_CONFIG = "en"
SPLIT = "train"
TARGET_SIZE_GB = 20
BLOB_NAME = "c4_20gb.jsonl"

# Quality filter settings
# Note: In production, add extensive filtering for PII, violence, sexual content, toxicity
MIN_TEXT_LENGTH = 10
MAX_TEXT_LENGTH = 100000

# Azure connection
conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = os.getenv("CONTAINER_NAME")


class DatasetProcessor:
    def __init__(self, conn_str, container_name, target_size_gb, blob_name):
        self.conn_str = conn_str
        self.container_name = container_name
        self.target_size_gb = target_size_gb
        self.blob_name = blob_name
        self.limit_bytes = target_size_gb * (1024**3)
        
    def is_good_text(self, text):
        """Filter for text quality"""
        if not text or not text.strip():
            return False
        if len(text) < MIN_TEXT_LENGTH:
            return False
        if len(text) > MAX_TEXT_LENGTH:
            return False
        return True
    
    def data_generator(self, dataset):
        """Generator that yields filtered data as bytes"""
        self.bytes_written = 0
        self.docs_processed = 0
        self.docs_accepted = 0
        
        pbar = tqdm(
            desc="Processing", 
            unit="docs",
            postfix={'GB': '0.00', 'accepted': '0%'}
        )
        
        try:
            for sample in dataset:
                self.docs_processed += 1
                text = sample.get("text", "").strip()
                
                if not self.is_good_text(text):
                    pbar.update(1)
                    continue
                    
                self.docs_accepted += 1
                line = json.dumps(sample, ensure_ascii=False) + '\n'
                line_bytes = line.encode('utf-8')
                self.bytes_written += len(line_bytes)
                
                if self.docs_accepted % 100 == 0:
                    accept_rate = (self.docs_accepted / self.docs_processed) * 100
                    pbar.set_postfix({
                        'GB': f'{self.bytes_written/1024**3:.2f}',
                        'accepted': f'{accept_rate:.1f}%'
                    })
                
                pbar.update(1)
                
                yield line_bytes
                
                if self.bytes_written >= self.limit_bytes:
                    print(f"\nReached {self.target_size_gb}GB with {self.docs_accepted:,} documents")
                    break
                    
        finally:
            pbar.close()
    
    def process_and_upload(self, dataset_name, dataset_config, split):
        """Process dataset and stream directly to Azure Blob"""
        try:
            print(f"Loading data from {dataset_name} until {self.target_size_gb}GB...")
            dataset = load_dataset(dataset_name, dataset_config, split=split, streaming=True)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return False
            
        try:
            print("Streaming JSONL directly to Azure Blob...")
            blob_client = BlobServiceClient.from_connection_string(self.conn_str).get_blob_client(
                container=self.container_name, 
                blob=self.blob_name
            )
        except Exception as e:
            print(f"Error connecting to Azure: {e}")
            return False
        
        try:
            # Single upload using generator - streams data without loading into memory
            print("Starting upload to Azure...")
            blob_client.upload_blob(self.data_generator(dataset), overwrite=True)
            print("Upload completed!")
            print(f"Successfully uploaded to {self.blob_name}")
            print(f"Final stats: {self.docs_accepted:,} documents accepted from {self.docs_processed:,} processed")
            print(f"Total size: {self.bytes_written/1024**3:.2f}GB")
            return True
            
        except Exception as e:
            print(f"Error during upload: {e}")
            return False


# Usage
processor = DatasetProcessor(conn_str, container_name, TARGET_SIZE_GB, BLOB_NAME)
success = processor.process_and_upload(DATASET_NAME, DATASET_CONFIG, SPLIT)
if not success:
    print("Processing failed!")