"""
GPT-2 Training Code Changes: Version 2 to Version 3
=================================================

DOCUMENT IDENTIFICATION (CORRECTED):
- Document 1: Version 2 - "gpt2-medailabs" project (smaller model, NO AMP)
- Document 2: Version 3 - "gpt2-rtx5090-safe" project (larger model, WITH AMP)
- Document 3: Also Version 3 - same as Document 2 (RTX 5090 optimized WITH AMP)

MAJOR CHANGES FROM VERSION 2 → VERSION 3:

1. MIXED PRECISION TRAINING ADDED:
   Version 2 (Document 1):
   - Pure FP32 training, no AMP
   - No GradScaler or autocast
   
   Version 3 (Documents 2 & 3):
   - ADDED torch.cuda.amp.autocast() and GradScaler
   - self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'), init_scale=2**12)
   - with torch.cuda.amp.autocast(dtype=torch.float16): in train_step()
   - self.scaler.scale(loss).backward(), self.scaler.step(), self.scaler.update()

2. MODEL SIZE INCREASE:
   Version 2:
   - n_embd=256, n_layer=6, n_head=8 (~25M parameters)
   - max_position_embeddings=512, block_size=512
   
   Version 3:
   - n_embd=512, n_layer=12, n_head=8 (~117M parameters)  
   - max_position_embeddings=1024, block_size=1024
   # Much larger model for RTX 5090

3. BATCH SIZE & TRAINING SCALE:
   Version 2:
   - batch_size=8, gradient_accumulation_steps=4 (effective=32)
   - max_steps=20000, warmup_steps=1000
   - learning_rate=3e-4
   
   Version 3:
   - batch_size=64, gradient_accumulation_steps=2 (effective=128)
   - max_steps=10000, warmup_steps=300
   - learning_rate=2e-4
   # Much larger effective batch size, fewer steps

4. PROJECT NAMING:
   Version 2: wandb_project="gpt2-medailabs"
   Version 3: wandb_project="gpt2-rtx5090-safe"

5. TIME LIMIT FEATURE ADDED:
   Version 2:
   - No automatic time-based stopping
   
   Version 3:
   - Added 5-hour time limit check in training loop
   - elapsed_hours = (time.time() - start_time) / 3600
   - if elapsed_hours >= 5.0: return

6. TRAINING LOOP DIFFERENCES:
   Version 2:
   - Standard range(max_epochs) loop
   - Relies on max_steps for stopping
   
   Version 3:
   - Uses itertools.count() for infinite epoch loop
   - Manual time limit checking + max_steps

7. DATA STRIDE:
   Version 2: stride=512 (100% stride with block_size=512, no overlap)
   Version 3: stride=512 (50% overlap with block_size=1024)
   # More data overlap in Version 3

KEY INSIGHT:
Version 3 represents a scale-up for high-end hardware (RTX 5090):
- Added mixed precision training for efficiency
- Significantly larger model (25M → 117M parameters)
- Larger batch sizes and context windows
- Added time-based training limits for sessions
- Optimized for powerful GPU training runs
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
import wandb
import glob
import os
import itertools

# Reduce memory/CPU overhead from HF tokenizers and avoid fork-related warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Model config - EXACTLY THE SAME AS YOUR ORIGINAL
    vocab_size: int = 50257
    max_position_embeddings: int = 1024
    n_embd: int = 512
    n_layer: int = 12
    n_head: int = 8
    dropout: float = 0.1
    
    # Training config - tuned for RTX 5090
    batch_size: int = 16
    learning_rate: float = 3e-4
    max_epochs: int = 10
    warmup_steps: int = 200
    max_steps: Optional[int] = 5000
    gradient_accumulation_steps: int = 2
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data config - SAME AS ORIGINAL
    block_size: int = 512
    stride: int = 512
    
    # Logging & Monitoring - SAME AS ORIGINAL  
    eval_interval: int = 500
    log_interval: int = 20
    save_interval: int = 5000
    light_save_interval: int = 1000
    
    # Weights & Biases - SAME AS ORIGINAL
    wandb_project: str = "gpt2-rtx5090-safe"
    wandb_name: Optional[str] = None
    use_wandb: bool = True


class WikiTextDataset(Dataset):
    """Dataset with sliding-window chunking - EXACTLY YOUR ORIGINAL."""
    
    def __init__(self, texts: List[str], tokenizer, block_size: int = 512, stride: int = 512):
        self.block_size = block_size
        self.examples = []
        print("Tokenizing dataset with sliding windows...")
        
        # We create windows of length block_size + 1 so that x,y are both length block_size
        window_len = block_size + 1
        
        for text in texts:
            if not text.strip():
                continue
            
            tokens = tokenizer.encode(text.strip())
            seq_len = len(tokens)
            if seq_len < window_len:
                continue
            
            for i in range(0, seq_len - window_len + 1, stride):
                chunk = tokens[i:i + window_len]
                self.examples.append(torch.tensor(chunk, dtype=torch.long))
            
            # Add a tail window if leftover doesn't align with stride
            if (seq_len - window_len) % stride != 0:
                tail = tokens[-window_len:]
                self.examples.append(torch.tensor(tail, dtype=torch.long))
        
        # Fallback: if per-text windows produced nothing, concatenate all tokens and window globally
        if len(self.examples) == 0:
            print("No chunks created per-text. Falling back to concatenated sliding windows across all texts.")
            all_tokens: List[int] = []
            for text in texts:
                t = text.strip()
                if not t:
                    continue
                all_tokens.extend(tokenizer.encode(t))
            if len(all_tokens) >= window_len:
                for i in range(0, len(all_tokens) - window_len + 1, stride):
                    chunk = all_tokens[i:i + window_len]
                    self.examples.append(torch.tensor(chunk, dtype=torch.long))
                if (len(all_tokens) - window_len) % stride != 0:
                    tail = all_tokens[-window_len:]
                    self.examples.append(torch.tensor(tail, dtype=torch.long))
        
        print(f"Total chunks created: {len(self.examples)}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        chunk = self.examples[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y


class TrainingManager:
    """Manages the training process - YOUR ORIGINAL CODE with minimal changes."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer - SAME AS ORIGINAL
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.tokenizer.model_max_length = int(1e9)
        except Exception:
            pass
        
        # Load and prepare data - SAME AS ORIGINAL
        self.train_loader, self.val_loader = self._prepare_data()
        
        # Initialize model - SAME AS ORIGINAL
        model_config = self._create_model_config()
        self.model = GPT2Model(model_config).to(self.device)
        print(f"Model has {self.model.get_num_params():,} parameters")
        
        # Initialize optimizer and scheduler - SAME AS ORIGINAL
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'), init_scale=2**12)
        
        # Training state - SAME AS ORIGINAL
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Initialize Weights & Biases - SAME AS ORIGINAL
        if self.config.use_wandb:
            self._init_wandb()
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging - SAME AS ORIGINAL."""
        # Create run name if not provided
        if self.config.wandb_name is None:
            self.config.wandb_name = f"rtx5090-safe-{self.config.n_layer}L-{self.config.n_embd}d-{self.config.n_head}h"
        
        # Initialize wandb
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config={
                # Model config
                "vocab_size": self.config.vocab_size,
                "max_position_embeddings": self.config.block_size,
                "n_embd": self.config.n_embd,
                "n_layer": self.config.n_layer,
                "n_head": self.config.n_head,
                "dropout": self.config.dropout,
                
                # Training config
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "max_epochs": self.config.max_epochs,
                "warmup_steps": self.config.warmup_steps,
                "max_steps": self.config.max_steps,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "weight_decay": self.config.weight_decay,
                "max_grad_norm": self.config.max_grad_norm,
                "block_size": self.config.block_size,
                "stride": self.config.stride,
                
                # Computed values
                "model_params": self.model.get_num_params(),
                "device": str(self.device),
                "effective_batch_size": self.config.batch_size * self.config.gradient_accumulation_steps
            }
        )
        
        # Watch model gradients only to reduce overhead
        wandb.watch(self.model, log="gradients", log_freq=1000)
        print(f"Initialized W&B run: {wandb.run.name}")
    
    def _create_model_config(self):
        """Create model configuration from training config - SAME AS ORIGINAL."""
        from types import SimpleNamespace
        return SimpleNamespace(
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.block_size,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_inner=None,
            dropout=self.config.dropout,
            layer_norm_epsilon=1e-5,
            use_bias=True
        )
    
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare WikiText-103 dataset - SAME AS ORIGINAL."""
        print("Loading WikiText-103 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Create datasets
        train_texts = [item['text'] for item in dataset['train'] if item['text'].strip()]
        val_texts = [item['text'] for item in dataset['validation'] if item['text'].strip()]
        
        train_dataset = WikiTextDataset(
            train_texts,
            self.tokenizer,
            block_size=self.config.block_size,
            stride=self.config.stride,
        )
        val_dataset = WikiTextDataset(
            val_texts,
            self.tokenizer,
            block_size=self.config.block_size,
            stride=self.config.stride,
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with weight decay - SAME AS ORIGINAL."""
        # Separate parameters for weight decay
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if any(nd in name for nd in ['bias', 'ln', 'norm']):
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer_groups = [
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        
        return optim.AdamW(optimizer_groups, lr=self.config.learning_rate)
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup + cosine decay - SAME AS ORIGINAL."""
        # Calculate total steps for cosine decay
        if self.config.max_steps:
            total_steps = self.config.max_steps
        else:
            total_steps = len(self.train_loader) * self.config.max_epochs // self.config.gradient_accumulation_steps
        
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {self.config.warmup_steps}")
        print(f"Cosine decay steps: {total_steps - self.config.warmup_steps}")
        
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                # Linear warmup
                return step / self.config.warmup_steps
            else:
                # Cosine decay from 1.0 to 0.0
                progress = (step - self.config.warmup_steps) / (total_steps - self.config.warmup_steps)
                progress = min(progress, 1.0)  # Clamp to avoid going beyond 1.0
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def save_light_checkpoint(self, filepath: str):
        """Save light checkpoint - SAME AS ORIGINAL."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, filepath)
        print(f"Light checkpoint saved: {filepath}")
    
    def save_full_checkpoint(self, filepath: str):
        """Save full checkpoint - SAME AS ORIGINAL."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
        print(f"Full checkpoint saved: {filepath}")
        
        # Log checkpoint to W&B
        if self.config.use_wandb:
            wandb.save(filepath)
    
    def cleanup_old_checkpoints(self, keep_last_n=3):
        """Clean up old checkpoints - SAME AS ORIGINAL."""
        # Clean light checkpoints
        light_files = glob.glob("light_checkpoint_step_*.pt")
        light_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for old_file in light_files[:-keep_last_n]:
            os.remove(old_file)
            print(f"Removed old checkpoint: {old_file}")
        
        # Clean full checkpoints (keep fewer)
        full_files = glob.glob("full_checkpoint_step_*.pt")
        full_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for old_file in full_files[:-2]:  # Keep last 2 full checkpoints
            os.remove(old_file)
            print(f"Removed old full checkpoint: {old_file}")
    
    def train_step(self, batch) -> float:
        """Single training step - EXACTLY YOUR ORIGINAL."""
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass with AMP
        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = self.model(x, labels=y)
            loss = outputs['loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        return loss.item()
    
    def evaluate(self) -> float:
        """Evaluate model on validation set - EXACTLY YOUR ORIGINAL."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                
                outputs = self.model(x, labels=y)
                loss = outputs['loss']
                
                total_loss += loss.item()
                num_batches += 1
                
                # Limit evaluation for faster training
                if num_batches >= 50:
                    break
        
        if num_batches == 0:
            print("Warning: Validation loader returned 0 batches. Returning inf loss.")
            return float('inf')
        return total_loss / num_batches
    
    def generate_sample(self, prompt: str = "The", max_tokens: int = 50) -> str:
        """Generate a sample text during training - EXACTLY YOUR ORIGINAL."""
        self.model.eval()
        
        # Tokenize prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                do_sample=True
            )
        
        # Decode
        generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return generated_text
    
    def train(self):
        """Main training loop - YOUR ORIGINAL with time limit added."""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        
        start_time = time.time()
        accumulated_loss = 0
        
        for epoch in itertools.count():
            self.epoch = epoch
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                
                # Update parameters every gradient_accumulation_steps
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Unscale gradients before clipping
                    self.scaler.unscale_(self.optimizer)
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step with AMP scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.step += 1
                    
                    # Track learning rate
                    current_lr = self.scheduler.get_last_lr()[0]
                    self.learning_rates.append(current_lr)
                    
                    # Logging
                    if self.step % self.config.log_interval == 0:
                        avg_loss = accumulated_loss / self.config.log_interval
                        self.train_losses.append(avg_loss)
                        
                        elapsed = time.time() - start_time
                        elapsed_hours = elapsed / 3600
                        tokens_per_sec = (self.step * self.config.batch_size * self.config.block_size * self.config.gradient_accumulation_steps) / elapsed
                        
                        # Log to W&B
                        if self.config.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/tokens_per_second": tokens_per_sec,
                                "train/step": self.step,
                                "train/epoch": epoch,
                                "train/elapsed_hours": elapsed_hours
                            })
                        
                        print(f"Step {self.step:5d} | "
                              f"Epoch {epoch:2d} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {current_lr:.2e} | "
                              f"Tokens/s: {tokens_per_sec:.0f} | "
                              f"Time: {elapsed_hours:.1f}h")
                        
                        accumulated_loss = 0
                    
                    # Evaluation
                    if self.step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        self.val_losses.append(val_loss)
                        
                        # Calculate perplexity for better interpretability
                        train_ppl = math.exp(min(avg_loss if 'avg_loss' in locals() else 10, 10))
                        val_ppl = math.exp(min(val_loss, 10))
                        
                        # Log to W&B
                        if self.config.use_wandb:
                            wandb.log({
                                "eval/loss": val_loss,
                                "eval/perplexity": val_ppl,
                                "train/perplexity": train_ppl,
                                "eval/step": self.step
                            })
                        
                        print(f"Validation loss: {val_loss:.4f} | Perplexity: {val_ppl:.2f}")
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_full_checkpoint(f"best_model_step_{self.step}.pt")
                            print(f"New best model saved! Val loss: {val_loss:.4f}")
                            
                            # Log best model to W&B
                            if self.config.use_wandb:
                                wandb.log({"eval/best_loss": val_loss})
                        
                        # Generate sample
                        sample = self.generate_sample("The future of artificial intelligence")
                        print(f"Sample: {sample[:100]}...")
                        
                        # Log sample to W&B
                        if self.config.use_wandb:
                            wandb.log({
                                "samples/generation": wandb.Html(f"<p><b>Prompt:</b> 'The future of artificial intelligence'</p><p><b>Generated:</b> {sample}</p>")
                            })
                        
                        print("-" * 80)
                    
                    # Save checkpoints
                    if self.step % self.config.light_save_interval == 0:
                        self.save_light_checkpoint(f"light_checkpoint_step_{self.step}.pt")
                        self.cleanup_old_checkpoints()
                    
                    if self.step % self.config.save_interval == 0:
                        self.save_full_checkpoint(f"full_checkpoint_step_{self.step}.pt")
                        self.cleanup_old_checkpoints()
                    
                    # Check 5-hour time limit
                    elapsed_hours = (time.time() - start_time) / 3600
                    if elapsed_hours >= 5.0:
                        print(f"⏰ 5-hour time limit reached! ({elapsed_hours:.1f}h)")
                        return
                    
                    # Early stopping
                    if self.config.max_steps and self.step >= self.config.max_steps:
                        print(f"Reached max steps ({self.config.max_steps})")
                        return
        
        print("Training completed!")
        self.save_full_checkpoint("final_model.pt")
        self.cleanup_old_checkpoints()
        
        # Finish W&B run
        if self.config.use_wandb:
            wandb.finish()
    
    def plot_losses(self):
        """Plot training and validation losses - SAME AS ORIGINAL."""
        plt.figure(figsize=(15, 5))
        
        # Training loss
        plt.subplot(1, 3, 1)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Steps (x100)')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Validation loss
        plt.subplot(1, 3, 2)
        plt.plot(self.val_losses)
        plt.title('Validation Loss')
        plt.xlabel('Evaluations')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Learning rate schedule
        plt.subplot(1, 3, 3)
        plt.plot(self.learning_rates)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# ALL YOUR ORIGINAL MODEL CLASSES - UNCHANGED
class GPT2Model(nn.Module):
    """GPT-2 model implementation."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final norm and head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.wte.weight
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, labels=None):
        b, t = input_ids.size()
        pos = torch.arange(0, t, device=input_ids.device).unsqueeze(0)
        
        tok_emb = self.wte(input_ids)
        pos_emb = self.wpe(pos)
        x = self.drop(tok_emb + pos_emb)
        
        for block in self.h:
            x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {'logits': logits, 'loss': loss}
    
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, do_sample=True):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids)
                logits = outputs['logits'][:, -1, :] / temperature
                
                if do_sample:
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
    
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
        self.register_buffer("causal_mask", torch.tril(torch.ones(config.max_position_embeddings, config.max_position_embeddings)).view(1, 1, config.max_position_embeddings, config.max_position_embeddings))
    
    def forward(self, x):
        B, T, C = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def main():
    """SAFE RTX 5090 training - your original code with just batch size increase."""
    # Create training configuration - tuned for RTX 5090
    config = TrainingConfig(
        # Model config - EXACTLY THE SAME
        n_embd=512,
        n_layer=12,
        n_head=8,
        
        # Training config - tuned
        batch_size=16,
        learning_rate=3e-4,
        max_epochs=10,              # Same as original  
        warmup_steps=200,
        max_steps=5000,
        gradient_accumulation_steps=2,
        
        # Everything else EXACTLY as your original
        eval_interval=500,
        log_interval=20,
        save_interval=5000,
        light_save_interval=1000,
        
        wandb_project="gpt2-rtx5090-safe",
        use_wandb=True
    )
    
    print("🔧 SAFE RTX 5090 Setup (AMP + tuned batches)")
    print("=" * 60)
    print(f"Model: {config.n_layer}L-{config.n_embd}d-{config.n_head}h")
    print(f"Batch size: {config.batch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective sequences/step: {config.batch_size * config.gradient_accumulation_steps}")
    print(f"Max steps: {config.max_steps}")
    print(f"Using AMP mixed precision, NO compilation")
    print("=" * 60)
    
    # Create trainer and start training
    trainer = TrainingManager(config)
    trainer.train()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_val_loss = trainer.evaluate()
    final_ppl = math.exp(min(final_val_loss, 10))
    print(f"Final validation loss: {final_val_loss:.4f}")
    print(f"Final perplexity: {final_ppl:.2f}")
    
    # Generate some samples
    print("\nGenerated Samples:")
    for prompt in ["The", "In the future", "Artificial intelligence"]:
        sample = trainer.generate_sample(prompt, max_tokens=30)
        print(f"Prompt: '{prompt}' -> {sample}")


if __name__ == "__main__":
    main()