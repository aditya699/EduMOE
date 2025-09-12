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


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Model config
    vocab_size: int = 50257  #vocab for tokenizer
    max_position_embeddings: int = 512
    n_embd: int = 256
    n_layer: int = 6
    n_head: int = 8
    dropout: float = 0.1
    
    # Training config
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_epochs: int = 10
    warmup_steps: int = 1000
    max_steps: Optional[int] = None
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Data config
    block_size: int = 256  # sequence length for training
    
    # Logging
    eval_interval: int = 500
    log_interval: int = 100
    save_interval: int = 1000


class WikiTextDataset(Dataset):
    """Dataset for WikiText-2 with proper tokenization and chunking."""
    
    def __init__(self, texts: List[str], tokenizer, block_size: int = 256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        
        # Tokenize all texts and concatenate
        print("Tokenizing dataset...")
        all_tokens = []
        
        for text in texts:
            if text.strip():  # Skip empty lines
                tokens = tokenizer.encode(text.strip())
                all_tokens.extend(tokens)
                all_tokens.append(tokenizer.eos_token_id)  # Add EOS between articles
        
        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Dataset has {len(self.tokens)} tokens")
        
        # Calculate number of samples
        self.num_samples = max(0, len(self.tokens) - block_size)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get a chunk of tokens
        chunk = self.tokens[idx:idx + self.block_size + 1]
        
        # Input and target (shifted by 1)
        x = chunk[:-1]
        y = chunk[1:]
        
        return x, y


class TrainingManager:
    """Manages the training process."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load and prepare data
        self.train_loader, self.val_loader = self._prepare_data()
        
        # Initialize model
        model_config = self._create_model_config()
        self.model = GPT2Model(model_config).to(self.device)
        print(f"Model has {self.model.get_num_params():,} parameters")
        
        # Initialize optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
    
    def _create_model_config(self):
        """Create model configuration from training config."""
        from types import SimpleNamespace
        return SimpleNamespace(
            vocab_size=self.config.vocab_size,
            max_position_embeddings=self.config.max_position_embeddings,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_inner=None,
            dropout=self.config.dropout,
            layer_norm_epsilon=1e-5,
            use_bias=True
        )
    
    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and prepare WikiText-2 dataset."""
        print("Loading WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
        # Create datasets
        train_texts = [item['text'] for item in dataset['train'] if item['text'].strip()]
        val_texts = [item['text'] for item in dataset['validation'] if item['text'].strip()]
        
        train_dataset = WikiTextDataset(train_texts, self.tokenizer, self.config.block_size)
        val_dataset = WikiTextDataset(val_texts, self.tokenizer, self.config.block_size)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Train dataset: {len(train_dataset)} samples")
        print(f"Val dataset: {len(val_dataset)} samples")
        
        return train_loader, val_loader
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create AdamW optimizer with weight decay."""
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
        """Create learning rate scheduler with warmup."""
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / self.config.warmup_steps
            return 1.0
        
        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
    
    def train_step(self, batch) -> float:
        """Single training step."""
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        
        # Forward pass
        outputs = self.model(x, labels=y)
        loss = outputs['loss']
        
        # Scale loss for gradient accumulation
        loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def evaluate(self) -> float:
        """Evaluate model on validation set."""
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
        
        return total_loss / num_batches
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        torch.save(checkpoint, filepath)
    
    def generate_sample(self, prompt: str = "The", max_tokens: int = 50) -> str:
        """Generate a sample text during training."""
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
        """Main training loop."""
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        
        start_time = time.time()
        accumulated_loss = 0
        
        for epoch in range(self.config.max_epochs):
            self.epoch = epoch
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Training step
                loss = self.train_step(batch)
                accumulated_loss += loss
                
                # Update parameters every gradient_accumulation_steps
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.step += 1
                    
                    # Logging
                    if self.step % self.config.log_interval == 0:
                        avg_loss = accumulated_loss / self.config.log_interval
                        self.train_losses.append(avg_loss)
                        
                        lr = self.scheduler.get_last_lr()[0]
                        elapsed = time.time() - start_time
                        
                        print(f"Step {self.step:5d} | "
                              f"Epoch {epoch:2d} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {lr:.2e} | "
                              f"Time: {elapsed:.1f}s")
                        
                        accumulated_loss = 0
                    
                    # Evaluation
                    if self.step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        self.val_losses.append(val_loss)
                        
                        print(f"Validation loss: {val_loss:.4f}")
                        
                        # Save best model
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_checkpoint(f"best_model_step_{self.step}.pt")
                            print(f"New best model saved! Val loss: {val_loss:.4f}")
                        
                        # Generate sample
                        sample = self.generate_sample("The future of artificial intelligence")
                        print(f"Sample: {sample[:100]}...")
                        print("-" * 80)
                    
                    # Save checkpoint
                    if self.step % self.config.save_interval == 0:
                        self.save_checkpoint(f"checkpoint_step_{self.step}.pt")
                    
                    # Early stopping
                    if self.config.max_steps and self.step >= self.config.max_steps:
                        print(f"Reached max steps ({self.config.max_steps})")
                        return
        
        print("Training completed!")
        self.save_checkpoint("final_model.pt")
    
    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(12, 5))
        
        # Training loss
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses)
        plt.title('Training Loss')
        plt.xlabel('Steps (x100)')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # Validation loss
        plt.subplot(1, 2, 2)
        plt.plot(self.val_losses)
        plt.title('Validation Loss')
        plt.xlabel('Evaluations')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# Simplified GPT-2 model (use your full implementation)
class GPT2Model(nn.Module):
    """Simplified GPT-2 for training demo."""
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


# ================================
# Main Training Script
# ================================

def main():
    """Main training function."""
    # Create training configuration
    config = TrainingConfig(
        # Model config (small for quick training)
        n_embd=256,
        n_layer=6,
        n_head=8,
        
        # Training config
        batch_size=4,  # Small batch for demo
        learning_rate=1e-4,
        max_epochs=3,
        warmup_steps=500,
        max_steps=20000,  # Limit for demo
        gradient_accumulation_steps=2,
        
        # Data config
        block_size=128,  # Smaller sequences for faster training
        
        # Logging
        eval_interval=200,
        log_interval=50,
        save_interval=500
    )
    
    # Create trainer and start training
    trainer = TrainingManager(config)
    trainer.train()
    
    # Plot results
    trainer.plot_losses()
    
    # Final evaluation
    print("\nFinal Evaluation:")
    final_val_loss = trainer.evaluate()
    print(f"Final validation loss: {final_val_loss:.4f}")
    
    # Generate some samples
    print("\nGenerated Samples:")
    for prompt in ["The", "In the future", "Artificial intelligence"]:
        sample = trainer.generate_sample(prompt, max_tokens=30)
        print(f"Prompt: '{prompt}' -> {sample}")


if __name__ == "__main__":
    main()