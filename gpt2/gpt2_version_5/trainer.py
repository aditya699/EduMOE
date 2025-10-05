import math
import time
import os
import glob
import itertools
from typing import List, Tuple

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb
from transformers import PreTrainedTokenizerFast

try:
    from .config import TrainingConfig  # type: ignore
    from .dataset import BinaryTokenDataset  # type: ignore
    from .model import GPT2Model  # type: ignore
except Exception:  # noqa: E722 - fallback when running as a script
    from config import TrainingConfig  # type: ignore
    from dataset import BinaryTokenDataset  # type: ignore
    from model import GPT2Model  # type: ignore


class TrainingManager:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.amp_dtype = torch.bfloat16 if str(self.config.amp_dtype).lower() == 'bfloat16' else torch.float16

        # Load custom tokenizer from processed folder
        tokenizer_path = os.path.join(os.path.dirname(__file__), 'processed', 'tokenizer.json')
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.tokenizer.model_max_length = int(1e9)
        except Exception:
            pass

        self.train_loader, self.val_loader = self._prepare_data()

        model_config = self._create_model_config()
        self.model = GPT2Model(model_config).to(self.device)
        setattr(self.model, 'amp_dtype', self.amp_dtype)

        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        # Grad scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.device.type == 'cuda'), init_scale=2**12)

        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.learning_rates: List[float] = []

        if self.config.use_wandb:
            self._init_wandb()

    def _init_wandb(self):
        if self.config.wandb_name is None:
            self.config.wandb_name = f"v4-{self.config.n_layer}L-{self.config.n_embd}d-{self.config.n_head}h"
        wandb.init(
            project=self.config.wandb_project,
            name=self.config.wandb_name,
            config={
                "vocab_size": self.config.vocab_size,
                "max_position_embeddings": self.config.max_position_embeddings,
                "n_embd": self.config.n_embd,
                "n_layer": self.config.n_layer,
                "n_head": self.config.n_head,
                "dropout": self.config.dropout,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "warmup_steps": self.config.warmup_steps,
                "max_steps": self.config.max_steps,
                "max_epochs": self.config.max_epochs,
                "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                "weight_decay": self.config.weight_decay,
                "max_grad_norm": self.config.max_grad_norm,
                "block_size": self.config.block_size,
                "stride": self.config.stride,
                "device": str(self.device),
                "amp_dtype": str(self.amp_dtype),
            }
        )
        wandb.watch(self.model, log="gradients", log_freq=1000)

    def _safe_exp(self, x: float, max_x: float = 20.0) -> float:
        return float(math.exp(x)) if x < max_x else float('inf')

    def _create_model_config(self):
        from types import SimpleNamespace
        max_pos = max(self.config.max_position_embeddings, self.config.block_size)
        return SimpleNamespace(
            vocab_size=self.config.vocab_size,
            max_position_embeddings=max_pos,
            n_embd=self.config.n_embd,
            n_layer=self.config.n_layer,
            n_head=self.config.n_head,
            n_inner=None,
            dropout=self.config.dropout,
            layer_norm_epsilon=1e-5,
            use_bias=True,
        )

    def _prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        # Load pre-tokenized binary files from processed folder
        processed_dir = os.path.join(os.path.dirname(__file__), 'processed')
        train_bin_path = os.path.join(processed_dir, 'train.bin')
        val_bin_path = os.path.join(processed_dir, 'val.bin')

        # Create datasets from binary files
        train_dataset = BinaryTokenDataset(
            bin_path=train_bin_path,
            block_size=self.config.block_size,
            stride=self.config.stride,
        )
        val_dataset = BinaryTokenDataset(
            bin_path=val_bin_path,
            block_size=self.config.block_size,
            stride=self.config.stride,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _create_optimizer(self) -> optim.Optimizer:
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(nd in name for nd in ['bias', 'ln', 'norm']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return optim.AdamW(groups, lr=self.config.learning_rate)

    def _create_scheduler(self):
        if self.config.max_steps:
            total_steps = self.config.max_steps
        else:
            total_steps = len(self.train_loader) * (self.config.max_epochs or 1) // max(1, self.config.gradient_accumulation_steps)

        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            progress = (step - self.config.warmup_steps) / max(1, (total_steps - self.config.warmup_steps))
            progress = min(progress, 1.0)
            return 0.5 * (1 + math.cos(math.pi * progress))

        return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def save_light_checkpoint(self, filepath: str):
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        torch.save(checkpoint, filepath)

    def save_full_checkpoint(self, filepath: str):
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
            'config': self.config,
        }
        torch.save(checkpoint, filepath)

    def cleanup_old_checkpoints(self, keep_last_n=3):
        light_files = glob.glob("light_checkpoint_step_*.pt")
        light_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for old_file in light_files[:-keep_last_n]:
            try:
                os.remove(old_file)
            except OSError:
                pass

        full_files = glob.glob("full_checkpoint_step_*.pt")
        full_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for old_file in full_files[:-2]:
            try:
                os.remove(old_file)
            except OSError:
                pass

    def train_step(self, batch) -> float:
        self.model.train()
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda'), dtype=self.amp_dtype):
            outputs = self.model(x, labels=x)
            loss = outputs['loss']
            loss = loss / max(1, self.config.gradient_accumulation_steps)
        self.scaler.scale(loss).backward()
        return float(loss.item())

    def evaluate(self) -> float:
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        with torch.no_grad():
            for batch in self.val_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                with torch.cuda.amp.autocast(enabled=(self.device.type == 'cuda'), dtype=self.amp_dtype):
                    outputs = self.model(x, labels=x)
                    loss = outputs['loss']
                total_loss += float(loss.item())
                num_batches += 1
                if num_batches >= 50:
                    break
        if num_batches == 0:
            return float('inf')
        return total_loss / num_batches

    def generate_sample(self, prompt: str = "The", max_tokens: int = 50) -> str:
        self.model.eval()
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=0.8,
                do_sample=True,
            )
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def train(self):
        print("Starting training...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
        print(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")

        start_time = time.time()
        accumulated_loss = 0.0

        for epoch in itertools.count():
            self.epoch = epoch
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.train_step(batch)
                accumulated_loss += loss

                if (batch_idx + 1) % max(1, self.config.gradient_accumulation_steps) == 0:
                    # Unscale, clip, and guard against non-finite gradients
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    total_norm = 0.0
                    found_inf = False
                    with torch.no_grad():
                        for p in self.model.parameters():
                            if p.grad is None:
                                continue
                            param_norm = p.grad.data.float().norm(2)
                            if torch.isnan(param_norm) or torch.isinf(param_norm):
                                found_inf = True
                                break
                            total_norm += param_norm.item() ** 2
                    if not found_inf:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # Skip optimizer step and reset scaler on inf/nan
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.update(new_scale=max(self.scaler.get_scale() / 2, 1.0))
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.step += 1

                    current_lr = self.scheduler.get_last_lr()[0]
                    self.learning_rates.append(current_lr)

                    if self.step % self.config.log_interval == 0:
                        avg_loss = accumulated_loss / self.config.log_interval
                        self.train_losses.append(avg_loss)
                        elapsed = time.time() - start_time
                        tokens_per_sec = (
                            self.step
                            * self.config.batch_size
                            * self.config.block_size
                            * max(1, self.config.gradient_accumulation_steps)
                        ) / max(1.0, elapsed)
                        if self.config.use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": current_lr,
                                "train/tokens_per_second": tokens_per_sec,
                                "train/step": self.step,
                                "train/epoch": epoch,
                                "train/elapsed_hours": elapsed / 3600.0,
                            })
                        print(
                            f"Step {self.step:5d} | Epoch {epoch:2d} | Loss: {avg_loss:.4f} | "
                            f"LR: {current_lr:.2e} | Tokens/s: {tokens_per_sec:.0f} | Time: {elapsed/3600.0:.1f}h"
                        )
                        accumulated_loss = 0.0

                    if self.step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        self.val_losses.append(val_loss)
                        train_ppl = self._safe_exp(self.train_losses[-1]) if self.train_losses else None
                        val_ppl = self._safe_exp(val_loss)
                        print(f"[EVAL] step={self.step} | val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f} | train_ppl={(train_ppl or float('nan')):.2f}")
                        if self.config.use_wandb:
                            wandb.log({
                                "eval/loss": val_loss,
                                "eval/perplexity": val_ppl,
                                "train/perplexity": train_ppl,
                            }, step=self.step)

                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.save_full_checkpoint(f"best_model_step_{self.step}.pt")
                            if self.config.use_wandb:
                                wandb.log({"eval/best_loss": val_loss}, step=self.step)

                        sample = self.generate_sample("The future of artificial intelligence", max_tokens=50)
                        print(f"Sample: {sample[:100]}...")
                        if self.config.use_wandb:
                            wandb.log({
                                "samples/generation": wandb.Html(
                                    f"<p><b>Prompt:</b> 'The future of artificial intelligence'</p><p><b>Generated:</b> {sample}</p>"
                                )
                            }, step=self.step)
                        print("-" * 80)

                    if self.step % self.config.light_save_interval == 0:
                        self.save_light_checkpoint(f"light_checkpoint_step_{self.step}.pt")
                        self.cleanup_old_checkpoints()

                    if self.step % self.config.save_interval == 0:
                        self.save_full_checkpoint(f"full_checkpoint_step_{self.step}.pt")
                        self.cleanup_old_checkpoints()

                    if self.config.max_steps and self.step >= self.config.max_steps:
                        print(f"Reached max steps ({self.config.max_steps})")
                        self._finish_wandb()
                        return

    def _finish_wandb(self):
        if self.config.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass


