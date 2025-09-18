# Pretraining from Scratch

A clean, modular implementation of transformer language model pretraining built from scratch in PyTorch, designed for training on WikiText-103 dataset with modern optimization techniques.

## Overview

This implementation follows the transformer architecture for autoregressive language modeling, incorporating modern training optimizations and best practices for pretraining from scratch.

## Architecture

### Core Components

**Model Architecture (`model.py`)**
- **Token Embeddings (wte)**: Maps vocabulary IDs to dense vectors
- **Positional Embeddings (wpe)**: Encodes token order information
- **Transformer Blocks**: Stack of identical blocks with:
  - Layer Normalization → Causal Self-Attention → Residual Connection
  - Layer Normalization → MLP → Residual Connection
- **Final Layer Norm**: Stabilizes output before logits computation
- **Language Model Head**: Projects hidden states to vocabulary size (weight-tied with token embeddings)

**Attention Mechanism (`attention.py`)**
- Multi-head causal self-attention with triangular masking
- Uses PyTorch's `scaled_dot_product_attention` with Flash Attention support
- Automatic mixed precision compatibility
- Optimized for numerical stability and performance

**Feed-Forward Network (`mlp.py`)**
- Standard GPT-2 MLP: Linear → GELU → Linear → Dropout
- 4x expansion ratio (hidden_dim = 4 × embedding_dim)

### Model Configurations

**Default Configuration (GPT-2 Medium scale)**
- Vocabulary Size: 50,257 tokens
- Context Length: 1,024 tokens
- Embedding Dimension: 768
- Layers: 12
- Attention Heads: 12
- Parameters: ~117M (exact count via `model.get_num_params()`)

## Training Infrastructure

### Data Processing (`dataset.py`)

**WikiTextDataset Implementation**
- Sliding window chunking with configurable stride
- Efficient overlapping sequences to maximize data utilization
- EOS token separation between documents
- Autoregressive pair construction: `x = chunk[:-1]`, `y = chunk[1:]`

**Example Data Flow**
```
Raw tokens: [10, 20, 30, 40, 99]
With block_size=4:
- Input (x):  [10, 20, 30, 40]
- Target (y): [20, 30, 40, 99]
```

### Optimization Techniques

**Automatic Mixed Precision (AMP)**
- BFloat16 precision for forward/backward passes
- FP32 precision for gradient computation and parameter updates
- Gradient scaling with overflow detection
- Configurable dtype: `bfloat16` (default) or `float16`

**Gradient Management**
- Gradient accumulation for larger effective batch sizes
- Gradient clipping (max norm: 1.0)
- Inf/NaN gradient detection and recovery
- Separate weight decay for different parameter groups

**Learning Rate Scheduling**
- Linear warmup: 4,000 steps
- Cosine annealing decay
- Separate handling for bias, layer norm (no weight decay)

### Training Configuration

**Hyperparameters**
```python
batch_size: 4
gradient_accumulation_steps: 8  # Effective batch size: 32
learning_rate: 1.5e-4
warmup_steps: 4000
max_steps: 300,000
weight_decay: 0.01
max_grad_norm: 1.0
block_size: 1024
stride: 512
```

**Hardware Optimizations**
- TF32 enabled on Ampere+ GPUs
- CUDA kernel optimizations for attention
- Pin memory for DataLoader
- Gradient checkpointing ready

## Key Features

### Modern PyTorch Integration
- **Flash Attention**: Uses `torch.backends.cuda.sdp_kernel` for optimized attention
- **Causal Masking**: Built-in `is_causal=True` parameter
- **Memory Efficiency**: Optimized attention patterns and gradient handling

### Monitoring & Logging
- **Weights & Biases** integration for experiment tracking
- Real-time metrics: loss, perplexity, learning rate, tokens/second
- Gradient monitoring and visualization
- Generated text samples during training

### Checkpointing System
- **Light Checkpoints**: Model state only (every 2,000 steps)
- **Full Checkpoints**: Complete training state (every 10,000 steps)
- **Best Model Saving**: Automatic save on validation improvement
- **Cleanup**: Automatic removal of old checkpoints

### Text Generation
- Temperature-controlled sampling
- Configurable sequence length
- Multinomial and greedy decoding options

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Training
```python
from config import TrainingConfig
from trainer import TrainingManager

# Use default configuration or customize
config = TrainingConfig()
trainer = TrainingManager(config)
trainer.train()
```

### Quick Start
```bash
python train.py
```

### Generation
```python
# After training
sample = trainer.generate_sample(
    prompt="The future of artificial intelligence",
    max_tokens=50
)
```

## Implementation Details

### Attention Mechanism
- Causal masking prevents future token access
- Multi-head attention with head dimension = `n_embd // n_head`
- Dropout applied to attention weights and output projection
- Numerically stable computation via PyTorch's optimized kernels

### Loss Function
- Standard next-token prediction (autoregressive language modeling)
- Cross-entropy loss with label shifting
- Loss computed over all positions simultaneously

### Weight Initialization
- Uses PyTorch's default initialization
- Weight tying between embedding and output layers
- Separate parameter groups for weight decay

### Memory Management
- Gradient accumulation reduces memory pressure
- Mixed precision reduces activation memory
- Efficient DataLoader with optimal worker configuration

## Performance Characteristics

**Training Metrics**
- Tokens/second: ~2,000-4,000 (depending on hardware)
- Memory usage: ~8-12GB VRAM (RTX 5090)
- Convergence: Typical validation loss ~3.5-4.0 on WikiText-103

**Evaluation**
- Perplexity tracking on validation set
- Early stopping based on validation loss
- Sample generation quality assessment

## File Structure

```
├── __init__.py           # Package initialization
├── config.py             # Training configuration
├── model.py              # GPT-2 architecture
├── attention.py          # Multi-head attention
├── mlp.py                # Feed-forward network
├── blocks.py             # Transformer blocks
├── dataset.py            # WikiText dataset processing
├── trainer.py            # Training loop and management
├── train.py              # Main training script
└── requirements.txt      # Dependencies
```

## Technical Notes

### Numerical Stability
- BFloat16 for improved numerical range over FP16
- Gradient scaling with adaptive scaling factor
- Overflow detection and recovery mechanisms

### Reproducibility
- Configurable random seeds (implement as needed)
- Deterministic operations where possible
- Checkpoint-based resumability

### Scalability
- Modular design for easy configuration changes
- Support for different model sizes via config
- Ready for distributed training extensions

## References

1. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language Models are Unsupervised Multitask Learners.
2. Vaswani, A., et al. (2017). Attention Is All You Need.
3. PyTorch Flash Attention documentation
4. WikiText-103 dataset (Merity et al., 2016)

## Author

**Aditya Bhatt**  
Email: ab0358031@gmail.com

## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

*This implementation prioritizes code clarity, training stability, and modern PyTorch best practices while maintaining fidelity to the original GPT-2 architecture.*