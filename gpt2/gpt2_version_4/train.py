"""
Author : Aditya Bhatt 12:16PM
Please reach out to ab0358031@gmail.com for any issues and suggestions.
NOTE:
1.We are keeping tokenizer parallelization as false since we will run it in RUNPOD machines , sometimes we might see an issue.
(For local machines, we can keep it true)

2.The following code will run on RTX 5090 GPU.

3.We are using AMP 

AMP = Automatic Mixed Precision, a feature in PyTorch (torch.amp or torch.cuda.amp) that lets your model train using both:

FP16 (half precision, 16-bit floats) for most operations.

FP32 (full precision, 32-bit floats) for critical operations where accuracy matters.

4.We are going to stick to max_setps not epoochs



TODO:
1.Check for Dead imports

"""

# ==========================
# Standard Library Imports
# ==========================
import math
import time
import os
import glob
import itertools
from typing import Dict, List, Optional, Tuple

# ==========================
# Third-Party Libraries
# ==========================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np                
import wandb
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import TrainingConfig from separate config module (supports both module and script execution)
try:
    from .config import TrainingConfig  # type: ignore
except Exception:  # noqa: E722 - fallback when running as a script
    from config import TrainingConfig  # type: ignore

## Import dataset class from separate module
try:
    from .dataset import WikiTextDataset  # type: ignore
except Exception:  # noqa: E722 - fallback when running as a script
    from dataset import WikiTextDataset  # type: ignore
