import os
import random

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and PyTorch for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
