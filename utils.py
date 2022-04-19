import numpy as np
import torch
import os


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    for reprocibility.

    Args:
        seed (int, optional): seed. Defaults to 42.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
