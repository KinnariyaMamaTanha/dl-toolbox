import torch
import numpy as np
import random, os


def set_seed(seed, use_cuda=True):
    """
    set random seed everywhere.
    """
    torch.set_default_tensor_type("torch.FloatTensor")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
