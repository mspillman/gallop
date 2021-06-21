# coding: utf-8
# Copyright (c) Mark Spillman.
# Distributed under the terms of the GPL v3 License.
"""
Provides functions for local optimisation and a class for particle swarm.
"""

import os
import random
import torch
import numpy as np

from . import local
from .swarm import Swarm





def seed_everything(seed=1234, change_backend=False):
    """
    Set random seeds for everything.
    Note that at the moment, CUDA (which is used by PyTorch) is not
    deterministic for some operations and as a result, GALLOP runs from
    the same seed may still produce different result.
    See here for more details:
        https://pytorch.org/docs/stable/notes/randomness.html

    Args:
        seed (int, optional): Set the random seed to be used.
            Defaults to 1234.
        change_backend (bool, optional): Whether to change the backend used to
            try to make the code more reproducible. At the moment, it doesn't
            seem to help... default to False
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if change_backend:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False