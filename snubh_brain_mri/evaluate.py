from __future__ import print_function

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.fastest = True
from torch.optim.lr_scheduler import CosineAnnealingLR

import os
import random
import numpy as np
from shutil import rmtree

from options import parse_option
from network import create_model
from utils import get_optimizer, get_loss_function, lr_update
from utils.core import evaluate
from datasets import get_dataloader
import warnings
warnings.filterwarnings('ignore')


# Seed
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

#NOTE: main loop for training
if __name__ == "__main__":
    # Option
    opt = parse_option(print_option=True)

    # Data Loader
    _, dataset_val = get_dataloader(opt)

    # Network
    net = create_model(opt)

    # Train
    evaluate(dataset_val, net, opt)
    rmtree(opt.exp)