import os
import random
import numpy as np
import torch
import time
import math

from config import CFG


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def as_minute(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(start_time, percent):
    now = time.time()
    s = now - start_time
    es = s / percent
    rs = es - s
    return '%s (remain %s)' % (as_minute(s), as_minute(rs))


def save_model(model, fold: int):
    if not os.path.exists(CFG.model_dir):
        os.makedirs(CFG.model_dir)
    file_name = os.path.join(CFG.model_dir, f'model_{fold}.pt')
    torch.save(model.state_dict(), file_name)


def load_model(fold: int):
    return torch.load(os.path.join(CFG.model_dir, f'model_{fold}.pt'))
