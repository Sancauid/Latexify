import random
import numpy as np
import torch
import os
import sys
import json
import gc


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(config):
    """monotonous bookkeeping"""
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        f.write(json.dumps(config.to_dict(), indent=4))


def free_mem(**args):
    torch.cuda.empty_cache()
    for arg in args:
        del arg
    gc.collect()
