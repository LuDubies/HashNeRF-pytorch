from os import path
import json
import numpy as np
import torch
from torch.utils.data import Dataset

from neaf_operations import get_random_directions, get_random_receivers_for_listener

def use_cpu_tensor(func):
    def wrapper(*args, **kwargs):
        torch.set_default_tensor_type(torch.FloatTensor)
        to_ret = func(*args, **kwargs)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        return to_ret
    return wrapper

class NeafDataset(Dataset):

    def __init__(self, basedir, rayfile, batchsize, clargs):
        super().__init__()
        with open(path.join(basedir, rayfile), 'r') as rf:
            loaded_json = json.load(rf)
        self.data = loaded_json
        self.states = loaded_json['states']
        self.listener_count = len(self.states)

        self.batchsize = batchsize
        self.clargs = clargs

    def __len__(self):
        return self.listener_count

    @use_cpu_tensor
    def __getitem__(self, idx):
        np.random.seed(idx)  # TODO make optional? flag?
        recs_d = get_random_directions(self.batchsize)
        recs, targets, times = get_random_receivers_for_listener(self.states[idx], self.batchsize, recs_d, self.clargs)
        return recs, targets, times




