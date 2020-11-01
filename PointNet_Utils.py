import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset
class ModelNet40Dataset(Dataset):
    def __init__(self, datapath):
        super.__init__(ModelNet40Dataset,self)
        self.dataPath = datapath
    def __len__(self):
        return 1
    def __getitem__(self, item):
        return 0




