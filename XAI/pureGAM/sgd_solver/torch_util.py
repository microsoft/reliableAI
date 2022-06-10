import torch as th
import torch.nn as nn
import torch.nn.functional as F

def init_model(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model