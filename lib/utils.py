import torch
import torch.nn as nn
from copy import deepcopy


def _is_prunable_module(m):
    return (isinstance(m,nn.Linear) or isinstance(m,nn.Conv2d))

def get_weights(model):
    weights = []
    for m in model.modules():
        if _is_prunable_module(m):
            weights.append(m.weight)
    return weights

def get_modules(model):
    modules = []
    for m in model.modules():
        if _is_prunable_module(m):
            modules.append(m)
    return modules
