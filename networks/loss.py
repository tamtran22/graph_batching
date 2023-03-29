import torch

def mean_squared_error(input, target):
    return torch.mean(torch.square(input - target))

def weighted_mean_squared_error(input, target, weight):
    return torch.mean(weight * torch.square(input - target))