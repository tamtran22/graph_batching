import torch

def mean_squared_error(input, target):
    return torch.mean((input - target) ** 2)

def weighted_mean_squared_error(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)