import numpy as np
import torch
from scipy.stats import norm

def exp_cdf(x, l):
    return 1 - torch.exp(-l*x)


def exp_percentile(p, l):
    return -torch.log(1-p)/l


def exp_normalize(intervals, l=0.0111460429649723213, mean=0.0, std=0.0):
    new_intervals = exp_cdf(intervals, 0.0111460429649723213)
    # standard_intervals = (new_intervals - mean)/std
    
    return new_intervals
    # return standard_intervals

def exp_unnormalize(intervals, l=0.0111460429649723213, mean=0.0, std=0.0):
    # inverse_intervals = intervals*std + mean
    
    # inverse_intervals = exp_percentile(inverse_intervals, 0.0111460429649723213)
    inverse_intervals = exp_percentile(intervals, 0.0111460429649723213)
    #将inf数据设为0
    inverse_intervals = torch.where(torch.isinf(inverse_intervals), torch.zeros_like(inverse_intervals), inverse_intervals)
    return inverse_intervals