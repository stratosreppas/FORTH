import numpy as np
from scipy.stats import norm

def indicator_function(a: bool):
    return 1 if a else 0

def normalize(a):
# normalize by subtracting the minimum and dividing by the range
    min_val = np.min(a)
    max_val = np.max(a)
    return [(x - min_val) / (max_val - min_val) for x in a]

def gaussian(length, mean, std_dev):
    x = np.arange(length)
    gaussian_values = norm.pdf(x, mean, std_dev)
    return gaussian_values.tolist()

