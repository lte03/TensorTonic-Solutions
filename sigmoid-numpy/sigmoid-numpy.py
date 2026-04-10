import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    x_ = np.asarray(x)
    return 1 / (1 + np.exp(-x_))