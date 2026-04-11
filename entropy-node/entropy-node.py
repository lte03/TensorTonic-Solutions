import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    y_vector = np.array(y)
    unique,freqs = np.unique(y_vector,return_counts=True)
    p = freqs / y_vector.shape[0]
    return -np.sum((p*np.log2(p)))