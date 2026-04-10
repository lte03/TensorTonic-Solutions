import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    if np.sum(p) != 1:
        raise ValueError()
    x_ = np.array(x)
    p_ = np.array(p)
    return np.sum(x_ * p_)
