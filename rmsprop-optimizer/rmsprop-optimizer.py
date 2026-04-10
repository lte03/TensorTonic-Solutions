import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    s_ = np.array(s)
    w_ = np.array(w)
    g_ = np.array(g)
    st = beta*s_ + (1-beta) *g_*g_
    wt = w_ - (lr/np.sqrt(st+eps))*g_
    return wt,st