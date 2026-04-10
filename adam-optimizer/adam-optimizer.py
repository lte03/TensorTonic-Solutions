import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    grad_arr = np.array(grad)
    param_arr = np.array(param)
    m_arr = np.array(m)
    v_arr = np.array(v)
    mt = beta1 * m_arr + (1-beta1)*grad_arr
    vt = beta2 * v_arr + (1-beta2)*(grad_arr ** 2)
    m_t = mt/(1-beta1**t)
    v_t = vt/(1-beta2**t)
    param_new = param_arr - lr * (m_t / (np.sqrt(v_t) + eps))
    return param_new,mt,vt