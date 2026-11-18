import numpy as np


def si_sdr(est, ref, eps=1e-9):
    """Scale-invariant SDR in dB (Le Roux et al., 2019)."""
    est = np.asarray(est, np.float64) - np.mean(est)
    ref = np.asarray(ref, np.float64) - np.mean(ref)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    t = alpha * ref
    return float(10 * np.log10((np.dot(t, t) + eps) / (np.dot(est - t, est - t) + eps)))
