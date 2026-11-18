"""Mixing-console style audio effects used to build realistic stage and mix scenes.

Everything is plain NumPy/SciPy so the same code path is used for data generation
and for unit tests. Filters follow the RBJ "Audio EQ Cookbook" formulas.
"""
from __future__ import annotations

import numpy as np
from scipy import signal


# --------------------------------------------------------------------------- EQ
def biquad(kind: str, f0: float, sr: int, q: float = 0.707, gain_db: float = 0.0):
    """Return normalised (b, a) for an RBJ biquad.

    kind: lowpass | highpass | bandpass | notch | peak | lowshelf | highshelf
    """
    f0 = float(np.clip(f0, 10.0, 0.49 * sr))
    w0 = 2 * np.pi * f0 / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    A = 10 ** (gain_db / 40)
    if kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bandpass":
        b = [alpha, 0, -alpha]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1, -2 * cw, 1]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]; a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind in ("lowshelf", "highshelf"):
        s = 1 if kind == "lowshelf" else -1
        sq = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - s * (A - 1) * cw + sq),
             s * 2 * A * ((A - 1) - s * (A + 1) * cw),
             A * ((A + 1) - s * (A - 1) * cw - sq)]
        a = [(A + 1) + s * (A - 1) * cw + sq,
             -s * 2 * ((A - 1) + s * (A + 1) * cw),
             (A + 1) + s * (A - 1) * cw - sq]
    else:
        raise ValueError(kind)
    b, a = np.asarray(b, float), np.asarray(a, float)
    return b / a[0], a / a[0]


def filt(x, kind, f0, sr, q=0.707, gain_db=0.0):
    b, a = biquad(kind, f0, sr, q, gain_db)
    return signal.lfilter(b, a, x)


def eq(x, sr, bands):
    """Apply a list of (kind, f0, q, gain_db) bands in series, like a channel-strip EQ."""
    for kind, f0, q, g in bands:
        x = filt(x, kind, f0, sr, q, g)
    return x


def random_channel_eq(rng, sr):
    """A plausible random channel EQ: high-pass, one or two bells, optional shelf."""
    bands = [("highpass", rng.uniform(40, 120), 0.707, 0.0)]
    for _ in range(rng.integers(1, 3)):
        bands.append(("peak", np.exp(rng.uniform(np.log(150), np.log(6000))),
                      rng.uniform(0.5, 2.5), rng.uniform(-6, 6)))
    if rng.random() < 0.5:
        bands.append(("highshelf", rng.uniform(3000, 8000), 0.707, rng.uniform(-4, 4)))
    return bands


# ------------------------------------------------------------------- dynamics
def compressor(x, sr, threshold_db=-18.0, ratio=4.0, attack_ms=5.0, release_ms=80.0,
               makeup_db=0.0, block=32):
    """Feed-forward peak compressor with attack/release smoothing (block-rate detector)."""
    n = len(x)
    nb = int(np.ceil(n / block))
    pad = np.zeros(nb * block); pad[:n] = np.abs(x)
    peak = pad.reshape(nb, block).max(1)
    lvl = 20 * np.log10(peak + 1e-9)
    over = np.maximum(lvl - threshold_db, 0.0)
    target_gr = over * (1 - 1 / ratio)                       # dB of gain reduction wanted
    a_att = np.exp(-block / (sr * attack_ms / 1000))
    a_rel = np.exp(-block / (sr * release_ms / 1000))
    gr = np.empty(nb); g = 0.0
    for i, t in enumerate(target_gr):
        coef = a_att if t > g else a_rel
        g = coef * g + (1 - coef) * t
        gr[i] = g
    gain_db = -gr + makeup_db
    centers = (np.arange(nb) + 0.5) * block
    gains = 10 ** (np.interp(np.arange(n), centers, gain_db) / 20)
    return x * gains


def limiter(x, ceiling=0.98):
    """Smooth tanh safety limiter."""
    return ceiling * np.tanh(x / ceiling)


# --------------------------------------------------------------------- space
def synthetic_rir(sr, rt60, rng, direct_delay=0, n_early=6, early_ms=(3, 40), tail_lp=6000.0):
    """Exponentially decaying noise tail + discrete early reflections. Direct path = 1."""
    n = int(max(rt60, 0.05) * sr) + direct_delay + 1
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail = filt(tail, "lowpass", tail_lp, sr)
    tail[: direct_delay + int(early_ms[0] * sr / 1000)] = 0.0
    rir = 0.08 * tail
    rir[direct_delay] += 1.0
    for _ in range(n_early):
        k = direct_delay + int(rng.uniform(*early_ms) * sr / 1000)
        if k < n:
            rir[k] += rng.choice([-1, 1]) * rng.uniform(0.15, 0.6)
    return rir


def convolve(x, h):
    return signal.fftconvolve(x, h)[: len(x)]


def reverb(x, sr, rt60, wet, rng):
    rir = synthetic_rir(sr, rt60, rng)
    rir[0] = 0.0                                             # wet path only
    w = convolve(x, rir)
    w *= np.sqrt(np.sum(x ** 2) + 1e-12) / np.sqrt(np.sum(w ** 2) + 1e-12)
    return (1 - wet) * x + wet * w


def delay(x, samples, gain=1.0):
    y = np.zeros_like(x)
    if samples < len(x):
        y[samples:] = x[: len(x) - samples]
    return gain * y


# --------------------------------------------------------------------- noise
def pink_noise(n, rng):
    """Voss-McCartney-ish pink noise via 1/f spectral shaping."""
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X)); f[0] = 1
    y = np.fft.irfft(X / np.sqrt(f), n)
    return y / (np.std(y) + 1e-12)


def mains_hum(n, sr, rng):
    f = rng.choice([50.0, 60.0]); t = np.arange(n) / sr
    y = sum(rng.uniform(0.2, 1.0) / k * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28))
            for k in (1, 2, 3, 5))
    return y / (np.std(y) + 1e-12)


def rms(x):
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def set_rms_db(x, db):
    return x * (10 ** (db / 20) / rms(x))
