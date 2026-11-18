"""Procedural multitrack generator: a source-filter singing voice and a small band.

The point is unlimited, perfectly isolated stems (the ground truth a real stage
recording never has), so bleed and feedback can be added with known targets.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import fx

# (F1, F2, F3) in Hz and bandwidths, roughly from Peterson & Barney averages
VOWELS = {
    "a": (800, 1150, 2900), "e": (450, 1900, 2600), "i": (300, 2300, 3000),
    "o": (500, 900, 2500), "u": (350, 800, 2400), "ae": (650, 1700, 2500),
}
BW = (80, 100, 140)
SCALE = np.array([0, 2, 4, 5, 7, 9, 11])          # major scale degrees


def _midi_to_hz(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


def _phrase_notes(rng, dur, beat):
    """Random melody: (start, end, midi) tuples with rests between phrases."""
    notes, t = [], rng.uniform(0, 2 * beat)
    root = rng.integers(48, 62)                   # C3..D4 tonic
    deg = rng.integers(0, 7)
    while t < dur - 0.2:
        if rng.random() < 0.18:                    # rest
            t += beat * rng.choice([0.5, 1, 2]); continue
        length = beat * rng.choice([0.5, 1, 1, 1.5, 2])
        deg = int(np.clip(deg + rng.integers(-2, 3), 0, 13))
        midi = root + 12 * (deg // 7) + SCALE[deg % 7]
        notes.append((t, min(t + length * rng.uniform(0.85, 1.0), dur), midi))
        t += length
    return notes


def singing_voice(dur, sr, rng, bpm=None):
    """Additive source-filter singer with vibrato, jitter, vowel morphs and consonants."""
    n = int(dur * sr)
    beat = 60.0 / (bpm or rng.uniform(80, 140))
    notes = _phrase_notes(rng, dur, beat)
    f0 = np.zeros(n); amp = np.zeros(n)
    vowel_idx = np.zeros(n, int)
    names = list(VOWELS)
    cons = np.zeros(n)
    for (s, e, m) in notes:
        a, b = int(s * sr), int(e * sr)
        if b - a < 32:
            continue
        L = b - a
        tt = np.arange(L) / sr
        glide = np.exp(-tt / 0.04) * rng.uniform(-0.6, 0.1)            # scoop into the note
        vib = rng.uniform(0.15, 0.45) * np.sin(2 * np.pi * rng.uniform(4.8, 6.2) * tt) \
            * np.clip(tt / 0.25, 0, 1)
        f0[a:b] = _midi_to_hz(m + glide + vib)
        env = np.minimum(1, tt / rng.uniform(0.02, 0.06)) * np.minimum(1, (L / sr - tt) / 0.05)
        amp[a:b] = np.clip(env, 0, 1) * rng.uniform(0.6, 1.0)
        vowel_idx[a:b] = rng.integers(0, len(names))
        if rng.random() < 0.6:                                           # consonant onset
            cl = int(rng.uniform(0.02, 0.06) * sr)
            burst = rng.standard_normal(cl) * np.exp(-np.arange(cl) / (0.3 * cl))
            burst = fx.filt(burst, "bandpass", rng.uniform(2500, 6500), sr, q=1.0)
            st = max(0, a - cl // 2)
            cons[st:st + cl] += 0.5 * burst[: len(cons[st:st + cl])]
    voiced = f0 > 0
    f0 = np.where(voiced, f0, 150.0)
    f0 *= 1 + 0.004 * signal.lfilter([0.02], [1, -0.98], rng.standard_normal(n))  # jitter
    phase = 2 * np.pi * np.cumsum(f0) / sr
    # smooth formant trajectories between vowels
    F = np.array([VOWELS[names[i]] for i in vowel_idx], float)
    F = signal.filtfilt(*signal.butter(2, 12 / (sr / 2)), F, axis=0)
    F *= rng.uniform(0.9, 1.15)                                          # vocal-tract length
    y = np.zeros(n)
    kmax = int((0.45 * sr) / 80)
    tilt = rng.uniform(1.0, 1.6)
    for k in range(1, kmax + 1):
        fk = k * f0
        live = fk < 0.45 * sr
        if not live.any():
            break
        env = 0.02
        for j in range(3):
            env = env + (1.0 / (j + 1)) / (1 + ((fk - F[:, j]) / (BW[j] / 2)) ** 2)
        y += live * env * np.sin(k * phase) / k ** tilt
    y *= amp
    y += rng.uniform(0.01, 0.04) * fx.filt(rng.standard_normal(n), "highpass", 2000, sr) * amp  # breath
    y += cons
    return y / (np.max(np.abs(y)) + 1e-9)


# ----------------------------------------------------------------- the band
def _env(L, sr, attack, decay):
    t = np.arange(L) / sr
    return np.minimum(1, t / max(attack, 1e-4)) * np.exp(-t / decay)


def drums(dur, sr, rng, beat):
    n = int(dur * sr); y = np.zeros(n)
    step = beat / 4
    pat_k = rng.random(16) < np.array([.9, 0, .1, 0, .1, 0, .3, 0, .8, 0, .2, .2, .1, 0, .3, 0])
    pat_s = rng.random(16) < np.array([0, 0, 0, 0, .95, 0, 0, .1, 0, 0, 0, 0, .95, 0, .1, .2])
    pat_h = rng.random(16) < 0.7
    i = 0; t = 0.0
    while t < dur:
        a = int(t * sr); s = i % 16
        if pat_k[s]:
            L = int(0.35 * sr); tt = np.arange(L) / sr
            f = 45 + 110 * np.exp(-tt / 0.03)
            hit = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-tt / 0.12)
            y[a:a + L] += hit[: len(y[a:a + L])]
        if pat_s[s]:
            L = int(0.2 * sr)
            nz = fx.filt(rng.standard_normal(L), "bandpass", 2500, sr, q=0.6) * _env(L, sr, .001, .06)
            tone = np.sin(2 * np.pi * 185 * np.arange(L) / sr) * _env(L, sr, .001, .04)
            hit = 0.8 * nz + 0.4 * tone
            y[a:a + L] += hit[: len(y[a:a + L])]
        if pat_h[s]:
            L = int(0.06 * sr)
            hit = 0.25 * fx.filt(rng.standard_normal(L), "highpass", 7000, sr) * _env(L, sr, .0005, .015)
            y[a:a + L] += hit[: len(y[a:a + L])]
        i += 1; t += step
    return y


def _chord_roots(rng, n_bars):
    root = rng.integers(36, 48)
    prog = rng.choice([[0, 5, 7, 5], [0, 9, 5, 7], [0, 7, 9, 5], [0, 4, 5, 7]])
    return [root + prog[b % 4] for b in range(n_bars)]


def bass(dur, sr, rng, beat, roots):
    n = int(dur * sr); y = np.zeros(n); bar = 4 * beat
    for b, r in enumerate(roots):
        for e in range(8):
            t = b * bar + e * beat / 2
            if t >= dur or rng.random() < 0.15:
                continue
            a = int(t * sr); L = int(beat / 2 * 0.9 * sr)
            m = r + (12 if rng.random() < 0.15 else 0)
            f = _midi_to_hz(m); tt = np.arange(L) / sr
            saw = 2 * ((f * tt) % 1) - 1
            note = saw * _env(L, sr, .005, .25)
            y[a:a + L] += note[: len(y[a:a + L])]
    return fx.filt(y, "lowpass", rng.uniform(500, 1200), sr, q=0.9)


def keys(dur, sr, rng, beat, roots):
    n = int(dur * sr); y = np.zeros(n); bar = 4 * beat
    for b, r in enumerate(roots):
        a = int(b * bar * sr)
        if a >= n:
            break
        L = min(int(bar * sr), n - a); tt = np.arange(L) / sr
        third = 4 if rng.random() < 0.6 else 3
        chord = np.zeros(L)
        for m in (r + 24, r + 24 + third, r + 31):
            f = _midi_to_hz(m)
            for h, ah in ((1, 1.0), (2, 0.4), (3, 0.2), (4, 0.1)):
                chord += ah * np.sin(2 * np.pi * f * h * tt) * np.exp(-tt * h / 0.9)
        y[a:a + L] += chord * _env(L, sr, .004, 2.0)
    return y


def guitar(dur, sr, rng, beat, roots):
    """Karplus-Strong strums."""
    n = int(dur * sr); y = np.zeros(n)
    t = 0.0; k = 0
    while t < dur:
        r = roots[min(int(t / (4 * beat)), len(roots) - 1)]
        for j, m in enumerate((r + 12, r + 19, r + 24, r + 28)):
            f = _midi_to_hz(m); N = max(2, int(sr / f))
            a = int((t + j * 0.012) * sr); L = min(int(1.2 * sr), n - a)
            if L <= N + 2:
                continue
            x = np.zeros(L); x[:N] = rng.uniform(-1, 1, N)
            den = np.zeros(N + 2); den[0] = 1; den[N] = -0.497; den[N + 1] = -0.497
            y[a:a + L] += signal.lfilter([1.0], den, x)
        k += 1; t += beat * rng.choice([1, 2])
    return y


def band(dur, sr, rng, bpm=None):
    """Return a dict of mixed-ready instrument stems (each through a channel EQ + comp)."""
    beat = 60.0 / (bpm or rng.uniform(80, 140))
    roots = _chord_roots(rng, int(np.ceil(dur / (4 * beat))) + 1)
    makers = {"drums": lambda: drums(dur, sr, rng, beat), "bass": lambda: bass(dur, sr, rng, beat, roots),
              "keys": lambda: keys(dur, sr, rng, beat, roots), "guitar": lambda: guitar(dur, sr, rng, beat, roots)}
    chosen = [k for k in makers if rng.random() < 0.75] or ["drums", "keys"]
    stems = {}
    for k in chosen:
        s = makers[k]()
        if np.max(np.abs(s)) < 1e-6:
            continue
        s = fx.eq(s, sr, fx.random_channel_eq(rng, sr))
        s = fx.compressor(fx.set_rms_db(s, -20), sr, threshold_db=rng.uniform(-28, -16),
                          ratio=rng.uniform(2, 6), attack_ms=rng.uniform(2, 20), release_ms=rng.uniform(50, 200))
        stems[k] = fx.set_rms_db(s, rng.uniform(-26, -18))
    return stems
