"""Live-stage vocal-mic scenes: direct vocal + instrument bleed + monitor feedback + noise.

The vocal mic signal is modelled as

    s(t)  = v_direct(t) + bleed(t) + noise(t)
    m(t)  = s(t) + (h_fb * g*sat(m))(t)       # closed acoustic loop: mic -> monitor -> mic

where h_fb is the monitor-wedge-to-mic path (speaker band-limit + distance delay + room)
and g is the loop gain relative to the howl threshold (0.90-1.30).  The monitor amplifier
saturates, so loops above unity howl at a bounded level, as a real stage does. The training target is v_direct.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from . import fx

SCENE_TYPES = ("bleed", "feedback", "bleed+feedback")


@dataclass
class Scene:
    mix: np.ndarray
    target: np.ndarray
    meta: dict = field(default_factory=dict)


def feedback_path(sr, rng):
    """Monitor wedge -> vocal mic impulse response, normalised so the loop sits exactly at
    the edge of howling for g = 1: max over f of Re H(f) = 1 (the Nyquist curve touches +1)."""
    dist_m = rng.uniform(1.0, 3.5)
    d = int(dist_m / 343.0 * sr)
    h = fx.synthetic_rir(sr, rng.uniform(0.1, 0.25), rng, direct_delay=d, n_early=4, early_ms=(2, 20))
    h = h * rng.uniform(0.6, 1.0)
    # wedge + mic response: band-limited with a couple of resonances
    h = fx.filt(h, "highpass", rng.uniform(100, 200), sr)
    h = fx.filt(h, "lowpass", rng.uniform(4000, 7000), sr)
    for _ in range(2):
        h = fx.filt(h, "peak", rng.uniform(400, 4000), sr, q=rng.uniform(2, 6), gain_db=rng.uniform(3, 9))
    h[:d] = 0.0                                   # filters must not leak energy before the delay
    H = np.fft.rfft(h, 1 << 16)
    return h / H.real.max(), d


def acoustic_feedback(s, h, loop_gain, sat_level=None):
    """Closed-loop mic signal m = s + h * (g * sat(m)).

    h[0..d-1] = 0, so the loop is a strict recursion.  With sat_level=None the loop is
    linear (must have g*max|H| < 1 to be stable).  With a saturation level the
    monitor amplifier clips softly (A*tanh(x/A)), so a loop gain above unity produces
    a real, self-sustaining howl that settles at the amplifier's limit, as on stage.
    """
    if sat_level is None:
        a = np.concatenate([[1.0], -loop_gain * h[1:]])
        return signal.lfilter([1.0], a, s)
    d = int(np.argmax(h != 0)); B = max(1, d); L = len(h)
    n = len(s)
    spk = np.zeros(n + L + 1)                         # spk[L + i] = speaker feed at sample i
    win = np.lib.stride_tricks.sliding_window_view(spk, L)
    hr = h[::-1].copy()
    m = np.zeros(n)
    for st in range(0, n, B):
        en = min(st + B, n)
        fb = win[st + 1:en + 1] @ hr                  # only speaker samples <= i - d are used
        m[st:en] = s[st:en] + fb
        spk[L + st:L + en] = loop_gain * sat_level * np.tanh(m[st:en] / sat_level)
    return m


def bleed_from(stems, sr, rng):
    """Spill of each backline source into the vocal mic: distance delay, off-axis tone, room."""
    out = 0.0
    for s in stems:
        d = int(rng.uniform(1.5, 6.0) / 343.0 * sr)
        x = fx.delay(s, d)
        x = fx.filt(x, "highshelf", 3000, sr, gain_db=rng.uniform(-9, -2))   # off-axis rejection
        x = fx.reverb(x, sr, rng.uniform(0.3, 1.0), rng.uniform(0.2, 0.6), rng)
        out = out + x
    return out


def make_scene(vocal, accomp_stems, sr, rng, kind=None, vbr_db=None):
    """Build one scene. vocal: dry source; accomp_stems: list of arrays (same length)."""
    n = len(vocal)
    kind = kind or rng.choice(SCENE_TYPES, p=[0.35, 0.25, 0.40])
    # direct vocal at the mic: proximity bass boost + capsule presence peak (part of target)
    v = fx.filt(vocal, "lowshelf", 200, sr, gain_db=rng.uniform(0, 6))
    v = fx.filt(v, "peak", rng.uniform(3000, 6000), sr, q=1.0, gain_db=rng.uniform(0, 4))
    v = fx.set_rms_db(v, -20.0)
    s = v.copy()
    meta = {"kind": str(kind)}
    if "bleed" in kind:
        b = bleed_from(accomp_stems, sr, rng)
        vbr = rng.uniform(-6, 12) if vbr_db is None else vbr_db        # vocal-to-bleed ratio
        b = fx.set_rms_db(b, -20.0 - vbr)
        s = s + b
        meta["vbr_db"] = float(vbr)
    noise = fx.set_rms_db(fx.pink_noise(n, rng), rng.uniform(-62, -45))
    if rng.random() < 0.3:
        noise += fx.set_rms_db(fx.mains_hum(n, sr, rng), rng.uniform(-60, -45))
    s = s + noise
    if "feedback" in kind:
        h, d = feedback_path(sr, rng)
        # g = 1 is the onset of instability: g < 1 rings (closer to 1 = longer), g > 1 howls
        g = rng.uniform(0.90, 1.30)
        sat = 10 ** ((-20 + rng.uniform(6, 14)) / 20)                    # monitor amp limit vs vocal level
        s = acoustic_feedback(s, h, g, sat_level=sat)
        meta.update(loop_gain=float(g), fb_delay_ms=1000 * d / sr)
    gain = 10 ** (rng.uniform(-10, 6) / 20)                              # input trim variation
    return Scene(mix=(s * gain).astype(np.float32), target=(v * gain).astype(np.float32), meta=meta)
